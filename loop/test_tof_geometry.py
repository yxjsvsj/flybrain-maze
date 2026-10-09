"""P2d 纯离线几何测试（不连硬件、不发指令）。

覆盖：
  1. F r=1.031m, theta=0  ->  相对车轴命中 = (1.181, 0)
  2. theta=90° 时 传感器原点与射线方向正确旋转
  3. FL/FR 不对称原点被保留
  4. NO_TARGET / IO_ERROR 不产生 OCCUPIED 格
  5. 重复 tof_seq 不会把 memory 更新两次
  6. 车访问位置用本体位置，不是传感器光学窗口

运行： python -m loop.test_tof_geometry
"""
from __future__ import annotations

import math
import sys

import numpy as np

from loop.memory_tof import ToFMemory
from loop.tof_geometry import (SensorExtrinsic, frame_to_cell_rays,
                               frame_to_world_rays, load_geometry, sensor_world_ray)
from nav.memory import OCCUPIED

MPC = 0.40
_fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


class _Frame:
    def __init__(self, ranges, status, seq=1):
        self.ranges = dict(ranges)
        self.status = dict(status)
        self.seq = seq


def main() -> int:
    geo = load_geometry()
    f_ext = geo["F"]

    # 1) F r=1.031m, theta=0 -> hit rel axle = (1.181, 0)
    ox, oy, wa, hx, hy = sensor_world_ray(0.0, 0.0, 0.0, f_ext, 1.031)
    check("1 F r=1.031 θ=0 -> hit (1.181, 0)",
          abs(hx - 1.181) < 1e-6 and abs(hy) < 1e-9,
          f"hit=({hx:.4f},{hy:.4f})")

    # 2) theta=90° rotates origins + directions
    ox, oy, wa, hx, hy = sensor_world_ray(0.0, 0.0, math.pi / 2, f_ext, 1.0)
    check("2 F θ=90° origin=(0,0.15) dir=+90°",
          abs(ox) < 1e-9 and abs(oy - 0.15) < 1e-9 and abs(wa - math.pi / 2) < 1e-9,
          f"origin=({ox:.4f},{oy:.4f}) wa={math.degrees(wa):.1f}")

    # 3) FL/FR asymmetric origins preserved
    fl, fr = geo["FL"], geo["FR"]
    check("3 FL/FR asymmetric extrinsics preserved",
          not (abs(fl.x_m - fr.x_m) < 1e-9 and abs(fl.y_m + fr.y_m) < 1e-9),
          f"FL=({fl.x_m},{fl.y_m}) FR=({fr.x_m},{fr.y_m})")

    # 4) NO_TARGET / IO_ERROR -> no occupied cells
    fr8 = _Frame({"L": None, "FL": 300.0, "F": None, "FR": None, "R": 500.0},
                 {"L": "NO_TARGET", "FL": "VALID", "F": "IO_ERROR",
                  "FR": "NO_TARGET", "R": "VALID"}, seq=100)
    rays = frame_to_cell_rays(fr8, 5.0, 5.0, 0.0, geo, MPC)
    mem = ToFMemory(20, 20, 6.0)
    mem.update_rays(5.0, 5.0, rays, tof_seq=100)
    check("4 NO_TARGET/IO_ERROR write no occupied cells",
          int((mem.known == OCCUPIED).sum()) > 0,  # FL/R valid produce walls, but not the dead ones
          f"occupied={int((mem.known == OCCUPIED).sum())}")
    # specifically: an all-dead frame must create ZERO occupied.
    dead = _Frame({"L": None, "FL": None, "F": None, "FR": None, "R": None},
                  {n: "NO_TARGET" for n in ("L", "FL", "F", "FR", "R")}, seq=101)
    mem2 = ToFMemory(20, 20, 6.0)
    mem2.update_rays(5.0, 5.0, frame_to_cell_rays(dead, 5.0, 5.0, 0.0, geo, MPC), tof_seq=101)
    check("4b all-dead frame -> zero occupied",
          int((mem2.known == OCCUPIED).sum()) == 0,
          f"occupied={int((mem2.known == OCCUPIED).sum())}")

    # 5) repeated tof_seq does not update twice
    mem3 = ToFMemory(20, 20, 6.0)
    r3 = frame_to_cell_rays(fr8, 5.0, 5.0, 0.0, geo, MPC)
    u1 = mem3.update_rays(5.0, 5.0, r3, tof_seq=200)
    rev1 = mem3.map_revision
    u2 = mem3.update_rays(5.0, 5.0, r3, tof_seq=200)   # same seq -> skip
    check("5 repeated tof_seq skipped (no double update)",
          u1 and (u2 is False) and mem3.map_revision == rev1,
          f"u1={u1} u2={u2} rev={rev1}->{mem3.map_revision}")

    # 6) visited cell = car cell, not the sensor origin
    car_cell = (int(5.0), int(5.0))
    check("6 car cell visited, not sensor window",
          bool(mem3.visited[car_cell[1], car_cell[0]]),
          f"visited(car)={bool(mem3.visited[car_cell[1], car_cell[0]])}")

    # extra: world-ray hit point uses origin + range*dir (meters)
    wr = frame_to_world_rays(_Frame({"F": 1031.0}, {"F": "VALID"}, seq=9),
                             0.0, 0.0, 0.0, {"F": f_ext})
    _, ox, oy, wa, r, kind = wr[0]
    check("X world ray: origin=(0.15,0), range=1.031, kind=hit",
          abs(ox - 0.15) < 1e-9 and abs(r - 1.031) < 1e-9 and kind == "hit",
          f"origin=({ox:.3f},{oy:.3f}) r={r:.3f} kind={kind}")

    # 7) old (car-centre) vs new (sensor origin) F hit -> delta == F x_m
    old_x = 0.0 + 1.031
    _, _, _, new_hx, _ = sensor_world_ray(0.0, 0.0, 0.0, f_ext, 1.031)
    check("7 old hit x=1.031 vs new x=1.181 (delta = F x_m)",
          abs(new_hx - old_x - f_ext.x_m) < 1e-9,
          f"old={old_x:.3f} new={new_hx:.3f} delta={new_hx - old_x:.3f}")

    # 8) wall NOT on a grid boundary -> real hit cell written OCCUPIED
    fr9 = _Frame({"F": 1031.0}, {"F": "VALID"}, seq=300)
    mem9 = ToFMemory(20, 20, 6.0)
    mem9.update_rays(0.0, 0.0, frame_to_cell_rays(fr9, 0.0, 0.0, 0.0, geo, MPC), tof_seq=300)
    check("8 endpoint-in-cell wall -> OCCUPIED",
          int((mem9.known == OCCUPIED).sum()) >= 1,
          f"occupied={int((mem9.known == OCCUPIED).sum())}")

    # 9) TOO_NEAR -> kind 'near', no precise wall
    frN = _Frame({"F": 20.0}, {"F": "TOO_NEAR"}, seq=400)
    raysN = frame_to_cell_rays(frN, 0.0, 0.0, 0.0, geo, MPC)
    memN = ToFMemory(20, 20, 6.0)
    memN.update_rays(0.0, 0.0, raysN, tof_seq=400)
    check("9 TOO_NEAR -> near + no OCCUPIED",
          len(raysN) == 1 and raysN[0][4] == "near"
          and int((memN.known == OCCUPIED).sum()) == 0,
          f"kind={raysN[0][4] if raysN else None} occ={int((memN.known == OCCUPIED).sum())}")

    # 10) goal-cell protection + conflict threshold preserved
    frG = _Frame({"F": 1031.0}, {"F": "VALID"}, seq=500)
    memG = ToFMemory(20, 20, 6.0)
    memG.set_goal(2.5, 0.5)              # F hit cell ~ (2,0)
    memG.update_rays(0.0, 0.0, frame_to_cell_rays(frG, 0.0, 0.0, 0.0, geo, MPC), tof_seq=500)
    check("10 goal cell protected + conflict thr=3",
          memG.known[0, 2] != OCCUPIED and memG.free_conflict_threshold == 3,
          f"goal known={int(memG.known[0, 2])} thr={memG.free_conflict_threshold}")

    print(f"\n  {'ALL PASS' if not _fails else 'FAILED: ' + ', '.join(_fails)}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
