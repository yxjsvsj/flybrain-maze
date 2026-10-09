"""OFF/ON 建图离线回放：同一份录制数据，旧（车中心射线） vs 新（逐原点射线）建图差异。

读一份含 `odom_x, odom_y, theta_deg` + `tof_{L,FL,F,FR,R}_mm` + `tof_*_st` 的 CSV
（例如 `loop/straight_run.py` 的 log），分别喂给：

  旧: 冻结 `OccupancyMemory.update(x, y, car_theta + sensor_angles, dists)`
      —— 所有射线从**车中心**、用 ToF 距离（cells）。
  新: `ToFMemory.update_rays(...)` + `loop.tof_geometry.frame_to_cell_rays`
      —— 每路从**真实传感器光学窗口**。

输出：
  * 两张占用图的差异（新图多出/少掉了哪些格）；
  * 指定传感器（默认 F）在两张图里各自落在哪个格 + 对应**世界坐标(米)**，
    便于和现场卷尺实测位置比对（差多少 cm）。

**纯离线，不连硬件、不动车。**
"""
from __future__ import annotations

import argparse
import csv
import math
import sys

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.tof_geometry import frame_to_cell_rays, load_geometry, rotate
from loop.memory_tof import ToFMemory
from nav.memory import OCCUPIED, OccupancyMemory

MIN_DIST_M = 0.05


class _Frame:
    def __init__(self, ranges, status, seq):
        self.ranges = dict(ranges)
        self.status = dict(status)
        self.seq = seq


def _mem(cls, w, h, max_range):
    return cls(w, h, max_range, unknown_cost=4.0, lookahead=0.9,
               arrive_dist=0.6, off_path_tol=1.5, free_conflict_threshold=3)


def replay(path, geometry, mpc, watch="F", w=60, h=60, pose=(30.0, 30.0, 0.0)):
    geo = load_geometry(geometry)
    mem_old = _mem(OccupancyMemory, w, h, 6.0)
    mem_new = _mem(ToFMemory, w, h, 6.0)
    angles = {n: math.radians(DEFAULT_ANGLES_DEG[n]) for n in ORDER}
    n_rows = 0
    old_hit = new_hit = None
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            # 位姿：有 odom 列就用（cells）；静态诊断（--odom-port 0）没有 -> 用固定 pose(cells)
            if r.get("odom_x") not in (None, ""):
                try:
                    x = float(r["odom_x"]); y = float(r["odom_y"])
                    th = math.radians(float(r["theta_deg"]))
                except (KeyError, ValueError):
                    continue
            else:
                x, y, th = pose[0], pose[1], math.radians(pose[2])

            ranges, status = {}, {}
            for n in ORDER:
                mm = r.get(f"tof_{n}_mm")
                if mm in (None, ""):
                    mm = r.get(f"{n}_mm", "")          # 静态诊断的列名
                ranges[n] = float(mm) if mm not in ("", None) else None
                st = r.get(f"tof_{n}_st") or r.get(f"{n}_status", "")
                status[n] = st or ""
            fr = _Frame(ranges, status, n_rows + 1)

            # --- OLD: 车中心射线 + ToF 距离(cells)，只取 VALID/TOO_NEAR ---
            idx, dists = [], []
            for i, n in enumerate(ORDER):
                st = status[n]
                if st == "VALID" and ranges[n] is not None:
                    idx.append(i); dists.append(max(MIN_DIST_M / mpc, (ranges[n] / 1000.0) / mpc))
                elif st == "TOO_NEAR":
                    idx.append(i); dists.append(MIN_DIST_M / mpc)
            if idx:
                import numpy as np
                a = np.array([th + angles[ORDER[i]] for i in idx], dtype=float)
                d = np.array(dists, dtype=float)
                mem_old.update(x, y, a, d)

            # --- NEW: 逐原点射线 ---
            rays = frame_to_cell_rays(fr, x, y, th, geo, mpc)
            mem_new.update_rays(x, y, rays, tof_seq=fr.seq)

            if watch in status and status[watch] == "VALID" and ranges[watch] is not None:
                rng_m = ranges[watch] / 1000.0
                ext = geo[watch]
                dx, dy = rotate(ext.x_m, ext.y_m, th)
                wa = th + ext.yaw_rad
                # 相对后轴原点（世界轴）的命中坐标，米
                o_rx, o_ry = rng_m * math.cos(wa), rng_m * math.sin(wa)
                n_rx, n_ry = dx + rng_m * math.cos(wa), dy + rng_m * math.sin(wa)
                o_cx = int(math.floor((x + o_rx / mpc)))
                o_cy = int(math.floor((y + o_ry / mpc)))
                n_cx = int(math.floor((x + n_rx / mpc)))
                n_cy = int(math.floor((y + n_ry / mpc)))
                old_hit = (o_cx, o_cy, o_rx, o_ry)
                new_hit = (n_cx, n_cy, n_rx, n_ry)
            n_rows += 1

    old_occ = set(zip(*[a.tolist() for a in (mem_old.known == OCCUPIED).nonzero()]))
    new_occ = set(zip(*[a.tolist() for a in (mem_new.known == OCCUPIED).nonzero()]))
    only_new = sorted(new_occ - old_occ)
    only_old = sorted(old_occ - new_occ)
    print(f"[replay] rows={n_rows}  old_occupied={len(old_occ)}  new_occupied={len(new_occ)}")
    print(f"[replay] only-in-NEW (first 20): {only_new[:20]}")
    print(f"[replay] only-in-OLD (first 20): {only_old[:20]}")
    if old_hit and new_hit:
        print(f"[replay] {watch} last VALID hit (relative to rear axle):")
        print(f"    OLD cell=({old_hit[0]},{old_hit[1]})  rel=(x {old_hit[2]:.3f}, y {old_hit[3]:.3f}) m")
        print(f"    NEW cell=({new_hit[0]},{new_hit[1]})  rel=(x {new_hit[2]:.3f}, y {new_hit[3]:.3f}) m")
        dxm = new_hit[2] - old_hit[2]; dym = new_hit[3] - old_hit[3]
        print(f"    NEW-OLD delta = ({dxm*100:+.1f}, {dym*100:+.1f}) cm  "
              f"(|d|={math.hypot(dxm, dym)*100:.1f} cm)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OFF/ON 建图离线回放（不动车）")
    ap.add_argument("csv_path")
    ap.add_argument("--tof-geometry", default="hardware/tof_geometry.json")
    ap.add_argument("--meters-per-cell", type=float, default=0.40)
    ap.add_argument("--watch", default="F", choices=list(ORDER))
    ap.add_argument("--pose", default="30,30,0",
                    help="无 odom 列时的固定地图位姿 X,Y,THETA_DEG (cells)")
    args = ap.parse_args(argv)
    px, py, pth = (float(v) for v in args.pose.split(","))
    return replay(args.csv_path, args.tof_geometry, args.meters_per_cell, args.watch,
                  pose=(px, py, pth))


if __name__ == "__main__":
    sys.exit(main())
