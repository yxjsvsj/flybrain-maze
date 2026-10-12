"""P2d-9 任务二/三回归：动态五路 ToF 环境 + 消融（fake brain，无硬件）。"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.decode import Decoder
from loop.dynamic_ablation import main as abl_main
from loop.dynamic_tof_sim import frame_from_pose, load_rig, run_dynamic
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup
from world.car import DiffDriveCar
from world.maze import Maze

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def _setup():
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="fake", map_name="gen", device="cpu", make_decoder=False, verbose=False)
    cfg.decoder.mode = "dn"
    brain.reset(64)
    dn = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=0.40)
    return cfg, brain, groups, dec, enc


def main() -> int:
    print("=========== dynamic_tof_sim + ablation regression ===========")
    cfg, brain, groups, dec, enc = _setup()
    rig = load_rig()
    check("rig has 5 sensors (L/FL/F/FR/R)", all(n in rig for n in ORDER))

    # frame_from_pose: 窄道内前方有近墙 -> VALID
    maze = Maze(["#####", "#S..#", "#...#", "#####"])
    fr = frame_from_pose(maze, 1.5, 1.5, 0.0, 0.0, 0, rig=rig, meters_per_cell=0.40, max_valid_mm=1200.0)
    check("frame_from_pose returns TofFrame", fr.seq == 0 and set(fr.status) == set(ORDER))
    check("straight-ahead F is VALID (near wall ahead)", fr.status["F"] == "VALID")
    check("behind R/FR likely NO_TARGET (range cap)",
          fr.status["R"] in ("NO_TARGET", "VALID"))

    # P2-only 一致性：前伸 F 传感器原点可能落入虚拟墙格 -> 记为 TOO_NEAR（非精确命中）
    m_wall = Maze(["####", "#..#", "#..#", "####"])
    fr2 = frame_from_pose(m_wall, 2.9, 1.5, 0.0, 0.0, 0, rig=rig,
                          meters_per_cell=0.40, max_valid_mm=1200.0)
    check("sensor origin inside wall -> TOO_NEAR (not exact hit)",
          fr2.status["F"] == "TOO_NEAR")

    # run_dynamic 在直道里（dn 模式）应向前推进并记录
    maze2 = Maze(["########", "#S....G#", "#......#", "########"])
    brain.reset(64)
    dec2 = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt,
                   dn_idx=np.asarray(brain.cells(["descending_neuron"])))
    enc2 = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                              sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                              car_radius=cfg.car.radius, meters_per_cell=0.40)
    sx, sy, sth = maze2.start
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    traj, s = run_dynamic(cfg, maze2, car, brain, enc2, dec2, memory_on=False,
                          seconds=3.0, tof_hz=10.0, rig=rig, seed=64)
    check("run_dynamic produced steps", s["steps"] == len(traj) and s["steps"] > 0)
    for k in ("reached_goal", "path_len_cells", "collision_steps", "collision_events",
              "path_efficiency", "replans", "timeout", "best_goal_dist",
              "min_time_lower_bound_s", "bfs_cells_over_driven_cells"):
        check(f"summary has {k}", k in s)
    check("tracks cells not metres (path_len_cells)", "path_len_cells" in s and "path_len_m" not in s)

    # ablation main (fake) over 1 maze seed, all 4 conditions
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = abl_main(["--brain", "fake", "--device", "cpu", "--mode", "dn",
                       "--seconds", "2", "--maze-seeds", "1000"])
    out = buf.getvalue()
    check("ablation main rc 0", rc == 0)
    check("ablation table has all conds", all(f" {c} " in out for c in ("A", "B", "C", "D")))

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
