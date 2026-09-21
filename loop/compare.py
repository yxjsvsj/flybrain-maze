"""假脑 vs 真脑 的对比表 —— 每次调参后的回归测试。

    python -m loop.compare
    python -m loop.compare --brains real --maps track corridor
    python -m loop.compare --steps 3000 --csv compare.csv

每种脑只构建一次（真脑加载 166,700 神经元要十几秒），然后在地图之间复用。
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from brain.neurons import resolve
from brain.wrap import make_brain
from config import Config, apply_preset
from loop.decode import Decoder
from loop.encode import Encoder
from loop.run import CHECK_SIM_SECONDS, DEFAULT_DT, make_maze, run
from world.car import DiffDriveCar
from world.maze import MAPS

FIELDS = ["map", "brain", "steps_done", "sim_time", "distance", "mean_speed",
          "coverage", "best_goal_dist", "reached_goal", "time_to_goal",
          "path_efficiency", "collision_events", "contact_ratio", "collisions",
          "escapes", "stalls", "front_safety_events", "replans",
          "brain_ms_per_step", "realtime_factor"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="假脑/真脑对比")
    ap.add_argument("--brains", nargs="+", choices=["fake", "real"], default=["fake", "real"])
    ap.add_argument("--maps", nargs="+", default=list(MAPS), help=f"内置地图 {sorted(MAPS)}")
    ap.add_argument("--sim-seconds", type=float, default=None,
                    help="仿真时长（秒）。两种脑的 dt 不同，用秒才能公平比较")
    ap.add_argument("--stop-on-goal", action="store_true", help="到终点立即结束")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--csv", default="", help="把结果写成 CSV")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印神经元组解析和逐秒日志")
    args = ap.parse_args(argv)

    rows = []
    for kind in args.brains:
        cfg = Config()
        cfg.brain.kind = kind
        cfg.brain.device = args.device
        cfg.brain.seed = args.seed
        cfg.brain.dt = DEFAULT_DT[kind]
        apply_preset(cfg, kind)
        sim_seconds = args.sim_seconds or CHECK_SIM_SECONDS[kind]
        steps = max(2, int(round(sim_seconds / cfg.brain.dt)))

        print(f"\n=== {kind} (dt={cfg.brain.dt * 1000:.0f} ms, "
              f"{sim_seconds:.0f}s = {steps} steps) ===")
        brain = make_brain(cfg.brain, verbose=args.verbose)
        groups = resolve(brain, verbose=args.verbose)

        for map_name in args.maps:
            maze = make_maze(map_name)
            sx, sy, sth = maze.start
            car = DiffDriveCar(sx, sy, sth, cfg.car)
            enc = Encoder(groups, cfg.encoder, brain.dt, car_max_speed=cfg.car.max_speed,
                          car_radius=cfg.car.radius)
            dn_idx = np.asarray(brain.cells(["descending_neuron"]))
            dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
            brain.reset(args.seed)

            stats = run(cfg, maze, car, brain, enc, dec, steps, verbose=args.verbose,
                        stop_on_goal=args.stop_on_goal)
            stats["map"] = map_name
            stats["brain"] = kind
            rows.append(stats)
            print(f"  {map_name:10s} dist={stats['distance']:6.1f} "
                  f"cov={stats['coverage'] * 100:5.1f}% hit={stats['collision_events']:3d} "
                  f"stall={stats['stalls']:3d} esc={stats['escapes']:3d} "
                  f"goal={stats['best_goal_dist']:4d}")

    print("\n" + "=" * 96)
    print(f"{'map':<11}{'brain':<7}{'dist':>7}{'speed':>7}{'cov%':>7}{'goal':>6}"
          f"{'hit':>5}{'stall':>6}{'esc':>5}{'ms/step':>9}{'realtime':>10}")
    print("-" * 96)
    for r in rows:
        goal = str(r["best_goal_dist"]) + ("*" if r["reached_goal"] else "")
        print(f"{r['map']:<11}{r['brain']:<7}{r['distance']:>7.1f}{r['mean_speed']:>7.3f}"
              f"{r['coverage'] * 100:>7.1f}{goal:>6}"
              f"{r['collision_events']:>5d}{r['stalls']:>6d}"
              f"{r['escapes']:>5d}{r['brain_ms_per_step']:>9.2f}{r['realtime_factor']:>10.2f}")
    print("=" * 96)
    print("goal = cells remaining to the goal (* = reached); -1 = map has no goal")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
