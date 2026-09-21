"""导航 benchmark：只用**未参与调参**的种子，输出成功率等指标。

    python -m loop.benchmark                      # 默认跑 real+memory 与 real 无记忆
    python -m loop.benchmark --controllers real_mem fake_mem
    python -m loop.benchmark --seeds 23 42 5

调参用 TUNE_SEEDS，最终结论只认 TEST_SEEDS。不要拿调参种子当测试结果。
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from brain.neurons import resolve
from brain.wrap import make_brain
from config import Config, apply_preset
from loop.decode import Decoder
from loop.encode import Encoder
from loop.run import DEFAULT_DT, make_maze, run
from world.car import DiffDriveCar

# 调参用（可以随便用这些 seed 试参数）
TUNE_SEEDS = [1, 3, 7, 11, 17]
# 最终测试用：绝不参与调参
TEST_SEEDS = [23, 42, 5, 31, 61, 71, 83, 97, 109, 127,
              131, 149, 163, 181, 197, 211, 227, 239, 251, 269]

CONTROLLERS = {
    "real_nomem": dict(kind="real", mode="dn", memory=0.0, brain_gain=1.0),
    "real_mem": dict(kind="real", mode="dn", memory=1.0, brain_gain=1.0),
    "real_readout_mem": dict(kind="real", mode="readout", memory=1.0, brain_gain=1.0,
                             readout="readout_real.npz"),
    "real_mem_nobrain": dict(kind="real", mode="dn", memory=1.0, brain_gain=0.0),
    "fake_mem": dict(kind="fake", mode="dn", memory=1.0, brain_gain=1.0),
}


def build(kind, mode, memory, brain_gain, readout="", device="auto"):
    cfg = Config()
    cfg.brain.kind = kind
    cfg.brain.device = device
    cfg.brain.dt = DEFAULT_DT[kind]
    cfg.brain.seed = 64
    apply_preset(cfg, kind)
    cfg.decoder.mode = mode
    cfg.decoder.memory_gain = memory
    cfg.decoder.brain_gain = brain_gain
    if readout:
        cfg.decoder.readout_path = readout
    brain = make_brain(cfg.brain, verbose=False)
    groups = resolve(brain, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    return cfg, brain, groups, dn_idx


def one(seed, spec, cells, scale, loop_chance, sim_seconds, device):
    cfg, brain, groups, dn_idx = build(
        spec["kind"], spec["mode"], spec["memory"], spec["brain_gain"],
        spec.get("readout", ""), device)
    maze = make_maze("gen", seed, cells, scale, loop_chance)
    sx, sy, sth = maze.start
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    enc = Encoder(groups, cfg.encoder, brain.dt, car_max_speed=cfg.car.max_speed,
                  car_radius=cfg.car.radius)
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    brain.reset(cfg.brain.seed)
    steps = max(2, int(round(sim_seconds / brain.dt)))
    return run(cfg, maze, car, brain, enc, dec, steps, verbose=False, stop_on_goal=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="导航 benchmark（只用未参与调参的种子）")
    ap.add_argument("--controllers", nargs="+", default=["real_nomem", "real_mem"],
                    choices=sorted(CONTROLLERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=TEST_SEEDS)
    ap.add_argument("--sim-seconds", type=float, default=900.0, help="每局上限（到终点会提前结束）")
    ap.add_argument("--cells", default="6x4")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--loop", type=float, default=0.08)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--csv", default="")
    args = ap.parse_args(argv)

    cells = tuple(int(v) for v in args.cells.lower().split("x"))
    print(f"迷宫 {cells[0]}x{cells[1]}  种子 {len(args.seeds)} 个（TEST_SEEDS）  "
          f"上限 {args.sim_seconds:.0f}s  到终点即停\n")

    rows = []
    for name in args.controllers:
        spec = CONTROLLERS[name]
        t0 = time.perf_counter()
        results = []
        for i, seed in enumerate(args.seeds):
            st = one(seed, spec, cells, args.scale, args.loop, args.sim_seconds, args.device)
            results.append(st)
            mark = "*" if st["reached_goal"] else " "
            print(f"  {name:<18} seed {seed:>4}  "
                  f"cov={st['coverage'] * 100:5.1f}%  goal={st['best_goal_dist']:3d}{mark}  "
                  f"t_goal={st['time_to_goal']:7.1f}  "
                  f"contact={st['contact_ratio']:.3f}  replan={st['replans']:5d}  "
                  f"front={st['front_safety_events']:4d}")
            sys.stdout.flush()

        n = len(results)
        reached = [r for r in results if r["reached_goal"]]
        agg = {
            "controller": name,
            "seeds": n,
            "success": len(reached),
            "success_rate": len(reached) / n,
            "time_to_goal_mean": float(np.mean([r["time_to_goal"] for r in reached])) if reached else -1.0,
            "time_to_goal_median": float(np.median([r["time_to_goal"] for r in reached])) if reached else -1.0,
            "path_efficiency_mean": float(np.mean([r["path_efficiency"] for r in reached])) if reached else -1.0,
            "distance_mean": float(np.mean([r["distance"] for r in results])),
            "coverage_mean": float(np.mean([r["coverage"] for r in results])),
            "contact_ratio_mean": float(np.mean([r["contact_ratio"] for r in results])),
            "collision_events_mean": float(np.mean([r["collision_events"] for r in results])),
            "stalls_mean": float(np.mean([r["stalls"] for r in results])),
            "front_safety_mean": float(np.mean([r["front_safety_events"] for r in results])),
            "replans_mean": float(np.mean([r["replans"] for r in results])),
            "map_explored_mean": float(np.mean([r["map_explored"] for r in results])),
            "brain_ms_per_step": float(np.mean([r["brain_ms_per_step"] for r in results])),
            "wall_seconds": time.perf_counter() - t0,
        }
        rows.append(agg)
        print(f"  -> {name}: {agg['success']}/{n} = {agg['success_rate'] * 100:.0f}%   "
              f"mean t_goal {agg['time_to_goal_mean']:.1f}s   "
              f"contact {agg['contact_ratio_mean']:.3f}   "
              f"replans {agg['replans_mean']:.0f}   "
              f"front {agg['front_safety_mean']:.1f}\n")
        sys.stdout.flush()

    print("=" * 108)
    print(f"{'controller':<20}{'success':>9}{'rate':>7}{'t_goal':>9}{'pathEff':>9}"
          f"{'contact':>9}{'dist':>8}{'cov%':>7}{'front':>7}{'replan':>8}{'explored%':>11}")
    print("-" * 108)
    for r in rows:
        expl = f"{r['map_explored_mean'] * 100:.1f}" if r["map_explored_mean"] >= 0 else "n/a"
        print(f"{r['controller']:<20}{r['success']:>4}/{r['seeds']:<4}"
              f"{r['success_rate'] * 100:>6.0f}%{r['time_to_goal_mean']:>9.1f}"
              f"{r['path_efficiency_mean']:>9.3f}{r['contact_ratio_mean']:>9.3f}"
              f"{r['distance_mean']:>8.1f}{r['coverage_mean'] * 100:>7.1f}"
              f"{r['front_safety_mean']:>7.1f}{r['replans_mean']:>8.0f}"
              f"{expl:>11}")
    print("=" * 108)
    print("t_goal = 到达终点的平均耗时(s)，只统计成功的局")
    print("pathEff = 最优格数 / 实际行驶距离（越接近 1 越高效）")
    print("contact = 顶墙步数 / 总步数")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
