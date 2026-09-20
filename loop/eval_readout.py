"""闭环评估：读出头 vs DNa02 直读 vs 手写贴墙。

三个控制器跑同一批**没参与训练**的迷宫，看谁能真的开出去。

    python -m loop.eval_readout --brain real --readout readout_real.npz
    python -m loop.eval_readout --brain fake --readout readout_fake.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from brain.neurons import resolve
from brain.wrap import make_brain
from config import Config, apply_preset
from loop.decode import Decoder
from loop.encode import Encoder
from loop.run import DEFAULT_DT, make_maze, run
from loop.train_readout import rollout, teacher_turn
from world.car import DiffDriveCar

CONTROLLERS = [
    ("scripted", "手写贴墙(天花板)"),
    ("dn", "DNa02 直读(现状)"),
    ("readout", "DN 群体读出头"),
]


def build(kind: str, device: str, mode: str, readout_path: str):
    cfg = Config()
    cfg.brain.kind = kind
    cfg.brain.device = device
    cfg.brain.dt = DEFAULT_DT[kind]
    cfg.brain.seed = 64
    apply_preset(cfg, kind)
    cfg.decoder.mode = mode
    if readout_path:
        cfg.decoder.readout_path = readout_path
    brain = make_brain(cfg.brain, verbose=False)
    groups = resolve(brain, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    return cfg, brain, groups, dn_idx


def main(argv=None):
    ap = argparse.ArgumentParser(description="闭环评估读出头")
    ap.add_argument("--brain", choices=["fake", "real"], default="real")
    ap.add_argument("--readout", default="", help="训练好的读出头 .npz")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[23, 42, 5, 17, 31, 61],
                    help="评估用迷宫种子（必须没参与训练）")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cells", default="9x6")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--loop", type=float, default=0.08)
    args = ap.parse_args(argv)

    cells = tuple(int(v) for v in args.cells.lower().split("x"))
    steps = args.steps or (20000 if args.brain == "fake" else 10000)

    # 离线评估：读出头在留出迷宫上逼近老师多少
    if args.readout:
        from flybrain import Readout
        cfg, brain, groups, dn_idx = build(args.brain, args.device, "scripted", "")
        print(f"[offline] 在留出迷宫 {args.seeds} 上采集老师数据 ...")
        Xe, Ye, Se = rollout(brain, groups, dn_idx, cfg, args.seeds, steps,
                             lambda i, f: teacher_turn(cfg, i), cells, args.scale, args.loop)
        base = float(np.mean((Ye - Ye.mean()) ** 2))
        for name, path, F in (("DN readout", args.readout, Xe),
                              ("sensor(ceiling)",
                               args.readout.replace(".npz", "_sensor.npz"), Se)):
            rd = Readout.load(path)
            mse = float(np.mean((rd.predict(F) - Ye) ** 2))
            print(f"          {name:16s} MSE {mse:.5f}  const {base:.5f}  R2 {1 - mse / base:.3f}")

    # 闭环评估
    rows = []
    for mode, label in CONTROLLERS:
        if mode == "readout" and not args.readout:
            continue
        cfg, brain, groups, dn_idx = build(args.brain, args.device, mode, args.readout)
        print(f"\n[closed-loop] {label}  ({len(args.seeds)} mazes x {steps} steps)")
        for seed in args.seeds:
            maze = make_maze("gen", seed, cells, args.scale, args.loop)
            sx, sy, sth = maze.start
            car = DiffDriveCar(sx, sy, sth, cfg.car)
            enc = Encoder(groups, cfg.encoder, brain.dt, car_max_speed=cfg.car.max_speed)
            dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
            brain.reset(cfg.brain.seed)
            s = run(cfg, maze, car, brain, enc, dec, steps, verbose=False)
            rows.append((label, seed, maze.optimal_path(), s))
            print(f"   seed {seed:3d}  opt={maze.optimal_path():3d}  "
                  f"cov={s['coverage'] * 100:5.1f}%  goal={s['best_goal_dist']:3d}"
                  f"{' REACHED' if s['reached_goal'] else ''}  "
                  f"hit={s['collision_events']:3d} stall={s['stalls']:3d}")

    print("\n" + "=" * 74)
    print(f"{'controller':<22}{'reached':>9}{'cov%':>8}{'goal(mean)':>12}"
          f"{'hit':>6}{'stall':>7}")
    print("-" * 74)
    for mode, label in CONTROLLERS:
        rs = [r for r in rows if r[0] == label]
        if not rs:
            continue
        reached = sum(r[3]["reached_goal"] for r in rs)
        cov = np.mean([r[3]["coverage"] for r in rs]) * 100
        goal = np.mean([r[3]["best_goal_dist"] for r in rs])
        hit = np.mean([r[3]["collision_events"] for r in rs])
        stall = np.mean([r[3]["stalls"] for r in rs])
        print(f"{label:<22}{reached:>5}/{len(rs):<3}{cov:>8.1f}{goal:>12.1f}"
              f"{hit:>6.1f}{stall:>7.1f}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
