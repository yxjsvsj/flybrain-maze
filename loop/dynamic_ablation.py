"""P2d-9 任务三：四组消融（brain_gain × memory_gain）在动态五路 ToF 闭环下的对照。

A: bg=0,   mem=0    B: bg=0.15, mem=0    C: bg=0, mem=1    D: bg=0.15, mem=1
同地图、同 seed、同模拟传感器与运动条件。跑直道 + 多个生成迷宫。**全离线，无硬件**。

用法：
  python -m loop.dynamic_ablation --brain real --seconds 15 --maze-seeds 1000,1001,1002,1003
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from loop.decode import Decoder
from loop.dynamic_tof_sim import load_rig, run_dynamic
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup
from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from world.car import DiffDriveCar
from world.maze import Maze, generate

CONDS = {"A": (0.0, 0.0), "B": (0.15, 0.0), "C": (0.0, 1.0), "D": (0.15, 1.0)}


def _build_maze(name, seed):
    if name == "straight":
        rows = ["#" * 8] + ["#S....G#"] + ["#......#"] * 2 + ["#" * 8]
        return Maze(rows)
    return Maze(generate(6, 4, seed=seed, scale=2, loop_chance=0.08))


def _sign_agree(traj):
    both = [(r["brain_turn"], r["pursuit_turn"]) for r in traj
            if abs(r["brain_turn"]) > 1e-4 and abs(r["pursuit_turn"]) > 1e-4]
    if not both:
        return None
    same = sum(1 for b, p in both if (b > 0) == (p > 0))
    return same / len(both)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                           # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["fake", "real"], default="real")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["readout", "dn", "scripted"], default="readout")
    ap.add_argument("--readout", default="readout_real.npz")
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--tof-hz", type=float, default=10.0)
    ap.add_argument("--max-valid-mm", type=float, default=1200.0)
    ap.add_argument("--maze-seeds", default="1000,1001,1002,1003")
    ap.add_argument("--meters-per-cell", type=float, default=0.40)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args(argv)

    cfg, _maze, _car, brain, groups, _e0, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=args.seed,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    cfg.decoder.mode = args.mode
    cfg.decoder.readout_path = args.readout
    dn = np.asarray(brain.cells(["descending_neuron"]))
    rig = load_rig()

    scenarios = [("straight", 0)] + [("gen", int(s)) for s in args.maze_seeds.split(",")]

    rows_out = []
    print(f"[ablation] brain={args.brain} seconds={args.seconds} tof={args.tof_hz}Hz "
          f"rig={len(rig)} sensors")
    print(f"{'scenario':12s} {'cond':4s} {'reach':>5s} {'t_goal':>7s} {'pathEff':>7s} "
          f"{'coll':>4s} {'path':>7s} {'replan':>6s} {'braw':>7s} {'pur':>7s} {'agree':>5s}")
    for name, seed in scenarios:
        maze = _build_maze(name, seed)
        for cond, (bg, mg) in CONDS.items():
            brain.reset(args.seed)
            cfg.decoder.brain_gain = bg
            cfg.decoder.memory_gain = mg
            dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)
            enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                                     sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                                     car_radius=cfg.car.radius, meters_per_cell=args.meters_per_cell)
            sx, sy, sth = maze.start
            car = DiffDriveCar(sx, sy, sth, cfg.car)
            traj, s = run_dynamic(cfg, maze, car, brain, enc, dec, memory_on=(mg > 0),
                                  seconds=args.seconds, tof_hz=args.tof_hz, rig=rig,
                                  max_valid_mm=args.max_valid_mm, seed=args.seed)
            braw = float(np.mean([r["brain_turn"] for r in traj]))
            pur = float(np.mean([r["pursuit_turn"] for r in traj]))
            agree = _sign_agree(traj)
            label = f"{name}:{seed}" if name == "gen" else name
            print(f"{label:12s} {cond:4s} {str(s['reached_goal'])[:5]:>5s} "
                  f"{s['time_to_goal_s']:7.2f} {s['path_efficiency']:7.2f} "
                  f"{s['collisions']:4d} {s['path_len_cells']:7.2f} {s['replans']:6d} "
                  f"{braw:+7.3f} {pur:+7.3f} "
                  f"{('-' if agree is None else f'{agree:.2f}'):>5s}")
            rows_out.append({"scenario": label, "cond": cond, **s, "brain_raw": braw,
                             "pursuit": pur, "agree": agree})
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"\n[ablation] -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
