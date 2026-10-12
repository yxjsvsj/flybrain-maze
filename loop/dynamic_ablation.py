"""P2d-10 任务三/二：动态五路 ToF 闭环的四组消融（长时窗）+ 完整诊断。

A: bg=0, mem=0  B: bg=0.15, mem=0  C: bg=0, mem=1  D: bg=0.15, mem=1
同 map/brain/sensor seed；几何、感知、动力学一致。**全离线，无硬件/Bridge/电机**。

每实验记录：reached_goal / time_to_goal_s / timeout / best_goal_dist / path_len_cells /
collision_steps / collision_events / front_safety_events / replans / no_path_events /
map_revision / brain_turn / brain_component / pursuit_component / turn_cmd / opposing_turn_fraction。
并对每条给出最短路径与最大速度理论下界 min_time_lower_bound_s。

用法：
  python -m loop.dynamic_ablation --brain real --seconds 120 --maze-seeds 1000,1001,1002,1003
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.decode import Decoder
from loop.dynamic_tof_sim import load_rig, run_dynamic
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup
from loop.tof_geometry import load_geometry
from world.car import DiffDriveCar
from world.maze import Maze, generate

CONDS = {"A": (0.0, 0.0), "B": (0.15, 0.0), "C": (0.0, 1.0), "D": (0.15, 1.0)}


def _build_maze(name, seed):
    if name == "straight":
        rows = ["########", "#S....G#", "#......#", "#......#", "########"]
        return Maze(rows)
    return Maze(generate(6, 4, seed=seed, scale=2, loop_chance=0.08))


def _components(traj, bg, mg):
    bt = np.array([r["brain_turn"] for r in traj])
    pu = np.array([r["pursuit_turn"] for r in traj])
    tc = np.array([r["turn_cmd"] for r in traj])
    both = (np.abs(bt) > 1e-4) & (np.abs(pu) > 1e-4)
    if both.sum() == 0:
        opp = None
    else:
        opp = float(np.mean((bt[both] > 0) != (pu[both] > 0)))
    return {"brain_turn_mean": float(bt.mean()),
            "brain_component": float(bg * np.abs(bt).mean()),
            "pursuit_component": float(mg * np.abs(pu).mean()),
            "turn_cmd_mean": float(tc.mean()),
            "opposing_turn_fraction": opp}


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
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--tof-hz", type=float, default=10.0)
    ap.add_argument("--max-valid-mm", type=float, default=1200.0)
    ap.add_argument("--maze-seeds", default="1000,1001,1002,1003")
    ap.add_argument("--meters-per-cell", type=float, default=0.40)
    ap.add_argument("--no-geometry", action="store_true", help="关闭逐原点 ToFMemory（退回车心）")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args(argv)

    cfg, _maze, _car, brain, groups, _e0, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=args.seed,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    cfg.decoder.mode = args.mode
    cfg.decoder.readout_path = args.readout
    dn = np.asarray(brain.cells(["descending_neuron"]))
    rig = load_rig()
    geometry = None if args.no_geometry else load_geometry()

    scenarios = [("straight", 0)] + [("gen", int(s)) for s in args.maze_seeds.split(",")]
    rows_out = []
    print(f"[ablation] brain={args.brain} seconds={args.seconds} tof={args.tof_hz}Hz "
          f"geometry={'on' if geometry else 'off'}")
    hdr = (f"{'scenario':12s} {'c':1s} {'reach':>5s} {'t_goal':>7s} {'best':>5s} {'path':>6s} "
           f"{'cstep':>5s} {'cev':>4s} {'fsafe':>5s} {'repl':>4s} {'nopath':>6s} {'mrev':>5s} "
           f"{'minLB':>6s} {'pathEff':>7s} {'bturn':>7s} {'bcomp':>6s} {'pcomp':>6s} "
           f"{'tc':>7s} {'opp':>5s}")
    print(hdr)
    for name, seed in scenarios:
        maze = _build_maze(name, seed)
        opt = maze.optimal_path()
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
                                  max_valid_mm=args.max_valid_mm, seed=args.seed,
                                  geometry=geometry)
            comp = _components(traj, bg, mg)
            label = f"{name}:{seed}" if name == "gen" else name
            opp = comp["opposing_turn_fraction"]
            print(f"{label:12s} {cond:1s} {str(s['reached_goal'])[:1]:>5s} "
                  f"{s['time_to_goal_s']:7.2f} {s['best_goal_dist']:5d} {s['path_len_cells']:6.2f} "
                  f"{s['collision_steps']:5d} {s['collision_events']:4d} {s['front_safety_events']:5d} "
                  f"{s['replans']:4d} {s['no_path_events']:6d} {s['map_revision']:5d} "
                  f"{s['min_time_lower_bound_s']:6.2f} {s['path_efficiency']:7.2f} "
                  f"{comp['brain_turn_mean']:+7.3f} {comp['brain_component']:6.3f} "
                  f"{comp['pursuit_component']:6.3f} {comp['turn_cmd_mean']:+7.3f} "
                  f"{('-' if opp is None else f'{opp:.2f}'):>5s}")
            rows_out.append({"scenario": label, "cond": cond, **s, **comp})
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
