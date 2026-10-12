"""P2d-11 任务三：脑/规划器融合离线消融（P2 适配层；不改冻结 decode.py / readout）。

方案（bg=brain_gain, mg=memory_gain）：
  C    planner-only：bg=0,   mg=1
  D    current blend：bg=0.15, mg=1
  E*   linear low-gain：bg=0.025/0.05/0.10, mg=1
  F    resting-bias corrected：D 但 omega 减 max_omega*bg*resting_bias（resting 只从静态
       标定 CSV 估计，绝不用评估迷宫；保留 time-varying 神经）
  G    conditional fusion：神经与 pursuit 显著反向时把 bg 降到 0.25×（规则评估前固定）
  SHAM 固定神经：去掉 time-varying brain_turn，只留 mg*pursuit + bg*resting

全离线，无 Bridge/电机/硬件。用法：
  python -m loop.dynamic_fusion_ablation --cal-csv <A_open静态CSV> --maze-seeds 1000,1001,1002,1003
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
from loop.static_tof_report import load_rows as load_static, summarize
from loop.tof_geometry import load_geometry
from world.car import DiffDriveCar
from world.maze import Maze, generate

SCHEMES = {
    "C": (0.0, 1.0, "plain"),
    "D": (0.15, 1.0, "plain"),
    "E025": (0.025, 1.0, "plain"),
    "E050": (0.05, 1.0, "plain"),
    "E100": (0.10, 1.0, "plain"),
    "F": (0.15, 1.0, "bias"),
    "G": (0.15, 1.0, "cond"),
    "SHAM": (0.15, 1.0, "sham"),
}


def resting_bias(cal_csv):
    """从静态标定 CSV（A_open）估计静息 brain_turn 均值。绝不用评估迷宫拟合。"""
    s = summarize(load_static(cal_csv))["brain_turn_raw"]
    return float(s["mean"]) if s else 0.0


def _build_maze(name, seed):
    if name == "straight":
        return Maze(["########", "#S....G#", "#......#", "#......#", "########"])
    return Maze(generate(6, 4, seed=seed, scale=2, loop_chance=0.08))


def _components(traj, bg, mg):
    bt = np.array([r["brain_turn"] for r in traj])
    pu = np.array([r["pursuit_turn"] for r in traj])
    both = (np.abs(bt) > 1e-4) & (np.abs(pu) > 1e-4)
    opp = (float(np.mean((bt[both] > 0) != (pu[both] > 0))) if both.sum() else None)
    return {"brain_turn_mean": float(bt.mean()),
            "brain_component": float(bg * np.abs(bt).mean()),
            "pursuit_component": float(mg * np.abs(pu).mean()),
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
    ap.add_argument("--maze-seeds", default="1000,1001,1002,1003")
    ap.add_argument("--meters-per-cell", type=float, default=0.40)
    ap.add_argument("--cal-csv", required=True, help="静态 A_open CSV（偏置标定，非评估迷宫）")
    ap.add_argument("--schemes", default="C,D,E050,F,G,SHAM")
    ap.add_argument("--no-geometry", action="store_true")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args(argv)

    bias = resting_bias(args.cal_csv)
    cfg, _maze, _car, brain, groups, _e0, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=args.seed,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    cfg.decoder.mode = args.mode
    cfg.decoder.readout_path = args.readout
    dn = np.asarray(brain.cells(["descending_neuron"]))
    rig = load_rig()
    geometry = None if args.no_geometry else load_geometry()
    mw = cfg.car.max_omega

    def hooks(bg, mg, mode):
        if mode == "cond":
            def gain_fn(k, d):                  # 因果：用上一步已得的分量门控
                opp = (d.last_brain_turn * d.last_pursuit_turn < 0
                       and abs(d.last_pursuit_turn) > 0.02)
                return (bg * (0.25 if opp else 1.0), mg)
        else:
            def gain_fn(k, d):
                return (bg, mg)
        if mode == "bias":
            def omega_fn(k, d, info, v, om):
                return om - mw * bg * bias
        elif mode == "sham":
            def omega_fn(k, d, info, v, om):
                return om - mw * bg * (d.last_brain_turn - bias)
        else:
            omega_fn = None
        return gain_fn, omega_fn

    scenarios = [("gen", int(s)) for s in args.maze_seeds.split(",")]
    out = []
    print(f"[fusion] brain={args.brain} seconds={args.seconds} resting_bias={bias:+.4f} "
          f"geometry={'on' if geometry else 'off'}")
    print(f"{'scenario':10s} {'sch':5s} {'reach':>5s} {'t_goal':>7s} {'path':>6s} {'cstep':>5s} "
          f"{'cev':>4s} {'fsafe':>5s} {'repl':>4s} {'nopath':>6s} {'bfs/drv':>7s} {'minLB':>6s} "
          f"{'bturn':>7s} {'bcomp':>6s} {'pcomp':>6s} {'opp':>5s}")
    for name, seed in scenarios:
        maze = _build_maze(name, seed)
        for sch in args.schemes.split(","):
            bg, mg, mode = SCHEMES[sch]
            brain.reset(args.seed)
            cfg.decoder.brain_gain = bg
            cfg.decoder.memory_gain = mg
            dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)
            enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                                     sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                                     car_radius=cfg.car.radius, meters_per_cell=args.meters_per_cell)
            sx, sy, sth = maze.start
            car = DiffDriveCar(sx, sy, sth, cfg.car)
            gain_fn, omega_fn = hooks(bg, mg, mode)
            traj, s = run_dynamic(cfg, maze, car, brain, enc, dec, memory_on=(mg > 0),
                                  seconds=args.seconds, tof_hz=args.tof_hz, rig=rig, seed=args.seed,
                                  geometry=geometry, gain_fn=gain_fn, omega_fn=omega_fn)
            comp = _components(traj, bg, mg)
            opp = "-" if comp["opposing_turn_fraction"] is None else f"{comp['opposing_turn_fraction']:.2f}"
            print(f"{name + ':' + str(seed):10s} {sch:5s} {str(s['reached_goal'])[:1]:>5s} "
                  f"{s['time_to_goal_s']:7.2f} {s['path_len_cells']:6.2f} {s['collision_steps']:5d} "
                  f"{s['collision_events']:4d} {s['front_safety_events']:5d} {s['replans']:4d} "
                  f"{s['no_path_events']:6d} {s['bfs_cells_over_driven_cells']:7.2f} "
                  f"{s['min_time_lower_bound_s']:6.2f} {comp['brain_turn_mean']:+7.3f} "
                  f"{comp['brain_component']:6.3f} {comp['pursuit_component']:6.3f} {opp:>5s}")
            out.append({"scenario": f"{name}:{seed}", "scheme": sch, **s, **comp})
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            for r in out:
                w.writerow(r)
        print(f"\n[fusion] -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
