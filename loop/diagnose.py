"""诊断单个种子的失败原因。

    python -m loop.diagnose --seed 131
    python -m loop.diagnose --seed 163 --video diag_163.mp4 --every 20
    python -m loop.diagnose --seed 181 --csv diag_181.csv --tail 40

输出每一步的 goal distance / target / bearing error / v / omega /
brain_turn / pursuit_turn / turn_cmd / dmin_F / front_blocked / replans / path_len，
最后给一个失败归类（planner / 终点最后一格 / front safety）。
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from brain.neurons import resolve
from brain.wrap import make_brain
from config import Config, apply_preset
from loop.benchmark import CONTROLLERS
from loop.decode import Decoder
from loop.encode import Encoder
from loop.run import DEFAULT_DT, make_maze, run
from world.car import DiffDriveCar

TAIL_FIELDS = ["t", "x", "y", "theta", "goal_dist", "target_dist", "bearing_err",
               "v", "omega", "brain_turn", "pursuit_turn", "turn_cmd",
               "dmin_F", "front_blocked", "replans", "path_len"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="单局失败诊断")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--controller", default="real_readout_mem",
                    choices=sorted(CONTROLLERS))
    ap.add_argument("--sim-seconds", type=float, default=600.0)
    ap.add_argument("--tail", type=float, default=30.0, help="打印最后多少秒的逐秒明细")
    ap.add_argument("--cells", default="6x4")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--loop", type=float, default=0.08)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--csv", default="")
    ap.add_argument("--video", default="")
    ap.add_argument("--every", type=int, default=20, help="录制时每 N 步一帧")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args(argv)

    cells = tuple(int(v) for v in args.cells.lower().split("x"))
    spec = CONTROLLERS[args.controller]

    cfg = Config()
    cfg.brain.kind = spec["kind"]
    cfg.brain.device = args.device
    cfg.brain.dt = DEFAULT_DT[spec["kind"]]
    cfg.brain.seed = 64
    apply_preset(cfg, spec["kind"])
    cfg.decoder.mode = spec["mode"]
    cfg.decoder.memory_gain = spec["memory"]
    cfg.decoder.brain_gain = spec["brain_gain"]
    if spec.get("readout"):
        cfg.decoder.readout_path = spec["readout"]

    maze = make_maze("gen", args.seed, cells, args.scale, args.loop)
    sx, sy, sth = maze.start
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    brain = make_brain(cfg.brain, verbose=False)
    groups = resolve(brain, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    enc = Encoder(groups, cfg.encoder, brain.dt, car_max_speed=cfg.car.max_speed,
                  car_radius=cfg.car.radius)
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    brain.reset(cfg.brain.seed)

    steps = max(2, int(round(args.sim_seconds / brain.dt)))
    print(f"[diag] {args.controller} seed={args.seed} "
          f"maze {cells[0]}x{cells[1]} optimal={maze.optimal_path()} cells  "
          f"goal=({maze.goal[0]:.1f},{maze.goal[1]:.1f})  {args.sim_seconds:.0f}s 上限")

    viewer, render_every = None, 5
    if args.video:
        from loop.run import Viewer
        viewer = Viewer(maze, video=args.video, fps=args.fps)
        render_every = float(args.every)

    records: list[dict] = []
    stats = run(cfg, maze, car, brain, enc, dec, steps, viewer=viewer,
                render_every=render_every, verbose=False, stop_on_goal=True,
                trace_fn=records.append, trace_every=1)
    if viewer is not None:
        viewer.close()

    t_end = stats["sim_time"]
    tail = [r for r in records if r["t"] >= t_end - args.tail]

    print(f"\n结果：{'REACHED' if stats['reached_goal'] else 'NOT REACHED'}  "
          f"closest={stats['best_goal_dist']}  steps_done={stats['steps_done']}  "
          f"t_end={t_end:.1f}s")
    print(f"      distance={stats['distance']:.1f}  contact={stats['contact_ratio']:.3f}  "
          f"stalls={stats['stalls']}  escapes={stats['escapes']}  "
          f"front_safety={stats['front_safety_events']}  replans={stats['replans']}  "
          f"explored={stats['map_explored'] * 100:.1f}%")

    print(f"\n最后 {args.tail:.0f} 秒逐秒明细：")
    print(f"  {'t':>7}{'x':>7}{'y':>7}{'goal':>6}{'tgt_d':>7}{'berr':>7}"
          f"{'v':>7}{'w':>7}{'brain':>7}{'purs':>7}{'cmd':>7}"
          f"{'dminF':>7}{'fblk':>6}{'repl':>6}{'plen':>6}")
    last_sec = None
    for r in tail:
        sec = int(r["t"])
        if sec == last_sec:
            continue
        last_sec = sec
        tgt = r["target"]
        tgt_d = r["target_dist"]
        print(f"  {r['t']:7.1f}{r['x']:7.2f}{r['y']:7.2f}{r['goal_dist']:6d}"
              f"{tgt_d:7.2f}{r['bearing_err']:+7.2f}"
              f"{r['v']:+7.2f}{r['omega']:+7.2f}{r['brain_turn']:+7.2f}"
              f"{r['pursuit_turn']:+7.2f}{r['turn_cmd']:+7.2f}"
              f"{r['dmin_F']:7.2f}{int(r['front_blocked']):6d}{r['replans']:6d}"
              f"{r['path_len']:6d}"
              + (f"   tgt=({tgt[0]:.1f},{tgt[1]:.1f})" if tgt else ""))

    # ---- 归类 ----
    if not tail:
        print("\n[verdict] 没有尾部数据")
        return 1
    front_ratio = float(np.mean([r["front_blocked"] for r in tail]))
    v_abs = float(np.mean([abs(r["v"]) for r in tail]))
    w_abs = float(np.mean([abs(r["omega"]) for r in tail]))
    replan_rate = ((tail[-1]["replans"] - tail[0]["replans"])
                   / max(1e-9, tail[-1]["t"] - tail[0]["t"]))
    path_len = float(np.mean([r["path_len"] for r in tail]))
    goal_lo = min(r["goal_dist"] for r in tail)
    goal_hi = max(r["goal_dist"] for r in tail)

    print(f"\n尾部统计：front_blocked 占比 {front_ratio:.2f}   |v| 均值 {v_abs:.3f}   "
          f"|w| 均值 {w_abs:.3f}   replan 速率 {replan_rate:.2f}/s   "
          f"平均 path_len {path_len:.1f}   goal_dist 区间 [{goal_lo},{goal_hi}]")

    if stats["reached_goal"]:
        verdict = "REACHED —— 不是失败局"
    elif front_ratio > 0.30:
        verdict = "front safety 长期硬停（正前方一直有东西，车被限速压死）"
    elif replan_rate > 1.0:
        verdict = "planner 抖动（重规划太频繁，路径一直在换）"
    elif goal_lo <= 2 and v_abs < 0.15:
        verdict = "终点最后一格控制（到终点旁边但停住/进不去）"
    elif goal_lo <= 2 and w_abs > 0.8:
        verdict = "终点最后一格控制（在终点旁边绕圈，进不去）"
    elif goal_lo <= 2:
        verdict = "终点最后一格控制（到终点旁边但没能进格）"
    else:
        verdict = "planner（根本没被送到终点附近）"
    print(f"[verdict] {verdict}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=TAIL_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(records)
        print(f"[csv] {args.csv}  ({len(records)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
