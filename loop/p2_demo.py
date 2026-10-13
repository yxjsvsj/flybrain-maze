"""P2d-16：真实 MaleCNS L50/R50 群体最小闭环的**离线演示**（虚拟仿真，非实体车）。

复现参考案例：brain seed 1001, must_right, 群体控制, 24.2s / 0 碰撞。
同图同 seed 同运动包络下并排展示 群体 / 置零 / 反号；另附一个失败案例并标注其失去进展处。

冻结不变：L50/R50（population_turn.py）、Trace τ=0.2、gain=8、sign=-1、每 seed 无遮挡 b0 校准、
BASE 编码器、P2Control、world.car 仿真动力学。**不**训练/调参/换神经元/改迷宫。

⚠ 归因边界（见 README）：MaleCNS 连接与模拟神经活动 / 人工选出的 L50/R50 / 人工神经元→电机映射 /
独立 front_safety / 虚拟碰撞≠实体安全。不得宣称"果蝇脑学会随机迷宫"或"稳定自主导航"。

用法：
  python -m loop.p2_demo --out-dir C:\\tmp\\p2demo
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.decode import Decoder
from loop.dn_steering import calibrate_baseline
from loop.dynamic_tof_sim import frame_from_pose, load_rig
from loop.encode_tof import PhysicalToFEncoder
from loop.local_avoidance_bench import SCENARIOS
from loop.p2_control import P2Control
from loop.population_turn import PopulationTurn, audit_indices, estimate_b0, make_pop_source
from loop.run import make_setup
from loop.turn_hold import make_source as dn_make_source
from world.car import DiffDriveCar
from world.maze import Maze

CONTROLS = ("POP", "POP_zero", "POP_swap")


def run_demo(cfg, brain, groups, dn_idx, seed, rows, mode, *, b0pop, b0dn, rig,
             seconds=40.0, tof_hz=10.0, mpc=0.40, source=None):
    """单次控制运行的逐帧记录（含诊断用群体差分 D(t)）。"""
    maze = Maze(rows)
    sx, sy, sth = maze.start
    cfg.decoder.mode = "dn"
    cfg.decoder.escape_thresh = 1e9
    fe = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    if source is not None:
        src = source
    elif mode == "DN_ema":
        src, _ = dn_make_source("DN_ema", b0dn, brain.dt)
    else:
        src, _ = make_pop_source(brain, brain.dt, b0pop, mode=mode, dn=dn_idx)
    ctl = P2Control(cfg.decoder, fe, src, name=mode)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    diag = PopulationTurn(brain, brain.dt, dn=dn_idx)          # 诊断 D(t)，不影响控制
    brain.reset(seed)
    dt = brain.dt
    n = int(round(seconds / dt))
    period = 1.0 / tof_hz
    frame, seq, last_t = None, 0, -1e9
    rec = {k: [] for k in ("t", "x", "y", "th", "D", "turn", "v", "om", "front", "gd")}
    reach_t = -1.0
    for k in range(n):
        t = k * dt
        if frame is None or (t - last_t) >= period - 1e-9:
            seq += 1
            frame = frame_from_pose(maze, car.x, car.y, car.theta, t, seq, rig=rig,
                                    meters_per_cell=mpc)
            last_t = t
        d = enc.sense(frame)
        enc.update(frame, d)
        inj, info = enc.inject()
        fired = brain.step(inject=inj)
        Dv = diag.D(fired)
        v, om = ctl.update(fired, info)
        car.set_command(v, om)
        car.step(maze, dt)
        gd = maze.dist_to_goal(car.x, car.y)
        for key, val in (("t", t), ("x", car.x), ("y", car.y), ("th", car.theta), ("D", Dv),
                         ("turn", ctl.last_turn_cmd), ("v", ctl.v), ("om", ctl.omega),
                         ("front", 1.0 if ctl.front_blocked else 0.0), ("gd", gd)):
            rec[key].append(val)
        if gd == 0 and reach_t < 0:
            reach_t = t
    return {"rec": {k: np.array(v, float) for k, v in rec.items()},
            "reach": reach_t >= 0, "t_goal": reach_t,
            "cev": int(car.collision_events), "cevs": int(car.collisions),
            "maze": maze}


def render(maze, demos, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    a = ax[0, 0]
    a.imshow(maze.grid, cmap="gray_r", origin="upper", extent=[0, maze.w, maze.h, 0], alpha=0.35)
    if maze.goal is not None:
        a.plot(maze.goal[0], maze.goal[1], "*", ms=18, color="tab:green")
    a.plot(maze.start[0], maze.start[1], "o", ms=8, color="k")
    for label, d, color in demos:
        r = d["rec"]
        a.plot(r["x"], r["y"], "-", lw=1.6, color=color, label=label)
    a.set_title(f"{title}\n[VIRTUAL SIMULATION - not a real vehicle]")
    a.set_aspect("equal")
    a.legend(fontsize=8)
    for j, (key, lbl) in enumerate((("D", "population D(t)=mean(L)-mean(R)"),
                                    ("turn", "turn_cmd (neural->motor)"),
                                    ("v", "v (forward)"))):
        bx = ax[(j + 1) // 2, (j + 1) % 2]
        for label, d, color in demos:
            bx.plot(d["rec"]["t"], d["rec"][key], lw=1.2, color=color, label=label)
        bx.set_title(lbl)
        bx.set_xlabel("t (s)")
        bx.grid(alpha=0.3)
        if key == "v":
            for label, d, color in demos:
                bx.plot(d["rec"]["t"], d["rec"]["om"], lw=0.8, ls="--", color=color)
            bx.set_title("v (solid) / omega (dashed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reference-seconds", type=float, default=40.0)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=args.device, seed=1001,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    aud = audit_indices(brain)
    print("[P2d-16] 索引审计: ok=%s  L_pos_ck=%s R_pos_ck=%s L_brain_ck=%s R_brain_ck=%s disjoint=%s"
          % (aud["ok"], aud["L_pos_ck"], aud["R_pos_ck"], aud["L_brain_ck"], aud["R_brain_ck"],
             aud["disjoint"]))
    rig = load_rig()

    def b0s(seed):
        return estimate_b0(cfg, brain, groups, dn_idx, seed), \
            calibrate_baseline(cfg, brain, groups, dn_idx, seed)

    # 参考成功案例：1001 must_right
    b0p, b0d = b0s(1001)
    demos = []
    for mode, color in (("POP", "tab:blue"), ("POP_zero", "tab:gray"), ("POP_swap", "tab:red")):
        d = run_demo(cfg, brain, groups, dn_idx, 1001, SCENARIOS["must_right"], mode,
                     b0pop=b0p, b0dn=b0d, rig=rig, seconds=args.reference_seconds)
        demos.append((mode, d, color))
        print(f"[ref 1001 must_right] {mode:9s} reach={d['reach']} t_goal={d['t_goal']:.1f} "
              f"collision_events={d['cev']}")
    render(Maze(SCENARIOS["must_right"]), demos, "seed 1001 / must_right / L50-R50 population",
           os.path.join(args.out_dir, "demo_success_1001_must_right.png"))

    # 失败案例：808 must_right（同参数，未通过）
    b0p8, b0d8 = b0s(808)
    d = run_demo(cfg, brain, groups, dn_idx, 808, SCENARIOS["must_right"], "POP",
                 b0pop=b0p8, b0dn=b0d8, rig=rig, seconds=args.reference_seconds)
    r = d["rec"]
    om = r["om"]
    spin = float(np.mean(np.abs(om) > 2.0)) if len(om) else 0.0
    print(f"[fail 808 must_right] POP reach={d['reach']} bestD={int(r['gd'].min())} "
          f"collision_events={d['cev']} |omega|>2 占比={spin * 100:.0f}% （持续旋转/失去进展）")
    render(Maze(SCENARIOS["must_right"]), [("POP (failed case)", d, "tab:orange")],
           "seed 808 / must_right / population FAILURE (spins, loses progress)",
           os.path.join(args.out_dir, "demo_failure_808_must_right.png"))

    print("\n[正式统计 P2d-15c] screen(64/101/202/303): POP 1/8, DN_ema 2/8, zero 0/8, swap 0/8")
    print("                    confirm(606/808/1001): POP 1/6, DN_ema 0/6, zero 0/6, swap 0/6")
    print("                    在线受控刺激：左右可分（左→负/右→正，全 screen seed）且撤除后减弱；")
    print("                    但动态转弯仍不稳定（首转向常对、累计转角大 → 打圈）。")
    print(f"\n[images] {os.path.join(args.out_dir, 'demo_success_1001_must_right.png')}")
    print(f"         {os.path.join(args.out_dir, 'demo_failure_808_must_right.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
