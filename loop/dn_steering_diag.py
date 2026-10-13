"""P2d-13c：最小动态绕障失败定位（只诊断，不训练/不调参/不改模型）。

记录每条动态轨迹的转弯决策上下文，并逐 (seed, 场景, 控制) 分类失败：
  A 方向错   B 方向对但太晚   C 方向对且及时但幅度/时间不足
  D 输出过大/反复换向/持续旋转   E front_safety 停车后起不来   F 感觉本身分不清左右

关键点：正确方向用**局部目标方位**判定（评估用，非控制器）；同时比较 must_left/must_right
在决策区的**实际 5 路 ToF 帧**与经 `PhysicalToFEncoder` 后的神经注入，若两种场景输入状态不可
区分必须披露。DN_readout 与 DN_zero 为主；DN_orig/DN_swap 沿用既有对照。

用法： python -m loop.dn_steering_diag --dyn-seeds 64,101,202,303,512
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.decode import Decoder
from loop.dn_steering import DNReadout, calibrate_baseline
from loop.dynamic_tof_sim import frame_from_pose, load_rig
from loop.encode_tof import PhysicalToFEncoder
from loop.local_avoidance_bench import CAR_R, SCENARIOS, wall_clearance
from loop.run import make_setup
from world.car import DiffDriveCar
from world.maze import Maze

D_FAR = 1.2          # dmin_F 低于它视为"决策区开始（接近前方墙）"
STRONG = 0.30        # |turn_cmd| 达到它视为"已产生明显转向"


def run_trace(cfg, brain, groups, dn, seed, kind, rows, *, b0, seconds=40.0, tof_hz=10.0,
              mpc=0.40, rig=None, enc_cls=PhysicalToFEncoder, turn_source=None):
    """带详细决策记录的动态闭环（P2Control 公平包络，memory_gain=0）。

    turn_source 非空时直接用它（外部提供的 stateful callable(ctl)->turn），忽略 kind。
    """
    from loop.p2_control import P2Control, turn_dn_single
    maze = Maze(rows)
    sx, sy, sth = maze.start
    cfg.decoder.mode = "dn"
    cfg.decoder.escape_thresh = 1e9
    fe = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)
    idx = fe.index
    rd = DNReadout(brain.dt, b0=b0)
    if turn_source is not None:
        src = turn_source
    elif kind == "DN_readout":
        src = lambda c: rd(c.last_brain_turn)          # noqa: E731
    elif kind == "DN_zero":
        src = lambda c: 0.0                            # noqa: E731
    elif kind == "DN_orig":
        src = turn_dn_single
    else:
        src = lambda c: -rd(c.last_brain_turn)         # noqa: E731
    ctl = P2Control(cfg.decoder, fe, src, name=kind)
    enc = enc_cls(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                  sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                  car_radius=cfg.car.radius, meters_per_cell=mpc)
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    brain.reset(seed)
    rig = rig or load_rig()
    dt = brain.dt
    n = int(round(seconds / dt))
    period = 1.0 / tof_hz
    frame, seq, last_t = None, 0, -1e9
    gx, gy = (maze.goal[0], maze.goal[1]) if maze.goal else (None, None)
    recs = []
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
        v, omega = ctl.update(fired, info)
        car.set_command(v, omega)
        car.step(maze, dt)
        gd = maze.dist_to_goal(car.x, car.y)
        cl = enc._cache.get("L", (0, 0, 0, 0, 0))
        cr = enc._cache.get("R", (0, 0, 0, 0, 0))
        need = (math.atan2(gy - car.y, gx - car.x) - car.theta + math.pi) % (2 * math.pi) - math.pi \
            if gx is not None else 0.0
        recs.append({
            "t": t, "x": car.x, "y": car.y, "theta": car.theta, "seq": seq, "gd": gd,
            "need_sign": float(np.sign(need)) if abs(need) > 0.05 else 0.0, "need_deg": math.degrees(need),
            "tof_F": frame.ranges.get("F"), "st_F": frame.status.get("F"),
            "tof_FL": frame.ranges.get("FL"), "tof_FR": frame.ranges.get("FR"),
            "dmin_F": info.get("dmin_F", 99.0),
            "size_L": cl[3], "size_R": cr[3], "chase_L": cl[1], "chase_R": cr[1],
            "loom_L": cl[0], "loom_R": cr[0], "threat_L": cl[2], "threat_R": cr[2],
            "rate_L": float(ctl.dec.rates[idx["turn_L"]]), "rate_R": float(ctl.dec.rates[idx["turn_R"]]),
            "raw": float(ctl.dec.last_brain_turn), "b0": float(b0), "corr": float(rd.ema),
            "turn": float(ctl.last_turn_cmd), "v": float(ctl.v), "om": float(ctl.omega),
            "front": bool(ctl.front_blocked), "cev": int(car.collision_events),
        })
        if gd == 0:
            break
    best, last_imp = float("inf"), 0.0
    for r in recs:
        if r["gd"] >= 0 and r["gd"] < best:
            best, last_imp = r["gd"], r["t"]
    T = recs[-1]["t"] if recs else 0.0
    summary = {"reach": bool(recs and recs[-1]["gd"] == 0), "bestD": int(best),
               "cev": recs[-1]["cev"] if recs else 0, "stuck": (best > 0) and (T - last_imp) > 8.0,
               "sat%": ctl.saturation_ratio() * 100, "stop%": ctl.stop_ratio() * 100,
               "front_events": int(ctl.front_blocks),
               "minGap": min((wall_clearance(maze, r["x"], r["y"], CAR_R) for r in recs),
                             default=float("nan"))}
    return summary, recs


def classify(summary, recs, dt):
    """按优先级给失败类别 + 关键度量（可审计）。"""
    if summary["reach"]:
        return "PASS", {}
    d = next((i for i, r in enumerate(recs) if r["dmin_F"] < D_FAR), None)
    if d is None:
        return "E", {"note": "no-decision-region"}
    w = recs[d:d + int(round(1.5 / dt))]
    cs = recs[d]["need_sign"]
    turns = np.array([r["turn"] for r in w])
    mean_turn = float(turns.mean()) if len(turns) else 0.0
    sign_ok = (cs == 0) or (np.sign(mean_turn) == cs and abs(mean_turn) > 0.05)
    first_strong = next((i for i, r in enumerate(w) if abs(r["turn"]) >= STRONG), None)
    first_front = next((i for i, r in enumerate(w) if r["front"]), None)
    zero_cross = int(np.sum(np.diff(np.sign(turns)) != 0)) if len(turns) > 1 else 0
    sat = float(np.mean(np.abs(turns) >= 0.999)) if len(turns) else 0.0
    met = {"need_sign": cs, "mean_turn": mean_turn, "sign_ok": bool(sign_ok),
           "first_strong": first_strong, "first_front": first_front,
           "zero_cross": zero_cross, "win_sat": sat, "stop%": summary["stop%"]}
    if cs != 0 and not sign_ok and abs(mean_turn) > 0.15:
        return "A", met
    if sat > 0.3 and zero_cross >= 4:
        return "D", met
    if summary["stop%"] > 35 and summary["stuck"]:
        return "E", met
    if sign_ok and first_front is not None and (first_strong is None or first_strong >= first_front):
        return "B", met
    return "C", met


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn-seeds", default="64,101,202,303,512")
    ap.add_argument("--scenarios", default="must_left,must_right")
    ap.add_argument("--kinds", default="DN_readout,DN_zero")
    ap.add_argument("--brain", default="real")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out-dir", default=None, help="决策窗 CSV（不提交）")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.dyn_seeds.split(",")]
    scenarios = args.scenarios.split(",")
    kinds = args.kinds.split(",")

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=seeds[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"]))
    rig = load_rig()
    b0 = {sd: calibrate_baseline(cfg, brain, groups, dn, sd) for sd in seeds}
    print(f"[P2d-13c] dyn seeds={seeds}  b0=" + " ".join(f"{sd}:{b0[sd]:+.3f}" for sd in seeds))

    print(f"\n{'seed':>5s} {'scenario':10s} {'kind':10s} {'reach':>5s} {'bestD':>5s} "
          f"{'fail':>4s} {'need':>4s} {'meanTurn':>8s} {'chaseL-R':>8s} {'sizeL-R':>8s} "
          f"{'dminF':>6s} {'frontStop':>9s} {'stop%':>6s}")
    win_inputs = {}                          # (seed,scenario)-> decision-window mean inputs
    for sd in seeds:
        for sc in scenarios:
            rows = SCENARIOS[sc]
            for kind in kinds:
                s, recs = run_trace(cfg, brain, groups, dn, sd, kind, rows, b0=b0[sd],
                                    seconds=args.seconds, rig=rig)
                cat, met = classify(s, recs, brain.dt)
                d = next((i for i, r in enumerate(recs) if r["dmin_F"] < D_FAR), None)
                w = recs[d:d + int(round(1.5 / brain.dt))] if d is not None else []
                if w:
                    win_inputs[(sd, sc, kind)] = {
                        "chase_LR": float(np.mean([r["chase_L"] - r["chase_R"] for r in w])),
                        "size_LR": float(np.mean([r["size_L"] - r["size_R"] for r in w])),
                        "dmin_F": float(np.mean([r["dmin_F"] for r in w])),
                        "FL": [r["tof_FL"] for r in w[:3]], "FR": [r["tof_FR"] for r in w[:3]]}
                fs = met.get("first_front")
                print(f"{sd:5d} {sc:10s} {kind:10s} {str(s['reach'])[:1]:>5s} {s['bestD']:5d} "
                      f"{cat:>4s} {met.get('need_sign', 0):+4.0f} {met.get('mean_turn', 0.0):+8.3f} "
                      f"{win_inputs.get((sd, sc, kind), {}).get('chase_LR', float('nan')):+8.3f} "
                      f"{win_inputs.get((sd, sc, kind), {}).get('size_LR', float('nan')):+8.3f} "
                      f"{win_inputs.get((sd, sc, kind), {}).get('dmin_F', float('nan')):6.2f} "
                      f"{str(fs is not None):>9s} {s['stop%']:6.1f}")
                if args.out_dir:
                    os.makedirs(args.out_dir, exist_ok=True)
                    p = os.path.join(args.out_dir, f"dec_{sd}_{sc}_{kind}.csv")
                    if w:
                        with open(p, "w", newline="", encoding="utf-8") as fh:
                            wr = csv.DictWriter(fh, fieldnames=list(w[0].keys()))
                            wr.writeheader()
                            for r in w:
                                wr.writerow(r)

    # L/R 输入可分性：must_left vs must_right 在同 seed 的决策区是否可分
    print("\n决策区输入可分性（must_left vs must_right，DN_readout）：")
    for sd in seeds:
        ml = win_inputs.get((sd, "must_left", "DN_readout"))
        mr = win_inputs.get((sd, "must_right", "DN_readout"))
        if ml and mr:
            dlr = ml["size_LR"] - mr["size_LR"]
            same = abs(dlr) < 0.05
            print(f"  seed {sd:3d}: sizeL-R L={ml['size_LR']:+.3f} R={mr['size_LR']:+.3f} "
                  f"diff={dlr:+.3f} -> {'不可分(F?)' if same else '可分'}   "
                  f"FL(mm) L={ml['FL'][:2]} R={mr['FL'][:2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
