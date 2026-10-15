"""P2d-17：MaleCNS 群体闭环"持续旋转"的根因诊断（只诊断，不改控制/不解算/不训练）。

对锁定案例逐时间步记录因果链并区分四类原因：
  A 神经持续输出：障碍已离开相关感觉范围，D-b0 仍维持转向（assertive 但相关侧 ToF 无目标）
  B 环境闭环反馈：小车旋转后障碍仍不断进入相关通道（assertive 且相关侧 ToF 有目标）
  C 人工映射放大：gain=8 长时间饱和，未限幅 raw 与实际 turn_cmd 差异大
  D 停车与旋转耦合：front_safety 令 v=0 但 |omega| 仍大 → 原地旋转
同时报告**有符号累计转角**与**绝对累计转角**。诊断数据不回喂控制器。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.decode import Decoder
from loop.dynamic_tof_sim import frame_from_pose, load_rig
from loop.encode_tof import PhysicalToFEncoder
from loop.local_avoidance_bench import SCENARIOS
from loop.p2_control import P2Control
from loop.population_turn import (GAIN, L_IDX, R_IDX, SIGN, TAU, estimate_b0, make_pop_source)
from loop.run import make_setup
from world.car import DiffDriveCar
from world.maze import Maze

IN_RANGE = ("VALID", "TOO_NEAR")


def spans(t, mask, min_s=0.1):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if t[j] - t[i] >= min_s:
                out.append((round(float(t[i]), 1), round(float(t[j]), 1)))
            i = j + 1
        else:
            i += 1
    return out


def run_diag(cfg, brain, groups, dn_idx, seed, rows, *, seconds=40.0, gain=GAIN, sign=SIGN,
             tau=TAU, tof_hz=10.0, mpc=0.40, rig=None, b0=None):
    from flybrain import Trace
    maze = Maze(rows)
    sx, sy, sth = maze.start
    cfg.decoder.mode = "dn"
    cfg.decoder.escape_thresh = 1e9
    fe = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    src, _obj = make_pop_source(brain, brain.dt, b0, mode="POP", dn=dn_idx, gain=gain, sign=sign, tau=tau)
    ctl = P2Control(cfg.decoder, fe, src, name="POP")
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    tr = Trace(brain, idx=np.asarray(dn_idx)[np.array(list(L_IDX) + list(R_IDX), int)], tau=tau)
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    brain.reset(seed)
    dt = brain.dt
    n = int(round(seconds / dt))
    period = 1.0 / tof_hz
    frame, seq, last_t = None, 0, -1e9
    keys = ("t", "x", "y", "th", "D", "Dmb", "raw", "turn", "v", "om", "front", "gd", "seq",
            "tofL", "tofFL", "tofF", "tofFR", "tofR",
            "stL", "stFL", "stF", "stFR", "stR", "nearL", "nearR",
            "loomL", "loomR", "chaseL", "chaseR", "threatL", "threatR",
            "inj_loomL", "inj_loomR", "inj_chaseL", "inj_chaseR", "inj_threatL", "inj_threatR",
            "inj_fwd", "meanL", "meanR")
    rec = {k: [] for k in keys}
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
        a = np.asarray(tr.observe(fired), np.float32)
        meanL, meanR = float(a[:len(L_IDX)].mean()), float(a[len(L_IDX):].mean())
        D = meanL - meanR
        raw = sign * gain * (D - b0)
        v, om = ctl.update(fired, info)
        car.set_command(v, om)
        car.step(maze, dt)
        gd = maze.dist_to_goal(car.x, car.y)
        cl, cr = enc._cache.get("L", (0, 0, 0, 0, 0)), enc._cache.get("R", (0, 0, 0, 0, 0))
        sk = {"VALID": 2.0, "TOO_NEAR": 1.0}
        rng, st = frame.ranges, frame.status
        ps = float(enc.per_step)
        jf = float(enc.cfg.forward_drive_dv * ps) if enc.cfg.forward_drive_dv > 0 else 0.0
        vals = {"t": t, "x": car.x, "y": car.y, "th": car.theta, "D": D, "Dmb": D - b0,
                "raw": raw, "turn": ctl.last_turn_cmd, "v": ctl.v, "om": ctl.omega,
                "front": 1.0 if ctl.front_blocked else 0.0, "gd": gd, "seq": float(frame.seq),
                "tofL": float(rng.get("L") or 0.0), "tofFL": float(rng.get("FL") or 0.0),
                "tofF": float(rng.get("F") or 0.0), "tofFR": float(rng.get("FR") or 0.0),
                "tofR": float(rng.get("R") or 0.0),
                "stL": sk.get(st.get("L"), 0.0), "stFL": sk.get(st.get("FL"), 0.0),
                "stF": sk.get(st.get("F"), 0.0), "stFR": sk.get(st.get("FR"), 0.0),
                "stR": sk.get(st.get("R"), 0.0),
                "nearL": 1.0 if (st.get("F") in IN_RANGE or st.get("FL") in IN_RANGE) else 0.0,
                "nearR": 1.0 if (st.get("F") in IN_RANGE or st.get("FR") in IN_RANGE) else 0.0,
                "loomL": float(info.get("loom_L", 0.0)), "loomR": float(info.get("loom_R", 0.0)),
                "chaseL": float(cl[1]), "chaseR": float(cr[1]),
                "threatL": float(info.get("threat_L", 0.0)), "threatR": float(info.get("threat_R", 0.0)),
                # 实际注入到各细胞群的每步幅度（dv*per_step），避免把 NO_TARGET 当作"无神经输入"
                "inj_loomL": cl[0] * ps if cl[0] > 0 else 0.0,
                "inj_loomR": cr[0] * ps if cr[0] > 0 else 0.0,
                "inj_chaseL": cl[1] * ps if cl[1] > 0 else 0.0,
                "inj_chaseR": cr[1] * ps if cr[1] > 0 else 0.0,
                "inj_threatL": cl[2] * ps if cl[2] > 0 else 0.0,
                "inj_threatR": cr[2] * ps if cr[2] > 0 else 0.0, "inj_fwd": jf,
                "meanL": meanL, "meanR": meanR}
        for kk, vv in vals.items():
            rec[kk].append(vv)
        if gd == 0:
            if reach_t < 0:
                reach_t = t
            break
    for kk in rec:
        rec[kk] = np.array(rec[kk], float)
    rec["reach"] = reach_t >= 0
    rec["cev"] = int(car.collision_events)
    rec["bestD"] = int(rec["gd"].min()) if len(rec["gd"]) else -1
    return rec


def analyze(rec):
    t = rec["t"]
    if len(t) < 2:
        return {}
    dt = t[1] - t[0]
    om, turn, front = rec["om"], rec["turn"], rec["front"]
    assertive = np.abs(turn) >= 0.5
    sat = np.abs(turn) >= 0.999
    near = np.where(turn > 0, rec["nearR"] > 0.5, rec["nearL"] > 0.5)
    A = assertive & (~near)                      # 持续转向但相关侧无目标
    B = assertive & near                         # 相关侧仍有目标
    Dd = (front > 0.5) & (np.abs(om) >= 0.5)     # 停车但仍在转
    return {
        "dur_s": float(t[-1] - t[0]), "n": len(t), "reach": rec["reach"],
        "cev": rec["cev"], "bestD": rec["bestD"],
        "A_frac": float(A.mean()), "B_frac": float(B.mean()),
        "C_frac": float(sat.mean()), "D_frac": float(Dd.mean()),
        "assertive_frac": float(assertive.mean()), "front_frac": float((front > 0.5).mean()),
        "mean_abs_Dmb": float(np.mean(np.abs(rec["Dmb"]))),
        "max_abs_Dmb": float(np.max(np.abs(rec["Dmb"]))),
        "cum_signed_deg": float(np.degrees(np.sum(om) * dt)),
        "cum_abs_deg": float(np.degrees(np.sum(np.abs(om)) * dt)),
        "A_B_overlap": float((A & B).mean()), "C_D_overlap": float((sat & Dd).mean()),
        "A_spans": spans(t, A), "B_spans": spans(t, B), "D_spans": spans(t, Dd),
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1001:must_right,808:must_right,606:must_right")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args(argv)

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=args.device, seed=1001,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    rig = load_rig()
    b0s = {}
    print(f"[P2d-17] gain={GAIN} sign={SIGN} tau={TAU}  只诊断（不回喂控制器）\n")
    for case in args.cases.split(","):
        seed_s, sc = case.split(":")
        seed = int(seed_s)
        if seed not in b0s:
            b0s[seed] = estimate_b0(cfg, brain, groups, dn_idx, seed)
        rec = run_diag(cfg, brain, groups, dn_idx, seed, SCENARIOS[sc], seconds=args.seconds,
                       rig=rig, b0=b0s[seed])
        a = analyze(rec)
        tag = "成功" if a["reach"] else "失败"
        print(f"=== seed {seed} / {sc} / POP  [{tag}]  b0={b0s[seed]:+.3f} ===")
        print(f"  到达={a['reach']} bestD={a['bestD']} 碰撞事件={a['cev']} 时长={a['dur_s']:.1f}s")
        print(f"  有符号累计转角={a['cum_signed_deg']:+.0f}°  绝对累计转角={a['cum_abs_deg']:.0f}°"
              f"  （差值={a['cum_signed_deg'] - a['cum_abs_deg']:.0f}° → 正负抵消程度）")
        print(f"  A 神经持续(assertive&无目标) 占比 {a['A_frac'] * 100:.1f}%  典型 {a['A_spans'][:4]}")
        print(f"  B 环境闭环(assertive&有目标) 占比 {a['B_frac'] * 100:.1f}%  典型 {a['B_spans'][:4]}")
        print(f"  C 映射饱和(|turn|>=1)        占比 {a['C_frac'] * 100:.1f}%  |D-b0|均 {a['mean_abs_Dmb']:.3f}"
              f"（相对 gain=8 → raw≈{a['mean_abs_Dmb'] * 8:.2f}）")
        print(f"  D 停车仍转(front & |omega|>=0.5) 占比 {a['D_frac'] * 100:.1f}%  典型 {a['D_spans'][:4]}")
        print(f"  assertive 占比 {a['assertive_frac'] * 100:.0f}%  front 占比 {a['front_frac'] * 100:.0f}%"
              f"  A∩B {a['A_B_overlap'] * 100:.1f}%  C∩D {a['C_D_overlap'] * 100:.1f}%\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
