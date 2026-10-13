"""P2d-15c：固定 L50/R50 群体的最小闭环转向验证（P2 离线）。

第一步已固化的群体在 population_turn.py（L_IDX/R_IDX + 校验和）。

第二步（在线门控）：A→左障碍→A 与 A→右障碍→A **连续在线**运行，检查实际输出 turn 的
  方向与撤除后恢复。b0 在运行前于**无遮挡 A**下估计后固定；**只用当前与过去**神经活动；
  P2d-15b 的时间匹配 AAA 差分仅离线评价，绝不进控制器。在线不可靠即停止，不进第三步。

第三步（动态闭环，仅在第二步通过后）：must_left / must_right；BASE 编码器、ToF 10Hz、脑 50Hz、
  memory_gain=0、P2Control 同 fwd/safety。对照：DN_ema / POP / POP_zero / POP_swap。
  先 screen seeds 64/101/202/303 检查接线；再在**新 seed 606/808/1001**独立确认（不改映射）。
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.dn_dynamics_probe import PLANS, make_frame, plan_segments, seg_means, STIMS
from loop.dn_steering import calibrate_baseline
from loop.dn_steering_diag import run_trace
from loop.encode_tof import PhysicalToFEncoder
from loop.local_avoidance_bench import SCENARIOS
from loop.population_turn import PopulationTurn, estimate_b0, make_pop_source, verify_indices, INDEX_CHECKSUM
from loop.run import make_setup
from loop.turn_hold import make_source as dn_make_source

GAIN, SIGN = 8.0, -1.0


def run_online(cfg, brain, groups, dn_idx, seed, plan, b0, *, tof_hz=10.0, mpc=0.40):
    """在线：只用当前/过去神经活动产生 turn；返回每段 turn 均值。"""
    src, _obj = make_pop_source(brain, brain.dt, b0, mode="POP", dn=dn_idx)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    brain.reset(seed)
    dt = brain.dt
    period = 1.0 / tof_hz
    frame, seq, last_t, total = None, 0, -1e9, 0
    turns = []
    for name, dur in plan:
        for _ in range(int(round(dur / dt))):
            tt = total * dt
            if frame is None or (tt - last_t) >= period - 1e-9:
                seq += 1
                frame = make_frame(STIMS[name], seq, tt)
                last_t = tt
            d = enc.sense(frame)
            enc.update(frame, d)
            inj, _info = enc.inject()
            fired = brain.step(inject=inj)
            turns.append(src(SimpleNamespace(fired=fired)))
            total += 1
    return seg_means(np.array(turns, float), plan, dt)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-seeds", default="64,101,202,303")
    ap.add_argument("--confirm-seeds", default="606,808,1001")
    ap.add_argument("--brain", default="real")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args(argv)
    screen = [int(s) for s in args.screen_seeds.split(",")]
    confirm = [int(s) for s in args.confirm_seeds.split(",")]

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=screen[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    print(f"[P2d-15c] 群体校验: verified={verify_indices(brain)}  checksum={INDEX_CHECKSUM} "
          f"|L|+|R|={len(__import__('loop.population_turn', fromlist=['L_IDX']).L_IDX) * 2}")
    print(f"         映射: D>0(左障碍)->负 omega, D<0(右障碍)->正 omega; gain={GAIN} sign={SIGN}"
          " (人工映射, 非天然运动连接)")

    # ---- 第二步：在线门控 ----
    from loop.dynamic_tof_sim import load_rig
    rig = load_rig()
    print(f"\n[在线] screen seeds A→刺激→A（在线 turn，b0 事前 A 估计后固定；AAA 差分不参与）")
    print(f"{'seed':>5s} {'stim':>5s} {'pre':>8s} {'stim':>8s} {'post':>8s} {'符号对?':>7s} {'衰减?':>6s}")
    b0s = {sd: estimate_b0(cfg, brain, groups, dn_idx, sd) for sd in screen}
    online_ok = True
    for sd in screen:
        tL = run_online(cfg, brain, groups, dn_idx, sd, PLANS["left"], b0s[sd])
        tR = run_online(cfg, brain, groups, dn_idx, sd, PLANS["right"], b0s[sd])
        sl = tL[1] < 0          # 左障碍(C): D>0 -> turn<0
        sr = tR[1] > 0          # 右障碍(B): D<0 -> turn>0
        dl = abs(tL[2]) < abs(tL[1])
        dr = abs(tR[2]) < abs(tR[1])
        online_ok &= (sl and sr and dl and dr)
        print(f"{sd:5d} {'left':>5s} {tL[0]:+8.3f} {tL[1]:+8.3f} {tL[2]:+8.3f} "
              f"{str(sl):>7s} {str(dl):>6s}")
        print(f"{sd:5d} {'right':>5s} {tR[0]:+8.3f} {tR[1]:+8.3f} {tR[2]:+8.3f} "
              f"{str(sr):>7s} {str(dr):>6s}")
    print(f"[在线] 方向可靠区分左右 且 撤除后减弱 全 seed: {online_ok}")
    if not online_ok:
        print("\n[STOP] 在线转向信号不可靠 -> 不进入动态小车测试（第三步）。")
        return 0

    # ---- 第三步：动态闭环 ----
    print(f"\n[动态] memory_gain=0, seconds={args.seconds:.0f}（screen 接线检查 + confirm 独立确认）")
    print(f"{'seed':>5s} {'scenario':10s} {'ctrl':9s} {'reach':>5s} {'t_goal':>7s} {'cev':>4s} "
          f"{'stop%':>6s} {'firstTurn':>9s} {'cumTurn°':>9s} {'holds?':>6s}")
    b0pop = {sd: estimate_b0(cfg, brain, groups, dn_idx, sd) for sd in screen + confirm}
    b0dn = {sd: calibrate_baseline(cfg, brain, groups, dn_idx, sd) for sd in screen + confirm}
    res = {}
    for sd in screen + confirm:
        for sc in ("must_left", "must_right"):
            for ctrl in ("DN_ema", "POP", "POP_zero", "POP_swap"):
                if ctrl == "DN_ema":
                    src, _o = dn_make_source("DN_ema", b0dn[sd], brain.dt)
                else:
                    src, _o = make_pop_source(brain, brain.dt, b0pop[sd], mode=ctrl, dn=dn_idx)
                s, recs = run_trace(cfg, brain, groups, dn_idx, sd, ctrl, SCENARIOS[sc],
                                    b0=b0dn[sd], seconds=args.seconds, rig=rig, turn_source=src)
                turns = np.array([r["turn"] for r in recs]) if recs else np.zeros(0)
                first = next((float(np.sign(t)) for t in turns if abs(t) >= 0.3), 0.0)
                cum = float(np.degrees(np.sum([r["om"] for r in recs]) * brain.dt)) if recs else 0.0
                res[(sd, sc, ctrl)] = s["reach"]
                print(f"{sd:5d} {sc:10s} {ctrl:9s} {str(s['reach'])[:1]:>5s} "
                      f"{(recs[-1]['t'] if s['reach'] else -1.0):7.2f} {s['cev']:4d} "
                      f"{s['stop%']:6.1f} {first:+9.0f} {cum:9.1f} {'-':>6s}")
    print("\n到达率：")
    for tag, seeds in (("screen", screen), ("confirm", confirm)):
        for ctrl in ("DN_ema", "POP", "POP_zero", "POP_swap"):
            n = sum(res[(sd, sc, ctrl)] for sd in seeds for sc in ("must_left", "must_right"))
            print(f"  [{tag}] {ctrl:9s} {n}/{len(seeds) * 2}")
    pc = sum(res[(sd, sc, "POP")] for sd in confirm for sc in ("must_left", "must_right"))
    dc = sum(res[(sd, sc, "DN_ema")] for sd in confirm for sc in ("must_left", "must_right"))
    ac = sum(res[(sd, sc, c)] for sd in confirm for sc in ("must_left", "must_right")
             for c in ("POP_zero", "POP_swap"))
    print(f"\n[判定] confirm 上 POP={pc} vs DN_ema={dc} vs 消融合计={ac} -> "
          + ("POP 提供有效闭环控制价值" if pc > max(dc, 0) and pc > 0 and ac < pc
             else "POP 未在新 seed 上胜过对照（离线可分但在线未获益）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
