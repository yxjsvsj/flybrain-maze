"""P2d-18：感觉输入反事实实验（只诊断；不改控制/不调参/不重选/不训练）。

从既有轨迹的**持续强转向区间**（事先固定规则：|turn|≥0.9 的最长连续段之起点）取分叉时刻，
用**相同 seed、相同初始化**重跑 MaleCNS：分叉前回放**完全相同**的 ToF 历史（校验 D/turn/注入一致），
分叉后只改传感器：
  orig  回放实际 ToF 后续数据      none  五路全 NO_TARGET
  noF   只撤前方 F                 noLR  只撤 FL/FR（F 保持原始）
只研究神经响应（不计算车辆运动）；只用当时与过去数据；不将反事实称作导航成功率。

⚠ 更正用语：A∩B=0 是代码互斥定义，不是独立证据；B=强转向且传感器有目标**只是相关性**；
C 的饱和比例需由**未限幅 raw 超过 ±1** 直接确认；**正前方 F 同属左右感觉池**，单独披露。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.encode_tof import PhysicalToFEncoder
from loop.local_avoidance_bench import SCENARIOS
from loop.p2_diag import run_diag
from loop.population_turn import GAIN, L_IDX, R_IDX, SIGN, TAU, estimate_b0
from loop.run import make_setup

_CODE2ST = {2.0: "VALID", 1.0: "TOO_NEAR", 0.0: "NO_TARGET"}
MODES = ("orig", "none", "noF", "noLR")


def pick_fork(turn, t, thr=0.5, min_s=0.5):
    """事先固定规则：|turn|>=thr(=诊断的 assertive 阈值) 的最长连续段起点；
    若无满足段，则退化为全程 |turn| 最大的一步（仍与假设无关）。成功/失败同规则。"""
    m = np.abs(turn) >= thr
    best, blen, i, n = -1, 0, 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            if t[j] - t[i] >= min_s and (j - i) > blen:
                best, blen = i, j - i
            i = j + 1
        else:
            i += 1
    if best < 0 and len(turn):
        best = int(np.argmax(np.abs(turn)))
    return best


def frame_at(rec, k, mode="orig"):
    """从记录的 ToF 序列重建第 k 步的帧；mode 决定分叉后哪些通道被撤除。"""
    rng, st = {}, {}
    for n in ORDER:
        code = float(rec["st" + n][k])
        val = float(rec["tof" + n][k])
        if mode == "none" or (mode == "noF" and n == "F") or (mode == "noLR" and n in ("FL", "FR")):
            rng[n], st[n] = None, "NO_TARGET"
        elif code <= 0.0:
            rng[n], st[n] = None, "NO_TARGET"
        else:
            rng[n], st[n] = val, _CODE2ST.get(code, "NO_TARGET")
    return TofFrame(session_id="cf", seq=int(rec["seq"][k]), t=float(rec["t"][k]), healthy=True,
                    ranges=rng, status=st, ages_ms={n: 100.0 for n in ORDER},
                    init_count=0, read_errors=0, reinit_count=0)


def run_cf(cfg, brain, groups, dn_idx, seed, rec, fork_i, mode, *, b0, gain=GAIN, sign=SIGN,
           tau=TAU, n_after=100, mpc=0.40):
    """分叉后重跑 n_after 步（0.02s/步 → 2s）；分叉前用原始帧确保状态一致。"""
    from flybrain import Trace
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    tr = Trace(brain, idx=np.asarray(dn_idx)[np.array(list(L_IDX) + list(R_IDX), int)], tau=tau)
    brain.reset(seed)
    n = min(fork_i + n_after, len(rec["t"]))
    Ds, turns = [], []
    for k in range(n):
        md = "orig" if k < fork_i else mode
        f = frame_at(rec, k, md)
        d = enc.sense(f)
        enc.update(f, d)
        inj, _info = enc.inject()
        fired = brain.step(inject=inj)
        a = np.asarray(tr.observe(fired), np.float32)
        Dv = float(a[:len(L_IDX)].mean() - a[len(L_IDX):].mean())
        Ds.append(Dv)
        turns.append(float(np.clip(sign * gain * (Dv - b0), -1.0, 1.0)))
    return np.array(Ds, float), np.array(turns, float)


def metrics(Ds, turns, fork_i, b0, dt, win_s=2.0):
    na = int(round(win_s / dt))
    post_D = Ds[fork_i:fork_i + na] - b0
    post_t = turns[fork_i:fork_i + na]
    pre_t = turns[max(0, fork_i - na):fork_i]
    pre_abs = float(np.mean(np.abs(pre_t))) if len(pre_t) else 0.0
    half = 0.5 * pre_abs
    tfall = -1.0
    for j in range(len(post_t)):
        if abs(post_t[j]) <= half:
            tfall = round(j * dt, 2)
            break
    return {"signed_Dmb": float(np.mean(post_D)) if len(post_D) else 0.0,
            "abs_Dmb": float(np.mean(np.abs(post_D))) if len(post_D) else 0.0,
            "turn_abs": float(np.mean(np.abs(post_t))) if len(post_t) else 0.0,
            "turn_dur": float(np.mean(np.abs(post_t) >= 0.5)) * win_s if len(post_t) else 0.0,
            "pre_turn_abs": pre_abs, "t_fall_half": tfall}


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
    from loop.dynamic_tof_sim import load_rig
    rig = load_rig()
    dt = brain.dt
    b0s = {}
    print(f"[P2d-18] 反事实：分叉规则=最长 |turn|>=0.9 段起点  分叉后 2s  只研究神经响应")
    for case in args.cases.split(","):
        sd_s, sc = case.split(":")
        seed = int(sd_s)
        if seed not in b0s:
            b0s[seed] = estimate_b0(cfg, brain, groups, dn_idx, seed)
        b0 = b0s[seed]
        rec = run_diag(cfg, brain, groups, dn_idx, seed, SCENARIOS[sc], seconds=args.seconds,
                       rig=rig, b0=b0)
        fork = pick_fork(rec["turn"], rec["t"])
        # 分叉前原始域内一致性（orig 模式重放的 D/turn 必须与原始一致）
        res = {}
        for mode in MODES:
            Ds, turns = run_cf(cfg, brain, groups, dn_idx, seed, rec, fork, mode, b0=b0)
            pre_ok = (np.allclose(Ds[:fork], rec["D"][:fork], atol=1e-4)
                      and np.allclose(turns[:fork], rec["turn"][:fork], atol=1e-4))
            res[mode] = (metrics(Ds, turns, fork, b0, dt), pre_ok)
        tag = "成功" if rec["reach"] else "失败"
        print(f"\n=== seed {seed} / {sc} [{tag}]  b0={b0:+.3f}  fork_t={rec['t'][fork]:.1f}s ===")
        print(f"{'mode':6s} {'signed(D-b0)':>12s} {'abs(D-b0)':>10s} {'|turn|均':>8s} "
              f"{'|turn|>=.5 时长':>12s} {'降至半值t':>9s} {'pre一致':>7s}")
        for mode in MODES:
            m, ok = res[mode]
            print(f"{mode:6s} {m['signed_Dmb']:+12.3f} {m['abs_Dmb']:10.3f} {m['turn_abs']:8.3f} "
                  f"{m['turn_dur']:12.2f} {m['t_fall_half']:9.2f} {str(ok):>7s}")
        # 相对贡献：撤除后 |turn| 相对 orig 的比例
        to = res["orig"][0]["turn_abs"] + 1e-9
        print(f"  持续刺激相对贡献: 全撤 none 后 |turn|={res['none'][0]['turn_abs']:.3f}"
              f"({res['none'][0]['turn_abs'] / to * 100:.0f}% of orig)  "
              f"撤F={res['noF'][0]['turn_abs']:.3f}({res['noF'][0]['turn_abs'] / to * 100:.0f}%)  "
              f"撤FL/FR={res['noLR'][0]['turn_abs']:.3f}({res['noLR'][0]['turn_abs'] / to * 100:.0f}%)")
        print("  披露：F 同时进入左右感觉池 → noF 会同时削弱两侧；noLR 保留 F。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
