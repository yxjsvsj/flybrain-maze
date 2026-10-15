"""P2d-21a：固定 L50/R50 神经方向检测器的**评价复核**（只改指标，不重优化阈值/不改模型）。

冻结：L50/R50、τ=0.2、gain=8、sign=-1、BASE 编码、b0 标定、窗口统计 `roll_mean(|D-b0|)`、
**THR=0.1246**（不得按新结果改）、screen 64/101/202/303、验证 1103/1201/1301/1409/1511。
可重跑同实验，但不重筛参数/神经元。

披露：原 THR = **screen 各 seed 背景 q99 的均值**，**不是**所有背景窗口的合并 q99。
在线检测只能用**单次运行的当前/过去**活动 + 固定 b0 + 固定 THR；离线净效应（时间匹配 A 对照）
是**神经科学指标**，两者分开汇报。

方向只由**检测时刻及之前**的有符号窗口 `roll_mean(D-b0)` 判定（正=FL，负=FR）；判不出计"方向不确定"。
连续超阈=**一次**事件；不把 screen+new 合并误报率当作验证集误报率。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.encode_tof import PhysicalToFEncoder
from loop.population_turn import GAIN, L_IDX, R_IDX, SIGN, TAU, estimate_b0
from loop.run import make_setup

STIM = {"A": {}, "FL": {"FL": 400.0}, "FR": {"FR": 400.0}}
PLANS = {"A":  [("A", 8.0)],
         "B":  [("A", 1.2), ("FL", 2.0), ("A", 4.8)],
         "C":  [("A", 1.2), ("FR", 2.0), ("A", 4.8)],
         "D":  [("A", 1.2), ("FR", 1.6), ("A", 1.2), ("FL", 1.6), ("A", 2.4)]}
WIN_S = 0.5
FAR_Q = 99.0
THR_FROZEN = 0.1246


def make_frame(stim, seq, t):
    rng = {n: None for n in ORDER}
    st = {n: "NO_TARGET" for n in ORDER}
    for n, mm in stim.items():
        rng[n], st[n] = float(mm), "VALID"
    return TofFrame(session_id="sig", seq=seq, t=t, healthy=True, ranges=rng, status=st,
                    ages_ms={n: 100.0 for n in ORDER}, init_count=0, read_errors=0, reinit_count=0)


def schedule(plan, dt):
    out = []
    for name, dur in plan:
        out += [name] * int(round(dur / dt))
    return out


def segments(plan, dt):
    """[(name, i0, i1), ...]"""
    out, acc = [], 0
    for name, dur in plan:
        m = int(round(dur / dt))
        out.append((name, acc, acc + m))
        acc += m
    return out


def run_job(cfg, brain, groups, dn_idx, seed, plan, *, b0, gain=GAIN, sign=SIGN, tau=TAU,
            tof_hz=10.0, mpc=0.40):
    from flybrain import Trace
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    tr = Trace(brain, idx=np.asarray(dn_idx)[np.array(list(L_IDX) + list(R_IDX), int)], tau=tau)
    brain.reset(seed)
    dt = brain.dt
    seqs = schedule(plan, dt)
    period = 1.0 / tof_hz
    frame, seq, last_t = None, 0, -1e9
    D = []
    for k, name in enumerate(seqs):
        t = k * dt
        if frame is None or (t - last_t) >= period - 1e-9:
            seq += 1
            frame = make_frame(STIM[name], seq, t)
            last_t = t
        d = enc.sense(frame)
        enc.update(frame, d)
        inj, _info = enc.inject()
        fired = brain.step(inject=inj)
        a = np.asarray(tr.observe(fired), np.float32)
        D.append(float(a[:len(L_IDX)].mean() - a[len(L_IDX):].mean()))
    D = np.array(D)
    return {"D": D, "Dmb": D - b0, "dt": dt, "names": seqs}


def roll_mean(x, w):
    if w <= 1:
        return x.copy()
    return np.convolve(x, np.ones(w) / w, mode="full")[:len(x)]


def over_runs(mask):
    """连续 True 段 -> [(i0, i1)]（i1 开区间）。"""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def event_far(wabs, thr, dt, seconds):
    """无遮挡背景误报：超阈步占比、每 seconds 秒**独立**事件数、事件时长。"""
    runs = over_runs(wabs > thr)
    durs = [(j - i) * dt for i, j in runs]
    return {"step_frac": float(np.mean(wabs > thr)), "n_events": len(runs),
            "per_s": len(runs) / (len(wabs) * dt) * seconds,
            "durs": durs}


def detect_segment(wabs, wsq, thr, i0, i1, dt, want):
    """单刺激段的因果检测：方向只用**检测时刻**的 signed 窗口；无检出=miss（不算正确）。"""
    runs = [(a, b) for a, b in over_runs(wabs > thr) if a < i1 and b > i0]
    if not runs:
        return "miss", -1.0, 0
    a = max(runs[0][0], i0)
    d = int(np.sign(wsq[a]))
    oc = "correct" if d == want else ("wrong" if d == -want else "uncertain")
    return oc, (a - i0) * dt, d


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-seeds", default="64,101,202,303")
    ap.add_argument("--new-seeds", default="1103,1201,1301,1409,1511")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bg-seconds", type=float, default=40.0)
    args = ap.parse_args(argv)
    screen = [int(s) for s in args.screen_seeds.split(",")]
    new = [int(s) for s in args.new_seeds.split(",")]

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=args.device, seed=screen[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dt = brain.dt
    w = max(1, int(round(WIN_S / dt)))
    THR = THR_FROZEN
    print(f"[P2d-21a] 冻结 THR={THR}（=screen 各 seed 背景 q99 的均值，非合并 q99）；"
          f"窗口={WIN_S}s；在线检测只用当前/过去 + 固定 b0/THR")
    b0s = {sd: estimate_b0(cfg, brain, groups, dn_idx, sd) for sd in screen + new}

    # 二：无遮挡误报（事件级、每 seed）
    print(f"\n=== 无遮挡背景误报（40s/seed，连续超阈=1 次事件）===")
    print(f"{'seed':>6s} {'set':>6s} {'stepFAR':>8s} {'events/40s':>10s} {'meanDur_s':>9s}")
    far_rows = []
    for sd in screen + new:
        r = run_job(cfg, brain, groups, dn_idx, sd, [("A", args.bg_seconds)], b0=b0s[sd])
        ev = event_far(roll_mean(np.abs(r["Dmb"]), w), THR, dt, 40.0)
        far_rows.append((sd, ev))
        md = float(np.mean(ev["durs"])) if ev["durs"] else 0.0
        print(f"{sd:6d} {'screen' if sd in screen else 'new':>6s} {ev['step_frac'] * 100:7.1f}% "
              f"{ev['n_events']:10d} {md:9.2f}")
    for tag, seeds in (("screen", screen), ("new", new)):
        fs = [ev for sd, ev in far_rows if sd in seeds]
        print(f"  [{tag}] stepFAR均值={np.mean([f['step_frac'] for f in fs]) * 100:.1f}%  "
              f"事件数均值={np.mean([f['n_events'] for f in fs]):.1f}/40s")

    # 三/四：因果方向检测（分段），离线净效应与在线检测分开
    print(f"\n=== 因果方向检测（检测量=roll_mean(|D-b0|)>THR；方向=检测时刻 signed roll_mean）===")
    print(f"{'seed':>6s} {'set':>6s} {'seg':>3s} {'preFA':>6s} {'outcome':>8s} {'lat_s':>6s} "
          f"{'dir':>4s} {'offΔD':>7s}")
    rows = []
    for sd in screen + new:
        b0, base = b0s[sd], run_job(cfg, brain, groups, dn_idx, sd, PLANS["A"], b0=b0s[sd])["Dmb"]
        for cond in ("B", "C", "D"):
            r = run_job(cfg, brain, groups, dn_idx, sd, PLANS[cond], b0=b0)
            x = r["Dmb"]
            wabs, wsq = roll_mean(np.abs(x), w), roll_mean(x, w)
            base_wabs = roll_mean(np.abs(base), w)
            for name, i0, i1 in segments(PLANS[cond], dt):
                if name == "A":
                    continue
                want = +1 if name == "FL" else -1
                pre = base_wabs[max(0, i0 - int(1.0 / dt)):i0] if i0 >= int(1.0 / dt) else base_wabs[:i0]
                pre_fa = bool(np.any(pre > THR))
                outcome, lat, d = detect_segment(wabs, wsq, THR, i0, i1, dt, want)
                m = min(len(x), len(base))
                off = float(np.mean(x[i0:min(i1, m)] - base[i0:min(i1, m)]))
                rows.append((sd, cond, name, pre_fa, outcome, lat, d, off))
                print(f"{sd:6d} {'scr' if sd in screen else 'new':>6s} {name:>3s} "
                      f"{str(pre_fa):>6s} {outcome:>8s} {lat:6.2f} {d:+4d} {off:+7.3f}")

    # 汇总（B/C=单方向；D 每段单列）
    for tag, seeds in (("screen", screen), ("new", new)):
        rs = [r for r in rows if r[0] in seeds and r[1] in ("B", "C")]
        det = [r for r in rs if r[4] != "miss"]
        cor = sum(1 for r in rs if r[4] == "correct")
        wrg = sum(1 for r in rs if r[4] == "wrong")
        unc = sum(1 for r in rs if r[4] == "uncertain")
        mis = sum(1 for r in rs if r[4] == "miss")
        pre = sum(1 for r in rs if r[3])
        lat = [r[5] for r in rs if r[5] >= 0]
        n = max(1, len(rs))
        print(f"\n[{tag} B/C, n={len(rs)} 次单方向刺激] 正确={cor} 错误={wrg} 不确定={unc} "
              f"漏检={mis} | 检出率={len(det) / n * 100:.0f}% 方向错率={wrg / n * 100:.0f}% "
              f"刺激前误报={pre}/{n} 首检延迟中位={np.median(lat) if lat else float('nan'):.2f}s")
    rp = [r for r in rows if r[0] in new and r[1] in ("B", "C")]
    dmis = [r for r in rows if r[0] in new and r[1] in ("B", "C") and r[4] == "miss"]
    print("[new 漏检明细] " + "  ".join(f"{sd}-{cond}({seg}) offΔD={off:+.3f}" for sd, cond, seg, pf, oc, la, dd, off in dmis)
          if dmis else "[new 漏检明细] 无")
    n_wrong = sum(1 for r in rp if r[4] == "wrong")
    n_new_det = sum(1 for r in rp if r[4] != "miss")
    far_new = np.mean([ev["n_events"] for sd, ev in far_rows if sd in new])
    pass_ = bool(n_new_det >= 0.6 * len(rp) and n_wrong <= 0.3 * len(rp) and far_new <= 5.0)
    print(f"\n[GATE] {'PASS' if pass_ else 'FAIL'}: new B/C 检出 {n_new_det}/{len(rp)} "
          f"方向错 {n_wrong}/{len(rp)}，new 无遮挡误报事件均值 {far_new:.1f}/40s → "
          + ("现有群体具有值得进一步研究的因果方向信号（等待批准读出实验）"
             if pass_ else "固定读出尚不可靠（误报频繁或触发方向错）→ 停止直接附加归一化/增益"))
    print("可信度：验证集仅 5 seed / 10 次单方向刺激，不等于已证明高可靠性。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
