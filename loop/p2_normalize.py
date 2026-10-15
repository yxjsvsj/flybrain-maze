"""P2d-22：固定群体神经读出的**背景噪声归一化**（严格离线对照；不读未来数据/不训练）。

候选（唯一）：
  b0 仍按原算法（estimate_b0：1.2s 无遮挡后半段 D 均值）；
  sigma = **同一次 1.2s 校准**中后半段 D 的标准差（不额外读未来 40s）；
  z(t) = (D(t)-b0) / max(sigma, sigma_min)，sigma_min 由 screen seeds 校准数据一次性规则确定；
  检测用同样的因果 0.5s 窗口 `roll_mean(|z|) > THR_z`，THR_z 由 screen 无遮挡背景一次性确定后冻结；
  方向只由当前/过去 signed z(t) 判定（正=FL，负=FR）。
披露：该归一化是**人工神经信号处理**，不是果蝇天然神经电路。旧读出（THR=0.1246，原样）作对照。

关键诊断：1.2s 校准能否可靠估计该次运行的背景尺度（sigma_cal vs 40s 真实背景 sigma）。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from loop.p2_signal import (FAR_Q, PLANS, THR_FROZEN, WIN_S, detect_segment, event_far, over_runs,
                            roll_mean, run_job, schedule, segments)
from loop.population_turn import GAIN, SIGN, TAU, estimate_b0
from loop.run import make_setup

CAL_S = 1.2


def calib(cfg, brain, groups, dn_idx, seed):
    """同一次 1.2s 无遮挡校准：返回 (b0, sigma)，b0 与原算法一致。"""
    r = run_job(cfg, brain, groups, dn_idx, seed, [("A", CAL_S)], b0=0.0)
    D = r["D"]
    n = len(D)
    h = D[n // 2:]
    return float(np.mean(h)), float(np.std(h))


def detect_pre(wabs, wsq, thr, i0, i1, dt, want, pre):
    """在 [i0,i1) 段内检测；`pre` 为该试次**自身**刺激前窗口的索引。
    只有**段内新起**的超阈事件算新检出；若超阈在建段前已持续 -> carryover（单列）。"""
    pre_fa = bool(np.any(wabs[pre] > thr)) if len(pre) else False
    runs = [(a, b) for a, b in over_runs(wabs > thr) if a < i1 and b > i0]
    new = [r for r in runs if r[0] >= i0]
    carry = any(r[0] < i0 for r in runs)
    if new:
        a = new[0][0]
        d = int(np.sign(wsq[a]))
        oc = "correct" if d == want else ("wrong" if d == -want else "uncertain")
        return oc, (a - i0) * dt, d, pre_fa, carry
    if carry:
        return "carryover", -1.0, 0, pre_fa, True
    return "miss", -1.0, 0, pre_fa, False


def _eval(series_map, thr, dt, w, plan, extra=None):
    """对一组条件（含 A 对照）用给定阈值评价；返回每段的 (outcome, lat, dir, preFA, carry)。"""
    out = []
    for cond, x in series_map.items():
        if cond == "A":
            continue
        wabs, wsq = roll_mean(np.abs(x), w), roll_mean(x, w)
        segs = segments(plan[cond], dt)
        for idx, (name, i0, i1) in enumerate(segs):
            if name == "A":
                continue
            want = +1 if name == "FL" else -1
            pre = list(range(max(0, i0 - int(1.0 / dt)), i0))
            out.append((cond, name, *detect_pre(wabs, wsq, thr, i0, i1, dt, want, pre)))
    return out


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-seeds", default="64,101,202,303")
    ap.add_argument("--new-seeds", default="1601,1709,1801,1907,2003,2111,2203,2309")
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
    print("[P2d-22] 归一化=人工神经信号处理；b0 原算法；sigma=同一次1.2s校准后半段 D 的 std")
    print(f"         旧读出 THR={THR_FROZEN}（原样）；新读出 0.5s 窗口 roll_mean(|z|)")

    # screen：校准 sigma + 40s 背景 sigma（可靠性诊断）+ 定 sigma_min / THR_z
    print("\n=== screen: 1.2s 校准 sigma 与 40s 真实背景 sigma 的可比性 ===")
    print(f"{'seed':>6s} {'b0':>7s} {'sig_cal':>8s} {'sig_bg40':>9s} {'sig_bg/b0校准后':>16s}")
    sigcals, b0s = {}, {}
    z_bg = {}
    for sd in screen + new:
        b0, sg = calib(cfg, brain, groups, dn_idx, sd)
        b0s[sd], sigcals[sd] = b0, sg
        bg = run_job(cfg, brain, groups, dn_idx, sd, [("A", args.bg_seconds)], b0=b0)["Dmb"]
        z_bg[sd] = bg                                   # 原始 D-b0（后面只除一次 sigma）
    for sd in screen:
        s40 = float(np.std(z_bg[sd]))
        print(f"{sd:6d} {b0s[sd]:+7.3f} {sigcals[sd]:8.4f} {s40:9.4f} "
              f"{'一致' if 0.5 < sigcals[sd] / (s40 + 1e-9) < 2.0 else '不一致'}")
    sigma_min = float(np.median([sigcals[sd] for sd in screen]))
    ratio = [sigcals[sd] / (float(np.std(z_bg[sd])) + 1e-9) for sd in screen]
    print(f"  sigma_min(规则=screen 校准 sigma 中位数)={sigma_min:.4f}；"
          f"校准/真实背景尺度比 中位={np.median(ratio):.2f}（接近1=校准能代表背景尺度）")
    # THR_z：screen 无遮挡背景 0.5s 窗口 |z| 的 q99（每 seed q99 取均值，与原规则同构）
    thrz = []
    for sd in screen:
        zz = z_bg[sd] / max(sigcals[sd], sigma_min)
        thrz.append(float(np.percentile(roll_mean(np.abs(zz), w), FAR_Q)))
    THR_Z = float(np.mean(thrz))
    print(f"  冻结 THR_z={THR_Z:.3f}（screen 背景窗口 |z| 的 q99 均值）")

    # 配对比较：screen/new 各 seed，旧 vs 新
    def evaluate(seed, use_new):
        b0, sg = b0s[seed], sigcals[seed]
        scale = max(sg, sigma_min) if use_new else 1.0
        thr = THR_Z if use_new else THR_FROZEN
        # 40s 背景（误报事件）
        bg = run_job(cfg, brain, groups, dn_idx, seed, [("A", args.bg_seconds)], b0=b0)["Dmb"]
        ev = event_far(roll_mean(np.abs(bg / scale), w), thr, dt, 40.0)
        # 刺激
        series = {"A": run_job(cfg, brain, groups, dn_idx, seed, PLANS["A"], b0=b0)["Dmb"]}
        for cond in ("B", "C", "D"):
            series[cond] = run_job(cfg, brain, groups, dn_idx, seed, PLANS[cond], b0=b0)["Dmb"]
        det = _eval({k: v / scale for k, v in series.items()}, thr, dt, w, PLANS)
        return ev, det

    print(f"\n=== 配对（同 seed/输入/脑）：旧 vs 新 ===")
    print(f"{'seed':>6s} {'set':>4s} {'read':>6s} {'FAev/40s':>9s} {'FAstp%':>7s} {'corr':>4s} "
          f"{'wrong':>5s} {'miss':>4s} {'carry':>5s} {'latMed':>7s}")
    agg = {"screen": {"old": [], "new": []}, "new": {"old": [], "new": []}}
    for grp, seeds in (("screen", screen), ("new", new)):
        for sd in seeds:
            for tag, un in (("old", False), ("new", True)):
                ev, det = evaluate(sd, un)
                oc = [d[2] for d in det]
                lat = [d[3] for d in det if d[3] >= 0]
                agg[grp][tag].append({"fa": ev["n_events"], "fa_step": ev["step_frac"],
                                      "corr": sum(1 for o in oc if o == "correct"),
                                      "wrong": sum(1 for o in oc if o == "wrong"),
                                      "miss": sum(1 for o in oc if o == "miss"),
                                      "carry": sum(1 for o in oc if o == "carryover"),
                                      "n": len(oc), "lat": float(np.median(lat)) if lat else float("nan")})
                a = agg[grp][tag][-1]
                print(f"{sd:6d} {grp:>4s} {tag:>6s} {ev['n_events']:9d} {ev['step_frac'] * 100:6.1f}% "
                      f"{a['corr']:4d} {a['wrong']:5d} {a['miss']:4d} {a['carry']:5d} {a['lat']:7.2f}")

    # 新旧汇总 + 验收
    def summ(grp, tag):
        rs = agg[grp][tag]
        n = sum(r["n"] for r in rs)
        return {"fa": float(np.mean([r["fa"] for r in rs])),
                "corr": sum(r["corr"] for r in rs), "n": n,
                "wrong": sum(r["wrong"] for r in rs), "carry": sum(r["carry"] for r in rs),
                "rate": sum(r["corr"] for r in rs) / max(1, n),
                "lat": float(np.nanmedian([r["lat"] for r in rs]))}
    print()
    for grp in ("screen", "new"):
        so, sn = summ(grp, "old"), summ(grp, "new")
        for tag, s in (("old", so), ("new", sn)):
            print(f"[{grp}/{tag}] 正确检出={s['corr']}/{s['n']} ({s['rate'] * 100:.0f}%) 错误={s['wrong']} "
                  f"carryover={s['carry']} FA事件/40s均值={s['fa']:.1f} 首检延迟中位={s['lat']:.2f}s")
    so, sn = summ("new", "old"), summ("new", "new")
    cond1 = sn["corr"] >= so["corr"] and sn["rate"] >= 0.80
    cond2 = sn["wrong"] <= so["wrong"]
    cond3 = sn["fa"] <= 0.5 * so["fa"]
    cond4 = (sn["lat"] <= so["lat"] + 0.2) or np.isnan(so["lat"]) or np.isnan(sn["lat"])
    ok = cond1 and cond2 and cond3 and cond4
    print(f"\n[验收·new] 1正确不降&≥80%:{cond1}  2不新增方向错:{cond2}  3误报≤旧50%:{cond3} "
          f"4延迟≤旧+0.2s:{cond4}")
    print(f"[GATE] {'PASS' if ok else 'FAIL'}: "
          + ("保留唯一离线候选，等待批准下一轮最小虚拟闭环对照"
             if ok else "归一化路线未达验收 → 停止该路线，不堆叠滤波/增益/死区，"
             "转向感觉—神经—运动接口建模审查"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
