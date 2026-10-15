"""P2d-21：固定 L50/R50 的无障碍底噪与刺激诱发信号可分性（只研究读出，不改控制）。

固定：L50/R50、真实脑 ID、Trace τ=0.2、gain=8、sign=-1、BASE 五路编码、10Hz ToF / 50Hz 脑、
正式控制器的 1.2s 无遮挡 b0 标定。禁止重选/gain 优化/训练/重定义方向。

背景 = **正常 BASE 无障碍 A**（五路 NO_TARGET，但保留编码器原背景 chase + fwd 注入），
**不是** P2d-20 的 zero_injection（P2d-20 零注入=不施加任何外部注入；此处保留 BASE 背景注入）。

规则：screen seeds 64/101/202/303 定阈值 → 冻结 → 未用过的新 seeds 1103/1201/1301/1409/1511 验证。
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
# 每条件的分段计划（秒）
PLANS = {"A":  [("A", 8.0)],
         "B":  [("A", 1.2), ("FL", 2.0), ("A", 4.8)],
         "C":  [("A", 1.2), ("FR", 2.0), ("A", 4.8)],
         "D":  [("A", 1.2), ("FR", 1.6), ("A", 1.2), ("FL", 1.6), ("A", 2.4)]}
WIN_S = 0.5            # 预先声明的窗口统计量
FAR_Q = 99.0           # 背景 99 百分位


def make_frame(stim, seq, t):
    rng = {n: None for n in ORDER}
    st = {n: "NO_TARGET" for n in ORDER}
    for n, mm in stim.items():
        rng[n], st[n] = float(mm), "VALID"
    return TofFrame(session_id="sig", seq=seq, t=t, healthy=True, ranges=rng, status=st,
                    ages_ms={n: 100.0 for n in ORDER}, init_count=0, read_errors=0, reinit_count=0)


def schedule(plan, dt):
    """返回每步的刺激名。"""
    out = []
    for name, dur in plan:
        out += [name] * int(round(dur / dt))
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
    bl = np.asarray(dn_idx)[np.array(list(L_IDX), int)]
    br = np.asarray(dn_idx)[np.array(list(R_IDX), int)]
    D, spkL, spkR = [], [], []
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
        spkL.append(int(np.isin(fired, bl).sum()) if len(fired) else 0)
        spkR.append(int(np.isin(fired, br).sum()) if len(fired) else 0)
    D = np.array(D)
    return {"D": D, "Dmb": D - b0, "raw": sign * gain * (D - b0),
            "turn": np.clip(sign * gain * (D - b0), -1, 1),
            "spkL": np.array(spkL), "spkR": np.array(spkR), "names": seqs, "dt": dt}


def roll_mean(x, w):
    if w <= 1:
        return x.copy()
    k = np.ones(w) / w
    return np.convolve(x, k, mode="full")[:len(x)]


def stats(x, dt):
    """背景统计：均值/标准差/分位/持续偏移/自相关时间。"""
    m, s = float(np.mean(x)), float(np.std(x))
    q = np.percentile(x, [1, 5, 50, 95, 99]).tolist()
    ac = np.correlate(x - m, x - m, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    tau_corr = next((i * dt for i in range(1, len(ac)) if ac[i] < 1 / np.e), float("nan"))
    return {"mean": m, "std": s, "q": q, "corr_tau_s": tau_corr,
            "block_means": {f"{T}s": float(np.mean(x[:int(T / dt)])) for T in (1, 5, 10, 20, 40)}}


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
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    screen = [int(s) for s in args.screen_seeds.split(",")]
    new = [int(s) for s in args.new_seeds.split(",")]

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=args.device, seed=screen[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dt = brain.dt
    w = max(1, int(round(WIN_S / dt)))
    print("[P2d-21] 背景=正常 BASE 无障碍 A（NO_TARGET + 编码器原背景 chase/fwd），"
          "非 P2d-20 的 zero_injection")
    print(f"         固定: L50/R50, τ={TAU}, gain={GAIN}, sign={SIGN}; 窗口统计={WIN_S}s; "
          f"阈值=背景 {FAR_Q:.0f} 百分位\n")

    def b0_of(seed):
        return estimate_b0(cfg, brain, groups, dn_idx, seed)

    # 任务二：screen seeds 的运行背景（每步统计；1/5/10/20/40s 尺度分开）
    print("=== 任务二 无障碍背景（40s）===")
    print(f"{'seed':>6s} {'b0':>7s} {'mean(D-b0)':>11s} {'std':>7s} {'q99|.|':>7s} "
          f"{'corr_tau':>8s} {'blk1/5/20/40':>22s}")
    bg_thr = []
    for sd in screen + new:
        b0 = b0_of(sd)
        r = run_job(cfg, brain, groups, dn_idx, sd, [("A", args.bg_seconds)], b0=b0)
        x = r["Dmb"]
        st = stats(x, dt)
        # 预声明窗口统计量：|D-b0| 的 0.5s 滚动均值
        wstat = roll_mean(np.abs(x), w)
        thr = float(np.percentile(wstat, FAR_Q))
        bg_thr.append((sd, thr, float(np.mean(np.abs(x))), st))
        print(f"{sd:6d} {b0:+7.3f} {st['mean']:+11.3f} {st['std']:7.3f} {thr:7.3f} "
              f"{st['corr_tau_s']:8.3f}   " +
              " ".join(f"{st['block_means'][f'{T}s']:+.3f}" for T in (1, 5, 20, 40)))
    screen_thr = [t for sd, t, _, _ in bg_thr if sd in screen]
    THR = float(np.mean(screen_thr))
    print(f"\n[锁定阈值] screen 背景 99 百分位(窗口 |D-b0| 均值的均值) 阈值 THR={THR:.4f}（冻结，"
          f"不在验证集上调）")
    print(f"  背景每步 |D-b0| 均值≈{np.mean([m for _, _, m, _ in bg_thr]):.3f} → 经 gain=8 "
          f"对应 ω≈{8 * np.mean([m for _, _, m, _ in bg_thr]) * 2.6:.2f} rad/s 量级的**假转向潜力**")

    # 任务三/四：刺激对照 + 可分性（screen 建规则，new 验证）
    print("\n=== 任务三/四 刺激 vs 背景（同 seed/同脑年龄）===")
    print(f"{'seed':>6s} {'cond':>4s} {'preΔmax':>8s} {'ΔD stim':>8s} {'ΔD post':>8s} "
          f"{'detect':>6s} {'dir':>4s} {'t_exceed':>8s}")
    res = []
    for sd in screen + new:
        b0 = b0_of(sd)
        base = run_job(cfg, brain, groups, dn_idx, sd, PLANS["A"], b0=b0)["Dmb"]
        for cond in ("B", "C", "D"):
            r = run_job(cfg, brain, groups, dn_idx, sd, PLANS[cond], b0=b0)
            x = r["Dmb"]
            names = r["names"]
            m = min(len(x), len(base))
            pre = [i for i in range(m) if names[i] == "A" and i < 60]
            # 刺激窗口 / 撤除后窗口
            stimw = [i for i in range(len(x)) if names[i] in ("FL", "FR")]
            postw = [i for i in range(len(x)) if names[i] == "A" and i > (stimw[-1] if stimw else 0)]
            pre_ok = float(np.max(np.abs(x[pre] - base[pre]))) if pre else 0.0
            eff_stim = float(np.mean(x[stimw] - base[stimw])) if stimw else 0.0
            eff_post = float(np.mean(x[postw] - base[postw])) if postw else 0.0
            wstat = roll_mean(np.abs(x), w)
            thr = THR
            hit = [i for i in stimw if wstat[i] > thr]
            detect = len(hit) > 0
            texc = (hit[0] - stimw[0]) * dt if hit else -1.0
            # 方向：刺激窗口 |D-b0| 的符号（先左 FL 期望 D 上升→Dmb>0；FR 期望 Dmb<0）
            sgn = float(np.mean(x[stimw])) if stimw else 0.0
            want = +1 if ("FL" in [names[i] for i in stimw]) else -1
            direrr = 0 if np.sign(sgn) == want and abs(sgn) > 1e-3 else 1
            res.append((sd, cond, detect, direrr, texc, pre_ok, eff_stim, eff_post))
            print(f"{sd:6d} {cond:>4s} {pre_ok:8.4f} {eff_stim:+8.3f} {eff_post:+8.3f} "
                  f"{str(detect):>6s} {direrr:4d} {texc:8.2f}")
    # FAR：无遮挡 A 上误报（窗口统计超阈值的时间占比）
    far = []
    for sd in screen + new:
        b0 = b0_of(sd)
        x = run_job(cfg, brain, groups, dn_idx, sd, [("A", args.bg_seconds)], b0=b0)["Dmb"]
        wstat = roll_mean(np.abs(x), w)
        far.append(float(np.mean(wstat > THR)))
    # 左右可分性只用**单一方向**刺激 B(FL)/C(FR)；D(交替)方向按构造不可判，单列
    lr = [(sd, c, d, e, t) for sd, c, d, e, t, a, b, cc in res if c in ("B", "C")]
    det_new = [d for sd, c, d, e, t in lr if sd in new]
    dir_new = [e for sd, c, d, e, t in lr if sd in new]
    texc_new = [t for sd, c, d, e, t in lr if sd in new and t >= 0]
    det_screen = [d for sd, c, d, e, t in lr if sd in screen]
    dir_screen = [e for sd, c, d, e, t in lr if sd in screen]
    dplan = [(sd, c, d) for sd, c, d, *_ in res if c == "D"]
    print(f"\n[汇总·new B/C] 检出率={np.mean(det_new) * 100:.0f}%  方向错率={np.mean(dir_new) * 100:.0f}%  "
          f"超阈值时间中位={np.median(texc_new) if texc_new else float('nan'):.2f}s  "
          f"无遮挡误报(步占比)={np.mean(far) * 100:.1f}%")
    print(f"[汇总·screen B/C] 检出率={np.mean(det_screen) * 100:.0f}%  方向错率={np.mean(dir_screen) * 100:.0f}%")
    print(f"[D 交替计划] 检出 {sum(d for _, _, d in dplan)}/{len(dplan)}（方向按构造不可判，"
          "仅用于切换/撤除检查）")
    pass_ = bool(np.mean(det_new) >= 0.6 and np.mean(dir_new) <= 0.3 and np.mean(far) < 0.1)
    print(f"[GATE] {'PASS' if pass_ else 'FAIL'}: 新 seed（左右单一方向 B/C）检出率 "
          f"{np.mean(det_new) * 100:.0f}% / 方向错率 {np.mean(dir_new) * 100:.0f}% / 无遮挡误报 "
          f"{np.mean(far) * 100:.1f}% → "
          + ("现有 L50/R50 存在可供未来因果控制的神经方向信号（仅建议下一步读出改进，等待批准）"
             if pass_ else "刺激与背景大量重叠 → 当前固定群体读出不适合直接承担稳定运动控制；"
             "停止对其追加增益/保持/回正，转向感觉—神经—运动接口整体有效性审查"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
