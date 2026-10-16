"""P2d-27 第二关：固定 MaleCNS 神经状态保持预检（**只用固定 46 EPG**；不改神经参数；无 yaw/控车）。

预注册见 `loop/STATE_PREREG.md`（**运行前冻结**）：时间结构、提示位置/幅度、窗口、阈值、种子全部冻结。
纯分析函数（`pop_vector`/`window_series`/`trial_metrics`/`angular_profile`）可离线测试；
`run_trial` 需要 flybrain 模拟器。
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ------- 冻结常量（STATE_PREREG.md） -------
PRE, CUE, OBS, WINDOW = 25, 25, 250, 5
TOTAL = PRE + CUE + OBS
CUE_ANGLES_DEG = [0.0, 120.0]
CUE_K, CUE_AMP = 4, 0.5
R_VALID, BUMP_R, MIN_SPK = 0.20, 0.30, 3
SUSTAIN_MIN_S = 1.0
TAU_REF = 0.1
PHASE_SEP_MIN = 90.0
SCREEN_SEEDS = [11, 12, 13]
CONFIRM_SEEDS = [21, 22, 23, 24]
NBINS = 20
WIN_S = WINDOW * 0.02


# ---------------- pure analysis ----------------
def pop_vector(counts, angles):
    counts = np.asarray(counts, float); tot = counts.sum()
    if tot <= 0:
        return 0.0, float("nan")
    C = float((counts * np.cos(angles)).sum() / tot)
    S = float((counts * np.sin(angles)).sum() / tot)
    return float(np.hypot(C, S)), float(np.arctan2(S, C))


def window_sums(counts_time, window=WINDOW):
    n = counts_time.shape[0]; k = n // window
    return counts_time[:k * window].reshape(k, window, -1).sum(1)


def window_series(counts_time, angles, window=WINDOW, R_valid=R_VALID, min_spk=MIN_SPK):
    """返回 (R[], phase[])；窗内发放<min_spk 或 R<R_valid 时 phase=NaN（相位无效）。"""
    W = window_sums(counts_time, window)
    Rs = np.zeros(len(W)); ph = np.full(len(W), np.nan)
    for i, w in enumerate(W):
        if w.sum() < min_spk:
            continue
        R, p = pop_vector(w, angles); Rs[i] = R
        if R >= R_valid:
            ph[i] = p
    return Rs, ph


def longest_valid_run(ph):
    best = cur = 0
    for x in ph:
        cur = cur + 1 if x == x else 0        # NaN != NaN
        best = max(best, cur)
    return best


def angular_profile(counts_pooled, angles, nbins=NBINS):
    h, _ = np.histogram(angles, bins=nbins, range=(-np.pi, np.pi), weights=counts_pooled)
    return h


def n_peaks(hist, frac=0.25):
    h = np.convolve(np.r_[hist[-1], hist, hist[0]], np.ones(3) / 3, mode="valid")
    if h.max() <= 0:
        return 0
    m = len(h)
    return sum(1 for i in range(m)
               if h[i] >= h[(i - 1) % m] and h[i] > h[(i + 1) % m] and h[i] >= frac * h.max())


def fwhm_deg(hist):
    if hist.max() <= 0:
        return float("nan")
    half = hist.max() / 2.0
    m = len(hist)
    # 从最大 bin 向两侧扩展，数 ≥half 的连续 bin（环形）
    c = int(np.argmax(hist)); cnt = 1
    i = (c - 1) % m
    while hist[i] >= half and cnt < m:
        cnt += 1; i = (i - 1) % m
    i = (c + 1) % m
    while hist[i] >= half and cnt < m:
        cnt += 1; i = (i + 1) % m
    return cnt * (360.0 / m)


def circ_mean_deg(ph):
    ph = np.asarray(ph, float); ph = ph[~np.isnan(ph)]
    if len(ph) == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(np.sin(ph).mean(), np.cos(ph).mean())))


def trial_metrics(counts_time, angles):
    """counts_time: (TOTAL, n_epg)。返回逐试验指标。"""
    Rs, ph = window_series(counts_time, angles)
    nw_pre, nw_cue = PRE // WINDOW, CUE // WINDOW
    obs = slice(nw_pre + nw_cue, len(Rs))
    R_obs, ph_obs = Rs[obs], ph[obs]
    valid_rate = float(np.mean(~np.isnan(ph_obs))) if len(ph_obs) else 0.0
    sustain_w = longest_valid_run(ph_obs)
    pooled = counts_time[(nw_pre + nw_cue) * WINDOW:].sum(0)
    hist = angular_profile(pooled, angles)
    return {"pre_medianR": round(float(np.median(Rs[:nw_pre])), 4) if nw_pre else None,
            "cue_medianR": round(float(np.median(Rs[nw_pre:nw_pre + nw_cue])), 4),
            "obs_medianR": round(float(np.median(R_obs)), 4) if len(R_obs) else None,
            "valid_rate": round(valid_rate, 3),
            "sustain_s": round(sustain_w * WIN_S, 2),
            "final_phase_deg": round(circ_mean_deg(ph_obs[-10:]), 1),
            "n_peaks": n_peaks(hist), "fwhm_deg": round(fwhm_deg(hist), 1),
            "n_spikes_obs": int(pooled.sum())}


# ---------------- simulator side ----------------
def epg_setup(data_dir):
    rows = list(csv.DictReader(open(os.path.join(HERE, "p2d26f_cells.csv"), encoding="utf-8")))
    epg = [(int(r["bodyId"]), float(r["circ_mean_deg"]))
           for r in rows if r["group"] == "EPG" and r["valid"] == "True"]
    meta = np.load(os.path.join(str(data_dir), "brain.npz"))
    ids = np.asarray(meta["ids"])
    idx_of = {int(b): i for i, b in enumerate(ids.tolist())}
    missing = [b for b, _ in epg if b not in idx_of]
    assert not missing, f"EPG bodyIds not in simulator: {missing}"
    order = np.argsort([idx_of[b] for b, _ in epg])
    epg_body = [epg[i][0] for i in order]
    epg_idx = np.array([idx_of[b] for b in epg_body], np.int64)
    epg_ang = np.radians([epg[i][1] for i in order])
    return epg_body, epg_idx, epg_ang


def nearest_epg(angles, target_deg, k=CUE_K):
    t = np.radians(target_deg)
    d = np.abs(np.arctan2(np.sin(angles - t), np.cos(angles - t)))
    return np.argsort(d)[:k]


def run_trial(brain, epg_idx, seed, cue_local=None, cue_amp=0.0, diffuse=False):
    """连续跑 TOTAL 步；CUE 段施加提示；撤除处不 reset。返回 (TOTAL, n_epg) 发放计数。"""
    brain.reset(seed)
    counts = np.zeros((TOTAL, len(epg_idx)), np.int32)
    for t in range(TOTAL):
        inject = ()
        if PRE <= t < PRE + CUE:
            if diffuse:
                inject = [(epg_idx, cue_amp * CUE_K / len(epg_idx))]
            elif cue_local is not None:
                inject = [(epg_idx[cue_local], cue_amp)]
        fired = np.asarray(brain.step(inject=inject))
        hit = fired[np.isin(fired, epg_idx)]
        if len(hit):
            j = np.searchsorted(epg_idx, hit)
            counts[t, j] += 1
    return counts


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                     # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "p2d27_trials.csv"))
    ap.add_argument("--series", default=os.path.join(HERE, "p2d27_timeseries.npz"))
    a = ap.parse_args(argv)
    from flybrain.brain import FlyBrain
    from flybrain.data import ensure_data
    data = ensure_data()
    epg_body, epg_idx, epg_ang = epg_setup(data)
    print(f"[P2d-27] EPG set = {len(epg_idx)} (fixed 46; 20818 NOT included)")
    brain = FlyBrain(data=data, device=a.device, seed=64, batch=1)

    trials = []
    series = {}
    conds = [("nocue", None), ("cue0", 0.0), ("cue120", 120.0), ("diffuse", "D")]
    for ph, seeds in (("screen", SCREEN_SEEDS), ("confirm", CONFIRM_SEEDS)):
        for seed in seeds:
            for cname, cval in conds:
                if cval is None:
                    counts = run_trial(brain, epg_idx, seed)
                elif cval == "D":
                    counts = run_trial(brain, epg_idx, seed, diffuse=True, cue_amp=CUE_AMP)
                else:
                    loc = nearest_epg(epg_ang, cval)
                    counts = run_trial(brain, epg_idx, seed, cue_local=loc, cue_amp=CUE_AMP)
                m = trial_metrics(counts, epg_ang)
                m.update({"phase": ph, "seed": seed, "condition": cname})
                trials.append(m)
                series[f"{ph}_{seed}_{cname}"] = counts.astype(np.int16)
                print(f"  {ph} seed={seed} {cname:7s} obsR={m['obs_medianR']} "
                      f"valid={m['valid_rate']} sustain={m['sustain_s']}s "
                      f"phase={m['final_phase_deg']} peaks={m['n_peaks']}")
    cols = ["phase", "seed", "condition", "pre_medianR", "cue_medianR", "obs_medianR",
            "valid_rate", "sustain_s", "final_phase_deg", "n_peaks", "fwhm_deg", "n_spikes_obs"]
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for t in trials:
            w.writerow({k: t.get(k, "") for k in cols})
    np.savez_compressed(a.series, **series)
    print("wrote", a.out, a.series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
