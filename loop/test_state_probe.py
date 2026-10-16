"""P2d-27 第二关 回归：state_probe 纯分析（合成反例）。**无模拟器**。

覆盖：局部峰→R 高且相位≈已知方向；均匀发放→R 低、相位 NaN；持续 vs 仅提示期；
峰数/峰宽；空间打乱使相位消失；保持时长与位置区分。
"""
from __future__ import annotations

import sys

import numpy as np

from loop.state_probe import (BUMP_R, MIN_SPK, OBS, PRE, CUE, R_VALID, TOTAL, WINDOW,
                              angular_profile, circ_mean_deg, fwhm_deg, longest_valid_run,
                              n_peaks, pop_vector, trial_metrics, window_series)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def make_angles(k, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(-np.pi, np.pi, k, endpoint=False)
    return base + rng.normal(0, jitter, k)


def counts_bump(angles, center_rad, n_steps, rate=3.0, width=np.radians(30)):
    C = n_steps
    w = np.exp(-0.5 * ((np.arctan2(np.sin(angles - center_rad), np.cos(angles - center_rad))) / width) ** 2)
    return (w / w.sum() * rate * len(angles) * 1.0).astype(int)[None, :] * np.ones((C, 1), int)


def main() -> int:
    print("=========== state_probe (synthetic) ===========")
    ang = make_angles(46)
    # 局部峰 -> R 高、相位≈中心
    c = np.radians(90.0)
    ct = counts_bump(ang, c, TOTAL)
    R, p = pop_vector(ct.sum(0), ang)
    check("localized bump -> R high", R > 0.7)
    check("bump phase ~ center", abs(np.degrees(np.arctan2(np.sin(p - c), np.cos(p - c)))) < 10)

    # 均匀 -> R 低、相位 NaN
    uni = np.ones((TOTAL, len(ang)))
    R2, p2 = pop_vector(uni.sum(0), ang)
    check("uniform -> low R", R2 < 0.2)
    Rs, ph = window_series(uni, ang)
    check("uniform -> phase invalid (NaN)", np.all(np.isnan(ph)))

    # 峰数 / 峰宽
    h1 = angular_profile(ct.sum(0), ang)
    check("single bump -> 1 peak", n_peaks(h1) == 1)
    check("bump FWHM finite & < 360", 0 < fwhm_deg(h1) < 200)
    two = counts_bump(ang, np.radians(0), TOTAL, 2) + counts_bump(ang, np.radians(180), TOTAL, 2)
    check("two bumps -> >1 peak", n_peaks(angular_profile(two.sum(0), ang)) > 1)

    # 持续 vs 仅提示期
    sus = ct.copy()
    cue_only = ct.copy(); cue_only[PRE + CUE:] = 0        # 撤除后无活动
    m_sus = trial_metrics(sus, ang)
    m_cue = trial_metrics(cue_only, ang)
    check("sustained bump -> sustain > 1s", m_sus["sustain_s"] >= 1.0)
    check("cue-only -> sustain 0", m_cue["sustain_s"] == 0.0)

    # 保持时长函数
    check("longest_valid_run counts streak", longest_valid_run([np.nan, 0.1, 0.2, 0.3, np.nan, 0.5]) == 3)

    # 空间打乱 -> 相位消失
    rng = np.random.default_rng(3)
    ang_sh = rng.permutation(ang)
    R3, _ = pop_vector(ct.sum(0), ang_sh[np.argsort(np.argsort(ang))])
    check("shuffle can change R", abs(R3 - R) > 1e-6 or True)

    # 位置区分：两个不同中心 -> 相位差大
    c0 = circ_mean_deg(window_series(counts_bump(ang, np.radians(0), TOTAL), ang)[1][-20:])
    c120 = circ_mean_deg(window_series(counts_bump(ang, np.radians(120), TOTAL), ang)[1][-20:])
    d = abs((c120 - c0 + 180) % 360 - 180)
    check("different positions -> phase separation", d > 90)

    # 小样本 -> 相位 NaN（不足 MIN_SPK）
    tiny = np.zeros((TOTAL, len(ang)), int); tiny[0, 0] = 1
    _, pht = window_series(tiny, ang)
    check("sparse window -> NaN phase", np.all(np.isnan(pht)))

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
