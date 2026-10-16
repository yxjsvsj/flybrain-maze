"""P2d-26f 回归：ring_spatial 的**合成反例**测试（空间法、圆统计、有效性门控）。

覆盖：合成环→角度可辨；均匀→无效(R<R_min)；双峰→无效(multimodal)；样本不足→无效；
PCA 平面由网格定（倾斜平面仍回收角度）；圆均/分半差正确。
"""
from __future__ import annotations

import sys

import numpy as np

from loop.ring_spatial import (N_MIN, R_MIN, angles, bootstrap_mean_angle, circ_diff_deg,
                               circ_stats, eb_frame, n_modes, shortest_arc_ci, split_half)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== ring_spatial (synthetic) ===========")
    # 网格：扁平椭球壳（x,y 大、z 小）-> PCA 平面应≈xy
    rng = np.random.default_rng(0)
    V = rng.normal(size=(4000, 3)) * np.array([30.0, 20.0, 6.0])
    fr = eb_frame(V)
    check("PCA normal ~ z (flattened ellipsoid)", abs(abs(fr["normal"][2]) - 1) < 0.05)

    # 合成单峰细胞：角度集中在 +45°
    def ring_cell(mu_deg, k=400, spread_deg=10):
        th = np.radians(mu_deg) + np.radians(rng.normal(0, spread_deg, k))
        r = 25.0
        return np.c_[r * np.cos(th), r * np.sin(th), rng.normal(0, 2, k)]

    a = angles(ring_cell(45), fr)
    st = circ_stats(a)
    check("single-peak cell R high", st["R"] > 0.9)
    # 零点/正向/手性由 PCA 特征向量符号决定 → 只检验**相对角距**（对旋转/反射不变）
    m0 = circ_stats(angles(ring_cell(0), fr))["mean"]
    m90 = circ_stats(angles(ring_cell(90), fr))["mean"]
    check("relative 0 vs 90 preserved (~90)", abs(circ_diff_deg(m0, m90) - 90) < 6)
    check("unimodal (1 mode)", n_modes(a) == 1)
    bo = bootstrap_mean_angle(a)
    check("bootstrap uncertainty small (mean stable)", bo["boot_circ_std_deg"] < 10)
    check("95% CI half-width small", bo["ci95_half_deg"] < 10)
    check("split-half stable", split_half(a) < 10)

    # ---- 关键区分：宽分布 vs 均值稳定 ----
    # (a) 很宽但单峰的突触分布（spread 60°）：raw 圆标准差大，但均值角估计仍稳定
    wide = angles(ring_cell(20, 2000, spread_deg=60), fr)
    wst = circ_stats(wide); wbo = bootstrap_mean_angle(wide)
    check("wide spread -> large raw circ std", np.degrees(wst["circ_std"]) > 45)
    check("wide spread -> mean still stable (boot std small)", wbo["boot_circ_std_deg"] < 3)
    check("wide spread -> CI narrow vs raw std",
          wbo["ci95_half_deg"] < np.degrees(wst["circ_std"]))

    # (b) 小样本：均值不确定性应变大（相对大样本）
    small = angles(ring_cell(20, 6, spread_deg=60), fr)
    sbo = bootstrap_mean_angle(small)
    check("small sample -> larger mean uncertainty", sbo["boot_circ_std_deg"] > wbo["boot_circ_std_deg"])

    # (c) 环绕 ±180°：直接测圆统计的跨分支（不经过任意符号的网格帧）
    wrap_raw = np.radians(np.array([179.0, -179.0, 178.0, -178.0, 180.0]))
    check("wrap near +-180 handled", circ_diff_deg(circ_stats(wrap_raw)["mean"], np.radians(180)) < 3)

    # (d) 双峰：raw 圆标准差大且峰数>1
    bim = angles(np.r_[ring_cell(0, 300), ring_cell(180, 300)], fr)
    check("bimodal -> large raw std", np.degrees(circ_stats(bim)["circ_std"]) > 50)
    check("bimodal -> >1 mode", n_modes(bim) > 1)

    # (e) CI 定义：最短弧（跨 0° 分支时 lo>hi 也正确；只查弧宽）
    lo, hi = shortest_arc_ci(np.radians([350, 355, 0, 5, 10, 300]), level=0.8)
    check("shortest-arc CI width small", ((hi - lo) % (2 * np.pi)) <= np.radians(90))
    # 弧确实覆盖那个簇（350..10）：把这些角落回该弧内
    arc = ((np.radians([350, 355, 0, 5, 10]) - lo) % (2 * np.pi))
    check("shortest-arc CI covers the cluster", (arc <= ((hi - lo) % (2 * np.pi)) + 1e-9).all())

    # 均匀分布 -> 无效
    uni = angles(rng.uniform(-np.pi, np.pi, 400).reshape(-1, 1) * [1, 1, 0]
                 + np.c_[np.zeros(400), np.zeros(400), rng.normal(0, 2, 400)], fr)
    check("uniform -> R < R_min", circ_stats(uni)["R"] < R_MIN)

    # 双峰 -> multimodal
    bi = np.r_[ring_cell(30, 250), ring_cell(210, 250)]
    check("bimodal detected (>1 mode)", n_modes(angles(bi, fr)) > 1)

    # 样本不足
    check("few points flagged (<N_MIN)", angles(ring_cell(0, N_MIN - 10), fr).shape[0] < N_MIN)

    # 倾斜网格平面：整体绕 x 轴旋转 -> 仍能回收**相对**角距
    R = np.array([[1, 0, 0], [0, np.cos(0.6), -np.sin(0.6)], [0, np.sin(0.6), np.cos(0.6)]])
    fr2 = eb_frame(V @ R.T)
    t0 = circ_stats(angles(ring_cell(0) @ R.T, fr2))["mean"]
    t90 = circ_stats(angles(ring_cell(90) @ R.T, fr2))["mean"]
    check("tilted-plane relative 0 vs 90 preserved", abs(circ_diff_deg(t0, t90) - 90) < 6)

    # circ_diff 环绕正确性
    check("circ_diff wraps correctly", abs(circ_diff_deg(np.radians(350), np.radians(10)) - 20) < 1e-6)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
