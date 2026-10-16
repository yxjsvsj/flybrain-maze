"""P2d-26 第二步（**被门控，未在真实数据运行**）：固定动力学"局部状态"预检的指标 + 合成测试。

门控：只有当第一步给出**可解释的环坐标**（ring_metrics 的 is_ring=True）才允许运行；
否则本模块拒绝运行（`main` raise）。本轮第一步判定 **is_ring=False（线序/无闭环）** → 未运行真实数据。

指标（阈值/窗口**先固定**）：环上局部性、峰宽、峰数、有效相位、相位漂移；峰/均值仅辅助。
全静默或无可靠局部峰 → 相位记 **NaN（无效）**，不输出任意角度。
"""
from __future__ import annotations

import numpy as np

# 预先固定的阈值/窗口（不按结果调）
PEAK_TO_MEAN_MIN = 1.5      # 峰/均值 低于它视为"无可信局部峰"
HALF_MAX = 0.5              # 半高
MIN_ACT = 1e-6              # 全静默判据


def ring_profile(act: np.ndarray) -> np.ndarray:
    return np.asarray(act, float)


def locality(act: np.ndarray) -> float:
    a = np.asarray(act, float)
    return float(a.max() / (a.mean() + 1e-9)) if a.size else 0.0


def peak_width(act: np.ndarray) -> float:
    """半高全宽（环上，单位=格；无峰返回 0）。"""
    a = np.asarray(act, float)
    if a.size == 0 or a.max() <= MIN_ACT:
        return 0.0
    above = a >= HALF_MAX * a.max()
    if not above.any():
        return 0.0
    # 环上最长连续 True 段长度
    n = len(a)
    best = run = 0
    for i in range(2 * n):
        run = run + 1 if above[i % n] else 0
        best = max(best, run)
    return float(min(best, n))


def n_peaks(act: np.ndarray) -> int:
    a = np.asarray(act, float)
    if a.size < 3 or a.max() <= MIN_ACT:
        return 0
    return int(np.sum((a > np.roll(a, 1)) & (a >= np.roll(a, -1)) & (a >= HALF_MAX * a.max())))


def effective_phase(act: np.ndarray) -> float:
    """有效相位（弧度，环上圆形均值）；无可信局部峰/全静默 -> NaN。"""
    a = np.asarray(act, float)
    if a.size == 0 or a.max() <= MIN_ACT or locality(a) < PEAK_TO_MEAN_MIN:
        return float("nan")
    th = 2 * np.pi * np.arange(len(a)) / len(a)
    return float(np.arctan2((a * np.sin(th)).sum(), (a * np.cos(th)).sum()))


def phase_drift(phases) -> float:
    """有效相位的漂移（最后-最先，环形差，弧度）；不足 2 个有效值 -> NaN。"""
    p = np.asarray([x for x in phases if np.isfinite(x)], float)
    if len(p) < 2:
        return float("nan")
    d = (p[-1] - p[0] + np.pi) % (2 * np.pi) - np.pi
    return float(d)


def main(argv=None) -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # 门控：无有效环序则拒绝运行
    from loop.compass_ring import ring_metrics
    ok = bool(argv and getattr(argv, "ring_ok", False))
    raise SystemExit("[P2d-26 step2] 门控未通过：第一步 is_ring=False（线序/无闭环），"
                     "空间坐标尚不可识别 → 不运行真实数据预检。" if not ok else
                     "[P2d-26 step2] 需先提供有效环序；本轮未运行。")


if __name__ == "__main__":
    raise SystemExit(main())
