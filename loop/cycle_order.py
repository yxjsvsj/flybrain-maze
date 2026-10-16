"""P2d-26a：**可测试的循环序重建流程**（特征矩阵 → 相似度 → 2-D 谱嵌入 → 判据）。

固定方法（事先声明，适合循环数据）：**对称归一化 Laplacian 的次小两特征向量 (v2,v3) 作 2-D 谱嵌入**；
环序 = atan2(v3,v2) 的角度排序。**不**再把单个 Fiedler 向量排序当通用环识别器。

判据（阈值先固定）——只有当下列**全部**成立才判 cyclic：
  A 相似度非退化（等效秩 > 1.5，避免全连接/全零）；
  B 低端**恰有一对近简并**模态 λ2≈λ3 且 λ4 明显分离（`deg_gap` 小、`sep4` 大）；
  C 2-D 嵌入半径集中（`r_cv` 小）。
`atan2` 能把点排成一圈 ≠ 数据存在可识别的环；故必须有非环对照与退化检查。
"""
from __future__ import annotations

import numpy as np

DEG_TOL = 0.25        # (λ3-λ2)/λ2 上限
SEP4_MIN = 0.15       # (λ4-λ3)/(λ3+eps) 下限
RCV_TOL = 0.35        # 半径变异系数上限
RANK_MIN = 1.5        # 等效秩下限


def rownorm(M):
    M = np.asarray(M, float)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def similarity(M):
    R = rownorm(M)
    S = R @ R.T
    np.fill_diagonal(S, 0.0)
    return 0.5 * (S + S.T)


def laplacian(S):
    d = S.sum(1)
    dinv = 1.0 / np.sqrt(d + 1e-12)
    return np.eye(len(S)) - dinv[:, None] * S * dinv[None, :]


def eff_rank(S):
    w = np.linalg.eigvalsh(0.5 * (S + S.T))
    w = np.clip(w, 0, None)
    return float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-18))


def ring_order(M):
    """返回 dict(order, angle, metrics, cyclic)。m 行=源细胞，列=特征。"""
    M = np.asarray(M, float)
    n = len(M)
    if n < 4 or not np.any(M):
        return {"order": np.arange(n), "angle": np.full(n, np.nan), "cyclic": False,
                "metrics": {"reason": "too small or all-zero"}, "n": n}
    S = similarity(M)
    L = laplacian(S)
    w, v = np.linalg.eigh(L)              # 升序
    v2, v3 = v[:, 1], v[:, 2]
    r = np.hypot(v2, v3)
    r_cv = float(np.std(r) / (np.mean(r) + 1e-12))
    deg_gap = float((w[2] - w[1]) / (w[1] + 1e-12))
    sep4 = float((w[3] - w[2]) / (w[2] + 1e-12)) if n > 3 else 0.0
    rank = eff_rank(S)
    ang = np.arctan2(v3, v2)
    order = np.argsort(ang)
    cyc = bool(rank > RANK_MIN and deg_gap < DEG_TOL and sep4 > SEP4_MIN and r_cv < RCV_TOL)
    return {"order": order, "angle": ang, "cyclic": cyc, "n": n,
            "metrics": {"r_cv": r_cv, "deg_gap": deg_gap, "sep4": sep4, "eff_rank": rank}}


def order_stability(M, seed=0, frac=0.8, reps=20):
    """稳定性：随机子采样后重算，看两两环序的**循环**一致性（相邻对命中率均值）。"""
    rng = np.random.default_rng(seed)
    n = len(M)
    k = max(4, int(round(frac * n)))
    base = ring_order(M)["order"]
    rank = {int(i): j for j, i in enumerate(base)}
    hits = []
    for _ in range(reps):
        idx = np.sort(rng.choice(n, k, replace=False))
        o = ring_order(np.asarray(M)[idx])["order"]
        # 循环相邻一致性：子序相邻对在 base 环上是否也相邻
        h = 0
        for t in range(len(o) - 1):
            a, b = int(idx[o[t]]), int(idx[o[t + 1]])
            d = abs(rank[a] - rank[b])
            h += 1 if min(d, n - d) == 1 else 0
        hits.append(h / max(1, len(o) - 1))
    return float(np.mean(hits)) if hits else float("nan")
