"""P9-1b0：真实 MaleCNS 连接组 -> COBA 引擎的**只读数据接入预检**（不跑 MaleCNS 动力学）。

- 加载 `weights.npz`（**有符号、逐突触后归一化相对权重**，无单位）与 `brain.npz`（身份/侧别/bodyId）。
- **精确分流**：`We = W⁺`、`Wi = |W⁻|`（两者**非负**），满足 **`W = We - Wi`**（逐元素精确）。
- **方向**：`W[post, pre]`（与 `flybrain.brain::_propagate` 一致；`g += g_unit·(W@S)`）。
- **严禁混淆三层**：① 连接组**原始突触计数**；② `weights.npz` 归一化**相对**权重（无单位）；③ **未知绝对电导**。
  权重**符号**来自**递质预测**（≠ 已测得的突触后受体效应）。
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
from scipy import sparse


def weights_sha256_16(path) -> str:
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()[:16]


def load_real():
    from flybrain.data import ensure_data
    data = ensure_data()
    meta = np.load(str(data / "brain.npz"))
    W = sparse.load_npz(str(data / "weights.npz")).tocsr()
    return {"data": data, "meta": meta, "W": W,
            "ids": np.asarray(meta["ids"]),
            "cell_type": np.array([str(x) for x in meta["cell_type"]]),
            "side": np.array([str(x) for x in meta["side"]]),
            "weights_sha256_16": weights_sha256_16(data / "weights.npz"),
            "brain_sha256_16": weights_sha256_16(data / "brain.npz")}


def split_signed(W) -> tuple:
    """有符号稀疏 W -> (We, Wi) 非负 CSR；满足 W = We - Wi（逐元素精确）。"""
    W = W.tocsr()
    pos = W.copy()
    pos.data = np.where(pos.data > 0, pos.data, 0.0)
    pos.eliminate_zeros()
    neg = W.copy()
    neg.data = np.where(neg.data < 0, -neg.data, 0.0)
    neg.eliminate_zeros()
    return pos.tocsr(), neg.tocsr()


def verify_split(W, We, Wi) -> dict:
    """静态断言：维数/边数/绝对权和/精确重构/非负/零值处理。"""
    W = W.tocsr()
    diff = (We - Wi - W)
    recon_max = float(np.abs(diff.data).max()) if diff.nnz else 0.0
    return {
        "shape": [int(x) for x in W.shape],
        "nnz_W": int(W.nnz), "nnz_We": int(We.nnz), "nnz_Wi": int(Wi.nnz),
        "edge_count_preserved": int(We.nnz + Wi.nnz) == int(W.nnz),
        "abs_sum_W": round(float(np.abs(W.data).sum()), 4),
        "abs_sum_split": round(float(We.data.sum() + Wi.data.sum()), 4),
        "abs_sum_preserved": abs(float(np.abs(W.data).sum()) - float(We.data.sum() + Wi.data.sum())) < 1e-3,
        "reconstruct_W_eq_We_minus_Wi_maxabs": recon_max,
        "exact_reconstruct": recon_max < 1e-9,
        "nonneg_We": bool((We.data >= 0).all()), "nonneg_Wi": bool((Wi.data >= 0).all()),
        "stored_zero_in_W": int((W.data == 0).sum()),
        "n_pos": int((W.data > 0).sum()), "n_neg": int((W.data < 0).sum()),
    }


def direction_check(D, pre_type="LC4", post_type="DNp04") -> dict:
    """方向核验：`W[post, pre]`；取该 (post,pre) 类型对的最大权重边，报告其 bodyId/权重。"""
    W = D["W"]; ct = D["cell_type"]; ids = D["ids"]
    pi = np.flatnonzero(ct == pre_type); qi = np.flatnonzero(ct == post_type)
    if not len(pi) or not len(qi):
        return {"found": False}
    sub = W[qi][:, pi]
    if sub.nnz == 0:
        return {"found": False, "note": "no edge in this (post,pre) type pair"}
    k = int(np.argmax(np.abs(sub.data)))
    q = qi[sub.tocoo().row[k]]; p = pi[sub.tocoo().col[k]]
    return {"found": True, "pre_type": pre_type, "post_type": post_type,
            "pre_bodyId": int(ids[p]), "post_bodyId": int(ids[q]),
            "w_post_pre": float(sub.data[k]), "convention": "W[post,pre] (row=post, col=pre)"}


def coba_inputs(W):
    """COBA 引擎输入 = (We, Wi, 均为 (post,pre) 非负 CSR)。**不改方向、不改权重绝对值。**"""
    return split_signed(W)


# 绝对电导不确定性表（静态，**不做**为增强运动输出而选值）
GUNIT_TABLE = [
    {"symbol": "g_unit", "unit": "nS / 单位归一化权重",
     "meaning": "每 **1 单位归一化相对权重** 对应的突触电导（**非**每条原始突触的实测 nS）",
     "literature": "果蝇单突触/量子电导量级多在 ~0.1–数 nS（nAChR/GABA 电生理综述）；**无本连接组逐连接测量**",
     "available": "无（本数据集无逐连接电导）",
     "missing": "计数->电导的换算；逐类型的受体组成与量子大小",
     "uncertainty": "**高**（跨类型/跨突触若干数量级可能）",
     "rule": "**禁止**为增强神经运动输出而选择 g_unit；须以**声明约定 + 有界范围**处理并标注不确定"},
]
