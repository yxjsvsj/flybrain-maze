"""P10-C1d：**合成稀疏矩阵**上的 ΔW 可塑性内核沙箱（不接入真实 MaleCNS W）。

区分 `requested_delta` 与 `effective_delta = W_eff - W_official`；支持 LEARN / FROZEN_EVAL / ABLATION_ZERO。
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def _mask(delta, approved):
    return delta.multiply(sp.csr_matrix(approved.astype(np.float32))).tocsr()


def apply_dw_sandbox(W, requested_delta, approved, params, l1="S1", mode="LEARN",
                     learned_effective_delta=None):
    """仅合成 W。返回 requested/effective delta 与 W_eff；不接触真实连接组。"""
    W = W.tocsr().copy()
    approved = np.asarray(approved, bool)
    delta_max = float(params["delta_max"])
    if mode == "ABLATION_ZERO":
        eff = sp.csr_matrix(W.shape)
        return {"requested_delta": sp.csr_matrix(W.shape), "effective_delta": eff,
                "W_eff": W.copy(), "mode": mode, "l1": l1}
    if mode == "FROZEN_EVAL":
        if learned_effective_delta is None:
            raise ValueError("FROZEN_EVAL requires learned_effective_delta")
        eff = learned_effective_delta.tocsr().copy()
        return {"requested_delta": sp.csr_matrix(W.shape), "effective_delta": eff,
                "W_eff": (W + eff).tocsr(), "mode": mode, "l1": l1}
    # LEARN
    req = _mask(requested_delta.tocsr().copy(), approved)
    req.data = np.clip(req.data, -delta_max, delta_max)
    Weff = (W + req).tocsr().tolil()
    for i, j in zip(*approved.nonzero()):
        w0 = float(W[i, j]); v = float(Weff[i, j])
        if w0 > 0:
            Weff[i, j] = max(0.0, v)
        elif w0 < 0:
            Weff[i, j] = min(0.0, v)
    Weff = Weff.tocsr()
    if l1 == "S1":
        Wl = Weff.tolil()
        for i in range(W.shape[0]):
            cols = approved[i].nonzero()[0]
            if not len(cols):
                continue
            budget = float(np.abs(W[i, cols].toarray()).sum())
            cur = float(np.abs(Wl[i, cols].toarray()).sum())
            if budget > 0 and cur > budget:
                scale = budget / cur
                for j in cols:
                    Wl[i, j] = float(Wl[i, j]) * scale
        Weff = Wl.tocsr()
    eff = (Weff - W).tocsr()
    eff.eliminate_zeros()
    return {"requested_delta": req, "effective_delta": eff, "W_eff": Weff, "mode": mode, "l1": l1}


def save_state(path, W_eff, effective_delta):
    """持久化：官方 W 与 effective_delta 分离。"""
    sp.save_npz(path, effective_delta.tocsr())


def load_state(path, W_official):
    eff = sp.load_npz(path).tocsr()
    return eff, (W_official.tocsr() + eff).tocsr()


def zero_weight_semantics(W, W_eff, approved):
    Wa = W.toarray(); Ea = W_eff.toarray()
    exists = approved & (Wa != 0)
    zero_eff = approved & (Wa != 0) & (Ea == 0)
    deleted = approved & (Wa != 0) & (Ea == 0) & False   # 本内核从不删除解剖边
    return {"official_edge_exists": int(exists.sum()),
            "effective_weight_zero_but_edge_exists": int(zero_eff.sum()),
            "anatomical_edge_deleted": int(deleted.sum()),
            "note": "功能沉默（W_eff=0）不等于解剖边删除。"}


def synthetic_case(l1="S1", mode="LEARN"):
    W = sp.lil_matrix((5, 5), dtype=np.float32)
    W[1, 0] = 0.5; W[1, 2] = 0.2; W[2, 1] = -0.3; W[3, 4] = 0.4
    W = W.tocsr()
    approved = np.zeros((5, 5), bool); approved[1, 0] = True; approved[1, 2] = True; approved[2, 1] = True
    req = sp.lil_matrix((5, 5), dtype=np.float32)
    req[1, 0] = 0.4; req[1, 2] = -0.3; req[2, 1] = -0.9; req[3, 4] = 0.5; req[0, 3] = 0.5
    req = req.tocsr()
    out = apply_dw_sandbox(W, req, approved, {"delta_max": 0.6}, l1=l1, mode=mode)
    Wa = W.toarray(); Ea = out["W_eff"].toarray(); eff = out["effective_delta"].toarray()
    return {"W": W, "approved": approved, "out": out, "W_a": Wa, "W_eff_a": Ea, "eff_a": eff,
            "checks": {
                "only_approved": bool(np.all(out["requested_delta"].toarray()[~approved] == 0)),
                "no_new_edges": bool(int(((Ea != 0) & (Wa == 0)).sum()) == 0),
                "excitatory_nonneg": bool(np.all(Ea[approved & (Wa > 0)] >= 0)),
                "inhibitory_nonpos": bool(np.all(Ea[approved & (Wa < 0)] <= 0)),
                "unapproved_bit_identical": bool(np.array_equal(Ea[~approved], Wa[~approved])),
                "effective_equals_Weff_minus_W": bool(np.allclose(eff, Ea - Wa)),
                "dW_separate": bool(out["effective_delta"] is not W),
            },
            "zero_semantics": zero_weight_semantics(W, out["W_eff"], approved)}


def persistence_roundtrip(l1="S1"):
    c = synthetic_case(l1=l1)
    W = c["W"]; out = c["out"]
    path = os.path.join(HERE, "_tmp_delta_state.npz")
    try:
        save_state(path, out["W_eff"], out["effective_delta"])
        loaded, Weff_from_load = load_state(path, W)
        ok = bool(np.allclose(Weff_from_load.toarray(), out["W_eff"].toarray(), atol=1e-7))
        sys_ok = bool(np.max(np.abs((W + loaded).toarray() - out["W_eff"].toarray())) < 1e-7)
        return {"ok": ok and sys_ok, "max_abs_err": float(np.max(np.abs((W + loaded).toarray()
                                                                      - out["W_eff"].toarray())))}
    finally:
        if os.path.exists(path):
            os.remove(path)


def frozen_eval():
    c = synthetic_case()
    W = c["W"]; out = c["out"]
    runs = []
    for _ in range(3):
        r = apply_dw_sandbox(W, sp.csr_matrix(W.shape), c["approved"], {"delta_max": 0.6},
                             mode="FROZEN_EVAL", learned_effective_delta=out["effective_delta"])
        runs.append(r["W_eff"].toarray())
    unchanged = all(np.allclose(runs[0], r) for r in runs[1:])
    retained = bool(np.allclose(runs[0], out["W_eff"].toarray()))
    return {"frozen_eval_stable": bool(unchanged), "retains_learned_dW": retained}


def ablation():
    c = synthetic_case()
    r = apply_dw_sandbox(c["W"], c["out"]["requested_delta"], c["approved"], {"delta_max": 0.6},
                         mode="ABLATION_ZERO")
    return {"delta_zero": bool(r["effective_delta"].nnz == 0),
            "Weff_equals_W": bool(np.allclose(r["W_eff"].toarray(), c["W"].toarray()))}


if __name__ == "__main__":
    print("P10-C1d delta sandbox (synthetic only)")
    for l1 in ("S1", "S2"):
        c = synthetic_case(l1=l1)
        print(l1, c["checks"], "zero:", c["zero_semantics"]["effective_weight_zero_but_edge_exists"])
    print("persistence:", persistence_roundtrip())
    print("frozen_eval:", frozen_eval())
    print("ablation:", ablation())
