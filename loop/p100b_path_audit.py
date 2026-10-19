"""P10-0b：真实 MaleCNS **感觉—学习—运动完整路径**只读审计。

链：`LPLC2/LC4 -> X -> KC -> MBON -> Y -> DN`（X/Y 为真实中间神经元；子段直接或 ≤2 跳）。
**只读**：不训练、不更新 ΔW、不跑神经动力学、不控车；不改 ToF 靶、不建连接、不找第二学习规则。
权重：`W[post,pre]`（row=post, col=pre）；**有符号相对权重（无单位，递质预测符号）**。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp

from loop.p100_evidence import audit, block, two_hop
from loop.visual_motor_audit import load_data

HERE = os.path.dirname(os.path.abspath(__file__))


def layer_block(W, hi, lo):
    if not len(hi) or not len(lo):
        return {"hi": 0, "lo": 0, "edges": 0, "sumW": 0.0, "absW": 0.0, "posW": 0.0, "negW": 0.0}
    M = W[hi][:, lo].copy()
    M.eliminate_zeros()
    d = M.data
    return {"hi": int(len(hi)), "lo": int(len(lo)), "edges": int(M.nnz),
            "sumW": round(float(d.sum()), 4), "absW": round(float(np.abs(d).sum()), 4),
            "posW": round(float(d[d > 0].sum()), 4), "negW": round(float(d[d < 0].sum()), 4)}


def chain_layers(D, A):
    """链的五级可达层（真实 W[post,pre]）：L0=LP, L1=收 LP, L2=KC 收 L1, L3=MBON 收 L2, L4=收 L3, L5=DN 收 L4。"""
    W = D["W"].tocsr().copy()
    W.eliminate_zeros()
    LP, KC, MBON, DN = A["LP"], A["KC"], A["MBON"], A["DN"]

    def receivers(src):
        return np.unique(W.tocsc()[:, src].tocoo().row) if len(src) else np.array([], dtype=int)

    L0 = LP
    L1 = receivers(L0)
    L2 = np.intersect1d(KC, receivers(L1)) if len(L1) else np.array([], dtype=int)
    L3 = np.intersect1d(MBON, receivers(L2)) if len(L2) else np.array([], dtype=int)
    L4 = receivers(L3) if len(L3) else np.array([], dtype=int)
    L5 = np.intersect1d(DN, receivers(L4)) if len(L4) else np.array([], dtype=int)
    return [L0, L1, L2, L3, L4, L5]


def dp_chain(D, layers):
    """分层 **walk** 统计（非 simple path）：路径数 / 带符号乘积和 / 绝对乘积和。"""
    W = D["W"].tocsr()
    v = np.ones(len(layers[0]))          # 带符号：每条 0 长 walk 权重 1
    va = np.ones(len(layers[0]))         # 绝对
    vc = np.ones(len(layers[0]))         # 计数
    stats = []
    for i in range(len(layers) - 1):
        lo, hi = layers[i], layers[i + 1]
        A = W[hi][:, lo].copy()
        A.eliminate_zeros()
        stats.append(layer_block(W, hi, lo))
        if A.nnz:
            v = A @ v
            va = abs(A) @ va
            vc = (A != 0).astype(np.float64) @ vc
        else:
            v = np.zeros(len(hi)); va = np.zeros(len(hi)); vc = np.zeros(len(hi))
    return {"layer_stats": stats,
            "n_paths_walk": float(vc.sum()), "signed_product": float(v.sum()),
            "abs_product": float(va.sum())}


def dan_modulation(D, A):
    KC, MBON, DAN = A["KC"], A["MBON"], A["DAN"]
    W = D["W"].tocsr()
    # DAN -> KC 的具体靶 KC（解剖）
    tgt_dan = {"DAN_to_KC": block(D, KC, DAN), "DAN_to_MBON": block(D, MBON, DAN)}
    # 能同时「DAN->KC 且 KC->MBON」的 KC（普通解剖可达）
    Wk = W[KC][:, DAN]
    kc_with_dan = set(np.unique(Wk.tocoo().row))
    kc_to_mbon = set(np.unique(W[MBON][:, KC].tocoo().col))
    both = kc_with_dan & kc_to_mbon
    return {"anatomy": tgt_dan, "n_KC_with_DAN_input": len(kc_with_dan),
            "n_KC_with_MBON_output": len(kc_to_mbon), "n_KC_both": len(both),
            "compartment_level": "MISSING (无逐突触/区室注释)",
            "receptor_level": "MISSING (无受体类型/位置数据)",
            "specific_synapse_modulation": "MISSING (无生理/钙成像/光遗传证据)",
            "note": "DAN→KC 权重**不能**自动推断对特定 KC→MBON 突触的调制成立。"}


def status(D=None):
    D = D or load_data()
    A = audit(D)
    L = chain_layers(D, A)
    dp = dp_chain(D, L)
    chain_exists = all(len(x) > 0 for x in L) and all(s["edges"] > 0 for s in dp["layer_stats"])
    # 定位瓶颈：边数最少/乘积最小的层
    edges = [s["edges"] for s in dp["layer_stats"]]
    bottleneck = int(np.argmin(edges)) if edges else -1
    return {"layers": L, "layer_sizes": [int(len(x)) for x in L], "dp": dp,
            "structural_path_exists": bool(chain_exists), "bottleneck_layer_index": bottleneck,
            "bottleneck_layer": (["LP->X", "X->KC", "KC->MBON", "MBON->Y", "Y->DN"][bottleneck]
                                 if 0 <= bottleneck < 5 else None),
            "dan": dan_modulation(D, A),
            "verdict": {
                "structural_path": "存在" if chain_exists else ("不存在" if any(e == 0 for e in edges) else "未确定"),
                "modulatory_evidence": "不足 (仅普通解剖连接；区室/受体/特定突触调制 MISSING)",
                "functional_transmission": "未验证",
                "biological_calibration": "g_unit 未解决 (P9-1b1)",
                "variable_goal_identifiability": "仍受目标信息缺失限制",
                "go": False,
                "why": "即使完整解剖链存在，也**不自动批准**学习实验；缺调制证据与 g_unit，且不得以路径数/Σw 代替因果功能验证。",
            }}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    S = status()
    print("[P10-0b] layer sizes:", S["layer_sizes"])
    for i, st in enumerate(S["dp"]["layer_stats"]):
        print(f"  L{i}->L{i+1}: edges={st['edges']} sumW={st['sumW']} absW={st['absW']} "
              f"pos={st['posW']} neg={st['negW']}")
    print("  walk paths:", S["dp"]["n_paths_walk"], "| signed_product:", S["dp"]["signed_product"],
          "| abs_product:", S["dp"]["abs_product"])
    print("  structural_path_exists:", S["structural_path_exists"], "| bottleneck:", S["bottleneck_layer"])
    print("  DAN:", S["dan"]["anatomy"], "| KC both:", S["dan"]["n_KC_both"], "|", S["dan"]["compartment_level"])
    print("  verdict:", S["verdict"])
    json.dump(S, open(os.path.join(HERE, "p100b_path_audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print("wrote p100b_path_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
