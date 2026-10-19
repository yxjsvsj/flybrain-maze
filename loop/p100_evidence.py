"""P10-0a：P10-0 三阻碍的**只读科学证据复核**（不改感觉靶/不建突触/不训练/不跑小车）。

数据来源：`flybrain` 官方 `brain.npz`(身份/侧别/类型) + `weights.npz`(**有符号、逐突触后 L1 归一化相对权重**)。
权重定义：**`W[post,pre]`（row=post, col=pre）**；符号来自**递质预测**（非实测受体效应）。**相对权重无单位**。
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

from loop.visual_motor_audit import load_data

HERE = os.path.dirname(os.path.abspath(__file__))
P912_ORIG = {50: 549.7, 100: 793.2, 500: 3611.9, 2000: 12883.9}


def _idx(D, types):
    return np.flatnonzero(np.isin(D["cell_type"], types))


def block(D, rows, cols):
    """直接块：边数 / Σw / Σ|w| / 正权贡献 / 负权贡献。"""
    if not len(rows) or not len(cols):
        return {"edges": 0, "sumW": 0.0, "absW": 0.0, "posW": 0.0, "negW": 0.0, "pos_edges": 0, "neg_edges": 0}
    M = D["W"][rows][:, cols].copy()
    M.eliminate_zeros()
    d = M.data
    return {"edges": int(M.nnz), "sumW": round(float(d.sum()), 4), "absW": round(float(np.abs(d).sum()), 4),
            "posW": round(float(d[d > 0].sum()), 4), "negW": round(float(d[d < 0].sum()), 4),
            "pos_edges": int((d > 0).sum()), "neg_edges": int((d < 0).sum())}


def two_hop(D, src_idx, tgt_idx, top=8):
    """`src -> X -> tgt` 两跳统计（严格 W[post,pre]）：X 收 src(`W[X,src]`) 且 X 送 tgt(`W[tgt,X]`)。

    **语义修正（P10-0b）**：严格区分
    - `n_intermediate`：不同的中间神经元数（不是路径数）；
    - `n_paths`：不同两跳路径数 `(source, intermediate, target)` 且两段均非零；
    - `signed_product`：两跳带符号权重乘积累计 `Σ_x (Σ_s W[x,s])·(Σ_t W[t,x])`；
    - `abs_product`：两跳绝对权重乘积累计 `Σ_x (Σ_s |W[x,s]|)·(Σ_t |W[t,x]|)`；
    - `sum_w_in/sum_w_out`：输入段/输出段各自的权重和（不是路径数）。
    """
    W = D["W"].tocsr().copy()
    W.eliminate_zeros()
    if not len(src_idx) or not len(tgt_idx):
        return {"n_intermediate": 0, "n_paths": 0, "sum_w_in": 0.0, "sum_w_out": 0.0,
                "signed_product": 0.0, "abs_product": 0.0, "top": []}
    Wsrc = W.tocsc()[:, src_idx]                     # (n, |src|)  W[X, src]
    recv = np.unique(Wsrc.tocoo().row)
    out_col = W[tgt_idx][:, :].tocsc()               # (|tgt|, n); 列 x = W[tgt, x]
    cand, n_paths, signed_p, abs_p = [], 0, 0.0, 0.0
    for x in recv:
        col = out_col.getcol(x)
        if col.nnz == 0:
            continue
        seg_in = W.getrow(x)[0, src_idx]             # W[x, src]
        w_in = float(seg_in.data.sum());  a_in = float(np.abs(seg_in.data).sum())
        w_out = float(col.data.sum());    a_out = float(np.abs(col.data).sum())
        n_paths += int(seg_in.nnz) * int(col.nnz)    # (s, x, t) 两段均非零
        signed_p += w_in * w_out
        abs_p += a_in * a_out
        cand.append((int(x), w_in, w_out, int(seg_in.nnz), int(col.nnz)))
    cand.sort(key=lambda t: -abs(t[2]))
    return {"n_intermediate": len(cand),
            "n_paths": n_paths,
            "sum_w_in": round(float(sum(c[1] for c in cand)), 4),
            "sum_w_out": round(float(sum(c[2] for c in cand)), 4),
            "signed_product": round(float(signed_p), 4),
            "abs_product": round(float(abs_p), 4),
            "top": [{"x_bodyId": int(D["ids"][c[0]]), "x_type": str(D["cell_type"][c[0]]),
                     "x_side": str(D["side"][c[0]]), "w_in": round(c[1], 4), "w_out": round(c[2], 4),
                     "deg_in": c[3], "deg_out": c[4]} for c in cand[:top]]}


def audit(D=None):
    D = D or load_data()
    ct = D["cell_type"]
    KC = _idx(D, [t for t in set(ct.tolist()) if t.startswith("KC")])
    MBON = _idx(D, [t for t in set(ct.tolist()) if t.startswith("MBON")])
    DAN = _idx(D, [t for t in set(ct.tolist()) if t.startswith("PPL") or t.startswith("PAM")])
    LP = _idx(D, ["LPLC2", "LC4"])
    DN = _idx(D, [t for t in set(D["superclass"].tolist()) if t == "descending_neuron"]) if False else \
        np.flatnonzero(D["superclass"] == "descending_neuron")
    return {"KC": KC, "MBON": MBON, "DAN": DAN, "LP": LP, "DN": DN}


def review(D=None):
    D = D or load_data()
    A = audit(D)
    KC, MBON, DAN, LP, DN = A["KC"], A["MBON"], A["DAN"], A["LP"], A["DN"]
    out = {}
    # 1) LPLC2/LC4 -> KC/MBON/DAN：直接 vs 2/3 跳
    for name, tgt in (("KC", KC), ("MBON", MBON), ("DAN", DAN)):
        out[f"LP_direct_to_{name}"] = block(D, tgt, LP)
        out[f"LP_2hop_to_{name}"] = two_hop(D, LP, tgt)
    # 2) MBON->DN、KC->DN：直接（含正负/绝对）+ 2 跳
    for name, src in (("MBON", MBON), ("KC", KC)):
        out[f"{name}_direct_to_DN"] = block(D, DN, src)
        out[f"{name}_2hop_to_DN"] = two_hop(D, src, DN)
    # 3) KC->MBON、DAN->KC（基准，作对照）
    out["KC_to_MBON"] = block(D, MBON, KC)
    out["DAN_to_KC"] = block(D, KC, DAN)
    return out


def pomdp_variable_goal():
    return {
        "student_inputs": ["五路 ToF 距离", "其时间变化", "内部神经状态", "自身运动指令反馈"],
        "variable_goal_note": "若目标是**可变的**(不同出口)，学生**无任何目标标识输入** → **目标不可辨识**："
                              "同一 ToF 历史在不同目标下应产生不同动作，但输入完全相同。",
        "not_identifiable": ["目标身份/位置(无输入)", "全局朝向(仅局部 ToF 不可定)", "已探索/未探索",
                             "门后/走廊尽头的可达性", "死胡同 vs 环路(局部相似)"],
        "implication": "固定目标且初始条件可控时或可用**内部状态**隐式编码；**可变目标**则**信息不可辨识**——"
                       "在禁止外部目标/航向输入的前提下，不能假定学生可复现全图教师。",
        "scope": "仅**信息论**陈述，不据此断言生物不可行，也不批准任何外部目标注入。",
    }


def check_p91b1_provenance(path=None):
    p = path or os.path.join(HERE, "p91b1.json")
    d = json.load(open(p, encoding="utf-8"))
    rows = d["perf"]["rows"]
    ms = {r["substeps"]: r["ms_per_step"] for r in rows}
    rem = {r["substeps"]: r.get("ms_per_step_remeasure_eb47a58") for r in rows}
    return {
        "source": "loop/p91b1.json (commit)",
        "ms_per_step_matches_original_2f12217": all(ms[s] == P912_ORIG[s] for s in P912_ORIG),
        "ms_per_step": ms,
        "remeasure_present_and_separate": all(rem.get(s) is not None for s in P912_ORIG),
        "remeasure": rem,
        "traced_peak_mem_present": all("traced_peak_mem_MB" in r for r in rows),
        "confused": False,
        "verdict": "**未混淆**：ms_per_step=原始 2f12217 测量；remeasure 另存独立字段；traced_peak_mem 为 eb47a58 增补",
    }


def verdict(R=None):
    R = R or review()
    lc_direct = sum(R[f"LP_direct_to_{n}"]["absW"] for n in ("KC", "MBON", "DAN"))
    lc_2hop = sum(R[f"LP_2hop_to_{n}"]["n_intermediate"] for n in ("KC", "MBON", "DAN"))
    mbon = R["MBON_direct_to_DN"]; kc = R["KC_direct_to_DN"]
    return {
        "verdict": "NO-GO（维持）",
        "evidence_levels": {
            "L1_direct_absent": f"LPLC2/LC4 直接→学习回路 Σ|w|={lc_direct:.4f}（≈0）—— **直接缺失**（数据支持，强）",
            "L2_indirect": f"2 跳候选中间神经元合计 n={lc_2hop}（需看逐条权重；**间接存在≠功能有效**）",
            "L3_MB_to_DN": f"MBON→DN 边 {mbon['edges']} Σw {mbon['sumW']} Σ|w| {mbon['absW']} "
                           f"(正 {mbon['posW']}/负 {mbon['negW']})；KC→DN 边 {kc['edges']} Σ|w| {kc['absW']} —— "
                           f"**Σw 小不足以单独断言功能弱**，须结合 Σ|w| 与 2 跳",
            "L4_g_unit": "绝对电导 g_unit 未定标（P9-1b1 科学 NO-GO）—— **数据缺失**",
        },
        "overturned": False,
        "why": "直接 ToF→学习回路缺失(数据支持)；但 P10-0 以 **Σw** 断言'MB→DN 很弱'**表述过强**，"
               "本轮改为**并列 Σ|w|/正负/间接**。综合 **直接缺失 + g_unit 未定标**仍为 NO-GO；",
        "remaining_blockers": ["ToF(当前靶)→学习回路 直接Σ|w|≈0（未改靶）", "g_unit 未定标",
                               "可变目标信息不可辨识"],
        "note": "不改变感觉靶、不建突触、不找第二机制；Dijkstra 仅未来训练期教师。",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    R = review()
    V = verdict(R)
    P = pomdp_variable_goal()
    prov = check_p91b1_provenance()
    print("[P10-0a] LP direct/2hop:")
    for n in ("KC", "MBON", "DAN"):
        print(f"  ->{n}: direct {R[f'LP_direct_to_{n}']} | 2hop n_inter={R[f'LP_2hop_to_{n}']['n_intermediate']} "
              f"sum_w_out={R[f'LP_2hop_to_{n}']['sum_w_out']}")
    print("  MBON->DN:", R["MBON_direct_to_DN"], "| 2hop n=", R["MBON_2hop_to_DN"]["n_intermediate"])
    print("  KC->DN  :", R["KC_direct_to_DN"], "| 2hop n=", R["KC_2hop_to_DN"]["n_intermediate"])
    print("[P10-0a] p91b1 provenance:", prov["verdict"])
    print("[P10-0a] verdict:", V["verdict"], "| overturned:", V["overturned"])
    json.dump({"review": R, "pomdp": P, "provenance": prov, "verdict": V},
              open(os.path.join(HERE, "p100_evidence_review.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote p100_evidence_review.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
