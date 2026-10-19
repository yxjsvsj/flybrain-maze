"""P10-0d：MaleCNS **真实突触定位 / DAN 区室对应 / 文献勘误**（只读）。

只回答三问：(1) 能否定位真实 KC→MBON 突触与区室？(2) 候选 DAN 释放位点是否同区室？(3) 证据能推进到哪一级？
**不训练、不更新 ΔW、不跑动力学、不控车**；不改靶/不建连接/不找第二机制。
数据：本地 `flybrain` MaleCNS + 官方 neuPrint `male-cns:v1.0`（`fetch_synapse_connections` 限定 source+target）。
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

from loop.p100_evidence import audit
from loop.p100b_path_audit import chain_layers
from loop.p100c_dan_compartment import fetch_metadata, mb_comps
from loop.visual_motor_audit import load_data

HERE = os.path.dirname(os.path.abspath(__file__))
MB_FINE = {"a1", "a2", "a3", "b1", "b2", "g1", "g2", "g3", "g4", "g5",
           "a'1", "a'2", "a'3", "b'1", "b'2"}
MB_LOBES = {"a", "b", "g", "a'", "b'"}


def side_of(roi: str) -> str:
    return "L" if str(roi).endswith("(L)") else ("R" if str(roi).endswith("(R)") else "?")


def lobe_of(roi: str):
    """完整 ROI -> (叶, 侧)，如 `g3(L)`->(`g`,`L`)、`a'L(R)`->(`a'`,`R`)、`gL(L)`->(`g`,`L`)。"""
    b = re.sub(r"\((L|R)\)$", "", str(roi))
    if b.startswith("a'"):
        return ("a'", side_of(roi))
    if b.startswith("b'"):
        return ("b'", side_of(roi))
    if b and b[0] in ("a", "b", "g"):
        return (b[0], side_of(roi))
    return (None, side_of(roi))


def side_aware_stats(D, layers, meta):
    """修正前后：合并左右（旧）vs 保留完整 ROI（新）。"""
    L2, L3 = layers[2], layers[3]
    W = D["W"].tocsr().copy(); W.eliminate_zeros()
    merged_kc = {int(D["ids"][i]): mb_comps(meta[int(D["ids"][i])]["roi_rows"], "pre", keep_side=False)
                 for i in L2 if int(D["ids"][i]) in meta}
    merged_mb = {int(D["ids"][i]): mb_comps(meta[int(D["ids"][i])]["roi_rows"], "post", keep_side=False)
                 for i in L3 if int(D["ids"][i]) in meta}
    full_kc = {int(D["ids"][i]): mb_comps(meta[int(D["ids"][i])]["roi_rows"], "pre", keep_side=True)
               for i in L2 if int(D["ids"][i]) in meta}
    full_mb = {int(D["ids"][i]): mb_comps(meta[int(D["ids"][i])]["roi_rows"], "post", keep_side=True)
               for i in L3 if int(D["ids"][i]) in meta}
    sub = W[L3][:, L2].tocoo()
    n = n_merged = n_full = n_lost = 0
    lost = []
    for r, c in zip(sub.row, sub.col):
        kc = int(D["ids"][L2[c]]); mb = int(D["ids"][L3[r]]); n += 1
        om = merged_kc.get(kc, set()) & merged_mb.get(mb, set())
        of = full_kc.get(kc, set()) & full_mb.get(mb, set())
        if om:
            n_merged += 1
        if of:
            n_full += 1
        elif om:
            n_lost += 1
            if len(lost) < 12:
                lost.append({"kc_bodyId": kc, "mbon_bodyId": mb, "merged_only": sorted(om)[:4],
                             "full_overlap": []})
    return {"n_edges": n, "overlap_merged_sides": n_merged, "overlap_full_roi": n_full,
            "n_cross_side_lost": n_lost, "cross_side_examples": lost,
            "note": "47 条左右合并后重叠、保留完整 ROI 后不重叠 -> 跨侧候选；**未删除**，但需同源突触数据证明。"}


def select_pairs(D, layers, meta, max_pairs=12):
    """固定、可复算的候选对选取（含 simple-path 对与阴性对照）。"""
    L2, L3 = layers[2], layers[3]
    W = D["W"].tocsr().copy(); W.eliminate_zeros()
    full_kc = {int(D["ids"][i]): mb_comps(meta[int(D["ids"][i])]["roi_rows"], "pre", keep_side=True)
               for i in L2 if int(D["ids"][i]) in meta}
    full_mb = {int(D["ids"][i]): mb_comps(meta[int(D["ids"][i])]["roi_rows"], "post", keep_side=True)
               for i in L3 if int(D["ids"][i]) in meta}
    sub = W[L3][:, L2].tocoo()
    edges = sorted({(int(D["ids"][L2[c]]), int(D["ids"][L3[r]])) for r, c in zip(sub.row, sub.col)})
    targets = [("g", "L"), ("g", "R"), ("a", "L"), ("a", "R"), ("b", "L"), ("b", "R"), ("a'", "L"), ("b'", "L")]
    pairs, used = [], set()
    for lb, sd in targets:
        for kc, mb in edges:
            if (kc, mb) in used:
                continue
            sh = full_kc.get(kc, set()) & full_mb.get(mb, set())
            if any(lobe_of(x) == (lb, sd) for x in sh):
                pairs.append({"kc": kc, "mb": mb, "target_lobe": lb, "target_side": sd,
                              "shared_full_rois": sorted(sh), "kind": "positive"})
                used.add((kc, mb)); break
    if (137302, 519128) not in used and len(pairs) < max_pairs:
        pairs.insert(0, {"kc": 137302, "mb": 519128, "target_lobe": "g", "target_side": "L",
                         "shared_full_rois": sorted(full_kc.get(137302, set()) & full_mb.get(519128, set())),
                         "kind": "simple_path"})
        used.add((137302, 519128))
    # 阴性对照：第一个 KC 与第一个 W=0 的 MBON
    neg = None
    for kc in [k for k, _ in edges]:
        for mb in [int(x) for x in D["ids"][L3]]:
            i_mb = int(np.flatnonzero(D["ids"] == mb)[0]); i_kc = int(np.flatnonzero(D["ids"] == kc)[0])
            if W[i_mb, i_kc] == 0:
                neg = {"kc": kc, "mb": mb, "kind": "negative_control", "target_lobe": None, "target_side": None}
                break
        if neg:
            break
    if neg and len(pairs) < max_pairs:
        pairs.append(neg)
    return pairs[:max_pairs]


def query_pairs(pairs, token_path=os.path.join(os.path.expanduser("~"), ".neuprint")):
    from neuprint import Client, fetch_synapse_connections
    tok = json.load(open(token_path))["token"]
    c = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=tok)
    out = []
    for p in pairs:
        rec = dict(p)
        try:
            df = fetch_synapse_connections(p["kc"], p["mb"], client=c, min_total_weight=1)
            rec["n_synapses"] = int(len(df))
            rec["complete"] = True
            if len(df):
                rec["roi_counts"] = {str(k): int(v) for k, v in df["roi_pre"].value_counts().items()}
                rec["compartment_pre"] = sorted(set(map(str, df["compartment_pre"])))
                rec["compartment_post"] = sorted(set(map(str, df["compartment_post"])))
                rec["confidence_pre_min"] = round(float(df["confidence_pre"].min()), 4)
                rec["confidence_pre_median"] = round(float(df["confidence_pre"].median()), 4)
                rec["coords_sample"] = [{"x": int(df.iloc[i]["x_pre"]), "y": int(df.iloc[i]["y_pre"]),
                                         "z": int(df.iloc[i]["z_pre"]), "roi": str(df.iloc[i]["roi_pre"])}
                                        for i in range(min(3, len(df)))]
            else:
                rec["roi_counts"] = {}
        except Exception as e:                    # noqa: BLE001
            rec["n_synapses"] = None
            rec["complete"] = False
            rec["error"] = repr(e)[:160]
        out.append(rec)
    return out


def dan_localization(D, A, meta, queried, max_c=False):
    """A/B/C/D 分级。只对已定位突触研究 DAN；B=lobe 级共区室。"""
    dan_lobes = {}
    for d in A["DAN"]:
        bid = int(D["ids"][d])
        m = meta.get(bid)
        if not m:
            continue
        comps = mb_comps(m["roi_rows"], "pre", keep_side=True)
        dan_lobes[bid] = {"type": str(D["cell_type"][d]), "lobes": {lobe_of(x) for x in comps}}
    res = []
    for q in queried:
        if q.get("kind") == "negative_control" or not q.get("n_synapses"):
            res.append({"kc": q["kc"], "mb": q["mb"], "A_synapse_localized": False,
                        "B_dan_same_lobe": [], "C_local_spatial": "N/A", "D_functional": "MISSING"})
            continue
        mb_lobes = set()
        for roi in q.get("roi_counts", {}):
            lb, sd = lobe_of(roi)
            if lb in MB_LOBES:
                mb_lobes.add((lb, sd))
        dans = [{"bodyId": b, "type": v["type"]} for b, v in dan_lobes.items() if v["lobes"] & mb_lobes]
        res.append({"kc": q["kc"], "mb": q["mb"], "synapse_rois": sorted(q.get("roi_counts", {})),
                    "A_synapse_localized": bool(q.get("n_synapses")),
                    "B_dan_same_lobe": dans[:12],
                    "B_n_dan": len(dans),
                    "B_specific": len(dans) <= 5,
                    "B_specificity": ("restricted" if len(dans) <= 5
                                      else "non-discriminative (broad lobe coverage)"),
                    "C_local_spatial": "未查询（需 DAN 释放点坐标；本轮仅 lobe 级）",
                    "D_functional": "MISSING（无受体/生理证据）"})
    return res


LITERATURE_CORRIGENDUM = {
    "wrong_doi": "10.1016/j.neuron.2014.12.045",
    "wrong_citation": "Hige et al. 2015, Neuron 85:1341",
    "correct_citation": "Hige et al. (2015), Heterosynaptic Plasticity Underlies Aversive Olfactory Learning in Drosophila, Neuron 88(5):985-998",
    "correct_doi": "10.1016/j.neuron.2015.11.003",
    "affected_files": ["loop/P100_REPORT.md", "loop/p10_design.py",
                       "loop/P100C_DAN_COMPARTMENT_REVIEW.md", "loop/p100c_dan_compartment.py"],
    "policy": "独立勘误；不静默覆盖冻结历史结果。",
}


def status(D=None, do_query=True):
    D = D or load_data()
    A = audit(D)
    layers = chain_layers(D, A)
    ids = [int(D["ids"][i]) for i in layers[2]] + [int(D["ids"][i]) for i in layers[3]] + \
          [int(D["ids"][i]) for i in A["DAN"]]
    meta = fetch_metadata(ids)
    stats = side_aware_stats(D, layers, meta)
    pairs = select_pairs(D, layers, meta)
    queried = query_pairs(pairs) if do_query else []
    dan = dan_localization(D, A, meta, queried)
    n_pos_localized = sum(1 for q in queried if q.get("kind") != "negative_control" and q.get("n_synapses"))
    n_neg = sum(1 for q in queried if q.get("kind") == "negative_control" and q.get("n_synapses") == 0)
    return {
        "data": {"dataset": "male-cns:v1.0", "source": "official neuPrint fetch_synapse_connections (source+target)",
                 "api": "fetch_synapse_connections(source, target, min_total_weight=1)",
                 "weight": "W[post,pre], signed L1-normalized relative, no units"},
        "layers": {"n_KC": int(len(layers[2])), "n_MBON": int(len(layers[3])), "n_DAN": int(len(A["DAN"]))},
        "side_aware_stats": stats,
        "n_pairs_selected": len(pairs), "pairs": queried,
        "n_positive_pairs_localized": n_pos_localized, "n_negative_controls_zero": n_neg,
        "dan_localization": dan,
        "literature_corrigendum": LITERATURE_CORRIGENDUM,
        "grading": {
            "Synapse_Localization": "部分（9 对阳性；lobe 级 ROI，非全部 4533 边）",
            "Compartment_Assignment": "非特异/部分（lobe 级 gL/aL/bL/a'L/b'L，非 fine g1-g5）",
            "DAN_Release_Co_localization": "有证据但非特异（B：同 lobe+side，187-317 DAN；C 未做；D 无）",
            "Functional_DA_Modulation": "未验证",
            "g_unit_Calibration": "未解决（P9-1b1）",
            "Variable_goal_Identifiability": "仍受限（P10-0a）",
            "Learning_Readiness": "NO-GO",
        },
        "stop": "若有限查询仍不足以建立具体突触—DAN 对应，记录缺口并停止该审计分支；不扩大到全脑。",
        "go": False,
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    S = status()
    st = S["side_aware_stats"]
    print("[P10-0d] side-aware:", st["overlap_merged_sides"], "->", st["overlap_full_roi"],
          "of", st["n_edges"], "| cross-side lost", st["n_cross_side_lost"])
    print("[P10-0d] pairs:", S["n_pairs_selected"], "| positive localized:", S["n_positive_pairs_localized"],
          "| negative zero:", S["n_negative_controls_zero"])
    for q in S["pairs"]:
        print(f"  {q['kind']:16s} {q['kc']}->{q['mb']} n={q.get('n_synapses')} roi={list(q.get('roi_counts',{}))[:4]}")
    print("[P10-0d] literature corrigendum:", S["literature_corrigendum"]["correct_doi"])
    print("[P10-0d] grading:", S["grading"])
    json.dump(S, open(os.path.join(HERE, "p100d_synapse_localization.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print("wrote p100d_synapse_localization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
