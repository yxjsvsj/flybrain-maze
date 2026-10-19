"""P10-0c：真实 MaleCNS **DAN—KC→MBON 区室调制证据**审计（只读）。

只回答：官方 MaleCNS v1.0 中能否找到**身份/脑区/来源可核验**的 DAN—KC→MBON 对应关系。
**不训练、不更新 ΔW、不跑动力学、不控车**；不改 ToF 靶/不建连接/不找第二机制。
数据：本地 `flybrain` MaleCNS（bodyId/类型/侧别/W）+ 官方 neuPrint `male-cns:v1.0` 的**逐神经元 ROI 计数**（targeted query）。
权重：`W[post,pre]`；有符号相对权重（无单位，递质预测符号）。
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

from loop.p100_evidence import audit
from loop.p100b_path_audit import chain_layers
from loop.visual_motor_audit import load_data

HERE = os.path.dirname(os.path.abspath(__file__))
MB_COMPARTMENTS = {"a1", "a2", "a3", "b1", "b2", "g1", "g2", "g3", "g4", "g5",
                   "a'1", "a'2", "a'3", "b'1", "b'2"}
TOKEN_PATH = os.path.join(os.path.expanduser("~"), ".neuprint")


def _base(roi: str, keep_side: bool = False) -> str:
    """ROI 规范化。`keep_side=True` 时**保留完整 ROI**（如 `g1(L)`），不合并左右。"""
    return str(roi) if keep_side else re.sub(r"\((L|R)\)$", "", str(roi))


def mb_comps(roi_rows, kind, keep_side: bool = False):
    """从逐神经元 ROI 计数取蘑菇体区室（kind=pre/post）。

    `keep_side=False`（旧行为，P10-0c）：合并 `(L)/(R)`；`keep_side=True`（P10-0d）：**保留完整 ROI**。
    """
    col = "pre" if kind == "pre" else "post"
    out = set()
    for r in roi_rows:
        if _base(r["roi"]) in MB_COMPARTMENTS and int(r.get(col, 0) or 0) > 0:
            out.add(str(r["roi"]) if keep_side else _base(r["roi"]))
    return out


def fetch_metadata(body_ids, token_path=TOKEN_PATH, chunk=100):
    """官方 neuPrint male-cns:v1.0：targeted 拉取 bodyId 的逐神经元 ROI 计数。"""
    from neuprint import Client, fetch_neurons
    tok = json.load(open(token_path))["token"]
    c = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=tok)
    meta = {}
    ids = [int(x) for x in body_ids]
    for i in range(0, len(ids), chunk):
        chunk_ids = ids[i:i + chunk]
        try:
            df, rr = fetch_neurons(chunk_ids, client=c)
        except Exception as e:                    # noqa: BLE001
            print(f"  fetch chunk {i} failed: {repr(e)[:100]}")
            continue
        rows_by_id = {}
        if rr is not None and len(rr):
            for bid, sub in rr.groupby("bodyId"):
                rows_by_id[int(bid)] = sub[["roi", "pre", "post"]].to_dict("records")
        for _, row in df.iterrows():
            bid = int(row["bodyId"])
            meta[bid] = {"type": str(row.get("type", "")), "instance": str(row.get("instance", "")),
                         "side": str(row.get("somaSide", "")), "roi_rows": rows_by_id.get(bid, [])}
    return meta


def build_evidence(D, layers, meta, max_examples=20):
    """对 4,533 条真实 KC→MBON 连接做区室/DAN 匹配（**神经元级 ROI 重叠**，非逐突触）。"""
    W = D["W"].tocsr().copy()
    W.eliminate_zeros()
    L2, L3 = layers[2], layers[3]
    KC_meta, MBON_meta = {}, {}
    KC_comp = {}
    for k in L2:
        bid = int(D["ids"][k])
        m = meta.get(bid)
        if m:
            KC_comp[bid] = mb_comps(m["roi_rows"], "pre")
            KC_meta[bid] = m
    MB_comp = {}
    for m_ in L3:
        bid = int(D["ids"][m_])
        mm = meta.get(bid)
        if mm:
            MB_comp[bid] = mb_comps(mm["roi_rows"], "post")
            MBON_meta[bid] = mm
    # DAN 区室
    A = audit(D)
    DAN_comp = {}
    for d in A["DAN"]:
        bid = int(D["ids"][d])
        m = meta.get(bid)
        if m:
            DAN_comp[bid] = mb_comps(m["roi_rows"], "pre")
    # 遍历真实 KC→MBON 边
    sub = W[L3][:, L2].tocoo()
    n_total = n_comp = n_dan = 0
    examples = []
    comp_to_dan = {}
    shared_sizes = {}
    shared_sets = set()
    for r, c in zip(sub.row, sub.col):
        kc = int(D["ids"][L2[c]]); mb = int(D["ids"][L3[r]])
        n_total += 1
        kcset = KC_comp.get(kc, set()); mbset = MB_comp.get(mb, set())
        shared = kcset & mbset
        if not shared:
            continue
        n_comp += 1
        shared_sizes[len(shared)] = shared_sizes.get(len(shared), 0) + 1
        shared_sets.add(frozenset(shared))
        dans = [b for b, cs in DAN_comp.items() if cs & shared]
        if dans:
            n_dan += 1
            for cm in shared:
                comp_to_dan.setdefault(cm, set()).update(dans)
        if len(examples) < max_examples:
            examples.append({"kc_bodyId": kc, "kc_type": str(D["cell_type"][L2[c]]),
                             "mbon_bodyId": mb, "mbon_type": str(D["cell_type"][L3[r]]),
                             "shared_compartments": sorted(shared),
                             "n_matching_DAN": len(dans),
                             "matching_DAN_sample": [{"bodyId": x, "type": str(D["cell_type"][np.flatnonzero(D["ids"] == x)[0]])
                                                      if np.any(D["ids"] == x) else ""} for x in dans[:5]]})
    return {"n_edges_total": n_total, "n_edges_compartment_overlap": n_comp, "n_edges_with_DAN": n_dan,
            "n_KC_with_compartment": sum(1 for s in KC_comp.values() if s),
            "n_MBON_with_compartment": sum(1 for s in MB_comp.values() if s),
            "n_DAN_with_compartment": sum(1 for s in DAN_comp.values() if s),
            "shared_size_distribution": dict(sorted(shared_sizes.items())),
            "n_distinct_shared_sets": len(shared_sets),
            "examples": examples,
            "compartment_to_DAN": {k: sorted(v)[:10] for k, v in sorted(comp_to_dan.items())}}


def find_simple_path(D, layers, max_iters=2_000_000):
    """从分层 walk 中找**至少一条**完整、非零权重、**无重复 bodyId**的 simple path。"""
    W = D["W"].tocsr().copy()
    W.eliminate_zeros()
    ids = D["ids"]
    it = [0]

    def neighbors(hi, lo, node):
        col = W[:, lo].tocsc().getcol(lo.index(node)) if False else None
        # 用 CSR 的列切片更稳妥：W[hi][:, [node]]
        sub = W[hi][:, [node]]
        return [(int(hi[r]), float(sub.data[k])) for k, r in enumerate(sub.tocoo().row)]

    # 预取每层的入边（避免重复切片）
    def cand(hi_layer, lo_layer, node):
        pos = {int(x): i for i, x in enumerate(lo_layer)}
        sub = W[hi_layer][:, [node]]
        out = []
        for r, v in zip(sub.tocoo().row, sub.data):
            if v != 0:
                out.append((int(hi_layer[r]), float(v)))
        out.sort(key=lambda t: -abs(t[1]))
        return out

    def dfs(depth, path, visited, weights):
        it[0] += 1
        if it[0] > max_iters:
            return None
        if depth == len(layers) - 1:
            return path, weights
        for nxt, w in cand(layers[depth + 1], layers[depth], path[-1]):
            if nxt in visited:
                continue
            if np.any(ids[path] == ids[nxt]):
                continue
            visited.add(nxt)
            res = dfs(depth + 1, path + [nxt], visited, weights + [w])
            if res:
                return res
            visited.remove(nxt)
        return None

    for start in layers[0]:
        res = dfs(0, [int(start)], {int(start)}, [])
        if res:
            path, weights = res
            return {"found": True, "bodyIds": [int(ids[i]) for i in path],
                    "types": [str(D["cell_type"][i]) for i in path],
                    "sides": [str(D["side"][i]) for i in path],
                    "layer_labels": ["LP", "X", "KC", "MBON", "Y", "DN"],
                    "segment_weights": weights}
    return {"found": False, "reason": "no simple path found within iteration budget"}


LITERATURE = [
    {"ref": "Aso et al. 2014 eLife 3:e04577", "level": "general + compartmental anatomy",
     "claim": "MB extrinsic neurons/DAN compartmental organization", "same_source_MaleCNS": False},
    {"ref": "Hige et al. 2015 Neuron 85:1341", "level": "specific compartment (alpha3) DA-dependent KC->MBON plasticity",
     "claim": "compartment-specific dopaminergic plasticity", "same_source_MaleCNS": False},
    {"ref": "Aso & Rubin 2016 Curr Biol 26:R225", "level": "circuit/compartment",
     "claim": "DAN/MBON recurrent architecture", "same_source_MaleCNS": False},
    {"ref": "Perisse et al. 2013 / Cohn et al. 2015 / Berry et al. 2018", "level": "compartment-specific",
     "claim": "compartmentalized DAN modulation of MB output", "same_source_MaleCNS": False},
]
LIT_CONCLUSION = {
    "general_KC_MBON_plasticity": "文献支持（其它数据集/生理）",
    "compartment_specific_DAN": "文献支持（其它数据集/区室）",
    "same_source_MaleCNS_bodyId_anatomy": "部分（本轮 neuron-level ROI 重叠；非逐突触）",
    "specific_MaleCNS_synapse_modulated": "MISSING（无该突触的直接生理/受体证据）",
    "caution": "DAN 与 KC 普通突触或共享 ROI ≠ 已证明该 KC→MBON 突触受 DA 功能调制；跨数据集不强行匹配。",
}


def status(D=None):
    D = D or load_data()
    A = audit(D)
    layers = chain_layers(D, A)
    ids = [int(D["ids"][i]) for i in layers[2]] + [int(D["ids"][i]) for i in layers[3]] + \
          [int(D["ids"][i]) for i in A["DAN"]]
    try:
        meta = fetch_metadata(ids)
    except Exception as e:                        # noqa: BLE001
        meta = {}
        print("fetch_metadata failed:", repr(e)[:120])
    ev = build_evidence(D, layers, meta)
    sp = find_simple_path(D, layers)
    return {"data": {"dataset": "male-cns:v1.0", "source": "official neuPrint + local flybrain",
                     "weight": "W[post,pre], signed L1-normalized relative, no units, transmitter-predicted sign"},
            "layers": {"n_KC": int(len(layers[2])), "n_MBON": int(len(layers[3])), "n_DAN": int(len(A["DAN"]))},
            "evidence": ev, "simple_path": sp, "literature": LITERATURE, "literature_conclusion": LIT_CONCLUSION,
            "grading": {
                "Structural_Path": "存在（P10-0b，数据支持）",
                "Compartment_Match": ("部分" if ev["n_edges_compartment_overlap"] > 0 else "不足")
                                     + "（neuron-level ROI 重叠，非逐突触）",
                "DAN_Modulation": "不足（文献一般性支持，但无同源 MaleCNS 逐突触调制证据）",
                "Synapse_Specificity": "MISSING（无逐突触 ROI/受体/生理证据）",
                "Biological_Calibration": "g_unit 未解决（P9-1b1）",
                "Learning_Readiness": "否（NO-GO 维持）",
            },
            "remaining_blockers": ["特定 KC→MBON 突触的 DA 调制证据（逐突触/受体/生理）",
                                   "g_unit 未定标", "可变目标信息不可辨识"],
            "go": False}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    S = status()
    print("[P10-0c] layers:", S["layers"])
    e = S["evidence"]
    print(f"  KC->MBON edges {e['n_edges_total']} | compartment-overlap {e['n_edges_compartment_overlap']} "
          f"| with DAN {e['n_edges_with_DAN']}")
    print(f"  KC/MBON/DAN with MB compartments: {e['n_KC_with_compartment']}/"
          f"{e['n_MBON_with_compartment']}/{e['n_DAN_with_compartment']}")
    print("  simple path found:", S["simple_path"].get("found"),
          (S["simple_path"].get("bodyIds") if S["simple_path"].get("found") else S["simple_path"].get("reason")))
    print("  grading:", S["grading"])
    json.dump(S, open(os.path.join(HERE, "p100c_dan_compartment.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print("wrote p100c_dan_compartment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
