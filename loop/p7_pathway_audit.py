"""P7-0：MaleCNS v1.0 **静态**转向通路可行性审查（只读连接组；**不运行 MaleCNS**）。

严格 `W[post, pre]`（与 `brain.py::_propagate` 一致，列=fired=pre）。核查 LC10a→DNa02 及其它
双侧感觉→转向下降通路，并与 LPLC2/LC4→逃逸 通路区分。**不**用导航成功率选神经元、不重筛 L50/R50、不调增益。
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

from loop.visual_motor_audit import load_data

HERE = os.path.dirname(os.path.abspath(__file__))


def conn(D, tgt_types, src_types):
    """W[post=tgt, pre=src] 的边数与正/负权和。"""
    ct = D["cell_type"]; W = D["W"].tocsr()
    ti = np.flatnonzero(np.isin(ct, tgt_types)); si = np.flatnonzero(np.isin(ct, src_types))
    if not len(ti) or not len(si):
        return {"n_edges": 0, "w_sum": 0.0, "w_pos": 0.0, "w_neg": 0.0}
    M = W[ti][:, si]; d = M.data
    return {"n_edges": int((M != 0).sum()), "w_sum": round(float(d.sum()), 4),
            "w_pos": round(float(d[d > 0].sum()), 4), "w_neg": round(float(d[d < 0].sum()), 4)}


def top_inputs(D, cell_type, side=None, k=5):
    """**修正版**：输入是 W[post,pre] 的**行** `W[cell, :]`（pre=源）。
    （旧版误用 `W.getcol` = 该神经元的**输出**列 → 方向颠倒。）"""
    ct = D["cell_type"]; sc = D["superclass"]; W = D["W"].tocsr()
    m = ct == cell_type
    if side:
        m = m & (D["side"] == side)
    j = np.flatnonzero(m)
    if not len(j):
        return []
    row = W.getrow(j[0])                       # 行 = 输入（pre）
    acc = collections.Counter()
    for kk, dd in zip(row.indices, row.data):
        acc[sc[kk]] += abs(dd)
    return [(str(t), round(float(w), 3)) for t, w in acc.most_common(k)]


def inputs_by_superclass(D, cell_type, side=None):
    """输入（行）按源 superclass：边数 / 正权 / 负权 / 代数和 / 绝对权和。"""
    ct = D["cell_type"]; sc = D["superclass"]; W = D["W"].tocsr()
    m = (ct == cell_type)
    if side:
        m = m & (D["side"] == side)
    j = np.flatnonzero(m)
    if not len(j):
        return {}
    row = W.getrow(j[0])
    acc = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    for kk, dd in zip(row.indices, row.data):
        a = acc[str(sc[kk])]; a[0] += 1
        a[1] += dd if dd > 0 else 0.0; a[2] += dd if dd < 0 else 0.0
        a[3] += dd; a[4] += abs(dd)
    return {t: {"edges": int(v[0]), "pos": round(float(v[1]), 4), "neg": round(float(v[2]), 4),
                "alg": round(float(v[3]), 4), "abs": round(float(v[4]), 4)}
            for t, v in sorted(acc.items(), key=lambda kv: -kv[1][4])}


def two_hop(D, src_types, tgt_types, top=12):
    """真实两跳 `src -> X -> tgt`（严格 `W[post,pre]`）：X 收 src（`W[X,src]`）且 X 送 tgt（`W[tgt,X]`）。
    保留 bodyId/类型/侧别/方向/符号；**不由路径存在推断动态功能**。"""
    ct = D["cell_type"]; side = D["side"]; ids = D["ids"]; W = D["W"].tocsr()
    src = np.flatnonzero(np.isin(ct, src_types)); tgt = np.flatnonzero(np.isin(ct, tgt_types))
    Wsrc = W[:, src].tocsc()                              # W[X, src]
    recv = np.unique(Wsrc.tocoo().row)
    inset = set()
    for t in tgt:
        inset |= set(W.getrow(t).indices.tolist())        # X -> tgt 存在
    excl = set(src.tolist()) | set(tgt.tolist())
    hits = []
    for x in recv:
        if x in inset and x not in excl:
            hits.append({"x_bodyId": int(ids[x]), "x_type": str(ct[x]), "x_super": str(D["superclass"][x]),
                         "x_side": str(side[x]), "src_to_x": round(float(W[x, src].sum()), 4),
                         "x_to_tgt": round(float(W[tgt, x].sum()), 4)})
    hits.sort(key=lambda r: -abs(r["x_to_tgt"]))
    return hits[:top], len(hits)


def audit(D):
    ct = D["cell_type"]; sc = D["superclass"]; side = D["side"]; W = D["W"].tocsr()
    dn = np.flatnonzero(sc == "descending_neuron"); vp = np.flatnonzero(sc == "visual_projection")
    M = W[dn][:, vp].tocoo()
    v2dn = collections.Counter()
    lat = collections.Counter()
    for i, j, w in zip(M.row, M.col, M.data):
        v2dn[str(ct[dn[i]])] += w
        lat["ipsi" if side[dn[i]] == side[vp[j]] == "L" or (side[dn[i]] == side[vp[j]]) else "contra"] += abs(w)
    return {
        "LC10a_to_DNa02": conn(D, ["DNa02"], ["LC10a"]),
        "LC10a_to_all_DN": conn(D, list(ct[dn]), ["LC10a"]),
        "DNa02_receives_visual": conn(D, ["DNa02"], list(ct[vp])),
        "DNa02_top_inputs": top_inputs(D, "DNa02"),
        "visual_to_DN_top_types": [(str(t), round(float(w), 3)) for t, w in v2dn.most_common(10)],
        "visual_to_DN_ipsi_contra": {k: round(float(v), 2) for k, v in lat.items()},
        "counts": {"LC10a": int((ct == "LC10a").sum()), "DNa02": int((ct == "DNa02").sum()),
                   "visual_projection": int(len(vp)), "descending_neuron": int(len(dn))},
    }


def three_functions(A):
    """三必要功能的**机制候选**判定。

    **科学解释更正**：静态连接**只能提供机制候选**（"是否存在可承载该功能的解剖路径"），
    **不能验证**以下**动力学能力**：开始转弯 / 左右选择（动态竞争）/ 结束转弯 / 回正。
    "supported" 一律读作"存在可承载该功能的解剖候选"，**非**"功能已实现"。
    """
    v2d = A["DNa02_receives_visual"]
    cb = dict(A["DNa02_top_inputs"]).get("cb_intrinsic", 0.0)
    return {
        "obstacle_triggers_turning": {
            "supported": bool(A["LC10a_to_DNa02"]["n_edges"] > 0),
            "evidence": f"机制候选：LC10a->DNa02 直接边={A['LC10a_to_DNa02']['n_edges']}；"
                        f"DNa02<-visual_projection 净 w={v2d['w_sum']}（pos {v2d['w_pos']}/neg {v2d['w_neg']} 近相消）"
                        f"≪ DNa02<-cb_intrinsic 绝对权和 {cb}。**动力学能力未验证。**"},
        "left_right_selection": {
            "supported": bool(A["LC10a_to_DNa02"]["n_edges"] > 0),
            "evidence": f"visual->DN 同侧主导 {A['visual_to_DN_ipsi_contra']}（但靶为 looming/逃逸 DN）；"
                        f"LC10a->DNa02 直接边=0。左右选择是**动态竞争**，静态连接不构成证据。"},
        "return_to_neutral": {
            "supported": False,
            "evidence": "回正/结束转弯是**动力学**性质；静态连接不可证，机制候选亦不足（本轮不运行）。"},
    }


def audit_corrected(D):
    """P7-0a 修正审计：输入=行；DNa02 左右分别；LC10a→X→DNa02 两跳。"""
    A = audit(D)
    th, n = two_hop(D, ["LC10a"], ["DNa02"])
    return {
        "convention": "W[post,pre]; inputs = row W[cell,:]",
        "LC10a_to_DNa02_direct": A["LC10a_to_DNa02"],          # 保留：0 边
        "note_direct": "direct edge remains 0; this does NOT mean all sensory->steering paths are absent",
        "DNa02_inputs_L": inputs_by_superclass(D, "DNa02", "L"),
        "DNa02_inputs_R": inputs_by_superclass(D, "DNa02", "R"),
        "DNa02_outputs_note": "old report's 'top inputs' were actually OUTPUTS (column); see P70A_REPORT",
        "LC10a_two_hop_to_DNa02": {"n_candidates": n, "top": th},
        "visual_to_DN_ipsi_contra": A["visual_to_DN_ipsi_contra"],
        "three_functions": three_functions(A),
        "verdict_scope": "original NO-GO applies to the DIRECT LC10a->DNa02 edge and to strong visual->steering; "
                         "weak 2-hop exists -> mechanism candidate pending approval, gate NOT auto-opened",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    D = load_data()
    A = audit(D)
    F = three_functions(A)
    print("[P7-0] counts:", A["counts"])
    print("  LC10a -> DNa02 :", A["LC10a_to_DNa02"])
    print("  LC10a -> all DN:", A["LC10a_to_all_DN"])
    print("  DNa02 <- visual_projection:", A["DNa02_receives_visual"])
    print("  DNa02 top inputs:", A["DNa02_top_inputs"])
    print("  visual -> DN top types:", A["visual_to_DN_top_types"])
    print("  visual -> DN ipsi/contra:", A["visual_to_DN_ipsi_contra"])
    for k, v in F.items():
        print("  fn", k, "supported=", v["supported"], "|", v["evidence"])
    # **不覆写** 原始 p70_audit.json（P7-0a 要求保留）；只写修正结果。
    C = audit_corrected(D)
    json.dump(C, open(os.path.join(HERE, "p70a_audit.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[P7-0a] DNa02_L inputs:", list(C["DNa02_inputs_L"].items())[:4])
    print("[P7-0a] DNa02_R inputs:", list(C["DNa02_inputs_R"].items())[:4])
    print("[P7-0a] LC10a->X->DNa02 candidates:", C["LC10a_two_hop_to_DNa02"]["n_candidates"],
          "top:", C["LC10a_two_hop_to_DNa02"]["top"][:3])
    print("[P7-0a] direct LC10a->DNa02 edges:", C["LC10a_to_DNa02_direct"]["n_edges"])
    print("wrote p70a_audit.json (p70_audit.json untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
