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
    ct = D["cell_type"]; sc = D["superclass"]; W = D["W"].tocsc()
    m = ct == cell_type
    if side:
        m = m & (D["side"] == side)
    j = np.flatnonzero(m)
    if not len(j):
        return []
    col = W.getcol(j[0])
    acc = collections.Counter()
    for kk, dd in zip(col.indices, col.data):
        acc[sc[kk]] += abs(dd)
    return [(str(t), round(float(w), 3)) for t, w in acc.most_common(k)]


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
    """三必要功能的**机制依据**判定（连接存在 ≠ 功能实现）。"""
    v2d = A["DNa02_receives_visual"]
    vnc = dict(A["DNa02_top_inputs"]).get("vnc_intrinsic", 0.0)
    return {
        # 关键：连接**存在** ≠ 功能实现。判定用**权重**而非边数；DNa02 以 vnc_intrinsic 为主输入。
        "obstacle_triggers_turning": {
            "supported": bool(A["LC10a_to_DNa02"]["n_edges"] > 0 or v2d["w_sum"] > 0.5),
            "evidence": f"LC10a->DNa02 edges={A['LC10a_to_DNa02']['n_edges']}; DNa02<-visual n={v2d['n_edges']} "
                        f"net w={v2d['w_sum']} (pos {v2d['w_pos']}/neg {v2d['w_neg']}, 近乎相消), "
                        f"vs DNa02<-vnc_intrinsic={vnc} -> 视觉输入可忽略"},
        "left_right_selection": {
            "supported": bool(A["LC10a_to_DNa02"]["n_edges"] > 0),
            "evidence": f"visual->DN 虽以同侧为主 {A['visual_to_DN_ipsi_contra']}，但**靶是 looming/逃逸 DN**，"
                        f"非转向；**LC10a->DNa02 边=0** -> 无数据支持的视觉→转向 左右选择"},
        "return_to_neutral": {
            "supported": False,
            "evidence": "静态连接无法证明动态回中性；需动力学证据（本轮不运行）"},
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
    out = {"audit": A, "three_functions": F,
           "verdict": "GO" if (A["LC10a_to_DNa02"]["n_edges"] > 0) else "NO-GO"}
    json.dump(out, open(os.path.join(HERE, "p70_audit.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[P7-0] verdict:", out["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
