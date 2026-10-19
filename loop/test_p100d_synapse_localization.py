"""P10-0d 回归：左右侧 ROI 不混淆（合成）、lobe 映射、伙伴特异性查询结果与裁决（不联网）。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp

from loop.p100c_dan_compartment import _base, mb_comps
from loop.p100d_synapse_localization import LITERATURE_CORRIGENDUM, lobe_of, side_aware_stats

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def make_D(n, edges, types, superclass=None):
    rows = [r for r, _, _ in edges]; cols = [c for _, c, _ in edges]; vals = [v for _, _, v in edges]
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return {"W": W, "ids": np.arange(1000, 1000 + n), "cell_type": np.array(types, dtype=object),
            "side": np.array(["L"] * n, dtype=object), "superclass": np.array(superclass or [""] * n, dtype=object)}


def test_lr_not_merged():
    rows = [{"roi": "g1(L)", "pre": 5, "post": 0}, {"roi": "g1(R)", "pre": 0, "post": 5}]
    check("_base keeps side", _base("g1(L)", keep_side=True) == "g1(L)")
    check("_base merges by default", _base("g1(L)") == "g1")
    check("mb_comps pre keep_side -> g1(L)", mb_comps(rows, "pre", keep_side=True) == {"g1(L)"})
    check("mb_comps post keep_side -> g1(R)", mb_comps(rows, "post", keep_side=True) == {"g1(R)"})
    check("mb_comps merged -> {g1}", mb_comps(rows, "pre", keep_side=False) == {"g1"})
    check("g1(L) != g1(R)", mb_comps(rows, "pre", keep_side=True) != mb_comps(rows, "post", keep_side=True))


def test_lobe_of():
    check("g3(L)->(g,L)", lobe_of("g3(L)") == ("g", "L"))
    check("a'1(R)->(a',R)", lobe_of("a'1(R)") == ("a'", "R"))
    check("b'2(L)->(b',L)", lobe_of("b'2(L)") == ("b'", "L"))
    check("gL(L)->(g,L)", lobe_of("gL(L)") == ("g", "L"))


def test_cross_side_lost():
    # KC g1(L) vs MBON g1(R): merged overlap yes, full-ROI overlap no
    types = ["LPLC2", "X", "KCg-d", "MBON01", "Y", "DN"]
    sc = ["", "", "", "", "", "descending_neuron"]
    D = make_D(6, [(1, 0, .5), (2, 1, .5), (3, 2, .5), (4, 3, .5), (5, 4, .5)], types, superclass=sc)
    layers = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
    meta = {1002: {"type": "KCg-d", "roi_rows": [{"roi": "g1(L)", "pre": 5, "post": 0}]},
            1003: {"type": "MBON01", "roi_rows": [{"roi": "g1(R)", "pre": 0, "post": 5}]}}
    st = side_aware_stats(D, layers, meta)
    check("cross-side: merged overlap = 1", st["overlap_merged_sides"] == 1)
    check("cross-side: full overlap = 0", st["overlap_full_roi"] == 0)
    check("cross-side: n_lost = 1", st["n_cross_side_lost"] == 1)


def test_committed():
    p = os.path.join(HERE, "p100d_synapse_localization.json")
    if not os.path.exists(p):
        check("committed json exists", False)
        return
    d = json.load(open(p, encoding="utf-8"))
    st = d["side_aware_stats"]
    check("real 4533 edges", st["n_edges"] == 4533, str(st["n_edges"]))
    check("real full-ROI overlap 4486", st["overlap_full_roi"] == 4486, str(st["overlap_full_roi"]))
    check("real cross-side lost 47", st["n_cross_side_lost"] == 47, str(st["n_cross_side_lost"]))
    pos = [q for q in d["pairs"] if q.get("kind") != "negative_control"]
    neg = [q for q in d["pairs"] if q.get("kind") == "negative_control"]
    check("9 positive pairs localized", len(pos) == 9 and all(q.get("n_synapses") for q in pos))
    check("negative control = 0 synapses", len(neg) == 1 and neg[0].get("n_synapses") == 0)
    check("positive pairs have lobe-level ROI", all(any(x.endswith(("L)", "R)")) for x in q.get("roi_counts", {}))
          for q in pos))
    check("DAN B matches non-specific", all(not r.get("B_specific") for r in d["dan_localization"] if r.get("B_n_dan")))
    g = d["grading"]
    check("Compartment non-specific", "非特异" in g["Compartment_Assignment"])
    check("Functional DA unverified", "未验证" in g["Functional_DA_Modulation"])
    check("g_unit unresolved", "未解决" in g["g_unit_Calibration"])
    check("Learning NO-GO", g["Learning_Readiness"] == "NO-GO")
    check("corrigendum DOI correct", d["literature_corrigendum"]["correct_doi"] == "10.1016/j.neuron.2015.11.003")
    check("corrigendum wrong DOI recorded", d["literature_corrigendum"]["wrong_doi"] == "10.1016/j.neuron.2014.12.045")
    check("verdict NO-GO (go=False)", d["go"] is False)


def main() -> int:
    print("=========== p100d_synapse_localization ===========")
    test_lr_not_merged()
    test_lobe_of()
    test_cross_side_lost()
    test_committed()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
