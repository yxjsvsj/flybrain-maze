"""P10-0c 回归：ID/ROI 匹配（合成）、simple path（合成）与已提交证据表的裁决（不联网）。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp

from loop.p100c_dan_compartment import build_evidence, find_simple_path, mb_comps

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def make_D(n, edges, types, sides=None, superclass=None):
    rows = [r for r, _, _ in edges]; cols = [c for _, c, _ in edges]; vals = [v for _, _, v in edges]
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return {"W": W, "ids": np.arange(100, 100 + n), "cell_type": np.array(types, dtype=object),
            "side": np.array(sides or ["L"] * n, dtype=object),
            "superclass": np.array(superclass or [""] * n, dtype=object)}


def test_mb_comps():
    rows = [{"roi": "g1(L)", "pre": 5, "post": 0}, {"roi": "a'2(R)", "pre": 2, "post": 3},
            {"roi": "CA(R)", "pre": 9, "post": 9}, {"roi": "MB(R)", "pre": 4, "post": 4}]
    check("mb_comps pre picks g1+a'2", mb_comps(rows, "pre") == {"g1", "a'2"}, str(mb_comps(rows, "pre")))
    check("mb_comps excludes non-MB ROI (CA/MB)", "CA" not in mb_comps(rows, "pre") and "MB" not in mb_comps(rows, "pre"))
    check("mb_comps empty when no MB rows", mb_comps([], "pre") == set())
    check("mb_comps post filter", mb_comps([{"roi": "b2(L)", "pre": 0, "post": 7}], "post") == {"b2"})


def test_build_evidence_synthetic():
    types = ["LPLC2", "X", "KCg-d", "MBON01", "Y", "DN", "PAM01"]
    sc = ["", "", "", "", "", "descending_neuron", ""]
    D = make_D(7, [(1, 0, .5), (2, 1, .5), (3, 2, .5), (4, 3, .5), (5, 4, -.5), (6, 2, .5)], types, superclass=sc)
    layers = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
    meta = {102: {"type": "KCg-d", "side": "L", "roi_rows": [{"roi": "g1(L)", "pre": 5, "post": 0}]},
            103: {"type": "MBON01", "side": "L", "roi_rows": [{"roi": "g1(L)", "pre": 0, "post": 5}]},
            106: {"type": "PAM01", "side": "L", "roi_rows": [{"roi": "g1(L)", "pre": 5, "post": 0}]}}
    ev = build_evidence(D, layers, meta)
    check("synthetic edges total = 1", ev["n_edges_total"] == 1, str(ev["n_edges_total"]))
    check("synthetic compartment overlap = 1", ev["n_edges_compartment_overlap"] == 1)
    check("synthetic DAN match = 1", ev["n_edges_with_DAN"] == 1)
    check("synthetic KC/MBON/DAN compartment counts", (ev["n_KC_with_compartment"], ev["n_MBON_with_compartment"],
          ev["n_DAN_with_compartment"]) == (1, 1, 1), str((ev["n_KC_with_compartment"], ev["n_MBON_with_compartment"], ev["n_DAN_with_compartment"])))
    check("synthetic shared set = {g1}", ev["examples"][0]["shared_compartments"] == ["g1"] if ev["examples"] else False)


def test_simple_path_synthetic():
    types = ["LPLC2", "X", "KCg-d", "MBON01", "Y", "DN"]
    sc = ["", "", "", "", "", "descending_neuron"]
    D = make_D(6, [(1, 0, .5), (2, 1, .5), (3, 2, .5), (4, 3, .5), (5, 4, .5)], types, superclass=sc)
    layers = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
    spath = find_simple_path(D, layers)
    check("simple path found", spath["found"] is True)
    check("simple path has 6 distinct bodyIds", len(spath["bodyIds"]) == 6 and len(set(spath["bodyIds"])) == 6)
    check("simple path weights all nonzero", all(w != 0 for w in spath["segment_weights"]))


def test_committed_evidence():
    p = os.path.join(HERE, "p100c_dan_compartment.json")
    if not os.path.exists(p):
        check("committed json exists", False)
        return
    d = json.load(open(p, encoding="utf-8"))
    e = d["evidence"]
    check("real 4533 KC->MBON edges", e["n_edges_total"] == 4533, str(e["n_edges_total"]))
    check("real all edges have neuron-level overlap", e["n_edges_compartment_overlap"] == 4533)
    check("real overlap is non-discriminative (many distinct sets)", e["n_distinct_shared_sets"] > 1)
    check("real DAN with MB comps < 340 (some MISSING)", e["n_DAN_with_compartment"] < 340)
    check("simple path is simple", d["simple_path"]["found"] and len(set(d["simple_path"]["bodyIds"])) == 6)
    g = d["grading"]
    check("Synapse_Specificity MISSING", "MISSING" in g["Synapse_Specificity"])
    check("Biological_Calibration unresolved", "g_unit" in g["Biological_Calibration"])
    check("Learning_Readiness = NO", g["Learning_Readiness"].startswith("否"))
    check("verdict NO-GO (go=False)", d["go"] is False)


def main() -> int:
    print("=========== p100c_dan_compartment ===========")
    test_mb_comps()
    test_build_evidence_synthetic()
    test_simple_path_synthetic()
    test_committed_evidence()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
