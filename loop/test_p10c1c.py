"""P10-C1c 回归：ΔW 不变量（S1/S2）、调制协议、子网边界、资源负例、A/B/C 独立裁决。不训练。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp

from loop.p10c1c_level_a_gate import (apply_dw, dw_invariants, l1_strategies, modulation_protocol,
                                      resource_gate, subnet_validity, verdicts)

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def test_invariants():
    d = dw_invariants()
    for k in ("S1", "S2"):
        s = d["strategies"][k]
        check(f"{k}: only approved modified", s["only_approved_modified"] is True)
        check(f"{k}: no new edges", s["no_new_edges"] is True)
        check(f"{k}: excitatory nonnegative", s["excitatory_nonnegative"] is True)
        check(f"{k}: inhibitory nonpositive", s["inhibitory_nonpositive"] is True)
        check(f"{k}: unapproved bit-identical", s["unapproved_bit_identical"] is True)
    check("S1 preserves L1", d["strategies"]["S1"]["l1_preserved"] is True)
    check("S2 may break L1", d["strategies"]["S2"]["l1_preserved"] is False)
    check("training off -> zero dW", d["strategies"]["training_off_zero"] is True)
    check("official W untouched", d["strategies"]["official_W_untouched"] is True)


def test_l1_break_explicit():
    W = sp.lil_matrix((2, 2), dtype=np.float32); W[0, 0] = 0.7; W[0, 1] = 0.3; W = W.tocsr()
    approved = np.array([[True, True], [False, False]])
    dW = sp.lil_matrix((2, 2), dtype=np.float32); dW[0, 0] = 0.4; dW = dW.tocsr()
    _, _ = apply_dw(W, dW, approved, l1="S1")
    Weff1, _ = apply_dw(W, dW, approved, l1="S1")
    Weff2, _ = apply_dw(W, dW, approved, l1="S2")
    check("S1 budget <= original row L1", float(np.abs(Weff1.toarray()[0]).sum()) <= 1.0 + 1e-6,
          str(float(np.abs(Weff1.toarray()[0]).sum())))
    check("S2 allows L1 exceed", float(np.abs(Weff2.toarray()[0]).sum()) > 1.0,
          str(float(np.abs(Weff2.toarray()[0]).sum())))


def test_protocol():
    p = modulation_protocol()
    check("D(t) is labelled, not sensory", "NOT sensory input" in p["D_t"]["source"])
    check("D(t) not biological calibration", p["D_t"]["not_biological_calibration"] is True)
    check("eligibility init e(0)=0", "e(0)=0" in p["eligibility"]["init"])
    check("eligibility reset specified", "reset" in p["eligibility"]["reset"])
    check("eligibility causal", p["eligibility"]["causal"] is True and p["eligibility"]["no_future_info"] is True)
    check("params are hypotheses", "computational hypotheses" in p["params"]["source"])


def test_l1_options():
    s = l1_strategies()
    check("two L1 strategies", "S1_plastic_budget" in s and "S2_allow_l1_exceed" in s)
    check("both pending approval", s["S1_plastic_budget"]["pending_approval"] and
          s["S2_allow_l1_exceed"]["pending_approval"])
    check("not selected/applied", "未选择" in s["note"])


def test_subnet():
    s = subnet_validity()
    check("subnet n=23218/nnz=3549643", s["n_neurons"] == 23218 and s["nnz"] == 3549643)
    check("candidate KC->MBON=4533", s["candidate_KC_MBON"] == 4533)
    check("sensory LPLC2/LC4 preserved", s["q3_sensory_preserved"] is True)
    check("truncation reported", s["truncation"]["lost_incoming_edges"] > 0)
    check("truncated-only results listed", len(s["q4_truncated_only"]) >= 3)
    check("full-brain recheck listed", len(s["q5_full_brain_recheck"]) >= 2)


def test_resource():
    r = resource_gate()
    check("measured budget ready", r["measured"]["resource_ready"] is True)
    check("class3 UNKNOWN", "UNKNOWN" in r["class3"])
    check("all negative resource cases fail", all(c["resource_ready"] is False for c in r["negative_cases"]),
          str([c["resource_ready"] for c in r["negative_cases"]]))
    check("1890 trials reported", r["measured"]["trials_1890"] == 1890)
    check("not called full training measurement", "不是**完整训练实测" in r["limitation"] or
          "不是" in r["limitation"])


def test_verdicts():
    dw_ok = all(s["only_approved_modified"] and s["no_new_edges"] and s["excitatory_nonnegative"]
                and s["inhibitory_nonpositive"] and s["unapproved_bit_identical"]
                for s in dw_invariants()["strategies"].values() if isinstance(s, dict))
    res_ok = resource_gate()["ok"]
    v = verdicts(dw_ok, res_ok)
    check("Biological NO-GO", v["Biological_Readiness"] == "NO-GO")
    check("Computational conditional GO", v["Computational_Hypothesis"].startswith("GO"))
    check("A design GO", v["Level_A_Design_Readiness"] == "GO")
    check("A execution NO-GO (erratum/update pending)", v["Level_A_Execution_Readiness"] == "NO-GO")
    check("A exec blockers explicit", "DW_ERRATUM_NOT_APPROVED" in v["a_exec_blockers"] and
          "UPDATE_NOT_IMPLEMENTED" in v["a_exec_blockers"])
    check("B NO-GO", v["Level_B_Execution_Readiness"] == "NO-GO")
    check("C NO-GO", v["Level_C_Execution_Readiness"] == "NO-GO")
    check("training not authorized", v["Training_Authorization"] is False)


def test_erratum():
    p = os.path.join(HERE, "p10c1c_dw_erratum.json")
    if not os.path.exists(p):
        check("erratum file exists", False); return
    d = json.load(open(p, encoding="utf-8"))
    check("erratum pending approval", "PENDING_APPROVAL" in d["status"])
    check("erratum not applied", "NOT applied" in d["status"])
    check("erratum has pending items", len(d["pending_approval_items"]) >= 4)
    check("erratum has two L1 strategies", "S1_plastic_budget" in d["l1_strategies"] and
          "S2_allow_l1_exceed" in d["l1_strategies"])


def main() -> int:
    print("=========== p10c1c ===========")
    test_invariants()
    test_l1_break_explicit()
    test_protocol()
    test_l1_options()
    test_subnet()
    test_resource()
    test_verdicts()
    test_erratum()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
