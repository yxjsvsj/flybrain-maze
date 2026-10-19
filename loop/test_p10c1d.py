"""P10-C1d 回归：A0 FAIL、合成 ΔW 内核（requested/effective、持久化、三模式）、S1/S2 审计、资源一致性。"""
from __future__ import annotations

import json
import os
import sys

from loop.p10c1d_delta_sandbox import (ablation, apply_dw_sandbox, frozen_eval, persistence_roundtrip,
                                       synthetic_case, zero_weight_semantics)

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def test_sandbox():
    for l1 in ("S1", "S2"):
        c = synthetic_case(l1=l1)
        for k, v in c["checks"].items():
            check(f"{l1}: {k}", v is True)
        p = persistence_roundtrip(l1=l1)
        check(f"{l1}: persistence round-trip", p["ok"] is True and p["max_abs_err"] < 1e-7)
        z = c["zero_semantics"]
        check(f"{l1}: zero-weight semantics", z["anatomical_edge_deleted"] == 0 and
              z["effective_weight_zero_but_edge_exists"] >= 1)
    fe = frozen_eval()
    check("FROZEN_EVAL stable", fe["frozen_eval_stable"] is True)
    check("FROZEN_EVAL retains learned dW", fe["retains_learned_dW"] is True)
    ab = ablation()
    check("ABLATION_ZERO delta=0", ab["delta_zero"] is True)
    check("ABLATION_ZERO W_eff=W", ab["Weff_equals_W"] is True)


def test_effective_definition():
    c = synthetic_case(l1="S1")
    check("effective = W_eff - W", c["checks"]["effective_equals_Weff_minus_W"] is True)
    check("dW separate from W", c["checks"]["dW_separate"] is True)
    check("no new edges", c["checks"]["no_new_edges"] is True)
    check("unapproved bit-identical", c["checks"]["unapproved_bit_identical"] is True)


def test_report():
    p = os.path.join(HERE, "p10c1d_report.json")
    if not os.path.exists(p):
        check("report exists", False); return
    d = json.load(open(p, encoding="utf-8"))
    a0 = d["a0"]
    check("A0 verdict FAIL", a0["a0_verdict"] == "FAIL")
    check("A0 go False", a0["a0_go"] is False)
    check("A0 stimulus not repeatable", a0["stim_effect_repeatable"] is False)
    check("A0 eligibility not discernible", a0["eligibility_discernible"] is False)
    check("A0 eligibility saturated by spontaneous", a0["eligibility_saturated_by_spontaneous"] is True)
    check("A0 paired KC effect tiny", abs(a0["paired"]["kc_diff_mean"]) < 0.01)
    s = d["s1_s2_structural_audit"]
    check("S1/S2 audit 95 MBON", s["n_MBON"] == 95)
    check("row L1 ~1", abs(s["row_L1"]["mean"] - 1.0) < 1e-3)
    check("BudgetA zero positive headroom on all", s["budgetA"]["n_mbon_zero_positive_headroom"] == 95)
    check("BudgetB positive headroom on all", s["budgetB"]["n_mbon_positive_headroom"] == 95)
    check("BudgetB headroom ~nonplastic sum", abs(s["budgetB"]["mean_headroom"] - s["nonplastic_abs"]["mean"]) < 1e-3)
    r = d["resource_consistency"]
    check("resource consistent", r["consistent"] is True)
    check("used = max(class1,class2)", abs(r["used_per_step_ms"] - max(r["class1_no_stimulus_ms"],
          r["class2_fixed_stimulus_ms"])) < 1e-6)
    check("3.29h formula matches", abs(r["est_hours_1890"] - r["formula_est_hours"]) < 1e-3)
    check("class3 UNKNOWN", "UNKNOWN" in r["class3"])
    m = d["modulation_identity"]
    check("D(t) not biological", m["teacher_derived_modulation"]["biological"] is False)
    check("actual DAN not measured", m["actual_DAN_activity"]["measured"] is False)
    check("no biological correspondence", m["biological_correspondence"]["evidence"] == "NONE")
    v = d["verdicts"]
    check("A0 NO-GO", v["A0_Sensory_Transmission"] == "NO-GO")
    check("Kernel GO", v["Synthetic_Plasticity_Kernel"] == "GO")
    check("S1 GO", v["S1_Mathematical_Feasibility"] == "GO")
    check("S2 GO", v["S2_Mathematical_Feasibility"] == "GO")
    check("A Design NO-GO (A0 fail)", v["Level_A_Design"] == "NO-GO")
    check("A Execution NO-GO", v["Level_A_Execution"] == "NO-GO")
    check("B NO-GO", v["Level_B"] == "NO-GO")
    check("C NO-GO", v["Level_C"] == "NO-GO")
    check("Biological NO-GO", v["Biological"] == "NO-GO")
    check("Training not authorized", v["Training_Authorization"] is False)


def main() -> int:
    print("=========== p10c1d ===========")
    test_sandbox()
    test_effective_definition()
    test_report()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
