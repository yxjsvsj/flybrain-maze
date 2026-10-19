"""P10-C1e 回归：A0 统计可达性、资格迹指标审计、S1 Budget B 勘误、合成内核元数据检查、阶段关闭。"""
from __future__ import annotations

import json
import os
import sys

from loop.p10c1e_method_audit import (a0_stats_audit, can_reach_alpha, candidate_loader_tests,
                                      eligibility_audit, kernel_audit, load_a0, min_achievable_p,
                                      s1_budget_erratum, sign_flip_p, verdicts)

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def test_a0_stats():
    a = a0_stats_audit()
    d = load_a0()
    check("saved KC diffs match", [round(x, 8) for x in d["paired"]["kc_diff_values"]] == a["diffs"]["kc"])
    check("KC n_nonzero=4, n_zero=1", a["kc"]["n_nonzero"] == 4 and a["kc"]["n_zero"] == 1)
    check("p (include zeros) = 0.625", abs(a["kc"]["p_include_zeros"] - 0.625) < 1e-9)
    check("p (exclude zeros) = 0.625", abs(a["kc"]["p_exclude_zeros"] - 0.625) < 1e-9)
    check("current min possible p = 0.125", abs(a["kc"]["min_p_possible_current_nonzero"] - 0.125) < 1e-9)
    check("5 nonzero min p = 0.0625", abs(min_achievable_p(5) - 0.0625) < 1e-12)
    check("5 nonzero cannot reach alpha 0.05", can_reach_alpha(5) is False)
    check("6 nonzero can reach alpha 0.05", can_reach_alpha(6) is True)
    check("statistical gate INVALID", a["stat_gate_valid"] is False)
    s = a["synthetic"]
    check("synthetic all-same-sign p = 0.0625", abs(s["all_same_sign_5"]["p"] - 0.0625) < 1e-9)
    check("synthetic one-zero p = 0.125", abs(s["one_zero_4_nonzero"]["p"] - 0.125) < 1e-9)
    check("synthetic all-zero p = 1.0", abs(s["all_zero_5"]["p"] - 1.0) < 1e-9)
    check("synthetic mixed p >= 0.0625", s["mixed_signs_5"]["p"] >= 0.0625)
    check("non-finite rejected", a["nonfinite_rejected"] is True)


def test_eligibility():
    e = eligibility_audit()
    check("uses same-step product (mismatch)", e["uses_same_step_product"] is True)
    check("definition does NOT match prereg", e["definition_matches_prereg"] is False)
    check("final-only binary fraction", e["final_only_binary_fraction"] is True)
    check("entrance positive control MISSING", e["entrance_positive_control_recorded"] is False)
    check("amplitude timecourse MISSING", e["amplitude_timecourse_saved"] is False)
    check("recover baseline not matched", e["recover_baseline_matched"] is False and e["recover_vs_stim_comparable"] is False)
    check("eligibility validity INVALID/LIMITED", e["validity"] == "INVALID/LIMITED")
    check("four interpretations separated", len(e["interpretations"]) == 4)
    check("no transmission NOT established", e["interpretations"]["stimulus_truly_no_transmission"] == "NOT ESTABLISHED")
    check("data cannot discriminate", e["interpretations"]["current_data_cannot_discriminate"] == "YES")


def test_s1_erratum():
    s = s1_budget_erratum()
    check("95 MBON", s["n_MBON"] == 95)
    check("original L1 all 1", s["l1_total"]["all_one"] is True)
    check("BudgetA headroom 0 (95 zero)", s["budgetA_headroom"]["n_zero"] == 95)
    check("BudgetB headroom ~0", abs(s["budgetB_headroom"]["mean"]) < 1e-6 and s["budgetB_headroom"]["max"] < 1e-6)
    check("BudgetB 95 within 1e-6", s["budgetB_headroom"]["n_zero_1e-6"] == 95)
    check("previous wrong value recorded", s["previous_wrong_budgetB_headroom"] == 0.9585)
    check("corrected value 0", s["corrected_budgetB_headroom"] == 0.0)
    check("synthetic negative exceeds 1", s["synthetic_negative"]["exceeds_one"] is True)
    check("erratum text present", "误当作" in s["erratum"])


def test_kernel():
    k = kernel_audit()
    check("requested/effective consistent", k["requested_effective_consistent"] is True)
    check("FROZEN_EVAL retains", k["frozen_eval_retains"] is True)
    check("zero-weight keeps anatomical identity", k["zero_weight_keeps_anatomical_identity"] is True)
    check("original save_state lacks metadata (limitation noted)", k["original_save_state_has_metadata"] is False)
    check("original FROZEN_EVAL does not validate source", k["original_frozen_eval_validates_source"] is False)
    t = candidate_loader_tests()
    check("candidate loader matching ok", t["matching_ok"] is True)
    check("candidate loader rejects wrong W", t["wrong_W"] is True)
    check("candidate loader rejects wrong mask", t["wrong_approved"] is True)
    check("candidate loader rejects wrong params", t["wrong_params"] is True)
    check("candidate loader rejects wrong L1", t["wrong_l1"] is True)


def test_verdicts():
    v = verdicts()
    check("A0 historical FAIL retained", v["A0_Historical_Protocol_Result"].startswith("FAIL"))
    check("A0 statistical gate INVALID", v["A0_Statistical_Gate_Validity"] == "INVALID")
    check("A0 sensory INCONCLUSIVE", v["A0_Sensory_Transmission_Evidence"] == "INCONCLUSIVE")
    check("eligibility INVALID/LIMITED", v["Eligibility_Measurement_Validity"] == "INVALID/LIMITED")
    check("S1 Budget B erratum", "erratum" in v["S1_Budget_B"])
    check("Level A Execution NO-GO", v["Level_A_Execution"] == "NO-GO")
    check("B/C NO-GO", v["Level_B"] == "NO-GO" and v["Level_C"] == "NO-GO")
    check("Biological NO-GO", v["Biological"] == "NO-GO")
    check("Training not authorized", v["Training_Authorization"] is False)
    check("route formally closed", v["route"].startswith("FORMALLY CLOSED"))
    check("restart prerequisites listed", len(v["restart_requires"]) >= 3)


def test_historical_untouched():
    p = os.path.join(HERE, "p10c1d_report.json")
    d = json.load(open(p, encoding="utf-8"))
    check("P10-C1d historical report retained", d["verdicts"]["Level_A_Execution"] == "NO-GO")
    check("P10-C1d A0 still FAIL", d["a0"]["a0_verdict"] == "FAIL")
    check("P10-C1d wrong BudgetB note retained (historical)",
          d["s1_s2_structural_audit"]["budgetB"]["mean_headroom"] > 0.9 and
          d["s1_s2_structural_audit"]["budgetB"]["n_mbon_positive_headroom"] == 95)


def main() -> int:
    print("=========== p10c1e ===========")
    test_a0_stats()
    test_eligibility()
    test_s1_erratum()
    test_kernel()
    test_verdicts()
    test_historical_untouched()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
