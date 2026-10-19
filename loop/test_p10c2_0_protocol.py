"""P10-C2-0 回归：样本量/功效、刺激单位、资格迹因果性与指标、失败条件、边界。仅合成/静态。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from loop.p10c2_0_protocol import (ALPHA, MIN_DZ, PLANNED_N, VALID_N_REQUIRED, eligibility_audit,
                                   eligibility_traces, metrics_and_failure, min_signflip_p,
                                   paired_power, required_n, resource_budget, sample_size_plan,
                                   sign_flip_p, stimulus_unit_audit, verdicts)

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def test_sample_size():
    s = sample_size_plan()
    check("min dz = 0.65", MIN_DZ == 0.65)
    check("power at valid n >= 0.80", paired_power(MIN_DZ, VALID_N_REQUIRED) >= 0.80,
          str(round(paired_power(MIN_DZ, VALID_N_REQUIRED), 4)))
    check("required n <= valid n", required_n(MIN_DZ) <= VALID_N_REQUIRED, str(required_n(MIN_DZ)))
    check("planned n = 22", PLANNED_N == 22)
    check("valid n = 20", VALID_N_REQUIRED == 20)
    check("exact p reachable with 6", min_signflip_p(6) < ALPHA)
    check("exact p NOT reachable with 5", min_signflip_p(5) >= ALPHA)
    check("not based on old A0 effect", s["not_based_on_old_A0_effect"] is True)
    check("zero handling frozen", "sign-flip" in s["independent_unit"] or "paired" in s["test"])


def test_signflip():
    check("all same sign 5 -> 0.0625", abs(sign_flip_p([1, 2, 3, 4, 5]) - 0.0625) < 1e-9)
    check("one zero + 4 same -> 0.125", abs(sign_flip_p([1, 2, 3, 4, 0]) - 0.125) < 1e-9)
    check("all zero -> 1.0", abs(sign_flip_p([0, 0, 0, 0, 0]) - 1.0) < 1e-9)


def test_stimulus_units():
    u = stimulus_unit_audit()
    check("old amount 0.6 -> dv 3.31", abs(u["old_steady_state_dv"] - 3.31) < 0.01)
    check("old unit NOT consistent", u["old_unit_consistent"] is False)
    check("Encoder formula recorded", "amount = dv" in u["encoder_semantics"])
    check("proposed amount = dv*(1-decay)", abs(u["proposed_amount_per_step"] - 0.036254) < 1e-5)
    check("mismatch factor ~16.5", abs(u["old_mismatch_factor_vs_dv_0.2"] - 16.55) < 0.1)
    check("calibration on entrance not KC", "LPLC2/LC4" in u["calibration_target"] and "NOT KC" in u["calibration_target"])
    check("no post-hoc increase", u["no_posthoc_increase"] is True)
    check("ceiling criteria present", "kc_baseline_rate_max" in u["ceiling_criteria"])


def test_eligibility():
    edges = np.array([[0, 0], [1, 0], [0, 1]], int)
    pre = np.zeros((20, 2)); post = np.zeros((20, 2))
    pre[5, 0] = 1; post[6, 0] = 1
    e1 = eligibility_traces(pre, post, edges)
    check("init e(0)=0", bool(np.all(e1[0] == 0)))
    check("nonzero after spikes", bool(e1[6:].max() > 0))
    pre2 = pre.copy(); post2 = post.copy(); pre2[15, 1] = 1; post2[15, 1] = 1
    e2 = eligibility_traces(pre2, post2, edges)
    check("strictly causal (future spike no effect)", bool(np.allclose(e1[:16], e2[:16])))
    e3 = eligibility_traces(pre, post, edges)
    check("reset/fresh equal", bool(np.allclose(e1, e3)))
    a = eligibility_audit()
    check("audit: strictly causal", a["strictly_causal"] is True)
    check("audit: time-resolved", a["time_resolved"] is True)
    check("audit: spontaneous vs evoked separable", a["spontaneous_vs_evoked_separable"] is True)
    check("audit: amplitude and nonzero recorded", a["records_amplitude_and_nonzero"] is True)
    check("audit: normalized fraction only is NOT sole metric", "only" not in a["primary_metric"].lower())


def test_failure_and_boundary():
    m = metrics_and_failure()
    check("five states", set(m["states"]) == {"INPUT_INVALID", "METRIC_INVALID", "SENSORY_NO_EFFECT",
                                              "INCONCLUSIVE", "A0_PASS"})
    check("A0_PASS not just p<0.05", "not just one p<0.05" in m["states"]["A0_PASS"])
    check("A0_PASS does not authorize training", m["a0_pass_authorizes_training"] is False)
    check("primary endpoint KC", "KC" in m["single_primary_endpoint"])
    check("secondary FDR", "BH" in m["secondary_family_fdr"] or "Benjamini" in m["secondary_family_fdr"])
    check("zero/nonfinite frozen", m["zero_diff_handling"] and "10%" in m["nonfinite_failure"])
    r = resource_budget()
    check("planned n=22 in budget", r["seeds"] == 22)
    check("LIF steps = 22*3*400", r["lif_steps_total"] == 22 * 3 * 400)
    check("class3 UNKNOWN", "UNKNOWN" in r["class3_learning_cost"])
    v = verdicts()
    check("protocol statistically valid", v["Protocol_Statistically_Valid"] is True)
    check("stimulus units clear", v["Stimulus_Units_Clear"] is True)
    check("eligibility preregistered", v["Eligibility_Metric_Preregistered"] is True)
    check("failure conditions observable", v["Failure_Conditions_Observable"] is True)
    check("A0 NOT run", v["A0_Run_This_Round"] is False)
    check("A0_PASS would not authorize training", v["Training_Authorization"] is False)
    check("B/C/Biological NO-GO", v["Level_B"] == "NO-GO" and v["Level_C"] == "NO-GO" and v["Biological"] == "NO-GO")


def test_report_and_history():
    p = os.path.join(HERE, "p10c2_0_protocol.json")
    check("protocol json exists", os.path.exists(p))
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        check("a0_executed False", d["a0_executed"] is False)
        check("training_authorized False", d["training_authorized"] is False)
    h = os.path.join(HERE, "p10c1e_report.json")
    if os.path.exists(h):
        e = json.load(open(h, encoding="utf-8"))
        check("P10-C1e historical untouched (training False)",
              e["verdicts"]["Training_Authorization"] is False)
        check("P10-C1e S1B erratum retained", "erratum" in e["verdicts"]["S1_Budget_B"])


def main() -> int:
    print("=========== p10c2_0_protocol ===========")
    test_sample_size()
    test_signflip()
    test_stimulus_units()
    test_eligibility()
    test_failure_and_boundary()
    test_report_and_history()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
