"""P10-C2-0a 回归：精确符号翻转快速算法、功效修订、资格迹时间索引、五状态判定器、资源。仅合成/静态。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from loop.p10c2_0a_gate import (ALPHA, MIN_DZ, PLANNED_N, SEEDS, VALID_N, adjudicate,
                                adjudicator_selftest, eligibility_audit, eligibility_traces_correct,
                                paired_power_normal, paired_power_t, resource_budget, sample_size_plan,
                                sign_flip_p_enum, sign_flip_p_fast, stimulus_audit, verdicts)

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def test_fast_p():
    rng = np.random.default_rng(0)
    ok = True
    for n in range(4, 13):
        for _ in range(10):
            d = rng.normal(0.3, 1.0, n)
            if abs(sign_flip_p_fast(d) - sign_flip_p_enum(d)) > 0:
                ok = False
    check("fast exact p == full enumeration (n=4..12)", ok)
    old = [-6.3e-05, 0.000563, -0.000313, -0.000687, 0.0]
    check("old A0 p = 0.625", abs(sign_flip_p_fast(old) - 0.625) < 1e-9)


def test_power():
    check("normal approx n=20 ~0.828", abs(paired_power_normal(0.65, 20) - 0.828) < 0.01)
    check("noncentral t n=20 ~0.787", abs(paired_power_t(0.65, 20) - 0.787) < 0.01,
          str(round(paired_power_t(0.65, 20), 4)))
    check("t power != normal power (methods distinct)",
          abs(paired_power_t(0.65, 20) - paired_power_normal(0.65, 20)) > 0.02)
    p = os.path.join(HERE, "p10c2_0a_protocol.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        sim = d["power_simulation"]
        check("sim n=20 < 0.80", sim["n20"]["power"] < 0.80, str(sim["n20"]["power"]))
        check("sim n=22 >= 0.80", sim["n22"]["power"] >= 0.80, str(sim["n22"]["power"]))
        check("MC error declared", sim["n22"]["mc_se"] > 0 and sim["n22"]["sims"] > 0)
        check("seed fixed", sim["n22"]["seed"] == 20260)
        s = d["sample_size"]
        check("valid n=22", s["valid_n"] == 22)
        check("planned n=25", s["planned_n"] == 25)
        check("seeds 101-125", s["seeds"] == list(range(101, 126)))
        check("max exclusions 3", s["max_exclusions"] == 3)
        check("condition order fixed", s["condition_order"] == ["none", "stim", "recover"])


def test_eligibility_index():
    edges = np.array([[0, 0]], int)
    pre = np.zeros((12, 1)); post = np.zeros((12, 1))
    pre[5, 0] = 1; post[5, 0] = 1
    e = eligibility_traces_correct(pre, post, edges)[:, 0]
    check("single pulse t=5 -> first nonzero e[6]", np.flatnonzero(e > 0)[0] == 6)
    pre2 = pre.copy(); post2 = post.copy(); pre2[9, 0] = 1; post2[9, 0] = 1
    e2 = eligibility_traces_correct(pre2, post2, edges)[:, 0]
    check("strictly causal", np.allclose(e[:10], e2[:10]))
    a = eligibility_audit()
    check("audit: frozen params", a["frozen_params"]["tau_pre"] == 0.05 and a["frozen_params"]["tau_e"] == 0.5)
    check("audit: exploratory grid separate", len(a["exploratory_only"]["tau_e"]) == 3)
    check("audit: no tau selection by significance", a["no_tau_selection_by_significance"] is True)
    check("audit: valid", a["valid"] is True)


def test_adjudicator():
    a = adjudicator_selftest()
    check("all five states trigger", a["all_correct"] is True)
    check("A0_PASS", a["cases"]["A0_PASS"] == "A0_PASS")
    check("INPUT_INVALID", a["cases"]["INPUT_INVALID"] == "INPUT_INVALID")
    check("METRIC_INVALID", a["cases"]["METRIC_INVALID"] == "METRIC_INVALID")
    check("SENSORY_NO_EFFECT", a["cases"]["SENSORY_NO_EFFECT"] == "SENSORY_NO_EFFECT")
    check("INCONCLUSIVE", a["cases"]["INCONCLUSIVE"] == "INCONCLUSIVE")
    check("missing field rejects PASS", a["cases"]["missing_field"] != "A0_PASS")
    check("version mismatch rejects PASS", a["cases"]["version_mismatch"] != "A0_PASS")
    check("unfrozen param rejects PASS", a["cases"]["unfrozen_dz"] != "A0_PASS")


def test_stimulus_resource():
    s = stimulus_audit()
    check("amount = 0.2*(1-decay) = 0.036254", abs(s["amount"] - 0.036254) < 1e-6)
    check("stimulus ready", s["ready"] is True)
    check("no post-hoc increase", s["no_posthoc_increase"] is True)
    r = resource_budget()
    check("resource ready", r["ready"] is True)
    check("worst case < cap", r["worst_case_2x_min"] <= r["wall_cap_min"])
    check("breakdown separated", all(k in r for k in ("core_step_s", "init_rebuild_s", "eligibility_s", "save_stats_s")))
    check("class3 UNKNOWN", "UNKNOWN" in r["class3_learning"])


def test_verdicts_and_report():
    v = verdicts(True, True, True, True)
    check("statistically valid", v["Protocol_Statistically_Valid"] is True)
    check("eligible to request one A0", v["Eligible_To_Request_One_NoLearning_A0"] is True)
    check("A0 NOT executed", v["A0_Executed"] is False)
    check("training not authorized", v["Training_Authorization"] is False)
    check("B/C/Biological NO-GO", v["Level_B"] == "NO-GO" and v["Level_C"] == "NO-GO" and v["Biological"] == "NO-GO")
    p = os.path.join(HERE, "p10c2_0a_protocol.json")
    d = json.load(open(p, encoding="utf-8"))
    check("report a0_executed False", d["a0_executed"] is False)
    check("report training False", d["training_authorized"] is False)
    check("delta record present", d["delta_from_p10c2_0"]["valid_n"] == "20 -> 22")
    # 历史 P10-C2-0 保留（旧 n=20/22）
    old = json.load(open(os.path.join(HERE, "p10c2_0_protocol.json"), encoding="utf-8"))
    check("P10-C2-0 historical retained", old["sample_size"]["valid_n_required"] == 20 and
          old["sample_size"]["planned_n"] == 22)


def main() -> int:
    print("=========== p10c2_0a ===========")
    test_fast_p()
    test_power()
    test_eligibility_index()
    test_adjudicator()
    test_stimulus_resource()
    test_verdicts_and_report()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
