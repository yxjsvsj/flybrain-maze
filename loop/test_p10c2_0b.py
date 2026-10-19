"""P10-C2-0b 回归：精确 p 修复、功效复验、原始记录五状态流水线、资源勘误。仅合成/静态。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from loop.p10c2_0b_adjudication import (adjudicate_raw, make_payload, resource_budget, selftest, verdicts)
from loop.p10c2_0b_exact_stats import (BOOTSTRAP_SEED, sign_flip_p, sign_flip_p_enum)

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def test_exact_p():
    check("all-zero [0,0] -> 1.0", sign_flip_p([0.0, 0.0]) == 1.0)
    check("[1,-1] -> 1.0", sign_flip_p([1.0, -1.0]) == 1.0)
    check("single zero in range", 0.0 <= sign_flip_p([1.0, 2.0, 0.0]) <= 1.0)
    check("multi zero in range", 0.0 <= sign_flip_p([1.0, -2.0, 0.0, 0.0, 3.0]) <= 1.0)
    check("tiny floats -> 1.0", sign_flip_p([1e-15, -1e-15]) == 1.0)
    check("duplicates in range", 0.0 <= sign_flip_p([0.5, 0.5, 0.5, -0.5]) <= 1.0)
    rng = np.random.default_rng(7)
    d = np.array([0.3, -0.1, 0.2, 0.0, 0.4, -0.2])
    check("permutation invariant", abs(sign_flip_p(d) - sign_flip_p(rng.permutation(d))) < 1e-12)
    check("positive scaling invariant", abs(sign_flip_p(d) - sign_flip_p(3.7 * d)) < 1e-12)
    agree = True
    for n in range(4, 17):
        for _ in range(6):
            x = rng.integers(-2, 3, n).astype(float) / 2.0
            if abs(sign_flip_p(x) - sign_flip_p_enum(x)) > 1e-12:
                agree = False
    check("enum agreement n=4..16", agree)
    x25 = rng.normal(0.2, 1.0, 25)
    p25 = sign_flip_p(x25)
    check("n=25 feasible and in range", 0.0 <= p25 <= 1.0, str(round(p25, 6)))
    for bad in ([], [1.0, float("nan")], [float("inf")]):
        try:
            sign_flip_p(bad); ok = False
        except ValueError:
            ok = True
        check(f"rejects invalid input {bad[:2]}", ok)


def test_power():
    p = os.path.join(HERE, "p10c2_0b_exact_stats.json")
    d = json.load(open(p, encoding="utf-8"))
    n20, n22 = d["power"]["n20"], d["power"]["n22"]
    check("n20 power 0.781", abs(n20["power"] - 0.781) < 0.01, str(n20["power"]))
    check("n22 power 0.831", abs(n22["power"] - 0.831) < 0.01, str(n22["power"]))
    check("n20 SE 0.0092", abs(n20["mc_se"] - 0.0092) < 0.001, str(n20["mc_se"]))
    check("n22 SE 0.0084", abs(n22["mc_se"] - 0.0084) < 0.001, str(n22["mc_se"]))
    check("n22 >= 0.80", n22["power"] >= 0.80)
    check("n20 < 0.80 (revised)", n20["power"] < 0.80)
    check("sims/seed declared", n22["sims"] == 2000 and n22["seed"] == 20260)
    r = json.load(open(os.path.join(HERE, "p10c2_0b_report.json"), encoding="utf-8"))
    check("0.831 is KC single-primary only", "NOT joint" in r["power_recheck"]["note"])


def test_adjudication():
    s = selftest()
    check("all cases correct", s["all_correct"] is True)
    check("A0_PASS", s["cases"]["A0_PASS"] == "A0_PASS")
    check("INPUT_INVALID", s["cases"]["INPUT_INVALID"] == "INPUT_INVALID")
    check("METRIC_INVALID", s["cases"]["METRIC_INVALID"] == "METRIC_INVALID")
    check("SENSORY_NO_EFFECT", s["cases"]["SENSORY_NO_EFFECT"] == "SENSORY_NO_EFFECT")
    check("INCONCLUSIVE", s["cases"]["INCONCLUSIVE"] == "INCONCLUSIVE")
    check("PENDING_APPROVAL -> INCONCLUSIVE", s["cases"]["PENDING_APPROVAL"] == "INCONCLUSIVE")
    for k in ("summary_only", "duplicate", "missing_condition", "version_mismatch"):
        check(f"malicious {k} not PASS", s["cases"][k] != "A0_PASS", s["cases"][k])


def test_raw_computation():
    payload = make_payload("pass")
    # 原始 payload 不含手填 p/effect
    check("raw payload has no hand-filled p", all("p" not in r for r in payload["records"]))
    r = adjudicate_raw(payload, eligibility_threshold=0.01)
    check("computed from raw -> PASS", r["state"] == "A0_PASS")
    check("entrance p computed", isinstance(r["entrance"]["p"], float) and r["entrance"]["p"] < 0.05)
    check("KC p computed", isinstance(r["kc"]["p"], float) and r["kc"]["p"] < 0.05)
    check("CI computed", "ci_low" in r["kc"]["ci"] and "ci_high" in r["kc"]["ci"])
    check("eligibility evoked computed", r["eligibility"]["evoked_mean"] > 0)
    # 无阈值 -> PENDING_APPROVAL -> 不可 PASS
    r2 = adjudicate_raw(payload)
    check("no threshold -> INCONCLUSIVE", r2["state"] == "INCONCLUSIVE")
    check("threshold status PENDING_APPROVAL", r2["eligibility"]["threshold_status"] == "PENDING_APPROVAL")
    # 手填汇总 -> 不可 PASS
    r3 = adjudicate_raw({"protocol_version": "P10-C2-0a/v1",
                         "params": payload["params"], "kc_p": 0.001, "effect": 0.5})
    check("summary-only rejected", r3["state"] == "INCONCLUSIVE")


def test_resource():
    r = resource_budget()
    check("30,000 LIF steps", r["lif_steps"] == 30000)
    check("core 39.0 s", abs(r["core_lif_s"] - 39.0) < 1e-6)
    check("worst 4.633 min < 15", r["worst_case_2x_min"] < 15.0)
    check("old 39,000/50.7 flagged", "39,000" in r["old_doc_error"] and "50.7" in r["old_doc_error"])
    check("class3 UNKNOWN", "UNKNOWN" in r["class3_learning"])


def test_verdicts():
    v = verdicts(True, True, True, True)
    check("exact fixed", v["Exact_P_Algorithm_Fixed"] is True)
    check("power ok", v["Primary_Power_At_n22_ge_0.80"] is True)
    check("raw adjudication ok", v["Raw_Record_Adjudication"] is True)
    check("eligibility PENDING_APPROVAL", v["Eligibility_Threshold"] == "PENDING_APPROVAL")
    check("resource ok", v["Resource_Consistent_And_Ready"] is True)
    check("ready to request one A0", v["Ready_To_Request_One_A0"] is True)
    check("A0 not executed", v["A0_Executed"] is False)
    check("training False", v["Training_Authorization"] is False)
    check("B/C/Biological NO-GO", v["Level_B"] == "NO-GO" and v["Level_C"] == "NO-GO" and v["Biological"] == "NO-GO")
    # 历史 P10-C2-0a 保留
    old = json.load(open(os.path.join(HERE, "p10c2_0a_protocol.json"), encoding="utf-8"))
    check("P10-C2-0a historical retained", old["sample_size"]["valid_n"] == 22 and
          old["sample_size"]["planned_n"] == 25)


def main() -> int:
    print("=========== p10c2_0b ===========")
    test_exact_p()
    test_power()
    test_adjudication()
    test_raw_computation()
    test_resource()
    test_verdicts()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
