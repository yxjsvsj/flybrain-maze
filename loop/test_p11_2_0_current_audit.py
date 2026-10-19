"""P11-2-0 回归：源码更新式、数学电流预算、KC 权重预算、哈希勘误、A1 准入。静态/合成。"""
from __future__ import annotations

import json
import os
import subprocess
import sys

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p11_2_0 current audit ===========")
    d = json.load(open(os.path.join(HERE, "p11_2_0_report.json"), encoding="utf-8"))
    s = d["source_audit"]
    check("source: decay/gain/threshold/reset/noise confirmed",
          all(s[k] for k in ("decay_source", "gain_applied_to_synaptic_current", "decay_before_add",
                             "tonic_added_with_current", "noise_bernoulli", "threshold_ge_1",
                             "reset_zero", "synaptic_uses_previous_fired", "postsynaptic_L1_normalization_source")))
    m = d["math_budget"]
    check("decay 0.818730753", abs(m["decay"] - 0.818730753) < 1e-9)
    check("tonic fixed point 0.772 < 1", abs(m["tonic_only_fixed_point"] - 0.772332) < 1e-5
          and m["tonic_alone_reaches_threshold"] is False)
    check("single-step post-gain 0.86", abs(m["single_step_input_from_reset_post_gain"] - 0.86) < 1e-9)
    check("single-step raw 0.286667", abs(m["single_step_input_from_reset_raw_Wx"] - 0.286667) < 1e-6)
    check("sustained post-gain 0.041269", abs(m["sustained_subthreshold_current_post_gain"] - 0.041269) < 1e-6)
    check("every-step raw 0.286667", abs(m["every_step_firing_raw_Wx"] - 0.286667) < 1e-6)
    check("noise p 0.024 / mean 0.00528", abs(m["noise_probability_per_step"] - 0.024) < 1e-9
          and abs(m["noise_mean_voltage_per_step"] - 0.00528) < 1e-9)
    check("tonic+noise mean fp 0.8008 < 1", abs(m["tonic_noise_mean_fixed_point"] - 0.8008) < 1e-3
          and m["tonic_noise_alone_reaches_threshold_in_expectation"] is False)

    k = d["kc_weight_budget"]
    check("4,064 KC rows", k["n_KC"] == 4064)
    check("pos mean ~0.883", abs(k["pos_sum"]["mean"] - 0.883136) < 1e-5)
    check("pos min ~0.802 / max ~0.957", abs(k["pos_sum"]["min"] - 0.801653) < 1e-5
          and abs(k["pos_sum"]["max"] - 0.957122) < 1e-5)
    check("neg mean ~0.1169", abs(k["neg_abs"]["mean"] - 0.116864) < 1e-5)
    check("L1 ~1", abs(k["L1"]["mean"] - 1.0) < 1e-3)
    check("edges mean 204 / median 190 / zero 0", abs(k["edges"]["mean"] - 204) < 1
          and k["edges"]["median"] == 190 and k["edges"]["zero_rows"] == 0)
    check("all pos >= every-step threshold", k["frac_pos_ge_threshold"] == 1.0
          and k["static_weights_can_reach_every_step_threshold"] is True)
    check("max post-gain current >= 2.4", k["max_postgain_current_from_static_budget"]["min"] >= 2.4)
    check("actual Wx UNKNOWN", "not saved" in k["actual_Wx_unknown"])
    tops = [x["type"] for x in k["top_presynaptic_types"]]
    check("top types include KCab/KCg-m/APL", "KCab" in tops and "KCg-m" in tops and "APL" in tops)

    e = d["source_errata"]
    check("LF hash present", len(e["current_LF_sha256"]) == 64)
    check("CRLF hash present", len(e["current_CRLF_sha256"]) == 64)
    check("report hash matches current CRLF", e["report_matches_current_CRLF"] is True)
    check("report hash does NOT match current LF", e["report_matches_current_LF"] is False)
    check("hash timing = compare-only post-processing", "compare-only" in e["report_hash_timing"])
    check("SOURCE_PROVENANCE PARTIAL", "PARTIAL" in e["conclusion"])

    v = d["verdicts"]
    check("tonic alone NO", v["TONIC_ALONE_REACHES_THRESHOLD"] == "NO")
    check("every-step raw threshold 0.286667", abs(v["EVERY_STEP_RAW_Wx_THRESHOLD"] - 0.286667) < 1e-6)
    check("static weights sufficient in principle", v["STATIC_WEIGHTS_SUFFICIENT_IN_PRINCIPLE"].startswith("YES"))
    check("actual Wx UNKNOWN", v["ACTUAL_Wx_CONTRIBUTION"].startswith("UNKNOWN"))
    check("tonic/noise excluded", len(v["MECHANISMS_MATHEMATICALLY_EXCLUDED"]) == 2)
    check("candidates include recurrent KC->KC", any("KC->KC" in x for x in v["CANDIDATE_MECHANISMS"]))
    check("root cause NOT_ESTABLISHED", v["MODEL_ROOT_CAUSE"] == "NOT_ESTABLISHED")
    check("A1 worth applying", v["A1_WORTH_APPLYING"].startswith("YES"))
    check("A2/A3/B NO-GO", v["A2_A3_B"] == "NO-GO")
    check("REAL_SYNAPTIC_LEARNING_READY NO", v["REAL_SYNAPTIC_LEARNING_READY"] == "NO")
    check("training False", v["Training_Authorization"] is False)
    check("Level A/B/C/Biological NO-GO", v["Level_A_Training"] == "NO-GO" and v["Level_B"] == "NO-GO"
          and v["Level_C"] == "NO-GO" and v["Biological"] == "NO-GO")
    check("official hashes ok", d["official_hashes"]["ok"] is True)
    src = open(os.path.join(HERE, "p11_2_0_current_audit.py"), encoding="utf-8").read()
    check("no flybrain import / instantiation", "from flybrain" not in src and "FlyBrain(" not in src)
    try:
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30).stdout
        check("pre-existing dirty files preserved", all(x in st for x in
              ("hardware/tof5_client.py", "loop/diagnose.py", "loop/p10c1b_decoder_freeze.json")))
        r = subprocess.run(["git", "diff", "--stat", "P1-mapfix", "--", "nav", "config.py",
                            "loop/decode.py", "loop/encode.py", "loop/run.py"],
                           cwd=ROOT, capture_output=True, text=True, timeout=30)
        check("P1 freeze empty", r.returncode == 0 and r.stdout.strip() == "")
    except Exception:                            # noqa: BLE001
        check("git checks", False)
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
