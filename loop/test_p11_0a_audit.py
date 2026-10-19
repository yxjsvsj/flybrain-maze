"""P11-0a 回归：net/mean 勘误、严格负值定义、pooled/per-seed、启动瞬态、P11-1 预注册边界。只读。"""
from __future__ import annotations

import json
import os
import sys

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p11_0a audit ===========")
    d = json.load(open(os.path.join(HERE, "p11_0a_report.json"), encoding="utf-8"))
    e = d["stat_errata"]
    check("archive verified", d["archive"]["status"] == "ARCHIVED_VERIFIED")
    check("net = per-cell mean x 320", e["net_equals_mean_times_320"] is True,
          f"rel_res={e['net_identity_rel_residual']:.3e}")
    check("net ~ -0.0064", abs(e["net_sum_over_cells"] + 0.0064) < 1e-4)
    check("per-cell mean ~ -2e-5", abs(e["effect_mean_per_cell"] + 2e-5) < 1e-7)
    s = e["sign_consistency"]
    check("strict all_25_negative = 0", s["all_25_negative_strict"] == 0)
    check("all_25_nonpositive = 278", s["all_25_nonpositive"] == 278)
    check("all_zero = 272", s["all_zero"] == 272)
    check("ge20_negative_strict = 0", s["ge20_negative_strict"] == 0)
    check("ge20_nonpositive = 303", s["ge20_nonpositive"] == 303)
    check("old wrong 278 documented", e["old_p11_0_errors"]["all_25_negative"] == 278
          and e["old_p11_0_errors"]["corrected_all_25_negative_strict"] == 0)
    c = e["cancellation"]
    check("weak numeric cancellation True", c["weak_numerical_cancellation"] is True)
    check("large opposite populations False", c["large_opposite_populations"] is False)
    check("cancellation distinction documented", "!=" in c["distinction"])

    p = d["pooled_per_seed"]
    check("pooled mean ~0.9118", abs(p["pooled_mean"] - 0.9118) < 1e-3)
    check("pooled >=0.85 ~0.863", abs(p["pooled_frac_ge_0.85"] - 0.86325) < 1e-3)
    check("pooled ==1 ~0.859", abs(p["pooled_frac_eq_1"] - 0.859375) < 1e-3)
    check("cross-seed per-cell median 0.96", abs(p["cross_seed_per_cell_median"] - 0.96) < 1e-6)
    pm = p["per_seed_population_mean"]
    check("per-seed range ~0.288-0.941", abs(pm["min"] - 0.2882) < 1e-3 and abs(pm["max"] - 0.9414) < 1e-3)
    check("per-seed SD ~0.130", abs(pm["sd"] - 0.130) < 1e-2)
    check("low/high states observed", p["low_high_states_observed"] is True)
    check("no seed selection", p["no_seed_selection"] is True)
    check("single seed-cell vs cross-seed median distinction documented", "different statistics" in p["note"])

    st = d["startup_transient"]
    k = st["KC"]
    check("KC initial bins rise 0.03 -> 0.82 -> 0.90", abs(k["initial_0_20_20_40_40_60_60_80_80_100"][0] - 0.0296) < 1e-3
          and abs(k["initial_0_20_20_40_40_60_60_80_80_100"][1] - 0.8185) < 1e-2
          and abs(k["initial_0_20_20_40_40_60_60_80_80_100"][2] - 0.8991) < 1e-2)
    check("KC t0.85 median 20, 24/25 before 100", k["time_to_0.85_steps"]["median"] == 20.0
          and k["time_to_0.85_steps"]["before_100"] == 24)
    check("KC high state before stimulus", k["high_state_before_stimulus"] is True)
    check("KC stim-none after 100 tiny", k["stim_minus_none_max_abs_after_100"] < 1e-3,
          str(k["stim_minus_none_max_abs_after_100"]))
    check("MBON t0.85 median 40, 22/25 before 100", st["MBON"]["time_to_0.85_steps"]["median"] == 40.0
          and st["MBON"]["time_to_0.85_steps"]["before_100"] == 22)
    check("MBON stim-none after 100 tiny", st["MBON"]["stim_minus_none_max_abs_after_100"] < 1e-3)

    pre = d["p11_1_prereg"]
    check("P11-1 design ready", pre["overall_design_ready"] is True)
    check("P11-1 execution NOT ready", pre["overall_execution_ready"] is False)
    check("training not authorized", pre["training_authorized"] is False)
    p1 = pre["phase1_truncation"]
    check("phase1 matches KCs by bodyId", "bodyId" in p1["matching"])
    check("phase1 full/subnet cost ratio ~7.2", abs(p1["cost_static_estimate"]["ratio"] - 7.207) < 1e-3)
    check("phase1 execution blockers", "full-brain LIF per-step benchmark" in p1["execution_blockers"][0])
    p2 = pre["phase2_sensitivity"]
    check("phase2 factors/values declared", all(x in p2["factors"] for x in ("gain", "tonic", "noise_hz")))
    check("phase2 no selection by response", "NO selection" in p2["selection_rule"])
    check("phase3 not designed", pre["phase3_sensory"]["design_ready"] is False)

    v = d["verdicts"]
    check("errata corrected", v["STATISTICAL_ERRATA_CORRECTED"] is True)
    check("sign consistency corrected", v["SIGN_CONSISTENCY_CORRECTED"] is True)
    check("cancellation distinction", "weak numeric only" in v["CANCELLATION_DISTINCTION"])
    check("HIGH_BASELINE_PHENOTYPE SUPPORTED", v["HIGH_BASELINE_PHENOTYPE"] == "SUPPORTED")
    check("CELL_LEVEL_HETEROGENEITY NOT_OBSERVED", v["CELL_LEVEL_HETEROGENEITY"] == "NOT_OBSERVED")
    check("MODEL_ROOT_CAUSE NOT_ESTABLISHED", v["MODEL_ROOT_CAUSE"] == "NOT_ESTABLISHED")
    check("startup high state before stim", v["STARTUP_HIGH_STATE_BEFORE_STIM"] is True)
    check("P11-1 execution not ready", v["P11_1_Execution_Ready"] is False)
    check("P11-1 not ready to apply", v["P11_1_Ready_To_Apply_For_New_Experiment"] is False)
    check("P10-C2-A0 verdicts preserved", v["P10C2_A0_Entrance"] == "PASS"
          and v["P10C2_A0_KC_Main_Endpoint"] == "NOT_DETECTED" and v["P10C2_A0_Canonical"] == "INCONCLUSIVE")
    check("training False", v["Training_Authorization"] is False)
    # no new dynamics in the audit module
    src = open(os.path.join(HERE, "p11_0a_audit.py"), encoding="utf-8").read()
    check("no FlyBrain.step / flybrain import", "FlyBrain.step" not in src and "from flybrain" not in src)
    # historical P11-0 retained
    old = json.load(open(os.path.join(HERE, "p11_0_report.json"), encoding="utf-8"))
    check("P11-0 historical retained", old["verdicts"]["HIGH_BASELINE_PHENOTYPE"] == "SUPPORTED"
          and old["training_authorized"] is False)
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
