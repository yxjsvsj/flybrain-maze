"""P11-2-S3-0: KC->MBON functional-chain causal experiment preregistration.

Static design only. No FlyBrain import/initialization, no step, no S3 authorization.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
S1R = os.path.join(HERE, "p11_2_s1r_report.json")
RUN_META = os.path.join(HERE, "p11_2_s1r_run_meta.json")
S2 = os.path.join(HERE, "p11_2_s2_report.json")
OUT = os.path.join(HERE, "p11_2_s3_0_report.json")
SEEDS = list(range(201, 226))
STEPS = 400
STIM = (100, 300)
DT = 0.020
TAU = 0.100
REF_DV = 0.20
PRIMARY_DV = 0.20
LOW_DV = 0.10
DECAY = math.exp(-DT / TAU)
AMOUNT_PRIMARY = PRIMARY_DV * (1.0 - DECAY)
AMOUNT_LOW = LOW_DV * (1.0 - DECAY)
REF_AMOUNT = REF_DV * (1.0 - DECAY)
CONDITIONS = ["sham", "kc_low", "kc_primary"]
N_STEPS_TOTAL = len(CONDITIONS) * len(SEEDS) * STEPS


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    s1r = _load(S1R)
    meta = _load(RUN_META)
    s2 = _load(S2)
    ids = np.load(os.path.join(meta["out_dir"], "ids.npz"), allow_pickle=True)
    target = np.asarray(ids["kc_matched"], dtype=np.int64)
    if len(target) != 320 or len(set(target.tolist())) != 320:
        raise RuntimeError("target definition is not 320 unique KC")
    target_sha = _sha_bytes(target.astype("<i8").tobytes())
    s2_match = s2["B_static_paths"]["matched_kc"]
    if not (s2_match["n"] == 320 and s2_match["reachable"] == 320):
        raise RuntimeError("S2 structural matched set is not 320/320 reachable")
    time_est = round(275.591 * N_STEPS_TOTAL / 20000.0, 3)
    peak_est = round(1943.16 + 5.0, 2)
    margin_est = round(2048.0 - peak_est, 2)
    rep = {
        "protocol": "P11-2-S3-0/v1",
        "baseline": "8461d91",
        "status": "PREREGISTRATION_COMPLETE; NOT_AUTHORIZED; NO_RUN",
        "new_s3_authorization": "NOT_AUTHORIZED",
        "new_full_brain_authorization": False,
        "training_authorization": False,
        "real_synaptic_learning_ready": "NO",
        "levels": "NO-GO",
        "frozen_s1r": {
            "status": s1r.get("status"), "grading": s1r.get("grading"),
            "completeness": s1r.get("completeness"),
            "entry_diff": s1r["entry"]["diff"]["mean"], "entry_p": s1r["entry"]["exact_p"],
            "matched_kc_diff": s1r["primary"]["diff"]["mean"], "matched_kc_p": s1r["primary"]["exact_p"],
            "matched_kc_ci": s1r["primary"]["ci"],
            "matched_mbon_diff": s1r["mbon"]["matched_95_diff"]["mean"], "matched_mbon_p": s1r["mbon"]["exact_p"],
            "preserved": True,
        },
        "s2_basis": {
            "status": s2.get("status"),
            "matched_kc": 320, "matched_kc_reachable": 320,
            "matched_kc_independent_paths": s2_match["independent_paths"],
            "other_kc_reachable": s2["B_static_paths"]["other_kc"]["reachable"],
            "note": "320/320 matched KC have static 2-hop paths; dynamic effective drive is unresolved.",
        },
        "target": {
            "rule": "S1R matched-320 KC set; frozen before S3",
            "count": 320, "unique": 320, "sha256_i64_le": target_sha,
            "first10": [int(x) for x in target[:10]], "last10": [int(x) for x in target[-10:]],
            "no_reselection": True,
        },
        "injection_semantics": {
            "api": "FlyBrain.step(inject=[(indices, amount)]) adds amount to v directly; amount is a voltage increment, not nA",
            "lif": "v <- exp(-dt/tau) v + gain*W@spikes + tonic + noise + inject; v>=1 -> spike; reset 0",
            "dt": DT, "tau": TAU, "decay": round(DECAY, 15), "tonic": 0.14,
            "steady_contribution_formula": "delta_v_inf = amount/(1-exp(-dt/tau))",
            "reference_s1r_dv": REF_DV, "reference_s1r_amount": REF_AMOUNT,
            "primary": {"dv_inf": PRIMARY_DV, "amount": AMOUNT_PRIMARY},
            "low": {"dv_inf": LOW_DV, "amount": AMOUNT_LOW},
            "no_empirical_dose_tuning": True,
        },
        "design": {
            "network": "official MaleCNS synthetic B KC->KC-zero; original LIF/noise/weight semantics",
            "seeds": SEEDS, "conditions": CONDITIONS, "steps_per_seed_condition": STEPS,
            "total_steps": N_STEPS_TOTAL, "window": list(STIM),
            "sham": "no KC injection",
            "kc_low": "matched-320 KC, low voltage increment",
            "kc_primary": "matched-320 KC, primary voltage increment",
            "injection_window": list(STIM),
            "no_sensory_injection": True, "no_mbon_injection": True,
            "cpu_only": True, "one_run": True, "auto_retry": False,
            "npz_estimate": len(CONDITIONS) * len(SEEDS),
            "record_per_condition": ["kc", "mb", "lp", "kc_matched", "mb_matched"],
        },
        "primary_endpoints": {
            "kc_intervention": {
                "target": "matched-320 KC firing fraction [100,300), kc_primary-sham",
                "min_positive_effect": 0.10, "test": "paired exact sign-flip, two-sided",
                "alpha": 0.05, "ci": "paired bootstrap 10,000, seed 20261019, lower bound > 0",
                "saturation_guard": "primary mean KC firing <0.85 and fraction of matched KCs with per-cell primary mean >=0.95 <=0.10",
                "pass_rule": "effect>=+0.10 AND p<0.05 AND CI low>0 AND not saturated",
            },
            "mbon_downstream": {
                "target": "matched-95 MBON firing fraction [100,300), kc_primary-sham",
                "min_positive_effect": 0.05, "test": "paired exact sign-flip, two-sided",
                "alpha": 0.05, "ci": "paired bootstrap 10,000, seed 20261019, lower bound > 0",
                "conditional": "interpreted only if KC intervention passes",
                "pass_rule": "effect>=+0.05 AND p<0.05 AND CI low>0",
            },
            "negative": "MBON effect<=-0.05 with p<0.05 and CI high<0 is reported as KC_MBON_NEGATIVE, not a positive functional pass",
        },
        "secondary_analysis": {
            "low_dose": "kc_low-sham for KC and MBON; descriptive; BH across the two low-dose endpoints; cannot replace primary",
            "all_cell_level": "descriptive only; no cell-level p-value selection",
            "no_endpoint_switching": True,
        },
        "decision_tree": [
            {"if": "KC effect<+0.10 or p>=0.05 or CI includes 0", "result": "INCONCLUSIVE_KC_NOT_DRIVEN", "downstream_interpretation": "none"},
            {"if": "saturation guard fails", "result": "INCONCLUSIVE_KC_SATURATED", "downstream_interpretation": "none"},
            {"if": "KC passes and MBON passes", "result": "KC_MBON_FUNCTIONAL_UNDER_DIRECT_DRIVE", "downstream_interpretation": "downstream KC->MBON functional under artificial drive; natural sensory limitation remains supported, not proven"},
            {"if": "KC passes and MBON effect>=+0.05 but p>=0.05/CI includes 0", "result": "INCONCLUSIVE_DOWNSTREAM", "downstream_interpretation": "cannot distinguish weak downstream from no downstream"},
            {"if": "KC passes and MBON effect<=-0.05 with p<0.05/CI high<0", "result": "KC_MBON_NEGATIVE", "downstream_interpretation": "inhibitory/negative downstream response; not a positive chain"},
            {"if": "KC passes and MBON neither positive nor negative significant", "result": "KC_MBON_NOT_ACTIVATED", "downstream_interpretation": "downstream not activated under artificial KC drive; artificial limitation"},
        ],
        "risk_matrix": [
            {"risk": "KC dose too weak", "mitigation": "predeclared low/primary; KC success gate; otherwise INCONCLUSIVE", "severity": "medium"},
            {"risk": "KC saturation/ceiling", "mitigation": "predeclared mean<0.85 and >=0.95 fraction<=0.10 gate; otherwise INCONCLUSIVE", "severity": "high"},
            {"risk": "artificial synchronous drive", "mitigation": "sham, paired seeds, both doses, explicit no-natural/learning claim", "severity": "high"},
            {"risk": "nonspecific network effects", "mitigation": "sham and predeclared target set; artificial conclusion only", "severity": "medium"},
            {"risk": "resource/timeout", "mitigation": "900 s/2048 MiB, one shot, no retry/no cap increase", "severity": "high"},
            {"risk": "multiplicity", "mitigation": "hierarchical KC gate; low-dose BH; no cell-level selection", "severity": "medium"},
            {"risk": "MBON ambiguous", "mitigation": "INCONCLUSIVE_DOWNSTREAM, no positive chain claim", "severity": "medium"},
        ],
        "resource": {
            "s1r_reference": {"steps": 20000, "wall_s": 275.591, "peak_MiB": 1943.16},
            "s3_steps": N_STEPS_TOTAL, "time_estimate_s": time_est,
            "peak_estimate_MiB": peak_est, "margin_estimate_MiB": margin_est,
            "wall_cap_s": 900.0, "mem_cap_MiB": 2048,
            "status": "RESOURCE_PLAUSIBLE_TIGHT", "verified": False,
            "note": "Linear time scaling from S1R; extra condition is not concurrent, peak estimate is approximate. Must be treated as unverified; no automatic increase.",
            "worth_applying_one_run": "YES_CONDITIONAL_ON_PREFLIGHT",
            "resource_not_ready": False,
        },
        "authorization": {
            "s3": "NOT_AUTHORIZED", "approved": False,
            "s1r_authorization_consumed": True,
            "no_s3_lock_created": True,
            "note": "This is a preregistration only; no S3 authorization or lock is created.",
        },
        "gates": {"NEW_FULL_BRAIN_AUTHORIZATION": False, "TRAINING_AUTHORIZATION": False,
                  "REAL_SYNAPTIC_LEARNING_READY": "NO", "Level_A/B/C/Biological": "NO-GO",
                  "Robot": "NO-GO", "Maze": "NO-GO", "Dijkstra": "NO-GO", "DeltaW": "NO-GO"},
        "no_new_dynamics": True, "no_s3_run": True,
        "answers": {
            "q1_target_and_current_freeze": "Use the S1R matched-320 KC global indices (SHA-256 frozen) and predeclare primary DV_inf=0.20 (amount 0.0362538494) with low DV_inf=0.10 (amount 0.0181269247); no pilot dose selection.",
            "q2_discriminating_result": "Only KC intervention success followed by MBON positive/negative/absent outcome can discriminate upstream effective drive from downstream KC->MBON function; KC failure is INCONCLUSIVE.",
            "q3_avoid_false_functional_claim": "Sham, paired seeds, two doses, KC success gate, saturation gate, and explicit artificial-synchronous-drive limitation; no natural pathway or learning claim.",
            "q4_worth_one_run": "YES, conditional on resource preflight: estimated 413.4 s and ~1948 MiB, tight but within caps; unverified, no retry/no increase.",
            "q5_inconclusive": "KC not driven, KC saturated, ambiguous MBON CI, missing seeds, resource/timeout, or any failure to meet predeclared gates.",
            "next_experiment_candidate": "Direct KC current injection (sham/low/primary) with matched-320 targets, new seeds 201-225; do not execute without a new S3 authorization.",
        },
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({"status": rep["status"], "targets": 320, "target_sha": target_sha,
                      "amount_primary": AMOUNT_PRIMARY, "amount_low": AMOUNT_LOW,
                      "total_steps": N_STEPS_TOTAL, "time_est": time_est,
                      "peak_est": peak_est, "resource": rep["resource"]["status"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
