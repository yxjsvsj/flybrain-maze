"""P11-2-S4-OFFLINE: diagnose weak direct KC drive from S3 NPZ + static model source.

No FlyBrain import/step; no full-brain dynamics; no new experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

import loop.p11_2_s3_experiment as s3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
S3R = os.path.join(HERE, "p11_2_s3_report.json")
S3M = os.path.join(HERE, "p11_2_s3_run_meta.json")
S3MAN = os.path.join(HERE, "p11_2_s3_raw_manifest.json")
S1M = os.path.join(HERE, "p11_2_s1r_run_meta.json")
BACKUP = r"C:\AA_project\guoying_archive\p11_2_s3_run_20261021T102823"
BRAIN_SRC = os.path.join(os.path.dirname(s3.__file__), "..", "p11_1c_full_brain_bench.py")
OUT = os.path.join(HERE, "p11_2_s4_report.json")
SEEDS = list(range(201, 226))
CONDS = ["sham", "kc_low", "kc_primary"]
BINS = [0, 100, 150, 200, 250, 300, 400]


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _stats(x):
    x = np.asarray(x, float)
    return {"mean": round(float(x.mean()), 9),
            "sd": round(float(x.std(ddof=1)), 9),
            "min": round(float(x.min()), 9), "max": round(float(x.max()), 9),
            "n_positive": int((x > 0).sum()), "n_negative": int((x < 0).sum()),
            "n_zero": int((x == 0).sum())}


def _cell_cancel(x):
    x = np.asarray(x, float)
    n = len(x); k = max(1, int(round(.01 * n)))
    order = np.argsort(x)
    return {**_stats(x),
            "positive_sum": round(float(x[x > 0].sum()), 9),
            "negative_sum": round(float(x[x < 0].sum()), 9),
            "net_sum": round(float(x.sum()), 9),
            "top_1pct_contribution_to_mean": round(float(x[order[-k:]].sum() / n), 12),
            "top_1pct_n": int(k),
            "max_abs": round(float(np.abs(x).max()), 9)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    s3r = _load(S3R); s3m = _load(S3M); s3man = _load(S3MAN)
    raw = s3m["out_dir"]
    s1m = _load(S1M)
    ids = np.load(os.path.join(s1m["out_dir"], "ids.npz"), allow_pickle=True)
    kc_idx = np.asarray(ids["kc_idx"])
    target = np.asarray(ids["kc_matched"])
    ml = np.searchsorted(kc_idx, target)
    other = np.setdiff1d(np.arange(len(kc_idx)), ml)
    # raw/manifest verification, including backup
    raw_bad = []; backup_bad = []
    for row in s3man["files"]:
        p = os.path.join(raw, row["name"])
        if not os.path.exists(p) or _sha(p) != row["sha256"] or os.path.getsize(p) != row["size"]:
            raw_bad.append(row["name"])
        q = os.path.join(BACKUP, row["name"])
        if not os.path.exists(q) or _sha(q) != row["sha256"]:
            backup_bad.append(row["name"])
    raw_validate = s3.validate_raw_75(raw, auth_sha=s3r["run_meta"]["auth_checksum"],
                                      script_sha=s3r["run_meta"]["script_sha256"])
    # per-seed / per-cell dynamic
    means = {c: {} for c in CONDS}
    cells = {c: {} for c in CONDS}
    tb = {c: {e: np.zeros(len(BINS) - 1) for e in ("kc", "mb")} for c in CONDS}
    for s in SEEDS:
        for c in CONDS:
            with np.load(os.path.join(raw, f"{s}_{c}.npz"), allow_pickle=True) as z:
                kcm = z["kc_matched"][100:300]
                mbm = z["mb_matched"][100:300]
                means[c][s] = {"kc": float(kcm.mean()), "mb": float(mbm.mean())}
                cells[c][s] = {"kc": kcm.mean(axis=0), "mb": mbm.mean(axis=0),
                               "kcf": z["kc"][100:300].mean(axis=0),
                               "mbf": z["mb"][100:300].mean(axis=0)}
                for j in range(len(BINS) - 1):
                    lo, hi = BINS[j], BINS[j + 1]
                    tb[c]["kc"][j] += float(z["kc_matched"][lo:hi].mean())
                    tb[c]["mb"][j] += float(z["mb_matched"][lo:hi].mean())
    for c in CONDS:
        for e in ("kc", "mb"):
            tb[c][e] /= len(SEEDS)
    def per_seed(c, e):
        return np.asarray([means[c][s][e] - means["sham"][s][e] for s in SEEDS], float)
    def pack(d):
        return {**_stats(d), "p": s3.exact_signflip(d), "ci": s3.boot_ci(d)}
    kc_p = pack(per_seed("kc_primary", "kc")); mb_p = pack(per_seed("kc_primary", "mb"))
    kc_l = pack(per_seed("kc_low", "kc")); mb_l = pack(per_seed("kc_low", "mb"))
    # per-cell primary differences
    def cell_diff(c, key):
        return np.mean([cells[c][s][key] - cells["sham"][s][key] for s in SEEDS], axis=0)
    kc_m_p = cell_diff("kc_primary", "kc"); kc_m_l = cell_diff("kc_low", "kc")
    kc_f_p = cell_diff("kc_primary", "kcf"); kc_f_l = cell_diff("kc_low", "kcf")
    mb_p_cell = cell_diff("kc_primary", "mb"); mb_l_cell = cell_diff("kc_low", "mb")
    primary_rate = np.mean([cells["kc_primary"][s]["kc"] for s in SEEDS], axis=0)
    sham_rate = np.mean([cells["sham"][s]["kc"] for s in SEEDS], axis=0)
    other_p = kc_f_p[other]
    # model static semantics from actual source
    import flybrain.brain as fb
    src = open(fb.__file__, encoding="utf-8").read()
    sem = {"source_file": fb.__file__, "source_sha256": _sha(fb.__file__),
           "inject_is_voltage": "self.v[xp.asarray(idx)] += self._amount(amount)" in src,
           "reset_to_zero": "self.v.ravel()[fired] = 0.0" in src,
           "threshold": "xp.flatnonzero(self.v >= 1.0)" in src,
           "dt": 0.020, "tau": 0.100, "decay": round(float(np.exp(-0.020 / 0.100)), 15),
           "amount_primary": s3.AMOUNT_PRIMARY, "amount_low": s3.AMOUNT_LOW,
           "dv_inf_primary": round(float(s3.AMOUNT_PRIMARY / (1 - np.exp(-0.020 / 0.100))), 12),
           "dv_inf_low": round(float(s3.AMOUNT_LOW / (1 - np.exp(-0.020 / 0.100))), 12),
           "observed_kc_rate_delta_primary": kc_p["mean"],
           "observed_kc_rate_delta_low": kc_l["mean"],
           "low_to_primary_rate_ratio": round(float(kc_l["mean"] / kc_p["mean"]), 9),
           "low_to_primary_dv_ratio": round(float(s3.AMOUNT_LOW / s3.AMOUNT_PRIMARY), 9)}
    rep = {
        "protocol": "P11-2-S4-OFFLINE/v1",
        "baseline": "775de5b",
        "status": "S4_OFFLINE_COMPLETE; NO_NEW_DYNAMICS; EXPLORATORY",
        "frozen_s3": {
            "status": s3r.get("status"), "decision": s3r.get("decision"),
            "scientific_go": s3r.get("scientific_go"),
            "kc_primary": s3r.get("kc_primary"), "mbon_primary": s3r.get("mbon_primary"),
            "saturation": s3r.get("saturation"),
        },
        "archive": {
            "raw_dir": raw, "backup_dir": BACKUP,
            "npz_count": raw_validate["npz_count"],
            "raw_hashes_ok": not raw_bad, "backup_hashes_ok": not backup_bad,
            "raw_bad": raw_bad, "backup_bad": backup_bad,
            "run_meta_sha256_match": _sha(os.path.join(raw, "run_meta.json")) == s3man.get("run_meta_sha256"),
            "manifest_files": len(s3man["files"]),
        },
        "reproduction": {
            "kc_primary": kc_p, "kc_low": kc_l,
            "mbon_primary": mb_p, "mbon_low": mb_l,
            "formal_kc_mean": s3r["kc_primary"]["mean"],
            "formal_kc_p": s3r["kc_primary"]["p"],
            "formal_mbon_mean": s3r["mbon_primary"]["mean"],
            "formal_mbon_p": s3r["mbon_primary"]["p"],
            "kc_match": abs(kc_p["mean"] - s3r["kc_primary"]["mean"]) < 1e-9,
            "kc_p_match": abs(kc_p["p"] - s3r["kc_primary"]["p"]) < 1e-12,
            "mbon_match": abs(mb_p["mean"] - s3r["mbon_primary"]["mean"]) < 1e-9,
            "mbon_p_match": abs(mb_p["p"] - s3r["mbon_primary"]["p"]) < 1e-12,
        },
        "dose_response": {
            "kc_primary": kc_p, "kc_low": kc_l,
            "low_to_primary_aggregate_ratio": sem["low_to_primary_rate_ratio"],
            "per_seed_low_to_primary_ratio_mean": round(float(np.mean(per_seed("kc_low", "kc") / per_seed("kc_primary", "kc"))), 9),
            "primary_direction_positive": int((per_seed("kc_primary", "kc") > 0).sum()),
            "low_direction_positive": int((per_seed("kc_low", "kc") > 0).sum()),
            "primary_sd": kc_p["sd"], "low_sd": kc_l["sd"],
            "note": "two fixed doses only; no extrapolation to untested stronger dose",
        },
        "cell_level": {
            "matched_kc_primary": _cell_cancel(kc_m_p),
            "matched_kc_low": _cell_cancel(kc_m_l),
            "full_kc_primary": _cell_cancel(kc_f_p),
            "other_kc_primary": _cell_cancel(other_p),
            "matched_mbon_primary": _cell_cancel(mb_p_cell),
            "matched_mbon_low": _cell_cancel(mb_l_cell),
            "primary_matched_rate_max": round(float(primary_rate.max()), 9),
            "sham_matched_rate_max": round(float(sham_rate.max()), 9),
            "primary_cells_ge085": int((primary_rate >= .85).sum()),
            "primary_cells_ge095": int((primary_rate >= .95).sum()),
            "note": "cell-level descriptive; not 25 independent samples; no cell selection",
        },
        "time_bins": {
            "bins": BINS,
            "kc_primary_minus_sham": [round(float(tb["kc_primary"]["kc"][j] - tb["sham"]["kc"][j]), 9) for j in range(6)],
            "kc_low_minus_sham": [round(float(tb["kc_low"]["kc"][j] - tb["sham"]["kc"][j]), 9) for j in range(6)],
            "mbon_primary_minus_sham": [round(float(tb["kc_primary"]["mb"][j] - tb["sham"]["mb"][j]), 9) for j in range(6)],
            "note": "sustained vs transient; EXPLORATORY",
        },
        "model_semantics": sem,
        "mechanism": {
            "OBSERVED": [
                "Inject is direct voltage; primary steady subthreshold increment DV_inf=0.20, low=0.10.",
                "Observed matched KC increment +0.022858 (primary) and +0.011632 (low); all 25 seeds positive.",
                "KC response is widespread: 311/320 matched KCs positive, 3 negative, 6 zero; top 1% contributes only ~4% of mean.",
                "No matched KC saturation: primary max rate 0.5734 <0.85; zero cells >=0.95.",
                "Matched KC positive, other KC mean -0.002287; full-4064 mean -0.000308.",
                "KC effect is sustained across 100-300, not a failed short-window transient; post-window residual small.",
            ],
            "SUPPORTED_HYPOTHESIS": [
                "The small response is consistent with threshold/subthreshold gating and post-spike reset limiting the effective rate increment; two-dose ratio ~0.509 is consistent with the DV ratio 0.5 but does not prove linearity.",
                "Non-injected KC negative shift is consistent with network interaction/redistribution; it is descriptive, not causal.",
            ],
            "UNKNOWN": [
                "Actual voltage trajectories, threshold distance, synaptic currents, and the relative contribution of reset vs network input are not recorded.",
                "Whether a stronger direct voltage dose would produce interpretable biology cannot be inferred by linear extrapolation.",
            ],
        },
        "decision": {
            "priority": "B",
            "A": "A new independent KC-drive dose experiment could test a prespecified dose range, but current data cannot justify stronger dose selection and no voltage traces exist.",
            "B": "Pause stronger stimulation; first validate LIF/network and biological-drive semantics, including subthreshold voltage and reset behavior.",
            "falsifiable_hypotheses": [
                "H1 threshold: most nonresponding KCs stay below threshold; responders sit near threshold. If voltage traces contradict, reject.",
                "H2 reset: post-spike reset dominates the rate ceiling; if traces/rate relation contradict, reject.",
                "H3 network: negative other-KC shift comes from shared network input; if isolated/voltage traces show none, reject.",
                "H4 aggregation: KC response is transient and hidden by window averaging; already contradicted by sustained bins.",
            ],
            "stop_conditions": [
                "No voltage/current traces or semantic mismatch between FlyBrain inject and intended biological drive.",
                "Proposed run exceeds 900 s / 2048 MiB or requires a new authorization absent.",
                "KC saturation or no monotonic dose response in a validated model.",
                "Any attempt to extrapolate a stronger dose without a new preregistration.",
            ],
            "next_experiment_candidate": "B-first: an independent model-semantics/voltage-trace study (or a validated LIF/current mapping) before any stronger KC stimulation. A dose experiment only after that, with prespecified doses and voltage monitoring.",
        },
        "answers": {
            "q1_formal_reproduced": bool(abs(kc_p["mean"] - s3r["kc_primary"]["mean"]) < 1e-9 and abs(mb_p["mean"] - s3r["mbon_primary"]["mean"]) < 1e-9),
            "q2_strongest_observation": "The KC response is significant but only 0.0229, far below the +0.10 gate, despite DV_inf=0.20; it is widespread, not a few-cell artifact, and not saturated.",
            "q3_likely_limiter": "SUPPORTED_HYPOTHESIS: threshold/reset/effective-voltage gating (and possibly network redistribution); exact cause UNKNOWN.",
            "q4_next_direction": "B: pause stronger stimulation and validate model/biological-drive semantics with voltage/current traces; A only after that.",
            "q5_no_extrapolation": "Do not infer a stronger dose from the approximate low/primary ratio.",
        },
        "gates": {"NEW_FULL_BRAIN_AUTHORIZATION": False, "TRAINING_AUTHORIZATION": False,
                  "REAL_SYNAPTIC_LEARNING_READY": "NO", "Level_A/B/C/Biological": "NO-GO",
                  "Robot": "NO-GO", "Maze": "NO-GO", "Dijkstra": "NO-GO", "DeltaW": "NO-GO"},
        "no_new_dynamics": True,
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({"status": rep["status"],
                      "kc_primary": kc_p["mean"], "mbon_primary": mb_p["mean"],
                      "low_primary_ratio": sem["low_to_primary_rate_ratio"],
                      "matched_pos_cells": rep["cell_level"]["matched_kc_primary"]["n_positive"],
                      "priority": rep["decision"]["priority"],
                      "hashes_ok": rep["archive"]["raw_hashes_ok"] and rep["archive"]["backup_hashes_ok"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
