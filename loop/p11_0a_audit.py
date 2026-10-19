"""P11-0a：P11-0 统计勘误 + 无刺激启动瞬态 + P11-1 模型有效性预注册（只读/合成）。

不运行新动力学、不重跑 A0、不改参数/W/ΔW、不训练、不进入 B/C。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from loop.p11_0_offline_diagnosis import load_archive

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_0a_report.json")
SEEDS = list(range(101, 126))
W = slice(100, 300)
BINS = 20


def _bins(arr):
    return np.array([arr[b].mean() for b in np.array_split(np.arange(400), BINS)])


def stat_errata(data):
    D = np.array([data[(s, "stim")]["kc"][W].mean(0) - data[(s, "none")]["kc"][W].mean(0) for s in SEEDS])
    em = D.mean(0)
    net = float(em.sum())
    mean_per_cell = float(em.mean())
    return {
        "n_cells": int(D.shape[1]), "n_seeds": int(D.shape[0]),
        "effect_mean_per_cell": mean_per_cell,
        "net_sum_over_cells": net,
        "net_equals_mean_times_320": bool(abs(net - mean_per_cell * D.shape[1])
                                          <= 1e-6 * max(1.0, abs(net))),
        "net_identity_abs_residual": abs(net - mean_per_cell * D.shape[1]),
        "net_identity_rel_residual": abs(net - mean_per_cell * D.shape[1]) / abs(net),
        "net_identity_note": "mathematically exact; float32/float64 accumulation residual ~2.3e-10 (rel 3.6e-8)",
        "net_over_320": float(net / D.shape[1]),
        "doc_correction": ("P11-0 wrote net (~-0.0064) as if it were the per-cell mean; "
                           "the per-cell mean is ~-2.0e-5. net = per-cell mean x 320."),
        "sign_consistency": {
            "all_25_negative_strict": int(np.all(D < 0, axis=0).sum()),
            "all_25_nonpositive": int(np.all(D <= 0, axis=0).sum()),
            "all_25_positive_strict": int(np.all(D > 0, axis=0).sum()),
            "all_25_nonnegative": int(np.all(D >= 0, axis=0).sum()),
            "all_zero": int(np.all(D == 0, axis=0).sum()),
            "any_negative": int(np.any(D < 0, axis=0).sum()),
            "mixed_pos_neg": int((np.any(D > 0, axis=0) & np.any(D < 0, axis=0)).sum()),
            "ge20_negative_strict": int((np.sum(D < 0, axis=0) >= 20).sum()),
            "ge20_nonpositive": int((np.sum(D <= 0, axis=0) >= 20).sum()),
            "ge20_positive_strict": int((np.sum(D > 0, axis=0) >= 20).sum()),
        },
        "old_p11_0_errors": {
            "all_25_negative": 278,
            "why_wrong": "P11-0 used D<=0, counting 272 all-zero cells as 'negative'",
            "corrected_all_25_negative_strict": 0,
        },
        "cancellation": {
            "sum_positive": float(em[em > 0].sum()), "sum_negative": float(em[em < 0].sum()),
            "net": net,
            "weak_numerical_cancellation": bool(em[em > 0].sum() > 0 and em[em < 0].sum() < 0
                                                and abs(net) < 0.5 * abs(em[em > 0].sum())),
            "large_opposite_populations": bool((em > 0.01).sum() >= 16 and (em < -0.01).sum() >= 16),
            "large_positive_cells": int((em > 0.01).sum()), "large_negative_cells": int((em < -0.01).sum()),
            "distinction": ("weak numeric cancellation of tiny per-cell means != large opposite-direction "
                            "response populations"),
        },
        "thresholds_unchanged": True, "exploratory": True,
    }


def pooled_per_seed(data):
    pers = np.array([data[(s, "none")]["kc"][W].mean(0) for s in SEEDS])  # seeds x cells
    pmean = pers.mean(1)
    return {
        "pooled_mean": float(pers.mean()),
        "pooled_frac_ge_0.85": float((pers >= 0.85).mean()),
        "pooled_frac_eq_1": float((pers >= 1.0 - 1e-9).mean()),
        "per_seed_population_mean": {
            "values": [round(float(x), 6) for x in pmean],
            "mean": float(pmean.mean()), "median": float(np.median(pmean)),
            "sd": float(pmean.std(ddof=1)), "min": float(pmean.min()), "max": float(pmean.max()),
            "low_seed": int(SEEDS[int(np.argmin(pmean))]), "high_seed": int(SEEDS[int(np.argmax(pmean))]),
        },
        "per_seed_frac_ge_0.85": {"min": float(((pers >= 0.85).mean(1)).min()),
                                   "median": float(np.median((pers >= 0.85).mean(1))),
                                   "max": float(((pers >= 0.85).mean(1)).max())},
        "cross_seed_per_cell_median": float(np.median(pers.mean(0))),
        "note": ("single seed-cell firing fraction (e.g. ==1.0 fraction) and cross-seed averaged "
                 "per-cell median (0.96) are different statistics; do not conflate."),
        "low_high_states_observed": bool(pmean.min() < 0.5 and pmean.max() > 0.85),
        "no_seed_selection": True,
    }


def startup_transient(data):
    out = {}
    for label, key in (("KC", "kc"), ("MBON", "mb")):
        traj = np.array([_bins(data[(s, "none")][key]) for s in SEEDS])
        stim = np.array([_bins(data[(s, "stim")][key]) for s in SEEDS])
        pop = traj.mean(0)
        times = []
        for s in SEEDS:
            idx = np.flatnonzero(_bins(data[(s, "none")][key]) >= 0.85)
            times.append(int(idx[0] * (400 // BINS)) if len(idx) else None)
        reached = [t for t in times if t is not None]
        out[label] = {
            "none_bin_means": np.round(pop, 6).tolist(),
            "initial_0_20_20_40_40_60_60_80_80_100": np.round(pop[:5], 6).tolist(),
            "time_to_0.85_steps": {"median": float(np.median(reached)) if reached else None,
                                   "n_reached": len(reached), "n_not_reached": len(times) - len(reached),
                                   "before_100": int(sum(1 for t in reached if t < 100)),
                                   "values": times},
            "high_state_before_stimulus": bool(len(reached) >= 20 and float(np.median(reached)) < 100),
            "stim_minus_none_max_abs_after_100": float(np.max(np.abs(stim[:, 5:].mean(0) - traj[:, 5:].mean(0)))),
        }
    out["note"] = ("descriptive startup dynamics only; does not establish gain/tonic/noise/truncation "
                   "as the cause")
    return out


def p11_1_prereg():
    return {
        "status": "PREREGISTRATION ONLY; NOT EXECUTED",
        "phase1_truncation": {
            "question": "does the 23,218-neuron truncated subnet have the same no-stim KC/MBON baseline/ceiling as the full MaleCNS?",
            "design": "full MaleCNS vs truncated subnet; same normalized LIF params, seeds 101-125, 3 conditions x 400 steps",
            "primary": ["no-stim KC population baseline", "no-stim MBON population baseline",
                        "ceiling fraction (>=0.85)", "ceiling fraction (>=0.95)"],
            "matching": ("subnet 320 KC are a subset of full-brain 4064 KC; match by bodyId; report the 320 matched KCs "
                         "separately from the full KC population; MBON 95/97 matched by bodyId"),
            "comparability_limits": ["full brain includes other KCs/MBONs/DANs and external inputs",
                                     "subnet removes ~4.2M incoming edges / sumW~4238",
                                     "differences cannot be attributed to a single cause"],
            "cost_static_estimate": {"full_nnz": 25582938, "subnet_nnz": 3549643, "ratio": 7.207,
                                     "core_lif_min_estimate": 4.69,
                                     "note": "static extrapolation from subnet 1.3 ms/step; NOT a full-brain benchmark"},
            "design_ready": True, "execution_ready": False,
            "execution_blockers": ["full-brain LIF per-step benchmark not measured",
                                   "peak memory may exceed the 2 GB cap",
                                   "written authorization not granted"],
        },
        "phase2_sensitivity": {
            "question": "which computational assumption (gain/tonic/noise) drives the high baseline?",
            "factors": {"gain": [1.5, 3.0], "tonic": [0.10, 0.14], "noise_hz": [0.6, 1.2]},
            "value_sources": "fly64/ornata published LIF calibration values; pre-registered ranges, not fitted to A0",
            "design": "one-factor-at-a-time vs baseline; seeds 101-125; no-stim 400 steps; no stimulus tuning",
            "primary": ["no-stim KC/MBON baseline", "ceiling fraction"],
            "controls": ["current baseline", "matched seeds", "same W/subnet"],
            "selection_rule": "NO selection by KC response/navigation success; report all factors; parameter sensitivity != physiological calibration",
            "stopping": ["if all candidates remain >0.85 -> NO-GO for that factor",
                         "if cost exceeds cap -> NO-GO"],
            "design_ready": True, "execution_ready": False,
            "execution_blockers": ["independent parameter source review", "full-brain benchmark", "written authorization"],
        },
        "phase3_sensory": {"status": "NOT DESIGNED; only if baseline is acceptable",
                           "design_ready": False, "execution_ready": False},
        "prohibitions": ["no extra steering mechanism", "no external planner", "no new connections",
                         "no real dW update/training", "no B/C"],
        "overall_design_ready": True, "overall_execution_ready": False,
        "training_authorized": False,
    }


def verdicts(err, pool, start, pre):
    return {
        "P10C2_A0_Entrance": "PASS", "P10C2_A0_KC_Main_Endpoint": "NOT_DETECTED",
        "P10C2_A0_Ceiling": "FAIL", "P10C2_A0_Canonical": "INCONCLUSIVE",
        "Level_A_Training": "NO-GO", "Level_B": "NO-GO", "Level_C": "NO-GO",
        "Biological": "NO-GO", "Training_Authorization": False,
        "STATISTICAL_ERRATA_CORRECTED": bool(err["net_equals_mean_times_320"]),
        "SIGN_CONSISTENCY_CORRECTED": bool(err["sign_consistency"]["all_25_negative_strict"] == 0
                                           and err["sign_consistency"]["all_25_nonpositive"] == 278),
        "CANCELLATION_DISTINCTION": ("weak numeric only; no large opposite populations"
                                     if err["cancellation"]["weak_numerical_cancellation"]
                                     and not err["cancellation"]["large_opposite_populations"] else "review"),
        "HIGH_BASELINE_PHENOTYPE": "SUPPORTED" if pool["pooled_frac_ge_0.85"] >= 0.5 else "LIMITED",
        "CELL_LEVEL_HETEROGENEITY": "NOT_OBSERVED" if (err["cancellation"]["large_positive_cells"] < 16
                                                       and err["cancellation"]["large_negative_cells"] < 16) else "OBSERVED",
        "MODEL_ROOT_CAUSE": "NOT_ESTABLISHED",
        "STARTUP_HIGH_STATE_BEFORE_STIM": bool(start["KC"]["high_state_before_stimulus"]),
        "P11_1_Design_Ready": bool(pre["overall_design_ready"]),
        "P11_1_Execution_Ready": bool(pre["overall_execution_ready"]),
        "P11_1_Ready_To_Apply_For_New_Experiment": False,
        "P11_1_Blockers": ["full-brain LIF benchmark", "peak-memory check", "independent parameter-source review",
                           "written authorization"],
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    arc = load_archive()
    if not arc.get("ok"):
        json.dump({"status": arc.get("status")}, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("RAW_DATA_UNAVAILABLE:", arc)
        return 1
    data = arc["data"]
    err = stat_errata(data); pool = pooled_per_seed(data); start = startup_transient(data)
    pre = p11_1_prereg(); ver = verdicts(err, pool, start, pre)
    report = {"archive": {"status": arc["status"], "dir": arc["archive_dir"],
                          "integrity": arc["integrity"]},
              "stat_errata": err, "pooled_per_seed": pool, "startup_transient": start,
              "p11_1_prereg": pre, "verdicts": ver, "training_authorized": False}
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else bool(o) if isinstance(o, np.bool_) else str(o))
    print("[P11-0a] net:", err["net_sum_over_cells"], "| per-cell:", err["effect_mean_per_cell"],
          "| net=mean*320:", err["net_equals_mean_times_320"])
    print("  strict all<0:", err["sign_consistency"]["all_25_negative_strict"],
          "| all<=0:", err["sign_consistency"]["all_25_nonpositive"],
          "| allzero:", err["sign_consistency"]["all_zero"],
          "| ge20neg:", err["sign_consistency"]["ge20_negative_strict"])
    print("  pooled mean:", pool["pooled_mean"], "| >=.85:", pool["pooled_frac_ge_0.85"],
          "| per-seed range:", pool["per_seed_population_mean"]["min"], "-", pool["per_seed_population_mean"]["max"])
    print("  startup KC bins:", start["KC"]["initial_0_20_20_40_40_60_60_80_80_100"],
          "| t0.85:", start["KC"]["time_to_0.85_steps"]["median"], "| before100:", start["KC"]["time_to_0.85_steps"]["before_100"])
    print("  verdicts:", ver)
    print("wrote p11_0a_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
