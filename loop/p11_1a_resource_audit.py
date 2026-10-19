"""P11-1a：完整 MaleCNS 模型有效性实验的**静态资源预检与范围冻结**。

只读官方 W/brain、归档与源码；**不初始化完整脑、不运行动力学、不训练、不更新 ΔW、不跑迷宫**。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp

from loop.p11_0_offline_diagnosis import load_archive
from loop.p100e_gate import w_readonly_check

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORT = os.path.join(HERE, "p11_1a_report.json")
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
W_PATH = os.path.join(DATA_DIR, "weights.npz")
B_PATH = os.path.join(DATA_DIR, "brain.npz")
FULL_N = 166700
FULL_NNZ = 25582938
SUBNET_N = 23218
SUBNET_NNZ = 3549643
SEEDS = list(range(101, 126))
STEPS = 400
NONE_ONLY_STEPS = len(SEEDS) * STEPS   # 10,000


def official_memory():
    W = sp.load_npz(W_PATH)
    csr = {"format": W.format, "shape": list(W.shape), "nnz": int(W.nnz),
           "dtype": str(W.dtype), "indices_dtype": str(W.indices.dtype),
           "indptr_dtype": str(W.indptr.dtype),
           "data_bytes": int(W.data.nbytes), "indices_bytes": int(W.indices.nbytes),
           "indptr_bytes": int(W.indptr.nbytes),
           "total_bytes": int(W.data.nbytes + W.indices.nbytes + W.indptr.nbytes)}
    Wc = W.tocsc()
    csc = {"total_bytes": int(Wc.data.nbytes + Wc.indices.nbytes + Wc.indptr.nbytes)}
    del W, Wc
    b = np.load(B_PATH, allow_pickle=True)
    meta = int(sum(b[k].nbytes for k in b.files))
    return {"csr": csr, "csc": csc, "metadata_bytes": meta,
            "weights_file_bytes": os.path.getsize(W_PATH), "brain_file_bytes": os.path.getsize(B_PATH)}


def memory_budget(mem, threads=16):
    state = {
        "v_bytes": FULL_N * 4,
        "noise_temp_float64": FULL_N * 8,
        "propagate_partial": threads * FULL_N * 4,
        "current": FULL_N * 4,
        "last_spike": 0,
    }
    state_total = sum(state.values())
    init_lower = mem["csr"]["total_bytes"] + mem["csc"]["total_bytes"] + mem["metadata_bytes"] + state_total
    with_file = init_lower + mem["weights_file_bytes"]
    overhead = 1.5
    upper = with_file * overhead
    return {
        "threads": threads, "state_bytes": state, "state_total_bytes": int(state_total),
        "init_peak_lower_MB": round(init_lower / 1e6, 2),
        "init_peak_with_file_MB": round(with_file / 1e6, 2),
        "upper_bound_with_1.5x_overhead_MB": round(upper / 1e6, 2),
        "within_2GB_upper_bound": bool(upper / 1e6 <= 2048),
        "measured": False,
        "resource_ready": False,
        "reason": ("static arrays fit well below 2 GB, but the load/tocsc/file-buffer/NumPy overhead peak "
                   "is not measured; no credible measured upper bound"),
        "recommendation_2GB": ("keep 2 GB for the benchmark; if measured peak exceeds 2 GB -> stop/INCONCLUSIVE; "
                               "do not auto-raise the cap"),
        "benchmark_required": True,
    }


def none_only_plan():
    per_step_ms = 1.3 * (FULL_NNZ / SUBNET_NNZ)   # static extrapolation, NOT a benchmark
    core_s = NONE_ONLY_STEPS * per_step_ms / 1000.0
    save_bytes = len(SEEDS) * STEPS * (4064 + 97) * 4
    return {
        "conditions": ["none"], "seeds": SEEDS, "steps_per_seed": STEPS,
        "planned_full_brain_steps": NONE_ONLY_STEPS,
        "plan": "one full-brain init + 25 resets; save KC/MBON arrays; NO eligibility computation",
        "static_core_seconds": round(core_s, 2),
        "static_core_minutes": round(core_s / 60.0, 2),
        "nnz_ratio": round(FULL_NNZ / SUBNET_NNZ, 3),
        "static_extrapolation_note": ("nnz ratio 7.207 is NOT a measured time/memory multiplier; "
                                      "cache/fired-count/fixed overhead differ"),
        "save_bytes_estimate": int(save_bytes),
        "eligibility": "not needed for the none-only baseline comparison",
        "benchmark_separate_authorization": True,
        "executed": False,
    }


def rng_audit():
    return {
        "implementation": "FlyBrain.reset(seed) sets self.rng=default_rng(seed); each step draws rng.random((N,B))",
        "full_N": FULL_N, "subnet_N": SUBNET_N,
        "same_seed_is_only_initial_rng": True,
        "draw_shape_differs": True,
        "common_bodyId_noise_sequence_same": False,
        "strict_noise_pairing": False,
        "comparison_type": "between-model comparison with a shared seed label; NOT a noise-paired comparison",
        "statistical_limitation": ("common-bodyId neurons occupy different positions and the RNG stream advances by N "
                                   "draws/step, so the per-neuron noise is not aligned"),
        "can_only_interpret_as": "overall truncation condition, not a specific external inhibition/excitation cause",
    }


def parameter_provenance():
    return {
        "phase2_candidates": {"gain": [1.5, 3.0], "tonic": [0.10, 0.14], "noise_hz": [0.6, 1.2]},
        "flybrain_code_defaults": {"gain": 3.0, "tonic": 0.14, "noise_hz": 1.2, "noise_amp": 0.22,
                                   "source": "flybrain/brain.py class attributes; author hand-set"},
        "fly64_code_values": {"gain": 1.5, "tonic": 0.18,
                              "source": "fly64/model.py mentioned in flybrain/brain.py docstring; code values"},
        "independent_physiological_calibration": "NONE found in the repository",
        "peer_reviewed_doi": None,
        "provenance_status": "INSUFFICIENT",
        "why": ("code defaults / author hand-set values and a code-to-code comparison are not independently "
                "published physiological calibrations; 'literature mentioned a similar model' != validated params"),
        "phase2_execution_ready": False,
        "no_selection_by_A0_result": True,
    }


def dz_errata():
    arc = load_archive()
    if not arc.get("ok"):
        return {"status": "RAW_DATA_UNAVAILABLE"}
    data = arc["data"]; W = slice(100, 300)
    D = np.array([data[(s, "stim")]["kc"][W].mean(0) - data[(s, "none")]["kc"][W].mean(0) for s in SEEDS])
    em = D.mean(0)
    cross_cell = float(em.mean() / em.std(ddof=1))
    pooled = float(D.mean() / D.std(ddof=1))
    sds = D.std(0, ddof=1)
    nz = D[:, sds > 0]
    per_cell_vals = np.mean(nz, 0) / np.std(nz, 0, ddof=1)
    return {
        "p11_0_reported_dz": round(cross_cell, 6),
        "p11_0_dz_definition": "mean(per-cell means) / sd(per-cell means) across 320 cells",
        "pooled_paired_dz": round(pooled, 6),
        "per_cell_paired_dz_median_nz": round(float(np.median(per_cell_vals)), 6) if len(per_cell_vals) else None,
        "n_cells_zero_variance": int((sds == 0).sum()),
        "mislabel": ("P11-0 dz is a cross-cell standardized mean, NOT a paired-by-seed Cohen's dz; "
                     "do not conflate the two"),
        "errata_recorded": True, "historical_unchanged": True,
    }


def gates():
    return {
        "Historical_P10C2_A0": "INCONCLUSIVE",
        "HIGH_BASELINE_PHENOTYPE": "SUPPORTED",
        "MODEL_ROOT_CAUSE": "NOT_ESTABLISHED",
        "Phase1_Design_Readiness": "GO",
        "Full_Brain_Resource_Readiness": "NO-GO",
        "Phase1_Execution_Readiness": "NO-GO",
        "Phase2_Parameter_Provenance": "INSUFFICIENT",
        "Phase2_Execution_Readiness": "NO-GO",
        "Level_A_Training": "NO-GO", "Level_B": "NO-GO", "Level_C": "NO-GO",
        "Biological": "NO-GO", "Training_Authorization": False,
        "why_phase1_execution_no_go": ["full-brain resource benchmark not measured",
                                       "peak memory vs 2 GB unverified",
                                       "written authorization for the 25-seed experiment not granted"],
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    mem = official_memory()
    budget = memory_budget(mem)
    plan = none_only_plan()
    rng = rng_audit()
    prov = parameter_provenance()
    dz = dz_errata()
    g = gates()
    arc = load_archive()
    w = w_readonly_check()
    report = {"archive": {"status": arc.get("status")},
              "hashes": {"ok": w.get("ok"),
                         "files": {k: v.get("sha256") for k, v in w.get("files", {}).items()}},
              "full_brain": mem, "memory_budget": budget,
              "none_only_plan": plan, "rng_audit": rng, "phase2_provenance": prov,
              "p11_0_dz_errata": dz, "gates": g, "training_authorized": False}
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else bool(o) if isinstance(o, np.bool_) else str(o))
    print("[P11-1a] full nnz:", mem["csr"]["nnz"], "| CSR/CSC MB:",
          round(mem["csr"]["total_bytes"] / 1e6, 2), "| metadata MB:", round(mem["metadata_bytes"] / 1e6, 2))
    print("  init lower/upper MB:", budget["init_peak_lower_MB"], budget["upper_bound_with_1.5x_overhead_MB"],
          "| resource_ready:", budget["resource_ready"])
    print("  none-only steps:", plan["planned_full_brain_steps"], "| static core min:", plan["static_core_minutes"])
    print("  rng strict pairing:", rng["strict_noise_pairing"], "| provenance:", prov["provenance_status"])
    print("  dz cross-cell vs pooled paired:", dz.get("p11_0_reported_dz"), dz.get("pooled_paired_dz"))
    print("  gates:", g)
    print("wrote p11_1a_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
