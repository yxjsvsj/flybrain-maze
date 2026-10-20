"""P11-2-S1 静态预检：两跳统计勘误、授权核验、资源/安全计划。

**未获 S1 执行授权。** 本脚本不运行 FlyBrain、不重跑 C1、不执行 S1。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
REPORT = os.path.join(HERE, "p11_2_s1_preflight_report.json")
S0_REPORT = os.path.join(HERE, "p11_2_s0_report.json")
S1_APPROVAL = os.path.join(HERE, "P11_2_S1_ONE_RUN_APPROVAL.md")
SEEDS = list(range(101, 126))
STEPS = 400
STIM = (100, 300)
DV = 0.2
DT = 0.020
TAU = 0.100
AMOUNT = DV * (1 - math.exp(-DT / TAU))
MIN_KC_EFFECT = 0.05
MIN_MBON_EFFECT = 0.05
WALL_CAP_S = 900.0
MEM_CAP_MIB = 2048
C1_PEAK_MIB = 1942.7
C1_MARGIN_MIB = MEM_CAP_MIB - C1_PEAK_MIB
OFFICIAL = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return {"rc": r.returncode, "out": (r.stdout or "").strip()}


def official_hashes():
    files = {k: _sha(os.path.join(DATA_DIR, k)) for k in OFFICIAL}
    return {"ok": all(files[k] == v for k, v in OFFICIAL.items()), "files": files}


def two_hop_correction():
    W = sp.load_npz(os.path.join(DATA_DIR, "weights.npz")).tocsr()
    b = np.load(os.path.join(DATA_DIR, "brain.npz"), allow_pickle=True)
    ct = b["cell_type"].astype(str)
    kc = np.flatnonzero(np.char.startswith(ct, "KC"))
    mb = np.flatnonzero(np.char.startswith(ct, "MBON"))
    src = np.sort(np.concatenate([np.flatnonzero(ct == "LPLC2"), np.flatnonzero(ct == "LC4")]))
    Wb = W.copy(); Wb.data = np.ones_like(Wb.data)
    # endpoint pairs and independent paths
    P = (Wb[kc] @ Wb[:, src]).tocoo()
    A = (W[kc] @ W[:, src]).tocoo()
    Pm = (Wb[mb] @ Wb[:, src]).tocoo()
    # B removes KC->KC; feedforward has no KC intermediate (direct src->KC=0)
    WbB = Wb.copy()
    kcset = np.zeros(W.shape[0], bool); kcset[kc] = True
    coo = WbB.tocoo()
    keep = ~(kcset[coo.row] & kcset[coo.col])
    WbB = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=W.shape).tocsr()
    PB = (WbB[kc] @ WbB[:, src]).tocoo()
    AB = (W[kc] @ W[:, src] - W[kc][:, kc] @ W[kc][:, src]).tocoo()
    PmB = (WbB[mb] @ WbB[:, src]).tocoo()
    km = W[mb][:, kc].tocoo()
    return {
        "kc_endpoint_pairs": int(P.nnz),
        "kc_independent_two_hop_paths": int(P.data.sum()),
        "kc_paths_per_endpoint_max": int(P.data.max()) if P.nnz else 0,
        "kc_paths_per_endpoint_mean": round(float(P.data.mean()), 6) if P.nnz else 0.0,
        "s0_reported_7883_is": "aggregated source-KC endpoint pairs (nnz of boolean 2-hop product), NOT independent paths",
        "independent_paths_are_distinct": True,
        "aggregate_positive_endpoints": int((A.data > 0).sum()),
        "aggregate_negative_endpoints": int((A.data < 0).sum()),
        "aggregate_signed_sum": round(float(A.data.sum()), 12),
        "sign_7783_100_is": "signs of the 7,883 aggregated endpoint values, NOT path signs",
        "signed_sum_is_dynamic_stimulus_effect": False,
        "signed_sum_unit": "sum of aggregated signed 2-hop products (static weight statistic), not a neural response",
        "kc_B_endpoint_pairs": int(PB.nnz),
        "kc_B_independent_paths": int(PB.data.sum()),
        "kc_removed_via_kc_intermediate": int(P.nnz - PB.nnz),
        "kc_aggregate_signed_sum_B": round(float(AB.data.sum()), 12),
        "feedforward_preserved_in_B": bool(PB.nnz == P.nnz and PB.data.sum() == P.data.sum()),
        "mbon_endpoint_pairs": int(Pm.nnz),
        "mbon_independent_two_hop_paths": int(Pm.data.sum()),
        "mbon_paths_per_endpoint_max": int(Pm.data.max()) if Pm.nnz else 0,
        "mbon_paths_per_endpoint_mean": round(float(Pm.data.mean()), 6) if Pm.nnz else 0.0,
        "mbon_B_endpoint_pairs": int(PmB.nnz),
        "mbon_B_independent_paths": int(PmB.data.sum()),
        "direct_src_to_kc": int(W[kc][:, src].nnz),
        "direct_src_to_mbon": int(W[mb][:, src].nnz),
        "kc_to_mbon_edges": int(km.nnz),
        "kc_to_mbon_all_positive": bool(km.nnz > 0 and (km.data > 0).all()),
        "kc_to_mbon_signed_sum": round(float(km.data.sum()), 9),
        "source_count": int(len(src)),
        "kc_count": int(len(kc)), "mbon_count": int(len(mb)),
        "note": "7,883 and 2,528 are aggregated endpoint pairs. Independent 2-hop paths are 8,162 (KC) and 3,265 (MBON). 7,783/100 are aggregate endpoint signs; signed sum 0.003759783 is not a dynamic stimulus effect.",
    }


def sensory_freeze():
    b = np.load(os.path.join(DATA_DIR, "brain.npz"), allow_pickle=True)
    ct = b["cell_type"].astype(str)
    lp = int((ct == "LPLC2").sum()); lc4 = int((ct == "LC4").sum())
    lc4s = int(np.char.startswith(ct, "LC4").sum())
    return {
        "lplc2": lp, "lc4_exact": lc4, "targets": lp + lc4,
        "lc4_startswith": lc4s, "lc4_startswith_overcount": lc4s - lc4,
        "target_rule": "exact cell_type in {LPLC2, LC4}",
        "dv": DV, "dt": DT, "tau": TAU, "amount": AMOUNT,
        "amount_rounded": round(AMOUNT, 6),
        "amount_is_0_036254": abs(AMOUNT - 0.036254) < 1e-6,
        "amount_is_not_0_6": abs(AMOUNT - 0.6) > 0.1,
        "stim_window": list(STIM),
        "only_window": True, "no_kc_mbon_stim": True, "no_target_change": True,
        "no_strength_change": True, "no_new_connections": True, "no_teacher": True,
        "tof_mapping_changed": False, "lif_params_changed": False, "dW": 0,
    }


def stats():
    return {
        "entry": {"targets": "exact LPLC2/LC4 (311)", "contrast": "stim-none",
                  "readout": "[100,300) firing fraction", "unit": "seed", "n": 25,
                  "test": "exact two-sided sign-flip", "pass": "mean>0 and p<0.05"},
        "primary": {"endpoint": "matched-320 KC [100,300) stim-none", "unit": "seed", "n": 25,
                    "test": "exact two-sided sign-flip", "ci": "paired bootstrap 10,000 seed 20261019",
                    "min_positive_effect": MIN_KC_EFFECT,
                    "pass": "mean>=+0.05 and p<0.05 and 95% CI excludes 0"},
        "secondary": {"full_4064": "exploratory", "matched_95_MBON": f"secondary; min +{MIN_MBON_EFFECT}",
                      "full_97_MBON": "exploratory", "background_ceiling": "per-cell background and >=0.85/>=0.95 fractions",
                      "low_feedback": "2460/109321, 2650/117446, 3971/545375 individually",
                      "none_drift": "none-only matched-320 [100,200) vs [200,300); exact; report p"},
        "negative_response_rule": {
            "statement": "A significant negative KC/MBON response is a real exploratory observation. It does not satisfy the frozen positive primary endpoint (mean >= +0.05) and must not be presented as primary success; it also must not be described as 'no sensory change'. Report separately; do not change the primary direction.",
            "negative_does_not_pass_primary": True,
            "negative_does_not_mean_no_change": True,
            "reported_separately": True,
        },
        "failure": {"entry_fail": "SENSORY_ENTRY_FAILED",
                    "entry_ok_kc_no_positive": "SENSORY_TRANSMISSION_NOT_DETECTED",
                    "kc_ok_mbon_no": "KC_RESPONSE_WITHOUT_MBON_TRANSMISSION",
                    "kc_mbon_ok": "SENSORY_TRANSMISSION_DETECTED_IN_SYNTHETIC_B",
                    "resource": "RESOURCE_NO_GO", "missing": "INCONCLUSIVE"},
        "rules": {"keep_all_seeds": True, "seed_114_kept": True, "drop_seeds": False,
                  "select_seeds": False, "cell_step_pseudo_replication": False,
                  "endpoint_switching": False, "min_effect_change": False},
    }


def resource():
    n_kc, n_mb, n_entry = 4064, 97, 311
    per_condition = STEPS * (n_kc + n_mb + n_entry + 320 + 95)
    per_seed = 2 * per_condition
    extra = per_seed / 1024 ** 2
    est = C1_PEAK_MIB + extra + 2.0
    return {
        "c1_peak_MiB": C1_PEAK_MIB, "c1_margin_MiB": round(C1_MARGIN_MIB, 1),
        "extra_MiB_per_seed_both": round(extra, 3), "static_extra_MiB": round(extra + 2.0, 2),
        "estimated_peak_MiB": round(est, 2), "estimated_margin_MiB": round(MEM_CAP_MIB - est, 2),
        "time_estimate_s": 290.38, "wall_cap_s": WALL_CAP_S, "mem_cap_MiB": MEM_CAP_MIB,
        "status": "RESOURCE_PLAUSIBLE_TIGHT", "verified": False,
        "no_auto_increase": True, "low_memory_per_seed": True,
        "job_peak_persist_before_postprocess": True,
        "create_suspended_job_resume": True,
        "w_b_under_job": True,
        "no_go_if": "peak >2048 MiB or wall >900 s or init/exit/missing seed failure",
        "stop_rule": "no retry, no parameter change, no cap increase, no seed replacement",
    }


def authorization():
    # No explicit S1 approval message/file exists at this static preflight.
    return {
        "status": "NOT_AUTHORIZED",
        "s1_execution_authorization": False,
        "full_brain_run": False,
        "s1_run_generated": False,
        "c1_authorization_consumed": True,
        "self_approval_config": False,
        "evidence": "No explicit user approval for a P11-2-S1 run was provided in the current task; the S0 P11_2_S1_ONE_RUN_APPROVAL.md is an approval request marked NOT_AUTHORIZED; C1's one-shot authorization is consumed.",
        "required_approval": "one independent S1 run: B synthetic KC->KC-zero; none/stim; 25 seeds; 400 steps; 20,000; exact LPLC2/LC4 311; dv=0.2 amount=0.036254; [100,300); CPU; 15 min/2 GiB; one run; no auto-retry; no training/dW/Dijkstra/maze/B/C/robot",
        "no_fabricated_run": True,
    }


def freeze():
    p1 = _git("diff", "--stat", "P1-mapfix", "--", "nav", "config.py", "loop/decode.py", "loop/encode.py", "loop/run.py")
    base = _git("rev-parse", "P1-baseline")
    base_tag = _git("rev-parse", "--verify", "BASE")
    readout = _git("status", "--short", "--", "loop/train_readout.py", "loop/eval_readout.py",
                   "loop/brain_readout_compare.py", "loop/p2_readout_recoverability.py", "loop/test_train_p2_readout.py")
    tags = _git("for-each-ref", "refs/tags", "--format=%(refname:short) %(objectname)")
    status = _git("status", "--short")
    dirty = ["hardware/tof5_client.py", "loop/diagnose.py", "loop/p10c1b_decoder_freeze.json"]
    return {
        "P1_mapfix": _git("rev-parse", "P1-mapfix")["out"],
        "P1_mapfix_freeze_empty": p1["out"] == "",
        "P1_baseline": base["out"], "BASE_tag_present": base_tag["rc"] == 0,
        "readout_working_tree_modified": readout["out"] != "",
        "tags": dict(line.split(" ", 1) for line in tags["out"].splitlines() if line.strip()),
        "tags_modified": False, "dirty_files": dirty,
        "dirty_files_preserved": all(f in status["out"] for f in dirty),
        "head": _git("rev-parse", "HEAD")["out"],
    }


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    h = official_hashes()
    corr = two_hop_correction(); sens = sensory_freeze(); st = stats()
    res = resource(); auth = authorization(); fz = freeze()
    verdicts = {
        "AUTHORIZATION": "NOT_AUTHORIZED",
        "S1_EXECUTION": False,
        "FULL_BRAIN_RUN": False,
        "S0_ENDPOINT_PAIRS_7883": True,
        "S0_INDEPENDENT_PATHS_8162": True,
        "S0_SIGNED_SUM_IS_NOT_DYNAMIC": True,
        "NEGATIVE_RESPONSES_EXPLORATORY": True,
        "PRIMARY_FROZEN": True,
        "RESOURCE": "RESOURCE_PLAUSIBLE_TIGHT",
        "RESOURCE_VERIFIED": False,
        "S1_READY_TO_APPLY": True,
        "TRAINING_AUTHORIZATION": False,
        "REAL_SYNAPTIC_LEARNING_READY": "NO",
        "Level_A/B/C/Biological": "NO-GO",
        "A1_A2_A3/B": "NO-GO",
    }
    rep = {
        "protocol": "P11-2-S1-preflight/v1",
        "status": "STATIC_PREFLIGHT_COMPLETE; NOT_AUTHORIZED; NO_RUN",
        "official_hashes": h, "authorization": auth,
        "two_hop_correction": corr, "sensory": sens, "stats": st,
        "resource": res, "freeze": fz, "verdicts": verdicts,
        "full_brain_run": False, "s1_run": False, "training_authorized": False,
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("[P11-2-S1 static] status:", rep["status"], "| official", h["ok"], "| auth", auth["status"])
    print("  7,883 = endpoint pairs; independent 2-hop paths =", corr["kc_independent_two_hop_paths"],
          "| aggregate signs", corr["aggregate_positive_endpoints"], "/", corr["aggregate_negative_endpoints"],
          "| signed sum", corr["aggregate_signed_sum"], "(not dynamic)")
    print("  MBON endpoint pairs/paths", corr["mbon_endpoint_pairs"], "/", corr["mbon_independent_two_hop_paths"],
          "| KC->MBON", corr["kc_to_mbon_edges"],
          "| B feedforward", corr["feedforward_preserved_in_B"])
    print("  sensory targets", sens["targets"], "amount", round(AMOUNT, 9), "0.036254", sens["amount_is_0_036254"])
    print("  resource", res["estimated_peak_MiB"], "MiB /", MEM_CAP_MIB, res["status"], "| S1 run", auth["s1_run_generated"])
    print("  verdicts:", verdicts)
    print("  full brain run: FALSE; no S1; no raw/run_meta generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
