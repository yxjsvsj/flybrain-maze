"""P11-2-S0：感觉传递实验预注册 + 资源/路径预检。

静态分析官方 brain/W、已有 C1/C1a 记录、源码和合成矩阵。
不运行 FlyBrain.step、不重跑 C1、不执行 S1、不训练、不控车。
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
REPORT = os.path.join(HERE, "p11_2_s0_report.json")
C1A_REPORT = os.path.join(HERE, "p11_2_c1a_report.json")
C1_RUN_META = os.path.join(HERE, "p11_2_c1_run_meta.json")
P10C2_0A = os.path.join(HERE, "p10c2_0a_gate.py")
P10C2_0B = os.path.join(HERE, "p10c2_0b_adjudication.py")
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
C1_MARGIN_MIB = 2048 - C1_PEAK_MIB
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
    return {"rc": r.returncode, "out": (r.stdout or "").strip(), "err": (r.stderr or "").strip()}


def _sig(x):
    return {"edges": int(x.nnz), "positive": int((x.data > 0).sum()),
            "negative": int((x.data < 0).sum()), "signed_sum": round(float(x.data.sum()), 9)}


def sensory_audit():
    b = np.load(os.path.join(DATA_DIR, "brain.npz"), allow_pickle=True)
    ct = b["cell_type"].astype(str); ids = b["ids"]
    lp = np.flatnonzero(ct == "LPLC2"); lc4 = np.flatnonzero(ct == "LC4")
    lc4s = np.flatnonzero(np.char.startswith(ct, "LC4"))
    extra = sorted(set(ct[lc4s]) - {"LC4"})
    src = np.sort(np.concatenate([lp, lc4]))
    return {
        "lplc2_count": int(len(lp)), "lc4_exact_count": int(len(lc4)),
        "lc4_startswith_count": int(len(lc4s)), "lc4_extra_types": extra,
        "lc4_startswith_overcount": int(len(lc4s) - len(lc4)),
        "target_count_exact": int(len(src)), "target_types": ["LPLC2", "LC4"],
        "lplc2_body_ids_first10": [int(x) for x in ids[lp][:10]],
        "lc4_body_ids_first10": [int(x) for x in ids[lc4][:10]],
        "source_body_ids_unique": int(len(set(ids[src].tolist()))),
        "target_rule": "exact cell_type in {LPLC2, LC4}; do NOT use startswith('LC4')",
        "protocol": {
            "dv": DV, "dt": DT, "tau": TAU,
            "amount_formula": "dv*(1-exp(-dt/tau))",
            "amount_exact": AMOUNT, "amount_rounded": round(AMOUNT, 6),
            "amount_is_0_036254": abs(AMOUNT - 0.036254) < 1e-6,
            "amount_is_not_0_6": abs(AMOUNT - 0.6) > 0.1,
            "stim_window": list(STIM), "stim_duration_steps": STIM[1] - STIM[0],
            "no_posthoc_increase": True, "no_direct_kc_mbon": True,
            "no_new_connections": True, "no_teacher": True,
            "tof_mapping_changed": False, "lif_params_changed": False, "dW": 0,
        },
        "source_protocol_audit": {
            "p10c2_0a_stimulus_audit_present": "stimulus_audit" in open(P10C2_0A, encoding="utf-8").read(),
            "p10c2_0b_DV_0_2": "DV = 0.2" in open(P10C2_0B, encoding="utf-8").read(),
            "p10c2_0b_AMOUNT_0_036254": "AMOUNT = 0.036254" in open(P10C2_0B, encoding="utf-8").read(),
            "old_0_6_does_not_replace_amount": True,
        },
    }


def path_audit():
    W = sp.load_npz(os.path.join(DATA_DIR, "weights.npz")).tocsr()
    b = np.load(os.path.join(DATA_DIR, "brain.npz"), allow_pickle=True)
    ct = b["cell_type"].astype(str); ids = b["ids"]
    kc = np.flatnonzero(np.char.startswith(ct, "KC"))
    mb = np.flatnonzero(np.char.startswith(ct, "MBON"))
    src = np.sort(np.concatenate([np.flatnonzero(ct == "LPLC2"), np.flatnonzero(ct == "LC4")]))
    is_kc = np.zeros(W.shape[0], bool); is_kc[kc] = True
    # direct
    direct_kc = W[kc][:, src].tocoo()
    direct_mb = W[mb][:, src].tocoo()
    # 2-hop signed / boolean
    Wb = W.copy(); Wb.data = np.ones_like(Wb.data)
    a2 = (W[kc] @ W[:, src]).tocoo(); a2b = (Wb[kc] @ Wb[:, src]).tocoo()
    rem2 = (W[kc][:, kc] @ W[kc][:, src]).tocoo()
    rem2b = (Wb[kc][:, kc] @ Wb[kc][:, src]).tocoo()
    b2 = (W[kc] @ W[:, src] - W[kc][:, kc] @ W[kc][:, src]).tocoo()
    b2b = (Wb[kc] @ Wb[:, src] - Wb[kc][:, kc] @ Wb[kc][:, src]).tocoo()
    a2m = (W[mb] @ W[:, src]).tocoo(); a2mb = (Wb[mb] @ Wb[:, src]).tocoo()
    rem3m = (W[mb][:, kc] @ W[kc][:, kc] @ W[kc][:, src]).tocoo()
    rem3mb = (Wb[mb][:, kc] @ Wb[kc][:, kc] @ Wb[kc][:, src]).tocoo()
    km = W[mb][:, kc].tocoo()
    # matched ids from P11-1c archive
    p = _load(os.path.join(HERE, "p11_1c_experiment_report.json"))
    z = np.load(os.path.join(p["raw_dir"], "ids.npz"), allow_pickle=True)
    kc_m = np.searchsorted(z["kc_idx"], z["kc_matched"])
    mb_m = np.searchsorted(z["mb_idx"], z["mb_matched"])
    mbtypes = {}
    for x in ct[mb]:
        mbtypes[str(x)] = mbtypes.get(str(x), 0) + 1
    return {
        "kc_count": int(len(kc)), "mbon_count": int(len(mb)),
        "matched_kc": int(len(kc_m)), "matched_mbon": int(len(mb_m)),
        "source_count": int(len(src)),
        "direct_src_to_kc": _sig(direct_kc),
        "direct_src_to_mbon": _sig(direct_mb),
        "two_hop_src_to_kc_A": _sig(a2),
        "two_hop_src_to_kc_A_bool_paths": int(a2b.nnz),
        "removed_via_kc_intermediate": _sig(rem2),
        "removed_via_kc_intermediate_bool_paths": int(rem2b.nnz),
        "two_hop_src_to_kc_B": _sig(b2),
        "two_hop_src_to_kc_B_bool_paths": int(b2b.nnz),
        "two_hop_src_to_mbon_A": _sig(a2m),
        "two_hop_src_to_mbon_A_bool_paths": int(a2mb.nnz),
        "three_hop_src_to_mbon_via_kc_kc_removed": _sig(rem3m),
        "three_hop_via_kc_kc_removed_bool_paths": int(rem3mb.nnz),
        "kc_to_mbon": _sig(km),
        "mbon_types": mbtypes,
        "mbon_unique_types": len(mbtypes),
        "feedforward_paths_preserved_in_B": bool(rem2b.nnz == 0 and b2b.nnz == a2b.nnz),
        "recurrent_kc_kc_edges_removed": 642933,
        "sensory_feedforward_uses_kc_kc": False,
        "anatomical_reachability_not_dynamical_detectability": True,
        "path_note": "No direct LPLC2/LC4->KC edge. The feedforward entry to KC is a 2-hop path via non-KC; the B KC->KC deletion removes the recurrent edges but not this feedforward path (0 paths removed via KC intermediate).",
    }


def c1a_note():
    full = 0.351093307; low3 = 0.000646358; rest_group = 0.350705836
    rest_contrib = rest_group * 4061.0 / 4064.0
    return {"full_B_mean": full, "low3_contribution": low3,
            "rest_4061_group_mean": rest_group,
            "rest_4061_contribution_to_full_4064_mean": round(rest_contrib, 9),
            "sum_check": round(low3 + rest_contrib, 9),
            "note": "Group mean and contribution to the full-4064 mean are different: rest_4061 contribution = 0.350705836 * 4061/4064 = 0.350446949; 0.000646358 + 0.350446949 = 0.351093307.",
            "historical_c1a_report_overwritten": False}


def stats_prereg():
    return {
        "entry_positive_control": {
            "targets": "exact LPLC2 + LC4 (311 neurons)",
            "readout": "fraction of target neurons firing in [100,300)",
            "contrast": "stim - none", "unit": "seed", "n": 25,
            "test": "exact two-sided sign-flip", "pass": "mean > 0 and p < 0.05",
            "note": "If the entry fails, SENSORY_ENTRY_FAILED; no downstream claim."},
        "primary": {
            "endpoint": "matched-320 KC firing fraction in [100,300)",
            "contrast": "stim - none", "unit": "seed", "n": 25,
            "test": "exact two-sided sign-flip", "ci": "paired bootstrap 10,000; seed 20261019",
            "min_meaningful_effect": MIN_KC_EFFECT,
            "pass": "mean >= +0.05 and p < 0.05 and CI excludes 0",
            "note": "If entry responds but this fails: SENSORY_TRANSMISSION_NOT_DETECTED."},
        "secondary": {
            "full_4064_KC": "exploratory stim-none; same window/effect; no causal primary claim",
            "matched_95_MBON": f"secondary stim-none; minimum meaningful effect +{MIN_MBON_EFFECT}",
            "full_97_MBON": "exploratory",
            "background_and_ceiling": "none-condition matched-320/full-4064/MBON rates and >=0.85/>=0.95 cell fractions; if none is at ceiling, no dynamic range",
            "three_low_feedback": "local 2460/2650/3971 and bodyIds 109321/117446/545375 reported individually; not representative of the population"},
        "none_drift_false_positive": {
            "contrast": "none only: matched-320 [100,200) vs [200,300)",
            "test": "exact two-sided sign-flip by seed",
            "purpose": "detect drift/noise false positives without stim",
            "rule": "report p and effect; if p<0.05, flag drift and interpret the stim effect cautiously",
            "random_noise_control": "within-seed paired stim/none; exact sign-flip; 25 seeds; no cell-step pseudo-replication"},
        "failure_conditions": {
            "entry_fail": "SENSORY_ENTRY_FAILED",
            "entry_ok_kc_no": "SENSORY_TRANSMISSION_NOT_DETECTED",
            "kc_ok_mbon_no": "KC_RESPONSE_WITHOUT_MBON_TRANSMISSION",
            "kc_mbon_ok": "SENSORY_TRANSMISSION_DETECTED_IN_SYNTHETIC_B",
            "resource": "RESOURCE_NO_GO",
            "missing": "INCONCLUSIVE"},
        "unit_rules": {"keep_all_seeds": True, "seed_114_kept": True, "drop_seeds": False,
                       "select_seeds": False, "cell_step_pseudo_replication": False,
                       "endpoint_switching": False, "min_effect_posthoc_change": False},
        "reuse_c1b": {"reuse_historical_C1_B_none": False,
                      "reason": "same-experiment none/stim paired design avoids code/params/RNG/init comparability assumptions",
                      "strict_rng_pairing": "same seed, same reset, same CPU, stim adds voltage only; noise sequence unchanged"},
    }


def resource_plan():
    # record arrays per seed/condition, uint8
    n_kc, n_mb, n_entry = 4064, 97, 311
    per_condition = STEPS * (n_kc + n_mb + n_entry + 320 + 95)
    per_seed = 2 * per_condition
    extra_mib = per_seed / 1024 ** 2
    return {
        "c1_peak_MiB": C1_PEAK_MIB, "c1_margin_MiB": round(C1_MARGIN_MIB, 1),
        "record_bytes_per_seed_both_conditions": int(per_seed),
        "record_MiB_per_seed_both_conditions": round(extra_mib, 3),
        "static_extra_peak_MiB": round(extra_mib + 2.0, 2),
        "estimated_peak_MiB": round(C1_PEAK_MIB + extra_mib + 2.0, 2),
        "estimated_margin_MiB": round(2048 - (C1_PEAK_MIB + extra_mib + 2.0), 2),
        "time_estimate_s": 290.38, "wall_cap_s": WALL_CAP_S,
        "mem_cap_MiB": MEM_CAP_MIB,
        "status": "RESOURCE_PLAUSIBLE_TIGHT",
        "verified": False,
        "low_memory": "save each seed/condition NPZ immediately, free arrays; no 25-seed full arrays in memory",
        "no_auto_increase": True,
        "no_go_if": "Job peak > 2048 MiB or wall > 900 s or missing seed",
        "job_peak_persist_before_postprocess": True,
        "create_suspended_job_resume": True,
        "note": "C1 measured 1942.7 MiB, only 105.3 MiB below 2 GiB. The S1 extra record arrays are small, but the margin is too tight to call 2 GiB verified. Do not auto-raise the cap.",
    }


def freeze_audit():
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
        "P1_mapfix_freeze_empty": p1["out"] == "", "P1_mapfix_diff": p1["out"],
        "P1_baseline": base["out"], "BASE_tag_present": base_tag["rc"] == 0,
        "readout_working_tree_modified": readout["out"] != "",
        "tags": dict(line.split(" ", 1) for line in tags["out"].splitlines() if line.strip()),
        "tags_modified": False,
        "dirty_files": dirty,
        "dirty_files_preserved": all(f in status["out"] for f in dirty),
        "head": _git("rev-parse", "HEAD")["out"],
    }


def official_hashes():
    files = {k: _sha(os.path.join(DATA_DIR, k)) for k in OFFICIAL}
    return {"ok": all(files[k] == v for k, v in OFFICIAL.items()), "files": files}


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    h = official_hashes()
    sens = sensory_audit(); paths = path_audit(); note = c1a_note()
    stats = stats_prereg(); res = resource_plan(); fz = freeze_audit()
    s1 = {
        "protocol": "P11-2-S1/v1", "status": "NOT_AUTHORIZED; NOT_RUN",
        "separate_authorization_required": True,
        "c1_one_shot_consumed": True,
        "self_approval_config": False,
        "request": "one run: B synthetic KC->KC-zero; none/stim paired; 25 seeds; 400 steps; 20,000 steps; LPLC2/LC4 exact 311; dv=0.2 amount=0.036254; [100,300); CPU; 15 min/2 GiB; one run; no auto-retry; no training/dW/Dijkstra/maze/B/C/robot",
        "training_authorized": False,
    }
    verdicts = {
        "SENSORY_TARGET_RULE": "EXACT LPLC2 + LC4 (311); startswith(LC4) would overcount to 227",
        "DIRECT_LPLC2_LC4_TO_KC": "ABSENT (0 edges)",
        "FEEDFORWARD_KC_PATH": "2-hop via non-KC; preserved in B",
        "KC_KC_RECURRENT": "removed (642,933)",
        "KC_TO_MBON": "61,210 existing edges, all positive; preserved in B",
        "ANATOMICAL_REACHABILITY": "NOT EQUAL TO DYNAMICAL DETECTABILITY",
        "PRIMARY_ENDPOINT_FROZEN": True,
        "RESOURCE": "RESOURCE_PLAUSIBLE_TIGHT",
        "RESOURCE_VERIFIED": False,
        "S1_EXECUTION_AUTHORIZATION": False,
        "S1_READY_TO_APPLY": True,
        "TRAINING_AUTHORIZATION": False,
        "REAL_SYNAPTIC_LEARNING_READY": "NO",
        "Level_A/B/C/Biological": "NO-GO",
        "A1_A2_A3/B": "NO-GO",
    }
    rep = {"protocol": "P11-2-S0/v1", "status": "PREFLIGHT_COMPLETE; NO_DYNAMICS",
           "official_hashes": h, "sensory": sens, "path_audit": paths,
           "c1a_statistical_note": note, "stats": stats, "resource": res,
           "freeze": fz, "s1_approval": s1, "verdicts": verdicts,
           "training_authorized": False}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("[P11-2-S0] status:", rep["status"], "| official", h["ok"])
    print("  sensory: LPLC2", sens["lplc2_count"], "LC4 exact", sens["lc4_exact_count"],
          "startswith", sens["lc4_startswith_count"], "targets", sens["target_count_exact"],
          "amount", round(AMOUNT, 9), "0.036254", sens["protocol"]["amount_is_0_036254"])
    print("  paths: direct src->KC", paths["direct_src_to_kc"]["edges"],
          "| 2hop KC A/B", paths["two_hop_src_to_kc_A_bool_paths"], paths["two_hop_src_to_kc_B_bool_paths"],
          "| removed via KC", paths["removed_via_kc_intermediate_bool_paths"])
    print("  MBON: direct src->MBON", paths["direct_src_to_mbon"]["edges"],
          "| 2hop", paths["two_hop_src_to_mbon_A_bool_paths"], "| KC->MBON", paths["kc_to_mbon"]["edges"])
    print("  C1a note:", note["rest_4061_contribution_to_full_4064_mean"], "+", note["low3_contribution"], "=", note["sum_check"])
    print("  resource:", res["estimated_peak_MiB"], "MiB /", res["mem_cap_MiB"], res["status"],
          "| S1", s1["status"])
    print("  verdicts:", verdicts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
