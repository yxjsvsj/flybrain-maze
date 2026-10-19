"""P11-2-0a：单神经元严格数学界限 + 历史 P11-1c KC→KC 递归输入只读重建。

**不运行 FlyBrain.step、不重跑 P11-1c、不做单神经元时间演化、不执行 A1/A2/A3/B。**
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
EXP_REPORT = os.path.join(HERE, "p11_1c_experiment_report.json")
FULL_MANIFEST = os.path.join(HERE, "p11_1c_full_manifest.json")
REPORT = os.path.join(HERE, "p11_2_0a_report.json")
DT = 0.020
TAU = 0.100
TONIC = 0.14
GAIN = 3.0
NOISE_P = 1.2 * DT
NOISE_AMP = 0.22
THRESHOLD = 1.0
SEEDS = list(range(101, 126))
STIM = (100, 300)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_archive():
    rep = json.load(open(EXP_REPORT, encoding="utf-8"))
    raw = rep["raw_dir"]
    man = json.load(open(FULL_MANIFEST, encoding="utf-8"))
    if not raw or not os.path.isdir(raw):
        return {"ok": False, "status": "OFFLINE_RECONSTRUCTION_UNVERIFIED", "reason": "raw dir missing"}
    files = sorted(os.listdir(raw))
    missing = [n for n in man if not os.path.exists(os.path.join(raw, n))]
    mismatch = [n for n in man if os.path.exists(os.path.join(raw, n)) and _sha(os.path.join(raw, n)) != man[n]["sha256"]]
    extra = [f for f in files if f not in man]
    return {"ok": not (missing or mismatch or extra), "status": "VERIFIED" if not (missing or mismatch or extra) else "OFFLINE_RECONSTRUCTION_UNVERIFIED",
            "raw_dir": raw, "file_count": len(files), "missing": missing, "mismatch": mismatch, "extra": extra}


def math_bounds():
    decay = math.exp(-DT / TAU)
    tonic_fp = TONIC / (1 - decay)
    mean_noise = NOISE_P * NOISE_AMP
    tonic_noise_fp = (TONIC + mean_noise) / (1 - decay)
    # max voltage from reset with a noise event every step
    v = 0.0; traj = []
    for _ in range(4):
        v = decay * v + TONIC + NOISE_AMP
        traj.append(v)
    earliest_refire = next((i + 1 for i, x in enumerate(traj) if x >= THRESHOLD), None)
    no_noise_raw = (THRESHOLD - TONIC) / GAIN
    max_noise_raw = (THRESHOLD - TONIC - NOISE_AMP) / GAIN
    return {
        "decay": round(decay, 12), "tonic_only_fixed_point": round(tonic_fp, 9),
        "tonic_alone_can_fire": bool(tonic_fp >= THRESHOLD),
        "tonic_noise_mean_fixed_point": round(tonic_noise_fp, 9),
        "tonic_noise_mean_can_fire": bool(tonic_noise_fp >= THRESHOLD),
        "max_voltage_reset_steps_1_4": [round(x, 9) for x in traj],
        "earliest_refire_step_after_reset": earliest_refire,
        "long_run_firing_fraction_upper_bound": round(1.0 / earliest_refire, 6) if earliest_refire else None,
        "raw_threshold_no_noise": round(no_noise_raw, 7),
        "raw_threshold_with_max_noise": round(max_noise_raw, 7),
        "strict_conclusion": ("with no synaptic input, tonic+noise cannot fire every step; earliest re-fire after reset is "
                              "step 4, so the long-run firing fraction is bounded by 0.25"),
        "caveat": "if the previous step did not fire, residual voltage exists; the reset=0 thresholds do not apply",
    }


def _st(x):
    return {"mean": round(float(x.mean()), 6), "median": round(float(np.median(x)), 6),
            "p5": round(float(np.percentile(x, 5)), 6), "p95": round(float(np.percentile(x, 95)), 6),
            "min": round(float(x.min()), 6), "max": round(float(x.max()), 6)}


def _direction_check(W, Wkk, kc_idx):
    nz = Wkk.nonzero(); i, j = int(nz[0][0]), int(nz[1][0])
    return {"local_post": i, "local_pre": j,
            "Wkk_value": float(Wkk[i, j]),
            "W_post_pre_value": float(W[kc_idx[i], kc_idx[j]]),
            "equal": bool(float(Wkk[i, j]) == float(W[kc_idx[i], kc_idx[j]]))}


def reconstruct(arc):
    W = sp.load_npz(os.path.join(DATA_DIR, "weights.npz")).tocsr()
    b = np.load(os.path.join(DATA_DIR, "brain.npz"), allow_pickle=True)
    ct = b["cell_type"].astype(str); ids = b["ids"]
    z_ids = np.load(os.path.join(arc["raw_dir"], "ids.npz"), allow_pickle=True)
    kc_idx = z_ids["kc_idx"]; matched = z_ids["kc_matched"]
    kc_m = np.searchsorted(kc_idx, matched)
    Wkk = W[kc_idx][:, kc_idx].tocsr()
    Wpos = Wkk.maximum(0); Wneg = Wkk.minimum(0)
    checks = {
        "brain_n": int(len(ids)), "W_shape": list(W.shape),
        "kc_idx_count": int(len(kc_idx)),
        "kc_idx_unique": bool(len(set(kc_idx.tolist())) == len(kc_idx)),
        "kc_idx_sorted": bool(np.all(np.diff(kc_idx) > 0)),
        "matched_count": int(len(matched)),
        "matched_subset": bool(np.array_equal(kc_idx[kc_m], matched)),
        "kc_cell_type_all_start_KC": bool(all(str(x).startswith("KC") for x in ct[kc_idx])),
        "W_KC_KC_direction_check": _direction_check(W, Wkk, kc_idx),
        "W_dtype": str(W.dtype), "W_KC_KC_nnz": int(Wkk.nnz),
        "W_KC_KC_pos_nnz": int(Wpos.nnz), "W_KC_KC_neg_nnz": int(Wneg.nnz),
    }
    rows = {"all": {"pos": [], "neg": [], "net": []},
            "matched": {"pos": [], "neg": [], "net": []}}
    per_seed = []
    consec = {"n": 0, "ge086": 0, "ge064": 0, "required_nonKC": [],
              "fired_not_again": 0, "fired_not_again_input_ge086": 0,
              "fired_not_again_input": [],
              "prev_all": 0, "prev_all_ge086": 0, "prev_all_ge064": 0}
    per_cell = []; main_zeros = []
    for s in SEEDS:
        z = np.load(os.path.join(arc["raw_dir"], f"{s}.npz"), allow_pickle=True)
        kc = z["kc"].astype(np.float32)          # 400 x 4064
        Spos = Wpos @ kc.T                       # 4064 x 400, input from kc[t]
        Sneg = Wneg @ kc.T
        Ipos = np.zeros_like(Spos); Ineg = np.zeros_like(Sneg)
        Ipos[:, 1:] = GAIN * Spos[:, :-1]
        Ineg[:, 1:] = GAIN * Sneg[:, :-1]
        Inet = Ipos + Ineg
        p = Ipos[:, STIM[0]:STIM[1]]; n = Ineg[:, STIM[0]:STIM[1]]; w = Inet[:, STIM[0]:STIM[1]]
        rows["all"]["pos"].append(p.ravel()); rows["all"]["neg"].append(n.ravel()); rows["all"]["net"].append(w.ravel())
        rows["matched"]["pos"].append(p[kc_m, :].ravel()); rows["matched"]["neg"].append(n[kc_m, :].ravel())
        rows["matched"]["net"].append(w[kc_m, :].ravel())
        per_seed.append({"seed": s, "pos_mean": round(float(p.mean()), 6), "neg_mean": round(float(n.mean()), 6),
                         "net_mean": round(float(w.mean()), 6), "net_median": round(float(np.median(w)), 6),
                         "net_p5": round(float(np.percentile(w, 5)), 6), "net_p95": round(float(np.percentile(w, 95)), 6),
                         "net_min": round(float(w.min()), 6), "net_max": round(float(w.max()), 6)})
        per_cell.append(w.mean(axis=1))
        zz = np.argwhere(kc[STIM[0]:STIM[1]] == 0)
        main_zeros.append({"seed": s, "count": int(len(zz)),
                           "positions": [[int(a) + STIM[0], int(bb)] for a, bb in zz[:10]]})
        prev = kc[STIM[0] - 1:STIM[1] - 1, :].astype(bool)
        cur = kc[STIM[0]:STIM[1], :].astype(bool)
        Iwin = w.T                             # 200 x 4064
        ev = prev & cur
        consec["n"] += int(ev.sum())
        consec["ge086"] += int((Iwin[ev] >= 0.86).sum())
        consec["ge064"] += int((Iwin[ev] >= 0.64).sum())
        consec["required_nonKC"].extend(np.maximum(0.0, 0.64 - Iwin[ev]).tolist())
        fn = prev & (~cur)
        consec["fired_not_again"] += int(fn.sum())
        consec["fired_not_again_input_ge086"] += int((Iwin[fn] >= 0.86).sum())
        consec["fired_not_again_input"].extend(Iwin[fn].tolist())
        consec["prev_all"] += int(prev.sum())
        consec["prev_all_ge086"] += int((Iwin[prev] >= 0.86).sum())
        consec["prev_all_ge064"] += int((Iwin[prev] >= 0.64).sum())
    def st3(d):
        return {k: _st(np.concatenate(v)) for k, v in d.items()}
    all_st = st3(rows["all"]); m_st = st3(rows["matched"])
    per_cell = np.mean(np.array(per_cell), axis=0)  # 4064 average over seeds
    req = np.array(consec["required_nonKC"])
    req_st = {**_st(req), "frac_gt_0": round(float((req > 0).mean()), 6) if len(req) else None}
    net_means = [x["net_mean"] for x in per_seed]
    return {
        "data_checks": checks,
        "W_KC_KC": {"shape": list(Wkk.shape), "nnz": int(Wkk.nnz), "dtype": str(Wkk.dtype),
                    "signed_preserved": True, "nnz_pos": int(Wpos.nnz), "nnz_neg": int(Wneg.nnz)},
        "all_4064_KC": all_st, "matched_320_KC": m_st,
        "per_seed": per_seed,
        "per_seed_net_mean_range": [min(net_means), max(net_means)],
        "per_cell_4064": {**_st(per_cell), "n_cells": int(len(per_cell))},
        "main_window_zero_spikes": {
            "seeds_with_zeros": [x for x in main_zeros if x["count"] > 0],
            "seeds_without_zeros": [x["seed"] for x in main_zeros if x["count"] == 0],
            "identical_except_listed": bool(sum(x["count"] for x in main_zeros) == 3)},
        "consecutive_firing": {
            "n_events": consec["n"],
            "n_ge_0.86": consec["ge086"], "n_ge_0.64": consec["ge064"],
            "n_lt_0.86": consec["n"] - consec["ge086"], "n_lt_0.64": consec["n"] - consec["ge064"],
            "frac_input_ge_0.86": round(consec["ge086"] / consec["n"], 6) if consec["n"] else None,
            "frac_input_ge_0.64": round(consec["ge064"] / consec["n"], 6) if consec["n"] else None,
            "prev_fired_all": consec["prev_all"],
            "prev_fired_frac_ge_0.86": round(consec["prev_all_ge086"] / consec["prev_all"], 6) if consec["prev_all"] else None,
            "prev_fired_frac_ge_0.64": round(consec["prev_all_ge064"] / consec["prev_all"], 6) if consec["prev_all"] else None,
            "fired_not_again": consec["fired_not_again"],
            "fired_not_again_input_ge_0.86": consec["fired_not_again_input_ge086"],
            "fired_not_again_input": _st(np.array(consec["fired_not_again_input"])),
            "required_nonKC_at_max_noise": req_st},
        "caveats": ["only KC->KC input reconstructed; other 162,636 neurons' spikes not saved",
                    "signed W preserved; positive/negative/net all reported",
                    "positive-feedback capacity != causal root cause",
                    "first step t=0 has no previous spikes and is excluded",
                    "float32 W accumulation; small numerical rounding possible"],
    }


def verdicts(m, r, arc):
    c = r["consecutive_firing"]
    return {
        "OFFLINE_RECONSTRUCTION": arc["status"],
        "TONIC_ONLY_EXCLUDED": bool(not m["tonic_alone_can_fire"]),
        "TONIC_NOISE_EVERY_STEP_EXCLUDED": bool(not m["tonic_noise_mean_can_fire"]),
        "EARLIEST_REFIRE_STEP": m["earliest_refire_step_after_reset"],
        "FIRING_FRACTION_UPPER_BOUND": m["long_run_firing_fraction_upper_bound"],
        "KC_KC_NET_STATIC_CAPACITY_SUFFICIENT": bool(r["all_4064_KC"]["net"]["p95"] >= 0.86),
        "KC_KC_NET_MEAN": r["all_4064_KC"]["net"]["mean"],
        "KC_KC_POS_MEAN": r["all_4064_KC"]["pos"]["mean"],
        "KC_KC_NEG_MEAN": r["all_4064_KC"]["neg"]["mean"],
        "CONSECUTIVE_INPUT_GE_086_FRAC": c["frac_input_ge_0.86"],
        "CONSECUTIVE_INPUT_GE_064_FRAC": c["frac_input_ge_0.64"],
        "PREV_FIRED_INPUT_GE_086_FRAC": c["prev_fired_frac_ge_0.86"],
        "PREV_FIRED_INPUT_GE_064_FRAC": c["prev_fired_frac_ge_0.64"],
        "CONSECUTIVE_REQUIRING_NON_KC": c["required_nonKC_at_max_noise"]["frac_gt_0"],
        "KC_KC_CAUSAL_ROOT_CAUSE": "NOT_ESTABLISHED (capacity only)",
        "SOURCE_PROVENANCE": "PARTIAL",
        "A1_STILL_OPTIONAL": "yes (sparse-probability estimate); A1 no longer needed to exclude tonic+noise every-step",
        "NETWORK_CAUSAL_EXPERIMENT_WORTH_APPLYING": "YES (KC->KC zeroed vs intact; separate authorization)",
        "A2_A3_B": "NO-GO",
        "REAL_SYNAPTIC_LEARNING_READY": "NO",
        "Training_Authorization": False, "Level_A_Training": "NO-GO",
        "Level_B": "NO-GO", "Level_C": "NO-GO", "Biological": "NO-GO",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    from loop.p100e_gate import w_readonly_check
    h = w_readonly_check()
    arc = load_archive()
    if not arc.get("ok"):
        json.dump({"archive": arc, "official_hashes": h, "status": "OFFLINE_RECONSTRUCTION_UNVERIFIED"},
                  open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("OFFLINE_RECONSTRUCTION_UNVERIFIED:", arc); return 1
    m = math_bounds(); r = reconstruct(arc); v = verdicts(m, r, arc)
    report = {"protocol": "P11-2-0a/v1", "archive": arc,
              "official_hashes": {"ok": h.get("ok"), "files": {k: x.get("sha256") for k, x in h.get("files", {}).items()}},
              "math_bounds": m, "reconstruction": r, "verdicts": v,
              "training_authorized": False}
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else bool(o) if isinstance(o, np.bool_) else str(o))
    print("[P11-2-0a] archive:", arc["status"], "| files:", arc["file_count"])
    print("  math: tonic fp", m["tonic_only_fixed_point"], "| tonic+noise fp", m["tonic_noise_mean_fixed_point"],
          "| v1..v4", m["max_voltage_reset_steps_1_4"], "| earliest refire", m["earliest_refire_step_after_reset"],
          "| fraction bound", m["long_run_firing_fraction_upper_bound"])
    print("  KC->KC net mean/median/p95:", r["all_4064_KC"]["net"]["mean"], r["all_4064_KC"]["net"]["median"], r["all_4064_KC"]["net"]["p95"],
          "| pos mean:", r["all_4064_KC"]["pos"]["mean"], "| neg mean:", r["all_4064_KC"]["neg"]["mean"])
    print("  matched 320 net mean/p95:", r["matched_320_KC"]["net"]["mean"], r["matched_320_KC"]["net"]["p95"])
    print("  consecutive events:", r["consecutive_firing"]["n_events"],
          "| >=0.86:", r["consecutive_firing"]["frac_input_ge_0.86"],
          "| >=0.64:", r["consecutive_firing"]["frac_input_ge_0.64"],
          "| need nonKC:", r["consecutive_firing"]["required_nonKC_at_max_noise"]["frac_gt_0"])
    print("  verdicts:", v)
    print("wrote p11_2_0a_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
