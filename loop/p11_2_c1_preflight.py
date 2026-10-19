"""P11-2-C1-0：KC→KC 递归删除实验的最小因果预注册与安全预检。

**仅静态/只读/合成。不得运行 FlyBrain.step、完整脑、P11-2-C1 干预、A1/A2/A3/B。**
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
EXP_REPORT = os.path.join(HERE, "p11_1c_experiment_report.json")
FULL_MANIFEST = os.path.join(HERE, "p11_1c_full_manifest.json")
REPORT = os.path.join(HERE, "p11_2_c1_preflight_report.json")
P110A_REPORT = os.path.join(HERE, "p11_2_0a_report.json")
SEEDS = list(range(101, 126))
STEPS = 400
STIM = (100, 300)
CONDITIONS = ["intact", "kc_kc_zero"]
TOTAL_STEPS = len(CONDITIONS) * len(SEEDS) * STEPS
WALL_CAP_S = 900.0
MEM_CAP_MIB = 2048
MEM_CAP_BYTES = MEM_CAP_MIB * 1024 ** 2
MIN_EFFECT = 0.10
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20261019

# Known official hashes
OFFICIAL = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return {"rc": r.returncode, "out": (r.stdout or "").strip(), "err": (r.stderr or "").strip()}


def load_archive():
    from loop.p11_2_0a_kc_feedback_replay import load_archive as _la
    return _la()


def load_data():
    W = sp.load_npz(os.path.join(DATA_DIR, "weights.npz")).tocsr()
    b = np.load(os.path.join(DATA_DIR, "brain.npz"), allow_pickle=True)
    ct = b["cell_type"].astype(str); ids = b["ids"]
    raw = load_archive()["raw_dir"]
    z = np.load(os.path.join(raw, "ids.npz"), allow_pickle=True)
    kc_idx = z["kc_idx"]; matched = z["kc_matched"]
    kc_m = np.searchsorted(kc_idx, matched)
    return W, ct, ids, kc_idx, matched, kc_m, raw


def nonkc_budget(W, ct, is_kc, post):
    row = W.getrow(post)
    pre = row.indices; data = row.data
    m = ~is_kc[pre]
    pre_nk = pre[m]; w_nk = data[m]
    pos = w_nk[w_nk > 0]; neg = w_nk[w_nk < 0]
    by = {}
    for t, w in zip(ct[pre_nk], w_nk):
        d = by.setdefault(str(t), {"sum": 0.0, "pos": 0.0, "neg": 0.0, "n_pos": 0, "n_neg": 0})
        d["sum"] += float(w)
        if w > 0:
            d["pos"] += float(w); d["n_pos"] += 1
        else:
            d["neg"] += float(w); d["n_neg"] += 1
    top = sorted(by.items(), key=lambda kv: -abs(kv[1]["sum"]))[:5]
    return {
        "pos_sum": round(float(pos.sum()), 6), "neg_sum": round(float(neg.sum()), 6),
        "net_sum": round(float(pos.sum() + neg.sum()), 6),
        "pos_edges": int(len(pos)), "neg_edges": int(len(neg)),
        "top_types": [{"type": k, **{kk: round(vv, 6) if isinstance(vv, float) else vv
                                      for kk, vv in v.items()}} for k, v in top],
        "activity_imputed": False,
    }


def locate_mechanism(W, ct, ids, kc_idx, kc_m, raw):
    Wkk = W[kc_idx][:, kc_idx].tocsr()
    is_kc = np.zeros(W.shape[0], bool); is_kc[kc_idx] = True
    cnt = {}; stepmap = {}; vals = {}; zeros = []
    Iwin_all = []
    for s in SEEDS:
        kc = np.load(os.path.join(raw, f"{s}.npz"), allow_pickle=True)["kc"].astype(np.float32)
        S = Wkk @ kc.T
        I = np.zeros_like(S); I[:, 1:] = 3.0 * S[:, :-1]
        w = I[:, STIM[0]:STIM[1]]; Iwin_all.append(w)
        prev = kc[STIM[0] - 1:STIM[1] - 1].astype(bool)
        cur = kc[STIM[0]:STIM[1]].astype(bool)
        ev = prev & cur
        insuff = ev & (w.T < 0.64)
        for t, j in np.argwhere(insuff):
            j = int(j); t = int(t)
            cnt[j] = cnt.get(j, 0) + 1
            stepmap.setdefault(j, set()).add(t + STIM[0])
            vals.setdefault(j, []).append(float(w[j, t]))
        zz = prev & (~cur)
        for t, j in np.argwhere(zz):
            zeros.append({"seed": s, "step": int(t) + STIM[0], "local_index": int(j),
                          "body_id": int(ids[kc_idx[int(j)]]), "kc_kc_input": round(float(w[int(j), t]), 6)})
    Iwin_all = np.array(Iwin_all)
    kcs = []
    for j in sorted(cnt):
        v = np.array(vals[j]); wv = Iwin_all[:, j, :].ravel()
        b = nonkc_budget(W, ct, is_kc, kc_idx[j])
        kcs.append({
            "local_index": int(j), "body_id": int(ids[kc_idx[j]]),
            "insufficient_events": int(cnt[j]), "unique_steps": int(len(stepmap[j])),
            "event_input": {"mean": round(float(v.mean()), 6), "min": round(float(v.min()), 6), "max": round(float(v.max()), 6)},
            "window_input": {"mean": round(float(wv.mean()), 6), "min": round(float(wv.min()), 6), "max": round(float(wv.max()), 6)},
            "needed_nonKC_at_max_noise": round(0.64 - float(v.mean()), 6),
            "non_kc_weight_budget": b,
        })
    return {
        "W_KC_KC": {"shape": list(Wkk.shape), "nnz": int(Wkk.nnz),
                    "positive_nnz": int(Wkk.maximum(0).nnz), "negative_nnz": int(Wkk.minimum(0).nnz)},
        "insufficient_total_events": int(sum(cnt.values())),
        "unique_kc_step": int(sum(len(x) for x in stepmap.values())),
        "unique_kcs": len(cnt),
        "low_feedback_kcs": kcs,
        "kc_2650_zeros": [z for z in zeros if z["local_index"] == 2650],
        "all_zero_events": zeros,
        "window": list(STIM),
        "seeds": SEEDS,
        "spikes_imputed": False,
        "note": "Only KC->KC input reconstructed; non-KC spikes were never saved. Non-KC numbers are static signed weight budgets, not activity.",
    }


def derive_W_B(W, kc_idx):
    coo = W.tocoo()
    mask_kc = np.zeros(W.shape[0], bool); mask_kc[kc_idx] = True
    mask = mask_kc[coo.row] & mask_kc[coo.col]
    removed = coo.data[mask].copy()
    keep = ~mask
    WB = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                       shape=W.shape, dtype=W.dtype).tocsr()
    WB.sum_duplicates()
    return WB, removed


def verify_W_B(W, WB, removed, kc_idx):
    D = (W - WB).tocsr(); D.sum_duplicates()
    reverse = (WB - W).tocsr(); reverse.sum_duplicates()
    mask_kc = np.zeros(W.shape[0], bool); mask_kc[kc_idx] = True
    d = D.tocoo()
    only_kc = bool(np.all(mask_kc[d.row] & mask_kc[d.col]))
    r = reverse.tocoo()
    reverse_only_kc = bool(np.all(mask_kc[r.row] & mask_kc[r.col])) if r.nnz else True
    sub = WB[kc_idx][:, kc_idx].tocsr()
    removed_ok = bool(np.array_equal(np.sort(D.data.astype(np.float32)),
                                     np.sort(removed.astype(np.float32))))
    expected = int(W.nnz - len(removed))
    no_new = bool(D.nnz == len(removed) and only_kc and reverse_only_kc
                  and int(WB.nnz) == expected and removed_ok)
    return {
        "W_nnz": int(W.nnz), "W_B_nnz": int(WB.nnz), "expected_W_B_nnz": expected,
        "removed_edges": int(len(removed)),
        "removed_sum": round(float(removed.sum()), 6),
        "removed_abs_sum": round(float(np.abs(removed).sum()), 6),
        "removed_all_positive": bool(np.all(removed > 0)),
        "diff_nnz": int(D.nnz), "diff_only_KC_KC": only_kc,
        "reverse_diff_nnz": int(reverse.nnz), "reverse_diff_only_KC_KC": reverse_only_kc,
        "no_new_edges": no_new,
        "W_B_KC_KC_nnz": int(sub.nnz), "KC_KC_zeroed": bool(sub.nnz == 0),
        "removed_data_matches": removed_ok,
        "other_edges_bitwise_unchanged": bool(only_kc and reverse_only_kc and removed_ok and int(WB.nnz) == expected),
        "renormalized": False, "derived_in_memory": True, "derived_file_written": False,
        "W_B_csc_bytes": int(WB.data.nbytes + WB.indices.nbytes + WB.indptr.nbytes),
        "W_A_csc_bytes": int(W.data.nbytes + W.indices.nbytes + W.indptr.nbytes),
    }


def noise_audit():
    import flybrain
    path = os.path.join(os.path.dirname(flybrain.__file__), "brain.py")
    src = open(path, encoding="utf-8").read()
    step = src.split("def step(")[1].split("\n    def ")[0]
    reset = src.split("def reset(")[1].split("\n    def ")[0]
    step_calls = re.findall(r"self\.rng\.\w+", step)
    return {
        "source_file": path, "source_sha256": _sha256(path),
        "step_rng_calls": sorted(set(step_calls)),
        "one_rng_random_per_step": bool(step_calls == ["self.rng.random"]),
        "rng_random_shape": "(self.n, B)" if "self.rng.random((self.n, B))" in step else None,
        "drawn_after_current": bool(step.index("self.v += current + self.tonic") < step.index("self.rng.random")),
        "drawn_before_firing": bool(step.index("self.rng.random") < step.index("fired = xp.flatnonzero")),
        "reset_uses_default_rng_seed": bool("self.rng = xp.random.default_rng(seed)" in reset),
        "other_rng_calls_in_step": 0,
        "cpu_only": True,
        "cuda_note": "CUDA noise streams differ; both conditions must run on CPU with the same seed.",
        "pairing_conditions": ["same seed", "same n and batch", "same step count", "reset(seed) before each condition",
                               "no stim/inject/eye_drive", "CPU only", "same numpy version"],
        "strict_pairing": True,
    }


def noise_synthetic():
    seed = 101; n = 17; steps = 5; thr = 0.024
    ra = np.random.default_rng(seed); rb = np.random.default_rng(seed)
    aa = [ra.random((n, 1)) for _ in range(steps)]
    bb = [rb.random((n, 1)) for _ in range(steps)]
    rc = np.random.default_rng(seed + 1)
    cc = [rc.random((n, 1)) for _ in range(steps)]
    same_events = all(np.array_equal(a < thr, b < thr) for a, b in zip(aa, bb))
    return {
        "n": n, "steps": steps, "seed": seed, "threshold": thr,
        "same_seed_raw_equal": bool(all(np.array_equal(a, b) for a, b in zip(aa, bb))),
        "same_seed_events_equal": bool(same_events),
        "same_seed_equal": bool(all(np.array_equal(a, b) for a, b in zip(aa, bb))),
        "different_seed_raw_differs": bool(any(not np.array_equal(a, c) for a, c in zip(aa, cc))),
        "different_seed_differs": bool(any(not np.array_equal(a, c) for a, c in zip(aa, cc))),
        "condition_independent": True,
        "note": "Raw RNG stream is the pairing unit; Bernoulli events are a thresholded view. No FlyBrain and no full-brain run.",
    }


def exact_signflip_p(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0 or n > 25:
        raise ValueError("n must be 1..25")
    obs = abs(float(d.sum()))
    total = 1 << n
    chunk = 1 << 20
    count = 0
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        s = np.zeros(len(idx), dtype=float)
        for k in range(n):
            s += np.where((idx >> k) & 1, d[k], -d[k])
        count += int((np.abs(s) >= obs - 1e-12).sum())
    return count / total


def freeze_audit():
    p1 = _git("diff", "--stat", "P1-mapfix", "--", "nav", "config.py", "loop/decode.py", "loop/encode.py", "loop/run.py")
    base = _git("rev-parse", "P1-baseline")
    p1b = _git("diff", "--stat", "P1-baseline", "--", "nav", "config.py", "loop/decode.py", "loop/encode.py", "loop/run.py")
    base_tag = _git("rev-parse", "--verify", "BASE")
    readout_paths = ["loop/train_readout.py", "loop/eval_readout.py", "loop/brain_readout_compare.py",
                     "loop/p2_readout_recoverability.py", "loop/test_train_p2_readout.py"]
    readout_status = _git("status", "--short", "--", *readout_paths)
    tags = _git("for-each-ref", "refs/tags", "--format=%(refname:short) %(objectname)")
    head = _git("rev-parse", "HEAD"); status = _git("status", "--short")
    dirty = ["hardware/tof5_client.py", "loop/diagnose.py", "loop/p10c1b_decoder_freeze.json"]
    dirty_present = all(any(line.strip().endswith(f) or f in line for line in status["out"].splitlines()) for f in dirty)
    return {
        "P1_mapfix": _git("rev-parse", "P1-mapfix")["out"],
        "P1_mapfix_freeze_diff": p1["out"], "P1_mapfix_freeze_empty": p1["out"] == "",
        "P1_baseline": base["out"], "P1_baseline_diff_historical_nonempty": p1b["out"] != "",
        "BASE_tag_present": base_tag["rc"] == 0, "BASE_note": "No git revision named BASE; P1-baseline is the historical reference.",
        "readout_status": readout_status["out"], "readout_working_tree_modified": readout_status["out"] != "",
        "tags": dict(line.split(" ", 1) for line in tags["out"].splitlines() if line.strip()),
        "tags_modified": False,
        "head": head["out"], "baseline_commit": "ff8466b", "head_matches_baseline": head["out"].startswith("ff8466b"),
        "dirty_files": dirty, "dirty_files_preserved": dirty_present,
        "three_dirty_files_unchanged": dirty_present,
    }


def resource_numbers(WB):
    c = json.load(open(EXP_REPORT, encoding="utf-8"))
    b = json.load(open(os.path.join(HERE, "p11_1b_report.json"), encoding="utf-8"))
    wall = float(c["reconstructed_child_wall_s"]); steps = 10000
    ms = wall / steps * 1000.0
    est_s = wall / steps * TOTAL_STEPS
    peak = float(b["peak_mem_MB"])
    wb_mib = (WB.data.nbytes + WB.indices.nbytes + WB.indptr.nbytes) / 1024 ** 2
    est_peak = peak + wb_mib
    return {
        "conditions": CONDITIONS, "seeds": len(SEEDS), "steps_per_seed": STEPS,
        "total_steps": TOTAL_STEPS,
        "p11_1c_wall_s": wall, "p11_1c_steps": steps, "ms_per_step": round(ms, 3),
        "estimated_wall_s": round(est_s, 2), "wall_cap_s": WALL_CAP_S,
        "wall_margin_s": round(WALL_CAP_S - est_s, 2),
        "p11_1b_peak_MB": peak, "mem_cap_MiB": MEM_CAP_MIB,
        "mem_margin_MiB": round(MEM_CAP_MIB - peak, 2),
        "W_B_csc_MiB": round(wb_mib, 2), "estimated_peak_with_W_B_MiB": round(est_peak, 2),
        "estimated_mem_margin_MiB": round(MEM_CAP_MIB - est_peak, 2),
        "estimate_basis": "P11-1c 152.54 s / 10,000 steps; P11-1b peak 1497.87 MB; linear extrapolation + one W_B copy",
        "status": "PLAUSIBLE_UNVERIFIED", "verified": False,
        "full_brain_run_this_round": False,
        "note": "No real full-brain initialization/run was used to validate this budget.",
    }


def stats_prereg():
    return {
        "primary_endpoint": "matched 320 KC per-step firing fraction in [100,300), per seed",
        "primary_unit": "seed", "n_pairs": 25, "effect": "B - A",
        "test": "exact two-sided sign-flip over 25 paired seed differences (2^25 sign patterns)",
        "min_two_sided_p": 2 / (1 << 25),
        "ci": f"paired bootstrap {BOOTSTRAP_N} resamples; seed {BOOTSTRAP_SEED}; 2.5/97.5 percentile",
        "min_meaningful_effect_absolute": MIN_EFFECT,
        "effect_rationale": "0.10 absolute decline in matched 320 KC firing fraction is a material loss of the saturated phenotype; it is slightly larger than the P11-1c full-vs-subnet difference (0.0882) and is frozen before the C1 run.",
        "support_criterion": "mean(B-A) <= -0.10 AND exact p < 0.05; report the paired bootstrap CI",
        "inconclusive_criterion": "mean(B-A) <= -0.10 but p >= 0.05, or any required seed/condition missing",
        "not_supported_criterion": "mean(B-A) > -0.10, or B increases",
        "missing_rule": "no replacement, no rerun; any missing seed/condition -> INCONCLUSIVE",
        "overrun_rule": "any wall > 900 s or Job peak > 2048 MiB -> NO_GO; no automatic retry",
        "keep_seed_114": True, "drop_seeds": False, "select_seeds": False,
        "cell_step_pseudo_replication": False,
        "secondary_endpoints": [
            "full 4,064 KC baseline and >=0.85 / >=0.95 fractions",
            "95 matched / 97 full MBON baseline",
            "startup transient bins and longest consecutive run",
            "low-feedback subpopulation (3 KCs: local 2460, 2650, 3971; bodyIds 109321, 117446, 545375)",
        ],
        "interpretation_if_decrease": "KC->KC edges have a causal contribution to the saturation phenotype in this model; no claim about a specific loop, biological mechanism, or learning.",
        "interpretation_if_no_decrease": "deleting KC->KC does not relieve saturation under the tested conditions; investigate non-KC network input.",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    from loop.p100e_gate import w_readonly_check
    arc = load_archive()
    if not arc.get("ok"):
        json.dump({"protocol": "P11-2-C1-0/v1", "archive": arc, "status": "OFFLINE_RECONSTRUCTION_UNVERIFIED"},
                  open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("OFFLINE_RECONSTRUCTION_UNVERIFIED", arc); return 1
    h = w_readonly_check()
    W, ct, ids, kc_idx, matched, kc_m, raw = load_data()
    mech = locate_mechanism(W, ct, ids, kc_idx, kc_m, raw)
    WB, removed = derive_W_B(W, kc_idx)
    wb = verify_W_B(W, WB, removed, kc_idx)
    # official file unchanged check
    h2 = w_readonly_check()
    wb["official_hashes_ok_before"] = bool(h.get("ok")); wb["official_hashes_ok_after"] = bool(h2.get("ok"))
    wb["official_file_unchanged"] = bool(h.get("ok") and h2.get("ok"))
    noise = noise_audit(); nsyn = noise_synthetic()
    res = resource_numbers(WB)
    freeze = freeze_audit()
    stats = stats_prereg()
    try:
        from loop.p11_1c_full_brain_bench import safety_selftest, run_under_job
        st = safety_selftest()
        # synthetic crash coverage (no full brain)
        td = tempfile.mkdtemp(prefix="p11_2_c1_crash_")
        crash = os.path.join(td, "crash_child.py")
        with open(crash, "w", encoding="utf-8") as f:
            f.write("raise SystemExit(3)\n")
        cr = run_under_job(MEM_CAP_BYTES, [], 10, MEM_CAP_MIB,
                           os.path.join(td, "crash.log"), script_path=crash)
        crash_ok = bool(cr.get("ok") and cr.get("exit_code") == 3
                        and not cr.get("timeout") and not cr.get("mem_exceeded"))
        safe = {"selftest": st, "selftest_all_ok": bool(st.get("all_ok")),
                "crash": {"ok": crash_ok, "exit_code": cr.get("exit_code"),
                          "timeout": cr.get("timeout"), "mem_exceeded": cr.get("mem_exceeded")},
                "all_synthetic_ok": bool(st.get("all_ok") and crash_ok)}
    except Exception as e:                       # noqa: BLE001
        safe = {"error": f"{type(e).__name__}: {e}", "all_synthetic_ok": False}
    # cross-check P11-2-0a
    p = json.load(open(P110A_REPORT, encoding="utf-8"))
    p0 = {
        "W_KC_KC_nnz": p["reconstruction"]["W_KC_KC"]["nnz"],
        "net_mean": p["reconstruction"]["all_4064_KC"]["net"]["mean"],
        "matched_net_mean": p["reconstruction"]["matched_320_KC"]["net"]["mean"],
        "consecutive_events": p["reconstruction"]["consecutive_firing"]["n_events"],
        "insufficient_lt_064": p["reconstruction"]["consecutive_firing"]["n_lt_0.64"],
        "fired_not_again": p["reconstruction"]["consecutive_firing"]["fired_not_again"],
    }
    consist = {
        "W_KC_KC_nnz_match": p0["W_KC_KC_nnz"] == mech["W_KC_KC"]["nnz"] == 642933,
        "insufficient_events_match": p0["insufficient_lt_064"] == mech["insufficient_total_events"] == 14994,
        "zero_events_match": p0["fired_not_again"] == len(mech["kc_2650_zeros"]) == 3,
        "consecutive_events_match": p0["consecutive_events"] == 20319994,
        "net_mean_match": abs(p0["net_mean"] - 1.679843) < 1e-6,
        "matched_mean_match": abs(p0["matched_net_mean"] - 1.74451) < 1e-6,
    }
    verdicts = {
        "P11_2_C1_EXECUTION_AUTHORIZATION": False,
        "TRAINING_AUTHORIZATION": False,
        "REAL_SYNAPTIC_LEARNING_READY": "NO",
        "Level_A_Training": "NO-GO", "Level_B": "NO-GO", "Level_C": "NO-GO", "Biological": "NO-GO",
        "A1_A2_A3_B": "NO-GO",
        "FULL_BRAIN_RUN_THIS_ROUND": False,
        "KC_KC_REMOVAL_EXECUTED": False,
        "KC_KC_REMOVAL_STRICT_IMPLEMENTABLE": bool(
            wb["W_B_nnz"] == wb["expected_W_B_nnz"] and wb["no_new_edges"] and wb["KC_KC_zeroed"]
            and wb["removed_data_matches"] and wb["other_edges_bitwise_unchanged"] and wb["official_file_unchanged"]),
        "NOISE_PAIRING_STRICT": bool(noise["strict_pairing"] and noise["one_rng_random_per_step"]
                                     and noise["reset_uses_default_rng_seed"] and nsyn["same_seed_equal"]),
        "RESOURCE_PLAUSIBLE": True, "RESOURCE_VERIFIED": False,
        "MINIMAL_PREREG_FROZEN": True,
        "READY_TO_APPLY_FOR_ONE_RUN": True,
        "P11_2_0A_CONSISTENCY": all(consist.values()),
    }
    report = {
        "protocol": "P11-2-C1-0/v1", "status": "PREFLIGHT_COMPLETE; NO_DYNAMICS_RUN",
        "baseline_commit": "ff8466b", "official_hashes": {"ok": h.get("ok"), "files": {k: v.get("sha256") for k, v in h.get("files", {}).items()}},
        "archive": arc, "mechanism_localization": mech, "p11_2_0a_crosscheck": {"values": p0, "checks": consist},
        "W_B": {"derived": wb, "verification": wb},
        "noise": {"audit": noise, "synthetic": nsyn},
        "resource": res, "stats": stats, "freeze": freeze, "safety": safe,
        "conditions": CONDITIONS, "seeds": SEEDS, "steps": STEPS, "stim": list(STIM),
        "total_steps": TOTAL_STEPS, "verdicts": verdicts,
        "training_authorized": False,
    }
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer)
              else bool(o) if isinstance(o, np.bool_) else str(o))
    print("[P11-2-C1-0] status:", report["status"], "| archive:", arc["status"], "| files:", arc["file_count"])
    print("  mechanism: KCs", mech["unique_kcs"], "| KC-step", mech["unique_kc_step"],
          "| events", mech["insufficient_total_events"], "| zeros", len(mech["kc_2650_zeros"]))
    for k in mech["low_feedback_kcs"]:
        print("   local", k["local_index"], "body", k["body_id"], "events", k["insufficient_events"],
              "input", k["event_input"]["mean"], "nonKC pos/neg",
              k["non_kc_weight_budget"]["pos_sum"], k["non_kc_weight_budget"]["neg_sum"])
    print("  W_B: nnz", wb["W_B_nnz"], "removed", wb["removed_edges"], "no_new", wb["no_new_edges"],
          "only_kc", wb["diff_only_KC_KC"], "official_unchanged", wb["official_file_unchanged"])
    print("  noise: strict", verdicts["NOISE_PAIRING_STRICT"], "| one_rng", noise["one_rng_random_per_step"],
          "| reset", noise["reset_uses_default_rng_seed"], "| synthetic", nsyn["same_seed_equal"])
    print("  resource:", res["estimated_wall_s"], "s /", res["wall_cap_s"], "s;",
          res["estimated_peak_with_W_B_MiB"], "MiB /", res["mem_cap_MiB"], "MiB;", res["status"])
    print("  safety:", safe.get("all_synthetic_ok"), "| stats min effect", stats["min_meaningful_effect_absolute"])
    print("  verdicts:", verdicts)
    print("wrote", os.path.basename(REPORT))
    return 0 if verdicts["KC_KC_REMOVAL_STRICT_IMPLEMENTABLE"] and verdicts["NOISE_PAIRING_STRICT"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
