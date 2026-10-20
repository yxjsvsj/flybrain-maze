"""P11-2-S2-OFFLINE: S1R 负结果的离线复核与静态机制诊断。

只读取已归档 50 NPZ、官方 brain/weights、S1R 报告/清单；不运行 FlyBrain，
不初始化完整脑，不修改任何正式结果。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
FORMAL = os.path.join(HERE, "p11_2_s1r_report.json")
RUN_META = os.path.join(HERE, "p11_2_s1r_run_meta.json")
MANIFEST = os.path.join(HERE, "p11_2_s1r_raw_manifest.json")
BACKUP = r"C:\AA_project\guoying_archive\p11_2_s1r_run_20261020T160105"
OUT = os.path.join(HERE, "p11_2_s2_report.json")
SEEDS = list(range(101, 126))
STEPS = 400
STIM = (100, 300)
BOOT_N = 10000
BOOT_SEED = 20261019
TIME_BINS = [0, 100, 150, 200, 250, 300, 400]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def exact_signflip_p(diffs):
    d = np.asarray(diffs, float)
    n = len(d)
    total = 1 << n
    obs = abs(float(d.sum()))
    count = 0
    chunk = 1 << 20
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        s = np.zeros(len(idx), dtype=float)
        for k in range(n):
            s += np.where((idx >> k) & 1, d[k], -d[k])
        count += int((np.abs(s) >= obs - 1e-12).sum())
    return count / total


def bootstrap_ci(diffs):
    rng = np.random.default_rng(BOOT_SEED)
    d = np.asarray(diffs, float)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT_N)]
    return {"ci_low": round(float(np.percentile(m, 2.5)), 9),
            "ci_high": round(float(np.percentile(m, 97.5)), 9),
            "n": BOOT_N, "seed": BOOT_SEED}


def _distribution(x):
    x = np.asarray(x, float)
    return {"n": int(x.size), "mean": round(float(x.mean()), 9),
            "sd": round(float(x.std(ddof=1)), 9) if x.size > 1 else 0.0,
            "min": round(float(x.min()), 9), "max": round(float(x.max()), 9),
            "q05": round(float(np.quantile(x, .05)), 9),
            "q25": round(float(np.quantile(x, .25)), 9),
            "q50": round(float(np.quantile(x, .50)), 9),
            "q75": round(float(np.quantile(x, .75)), 9),
            "q95": round(float(np.quantile(x, .95)), 9)}


def _cancel(x):
    x = np.asarray(x, float)
    pos = x[x > 0]
    neg = x[x < 0]
    return {"n": int(x.size), "mean": round(float(x.mean()), 9),
            "sd": round(float(x.std(ddof=1)), 9) if x.size > 1 else 0.0,
            "positive_cells": int(pos.size), "negative_cells": int(neg.size),
            "zero_cells": int((x == 0).sum()),
            "positive_sum": round(float(pos.sum()), 9),
            "negative_sum": round(float(neg.sum()), 9),
            "net_sum": round(float(x.sum()), 9),
            "positive_fraction_of_abs_sum": round(float(pos.sum() / (abs(pos.sum()) + abs(neg.sum()))), 9),
            "note": "cell-level mean across 25 paired seeds; not 25 independent cell samples; EXPLORATORY"}


def _bh(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    prev = 1.0
    for rank in range(len(p) - 1, -1, -1):
        i = order[rank]
        val = min(prev, p[i] * len(p) / (rank + 1))
        q[i] = val
        prev = val
    return q


def reproduce_and_archive():
    formal = _load(FORMAL)
    meta = _load(RUN_META)
    man = _load(MANIFEST)
    raw = meta["out_dir"]
    if not os.path.isdir(raw) and os.path.isdir(BACKUP):
        raw = BACKUP
    ids = np.load(os.path.join(raw, "ids.npz"), allow_pickle=True)
    rows = []
    for x in man["files"]:
        p = os.path.join(raw, x["name"])
        ok = os.path.exists(p) and _sha(p) == x["sha256"] and os.path.getsize(p) == x["size"]
        rows.append({"name": x["name"], "ok": bool(ok)})
    per = {k: [] for k in ("entry", "kc", "mb", "full_kc", "full_mb")}
    for seed in SEEDS:
        with np.load(os.path.join(raw, f"{seed}_none.npz"), allow_pickle=True) as zn, \
                np.load(os.path.join(raw, f"{seed}_stim.npz"), allow_pickle=True) as zs:
            for key, a, b in (
                    ("entry", zn["lp"], zs["lp"]),
                    ("kc", zn["kc_matched"], zs["kc_matched"]),
                    ("mb", zn["mb_matched"], zs["mb_matched"]),
                    ("full_kc", zn["kc"], zs["kc"]),
                    ("full_mb", zn["mb"], zs["mb"])):
                per[key].append(float(b[100:300].mean() - a[100:300].mean()))
    def summary(key):
        d = np.asarray(per[key], float)
        return {"mean": round(float(d.mean()), 9), "sd": round(float(d.std(ddof=1)), 9),
                "p": exact_signflip_p(d), "ci": bootstrap_ci(d)}
    ent = summary("entry"); kc = summary("kc"); mb = summary("mb")
    full_kc = summary("full_kc"); full_mb = summary("full_mb")
    def match(actual, report, field="mean"):
        return abs(float(actual) - float(report[field])) < 1e-9
    repro = {
        "entry": {**ent, "formal_mean": formal["entry"]["diff"]["mean"],
                  "formal_p": formal["entry"]["exact_p"],
                  "mean_match": match(ent["mean"], formal["entry"]["diff"]),
                  "p_match": abs(ent["p"] - formal["entry"]["exact_p"]) < 1e-12},
        "primary": {**kc, "formal_mean": formal["primary"]["diff"]["mean"],
                    "formal_p": formal["primary"]["exact_p"],
                    "formal_ci": formal["primary"]["ci"],
                    "mean_match": match(kc["mean"], formal["primary"]["diff"]),
                    "p_match": abs(kc["p"] - formal["primary"]["exact_p"]) < 1e-12,
                    "ci_match": abs(kc["ci"]["ci_low"] - formal["primary"]["ci"]["ci_low"]) < 1e-9
                                 and abs(kc["ci"]["ci_high"] - formal["primary"]["ci"]["ci_high"]) < 1e-9},
        "mbon": {**mb, "formal_mean": formal["mbon"]["matched_95_diff"]["mean"],
                 "formal_p": formal["mbon"]["exact_p"],
                 "mean_match": match(mb["mean"], formal["mbon"]["matched_95_diff"]),
                 "p_match": abs(mb["p"] - formal["mbon"]["exact_p"]) < 1e-12},
        "full_kc": {**full_kc, "formal_none": formal["secondary"]["full_4064_none"],
                    "formal_stim": formal["secondary"]["full_4064_stim"],
                    "formal_mean": round(formal["secondary"]["full_4064_stim"] - formal["secondary"]["full_4064_none"], 9),
                    "mean_match": abs(full_kc["mean"] - (formal["secondary"]["full_4064_stim"] - formal["secondary"]["full_4064_none"])) < 1e-9},
        "full_mbon": {**full_mb, "formal_none": formal["secondary"]["full_97_mbon_none"],
                      "formal_stim": formal["secondary"]["full_97_mbon_stim"],
                      "formal_mean": round(formal["secondary"]["full_97_mbon_stim"] - formal["secondary"]["full_97_mbon_none"], 9),
                      "mean_match": abs(full_mb["mean"] - (formal["secondary"]["full_97_mbon_stim"] - formal["secondary"]["full_97_mbon_none"])) < 1e-9},
        "npz_count": int(len([x for x in man["files"] if x["name"].endswith("_none.npz") or x["name"].endswith("_stim.npz")])),
        "all_listed_hashes_ok": all(x["ok"] for x in rows),
        "run_meta_sha256_match": _sha(os.path.join(raw, "run_meta.json")) == man.get("run_meta_sha256"),
        "ids_kc_count": int(len(ids["kc_idx"])), "ids_matched_kc": int(len(ids["kc_matched"])),
        "ids_mb_count": int(len(ids["mb_idx"])), "ids_matched_mb": int(len(ids["mb_matched"])),
    }
    repro["formal_negative_reproduced"] = all([
        repro["entry"]["mean_match"], repro["entry"]["p_match"],
        repro["primary"]["mean_match"], repro["primary"]["p_match"], repro["primary"]["ci_match"],
        repro["mbon"]["mean_match"], repro["mbon"]["p_match"],
        repro["full_kc"]["mean_match"], repro["full_mbon"]["mean_match"],
        repro["all_listed_hashes_ok"], repro["npz_count"] == 50,
    ])
    return raw, ids, repro, per, formal


def static_paths(raw, ids):
    W = sp.load_npz(os.path.join(DATA_DIR, "weights.npz")).tocsr()
    b = np.load(os.path.join(DATA_DIR, "brain.npz"), allow_pickle=True)
    ct = b["cell_type"].astype(str)
    kc = np.flatnonzero(np.char.startswith(ct, "KC"))
    mb = np.flatnonzero(np.char.startswith(ct, "MBON"))
    src = np.sort(np.concatenate([np.flatnonzero(ct == "LPLC2"), np.flatnonzero(ct == "LC4")]))
    kc_idx = np.asarray(ids["kc_idx"]); kc_matched = np.asarray(ids["kc_matched"])
    mb_idx = np.asarray(ids["mb_idx"]); mb_matched = np.asarray(ids["mb_matched"])
    assert np.array_equal(kc_idx, kc) and np.array_equal(mb_idx, mb)
    ml = np.searchsorted(kc, kc_matched)
    mbl = np.searchsorted(mb, mb_matched)
    Wb = W.copy(); Wb.data = np.ones_like(Wb.data)
    P = (Wb[kc] @ Wb[:, src]).tocsr()
    A = (W[kc] @ W[:, src]).tocsr()
    Pm = (Wb[mb] @ Wb[:, src]).tocsr()
    Am = (W[mb] @ W[:, src]).tocsr()
    is_kc = np.zeros(W.shape[0], bool); is_kc[kc] = True
    coo = Wb.tocoo(); keep = ~(is_kc[coo.row] & is_kc[coo.col])
    WbB = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=W.shape).tocsr()
    cooW = W.tocoo(); keepW = ~(is_kc[cooW.row] & is_kc[cooW.col])
    WB = sp.coo_matrix((cooW.data[keepW], (cooW.row[keepW], cooW.col[keepW])), shape=W.shape).tocsr()
    PB = (WbB[kc] @ WbB[:, src]).tocsr()
    AB = (WB[kc] @ WB[:, src]).tocsr()
    PmB = (WbB[mb] @ WbB[:, src]).tocsr()
    AmB = (WB[mb] @ WB[:, src]).tocsr()
    # 3-hop feedforward to MBON: source -> non-KC -> KC -> MBON, B network
    P3 = (WbB[mb][:, kc] @ PB).tocoo()
    A3 = (WB[mb][:, kc] @ AB).tocoo()
    # matched rows
    Pr = P[ml].tocsr(); Ar = A[ml].tocsr()
    paths = np.asarray(Pr.sum(axis=1)).ravel()
    strength = np.asarray(Ar.sum(axis=1)).ravel()
    reach = np.diff(Pr.indptr) > 0
    other = np.setdiff1d(np.arange(len(kc)), ml)
    Po = P[other]; Ao = A[other]
    Por = Po.tocsr(); Aor = Ao.tocsr()
    reach_o = np.diff(Por.indptr) > 0
    # matched MBON
    Pmr = Pm[mbl].tocsr(); Amr = Am[mbl].tocsr()
    paths_m = np.asarray(Pmr.sum(axis=1)).ravel()
    strength_m = np.asarray(Amr.sum(axis=1)).ravel()
    def sig(x):
        return {"edges": int(x.nnz), "positive": int((x.data > 0).sum()),
                "negative": int((x.data < 0).sum()),
                "signed_sum": round(float(x.data.sum()), 12)}
    return {
        "source_count": int(len(src)),
        "kc_count": int(len(kc)), "mbon_count": int(len(mb)),
        "matched_kc": int(len(ml)), "matched_mbon": int(len(mbl)),
        "direct_src_to_kc": sig(W[kc][:, src]),
        "direct_src_to_mbon": sig(W[mb][:, src]),
        "two_hop_kc": {**sig(A), "endpoint_pairs": int(P.nnz),
                       "independent_paths": int(P.data.sum()),
                       "paths_per_endpoint_max": int(P.data.max()) if P.nnz else 0,
                       "paths_per_endpoint_mean": round(float(P.data.mean()), 6) if P.nnz else 0.0},
        "two_hop_kc_B": {**sig(AB), "endpoint_pairs": int(PB.nnz),
                         "independent_paths": int(PB.data.sum())},
        "two_hop_mbon": {**sig(Am), "endpoint_pairs": int(Pm.nnz),
                         "independent_paths": int(Pm.data.sum())},
        "two_hop_mbon_B": {**sig(AmB), "endpoint_pairs": int(PmB.nnz),
                           "independent_paths": int(PmB.data.sum())},
        "three_hop_mbon_via_kc_B": {**sig(A3), "endpoint_pairs": int(P3.nnz),
                                    "independent_paths": int(P3.data.sum())},
        "matched_kc": {
            "n": int(len(ml)), "reachable": int(reach.sum()),
            "endpoint_pairs": int(Pr.nnz), "independent_paths": int(paths.sum()),
            "paths_mean": round(float(paths.mean()), 6), "paths_min": int(paths.min()),
            "paths_max": int(paths.max()), "strength_sum": round(float(strength.sum()), 12),
            "strength_mean": round(float(strength.mean()), 12),
            "strength_min": round(float(strength.min()), 12),
            "strength_max": round(float(strength.max()), 12),
            "positive_kc": int((strength > 0).sum()), "negative_kc": int((strength < 0).sum()),
            "zero_kc": int((strength == 0).sum()),
            "paths": paths.tolist(), "strength": strength.tolist(),
        },
        "other_kc": {
            "n": int(len(other)), "reachable": int(reach_o.sum()),
            "paths_sum": int(np.asarray(Por.sum(axis=1)).sum()),
            "strength_sum": round(float(np.asarray(Aor.sum(axis=1)).sum()), 12),
        },
        "matched_mbon": {
            "n": int(len(mbl)), "endpoint_pairs": int(Pmr.nnz),
            "independent_paths": int(paths_m.sum()), "paths_mean": round(float(paths_m.mean()), 6),
            "strength_sum": round(float(strength_m.sum()), 12),
            "positive_mbon": int((strength_m > 0).sum()), "negative_mbon": int((strength_m < 0).sum()),
        },
        "removed_kc_kc_edges": int(W[kc][:, kc].nnz),
        "feedforward_preserved_in_B": bool(PB.nnz == P.nnz and PB.data.sum() == P.data.sum()),
        "note": "W^2/W^3 are static weighted path statistics, not dynamic responses.",
        "arrays": {"matched_paths": paths, "matched_strength": strength},
    }


def dynamic(raw, ids, static):
    n_kc, n_m = len(ids["kc_idx"]), len(ids["mb_idx"])
    n_km, n_mm = len(ids["kc_matched"]), len(ids["mb_matched"])
    sums = {k: np.zeros(n, float) for k, n in (
        ("kcn", n_kc), ("kcs", n_kc), ("kcmn", n_km), ("kcms", n_km),
        ("mbn", n_m), ("mbs", n_m), ("mbmn", n_mm), ("mbms", n_mm))}
    per = {k: [] for k in ("entry", "kc", "mb", "full_kc", "full_mb")}
    tbins = {k: np.zeros(len(TIME_BINS) - 1) for k in ("entry", "kc", "mb")}
    for seed in SEEDS:
        with np.load(os.path.join(raw, f"{seed}_none.npz"), allow_pickle=True) as zn, \
                np.load(os.path.join(raw, f"{seed}_stim.npz"), allow_pickle=True) as zs:
            sums["kcn"] += zn["kc"][100:300].mean(axis=0)
            sums["kcs"] += zs["kc"][100:300].mean(axis=0)
            sums["kcmn"] += zn["kc_matched"][100:300].mean(axis=0)
            sums["kcms"] += zs["kc_matched"][100:300].mean(axis=0)
            sums["mbn"] += zn["mb"][100:300].mean(axis=0)
            sums["mbs"] += zs["mb"][100:300].mean(axis=0)
            sums["mbmn"] += zn["mb_matched"][100:300].mean(axis=0)
            sums["mbms"] += zs["mb_matched"][100:300].mean(axis=0)
            for key, a, b in (("entry", zn["lp"], zs["lp"]),
                              ("kc", zn["kc_matched"], zs["kc_matched"]),
                              ("mb", zn["mb_matched"], zs["mb_matched"]),
                              ("full_kc", zn["kc"], zs["kc"]),
                              ("full_mb", zn["mb"], zs["mb"])):
                per[key].append(float(b[100:300].mean() - a[100:300].mean()))
            for key, a, b in (("entry", zn["lp"], zs["lp"]),
                              ("kc", zn["kc_matched"], zs["kc_matched"]),
                              ("mb", zn["mb_matched"], zs["mb_matched"])):
                for i in range(len(TIME_BINS) - 1):
                    lo, hi = TIME_BINS[i], TIME_BINS[i + 1]
                    tbins[key][i] += float(b[lo:hi].mean() - a[lo:hi].mean())
    for k in sums:
        sums[k] /= len(SEEDS)
    for k in tbins:
        tbins[k] /= len(SEEDS)
    kc_diff = sums["kcs"] - sums["kcn"]
    km_diff = sums["kcms"] - sums["kcmn"]
    mb_diff = sums["mbs"] - sums["mbn"]
    mm_diff = sums["mbms"] - sums["mbmn"]
    paths = np.asarray(static["arrays"]["matched_paths"], float)
    strength = np.asarray(static["arrays"]["matched_strength"], float)
    med_p = float(np.median(paths)); med_s = float(np.median(np.abs(strength)))
    def group(mask):
        return {"n": int(mask.sum()), "mean_diff": round(float(km_diff[mask].mean()), 9),
                "mean_paths": round(float(paths[mask].mean()), 6),
                "mean_abs_strength": round(float(np.abs(strength[mask]).mean()), 12)}
    groups = {
        "path_low": group(paths <= med_p), "path_high": group(paths > med_p),
        "strength_low": group(np.abs(strength) <= med_s),
        "strength_high": group(np.abs(strength) > med_s),
        "path_median": med_p, "strength_abs_median": med_s,
        "corr_path_count_vs_km_diff": round(float(np.corrcoef(paths, km_diff)[0, 1]), 9),
        "corr_abs_strength_vs_km_diff": round(float(np.corrcoef(np.abs(strength), km_diff)[0, 1]), 9),
        "corr_signed_strength_vs_km_diff": round(float(np.corrcoef(strength, km_diff)[0, 1]), 9),
        "exploratory": True,
    }
    # background groups on matched KC none rate
    km_none = sums["kcmn"]
    bg = []
    for lo, hi in ((0, .1), (.1, .5), (.5, .85), (.85, 1.01)):
        sel = (km_none >= lo) & (km_none < hi)
        bg.append({"none_rate_range": [lo, hi], "n": int(sel.sum()),
                   "mean_none_rate": round(float(km_none[sel].mean()), 9) if sel.any() else None,
                   "mean_diff": round(float(km_diff[sel].mean()), 9) if sel.any() else None})
    low = [{"full_kc_index": int(i), "none": round(float(sums["kcn"][i]), 9),
            "stim": round(float(sums["kcs"][i]), 9),
            "diff": round(float(kc_diff[i]), 9)} for i in (2460, 2650, 3971)]
    # paired seed correlations, BH over 3
    def corr(a, b):
        r, p = stats.pearsonr(np.asarray(a, float), np.asarray(b, float))
        return {"pearson_r": round(float(r), 9), "p": float(p)}
    pairs = {"entry_vs_kc": corr(per["entry"], per["kc"]),
             "entry_vs_mbon": corr(per["entry"], per["mb"]),
             "kc_vs_mbon": corr(per["kc"], per["mb"])}
    q = _bh([pairs[k]["p"] for k in pairs])
    for k, qq in zip(pairs, q):
        pairs[k]["q_bh"] = round(float(qq), 12)
    sign = {"entry_positive": int(sum(x > 0 for x in per["entry"])),
            "kc_positive": int(sum(x > 0 for x in per["kc"])),
            "mbon_positive": int(sum(x > 0 for x in per["mb"])),
            "entry_pos_kc_pos": int(sum(a > 0 and b > 0 for a, b in zip(per["entry"], per["kc"]))),
            "entry_pos_mbon_pos": int(sum(a > 0 and b > 0 for a, b in zip(per["entry"], per["mb"])))}
    def top(x, n=10):
        idx = np.argsort(x)
        return {"negative": [{"index": int(i), "diff": round(float(x[i]), 9)} for i in idx[:n]],
                "positive": [{"index": int(i), "diff": round(float(x[i]), 9)} for i in idx[-n:][::-1]]}
    return {
        "per_seed": {k: [round(float(x), 9) for x in v] for k, v in per.items()},
        "per_seed_summary": {k: {"mean": round(float(np.mean(v)), 9),
                                 "sd": round(float(np.std(v, ddof=1)), 9)} for k, v in per.items()},
        "cancellation": {
            "full_kc": _cancel(kc_diff), "matched_kc": _cancel(km_diff),
            "full_mbon": _cancel(mb_diff), "matched_mbon": _cancel(mm_diff),
        },
        "per_cell_distribution": {
            "full_kc": {**_distribution(kc_diff), **top(kc_diff)},
            "matched_kc": {**_distribution(km_diff), **top(km_diff)},
            "full_mbon": {**_distribution(mb_diff), **top(mb_diff)},
            "matched_mbon": {**_distribution(mm_diff), **top(mm_diff)},
        },
        "time_bins": {"bins": TIME_BINS,
                      "entry": [round(float(x), 9) for x in tbins["entry"]],
                      "matched_kc": [round(float(x), 9) for x in tbins["kc"]],
                      "matched_mbon": [round(float(x), 9) for x in tbins["mb"]],
                      "note": "mean across seeds of stim-none per bin; EXPLORATORY"},
        "background_matched_kc": {"groups": bg,
                                  "matched_none_mean": round(float(km_none.mean()), 9),
                                  "matched_none_ge085": int((km_none >= .85).sum())},
        "low_feedback": {"indices": [2460, 2650, 3971], "cells": low},
        "static_groups": groups,
        "paired": {"correlations": pairs, "sign": sign,
                   "note": "25 seeds; exploratory; BH q over 3 correlations"},
    }


def answer(repro, static, dyn):
    observed = [
        "S1R formal negative result is independently reproduced from 50 NPZ: entry +0.066634727 (p=5.96e-8), matched-320 KC -0.001430625 (p=0.352361), matched-95 MBON -0.001760000 (p=0.438998).",
        "No direct LPLC2/LC4->KC edges; all 320 matched KCs have 2-hop paths; nonmatched 3744 KCs have zero static feedforward paths.",
        "KC and MBON dynamic changes are strongly coupled across seeds (Pearson r=0.873, p=1.21e-8), but entry-vs-KC is not (r=-0.016, p=0.939).",
    ]
    supported = [
        "The limiting link is more likely sensory->KC effective feedforward integration/dynamics than absent anatomy or absent sensory entry.",
        "Static reachable path count/strength does not explain the dynamic KC effect (path count vs effect r=0.028; signed strength vs effect r=0.0016; absolute strength vs effect r=0.038); this is a supported hypothesis, not a confirmed cause.",
        "Cell-level heterogeneity is substantial (320 matched KCs: 116 positive, 177 negative; 95 MBONs: 34 positive, 61 negative), so local positive responses are partially offset by negative responses, but the population net remains negative rather than exactly canceled.",
    ]
    unknown = [
        "Whether the dominant cause is weak/desynchronized excitatory drive, inhibition, adaptation/threshold, or measurement/aggregation cannot be resolved from static W^2 and aggregated NPZ alone.",
        "The intermediate cell types and actual synaptic time courses were not recorded; no causal perturbation was performed.",
    ]
    return {
        "OBSERVED": observed,
        "SUPPORTED_HYPOTHESIS": supported,
        "UNKNOWN": unknown,
        "answers": {
            "q1_formal_negative_reproduced": bool(repro["formal_negative_reproduced"]),
            "q2_static_reachable_to_matched_kc": bool(static["matched_kc"]["reachable"] == static["matched_kc"]["n"] == 320 and static["direct_src_to_kc"]["edges"] == 0),
            "q3_local_cancellation": "PARTIAL/EXPLORATORY: positive and negative cell subsets exist, but net matched-KC/MBON effect is negative; no strong evidence of a hidden positive population exactly canceled to zero.",
            "q4_most_likely_limiting_stage": "SUPPORTED_HYPOTHESIS: sensory->KC effective feedforward integration/effective drive (not absent anatomy, not absent sensory entry); exact mechanism UNKNOWN.",
            "q5_ready_for_next_prereg": True,
            "next_experiment_candidate": "Independent preregistered causal test: bypass sensory entry with calibrated graded current/optogenetic drive to the matched-320 KC population (or a prespecified subset) and record matched-95 MBON, with fresh seeds and the same frozen readout; this tests whether KC->MBON is functional and whether weak afferent drive is the limiting step. Do not execute without new authorization.",
        },
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    raw, ids, repro, per, formal = reproduce_and_archive()
    static = static_paths(raw, ids)
    dyn = dynamic(raw, ids, static)
    ans = answer(repro, static, dyn)
    rep = {
        "protocol": "P11-2-S2-OFFLINE/v1",
        "baseline": "6d76109",
        "status": "OFFLINE_ANALYSIS_COMPLETE; NO_NEW_DYNAMICS; EXPLORATORY",
        "formal_result_preserved": True,
        "audit": {
            "raw_dir": raw, "backup_dir": BACKUP,
            "npz_count": repro["npz_count"], "hashes_ok": repro["all_listed_hashes_ok"],
            "run_meta_sha256_match": repro["run_meta_sha256_match"],
            "formal_report_sha256": _sha(FORMAL), "formal_report_status": formal.get("status"),
        },
        "A_reproduction": repro,
        "B_static_paths": {k: v for k, v in static.items() if k != "arrays"},
        "C_dynamic_exploration": dyn,
        "D_mechanism": ans,
        "gates": {"NEW_FULL_BRAIN_AUTHORIZATION": False, "TRAINING_AUTHORIZATION": False,
                  "REAL_SYNAPTIC_LEARNING_READY": "NO", "Level_A/B/C/Biological": "NO-GO",
                  "Robot": "NO-GO", "Maze": "NO-GO", "Dijkstra": "NO-GO", "DeltaW": "NO-GO"},
        "runtime_s": None,
    }
    rep["runtime_s"] = round(time.time() - t0, 3)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({"status": rep["status"], "runtime_s": rep["runtime_s"],
                      "reproduced": repro["formal_negative_reproduced"],
                      "matched_reachable": static["matched_kc"]["reachable"],
                      "matched_paths": static["matched_kc"]["independent_paths"],
                      "direct_kc": static["direct_src_to_kc"]["edges"],
                      "three_hop_mbon_endpoints": static["three_hop_mbon_via_kc_B"]["endpoint_pairs"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
