"""P11-0：P10-C2-A0 离线诊断（高基线 / KC 细胞级 / 资格迹 / 源码审计）。

**只读已归档数据与源码**；不运行神经动力学、不重跑 A0、不改参数/W/ΔW、不训练、不进入 B/C。
所有结果均为事后探索性描述，不改写冻结的 A0 结论。
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time

import numpy as np

from loop.p10c2_0b_exact_stats import sign_flip_p

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARCHIVE_MANIFEST = os.path.join(HERE, "p10c2_a0_archive_manifest.json")
REPORT = os.path.join(HERE, "p11_0_report.json")
SEEDS = list(range(101, 126))
CONDITIONS = ("none", "stim", "recover")
STEPS = 400
PRE = (0, 100)
STIM = (100, 300)
POST = (300, 400)
NEAR_ZERO = 0.01          # 描述性“近零”阈值，非显著性阈值
BH_Q = 0.05
FDR_MAX_S = 180.0


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_archive():
    if not os.path.exists(ARCHIVE_MANIFEST):
        return {"ok": False, "status": "RAW_DATA_UNAVAILABLE", "reason": "manifest missing"}
    man = json.load(open(ARCHIVE_MANIFEST, encoding="utf-8"))
    a = man.get("archive", {})
    d = a.get("archive_dir")
    if not d or not os.path.isdir(d):
        return {"ok": False, "status": "RAW_DATA_UNAVAILABLE", "reason": "archive dir missing"}
    per = a.get("per_file", {})
    missing, mismatch = [], []
    for n, meta in per.items():
        p = os.path.join(d, n)
        if not os.path.exists(p):
            missing.append(n)
        elif meta.get("sha256") != _sha(p):
            mismatch.append(n)
    files = sorted(os.listdir(d))
    integrity = {"file_count": len(files), "npz": sum(f.endswith(".npz") for f in files),
                 "records": "records.json" in files, "missing": missing, "mismatch": mismatch}
    if missing or mismatch or integrity["file_count"] != 76:
        return {"ok": False, "status": "RAW_DATA_UNAVAILABLE", "integrity": integrity}
    data = {}
    for s in SEEDS:
        for c in CONDITIONS:
            z = np.load(os.path.join(d, f"{s}_{c}.npz"), allow_pickle=True)
            data[(s, c)] = {"lp": z["lp"], "kc": z["kc"], "mb": z["mb"],
                            "elig": z["eligibility"], "elig_mean": z["eligibility_mean"]}
    return {"ok": True, "status": "ARCHIVED_VERIFIED", "archive_dir": d, "integrity": integrity,
            "data": data}


def _stats(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "q1": float(np.percentile(x, 25)), "q3": float(np.percentile(x, 75)),
            "min": float(x.min()), "max": float(x.max())}


def _fractions(kc, c):
    return (kc[PRE[0]:PRE[1]].mean(0), kc[STIM[0]:STIM[1]].mean(0), kc[POST[0]:POST[1]].mean(0))


def _concentration(counts):
    total = float(counts.sum())
    if total <= 0:
        return {"top10_share": None, "n_cells_80pct": None, "n_cells": int(len(counts))}
    sd = np.sort(counts)[::-1]
    k = max(1, int(math.ceil(0.1 * len(sd))))
    cum = np.cumsum(sd) / total
    n80 = int(np.searchsorted(cum, 0.8) + 1)
    return {"top10_share": float(sd[:k].sum() / total), "n_cells_80pct": n80, "n_cells": int(len(counts))}


def high_baseline(data):
    out = {}
    for label, key in (("KC", "kc"), ("MBON", "mb")):
        per_seed = {"none": [], "stim": []}
        conc = {"none": [], "stim": []}
        for s in SEEDS:
            for c in ("none", "stim"):
                arr = data[(s, c)][key]
                fr = _fractions(arr, c)
                per_seed[c].append({"pre": fr[0], "stim": fr[1], "post": fr[2]})
                conc[c].append(_concentration(arr[STIM[0]:STIM[1]].sum(0)))
        res = {}
        for c in ("none", "stim"):
            w = {}
            for wi, name in enumerate(("pre", "stim", "post")):
                mats = np.array([pse[name] for pse in per_seed[c]])  # seeds x neurons
                pop = mats.mean(0)
                w[name] = {"population_mean": _stats(pop),
                           "per_seed_mean_range": [float(mats.mean(1).min()), float(mats.mean(1).max())],
                           "frac_ge_0.85": float(np.mean([(m >= 0.85).mean() for m in mats])),
                           "frac_ge_0.95": float(np.mean([(m >= 0.95).mean() for m in mats])),
                           "frac_eq_1": float(np.mean([(m >= 0.999).mean() for m in mats]))}
            res[c] = w
            res[c]["concentration"] = {"top10_share_mean": float(np.mean([x["top10_share"] for x in conc[c]])),
                                       "n_cells_80pct_mean": float(np.mean([x["n_cells_80pct"] for x in conc[c]])),
                                       "n_cells": conc[c][0]["n_cells"]}
            # time course, 20-step bins, population mean
            tc = []
            for b in np.array_split(np.arange(STEPS), 20):
                tc.append(float(np.mean([data[(s, c)][key][b].mean() for s in SEEDS])))
            res[c]["time_course_20step"] = tc
        out[label] = res
    return out


def _bh(pvals, q=BH_Q):
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool), None
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * np.arange(1, n + 1) / n
    below = ranked <= thresh
    if not below.any():
        return np.zeros(n, bool), None
    k = int(np.max(np.where(below)[0]))
    cutoff = float(ranked[k])
    return p <= cutoff, cutoff


def kc_cell_level(data):
    diffs = []
    for s in SEEDS:
        d = (data[(s, "stim")]["kc"][STIM[0]:STIM[1]].mean(0)
             - data[(s, "none")]["kc"][STIM[0]:STIM[1]].mean(0))
        diffs.append(d)
    D = np.array(diffs)  # seeds x neurons
    mean_d = D.mean(0)
    pos = int((mean_d > NEAR_ZERO).sum()); neg = int((mean_d < -NEAR_ZERO).sum())
    near = int((np.abs(mean_d) <= NEAR_ZERO).sum())
    sd = float(np.std(mean_d, ddof=1)) if len(mean_d) > 1 else 0.0
    pvals = np.array([sign_flip_p(D[:, j]) for j in range(D.shape[1])])
    sig, cutoff = _bh(pvals)
    sign_counts = (D > 0).sum(0)
    return {
        "n": int(D.shape[1]), "positive": pos, "negative": neg, "near_zero": near,
        "near_zero_threshold": NEAR_ZERO,
        "effect_mean": _stats(mean_d), "dz": (float(mean_d.mean() / sd) if sd > 0 else None),
        "sum_positive": float(mean_d[mean_d > 0].sum()), "sum_negative": float(mean_d[mean_d < 0].sum()),
        "net": float(mean_d.sum()),
        "cancellation_present": bool(float(mean_d[mean_d > 0].sum()) > 0
                                     and float(mean_d[mean_d < 0].sum()) < 0
                                     and abs(float(mean_d.sum())) < 0.5 * abs(float(mean_d[mean_d > 0].sum()))),
        "sign_consistency": {"all_25_positive": int((sign_counts == len(SEEDS)).sum()),
                             "all_25_negative": int((sign_counts == 0).sum()),
                             "ge20_positive": int((sign_counts >= 20).sum()),
                             "ge20_negative": int((sign_counts <= len(SEEDS) - 20).sum())},
        "bh_fdr": {"q": BH_Q, "n_significant": int(sig.sum()), "cutoff": cutoff,
                   "exploratory": True, "n_tests": int(len(pvals))},
        "note": "exploratory; no responsive-subset replacement of the primary KC endpoint",
    }


def eligibility_distribution(data):
    diffs, bg = [], []
    final_binary = {"none": [], "stim": []}
    tcourse = {"none": [], "stim": []}
    for s in SEEDS:
        es = data[(s, "stim")]["elig"]; en = data[(s, "none")]["elig"]
        d = es[STIM[0]:STIM[1]].mean(0) - en[STIM[0]:STIM[1]].mean(0)
        diffs.append(d); bg.append(en[STIM[0]:STIM[1]].mean(0))
        for c, arr in (("none", en), ("stim", es)):
            final_binary[c].append(float((arr[-1] > 1e-6).mean()))
            tcourse[c].append(np.array([float(arr[b].mean()) for b in np.array_split(np.arange(STEPS), 20)]))
    D = np.array(diffs); B = np.array(bg)
    mean_d = D.mean(0)
    pvals = np.array([sign_flip_p(D[:, j]) for j in range(D.shape[1])])
    sig, cutoff = _bh(pvals)
    return {
        "n_edges": int(D.shape[1]),
        "background_amplitude": {"mean": float(B.mean()), "median": float(np.median(B)),
                                 "q05": float(np.percentile(B, 5)), "q95": float(np.percentile(B, 95))},
        "paired_change": {"mean": float(mean_d.mean()), "median": float(np.median(mean_d)),
                          "q05": float(np.percentile(mean_d, 5)), "q95": float(np.percentile(mean_d, 95)),
                          "positive": int((mean_d > 0).sum()), "negative": int((mean_d < 0).sum()),
                          "sum_positive": float(mean_d[mean_d > 0].sum()),
                          "sum_negative": float(mean_d[mean_d < 0].sum()), "net": float(mean_d.sum())},
        "final_binary_fraction": {"none_mean": float(np.mean(final_binary["none"])),
                                  "stim_mean": float(np.mean(final_binary["stim"])),
                                  "note": "binary e>1e-6 fraction; NOT amplitude saturation"},
        "time_course_none": np.mean(tcourse["none"], axis=0).round(6).tolist(),
        "time_course_stim": np.mean(tcourse["stim"], axis=0).round(6).tolist(),
        "bh_fdr": {"q": BH_Q, "n_significant": int(sig.sum()), "cutoff": cutoff,
                   "exploratory": True, "n_tests": int(len(pvals))},
    }


def source_audit():
    brain_py = os.path.join(ROOT, ".venv", "Lib", "site-packages", "flybrain", "brain.py")
    build_py = os.path.join(ROOT, ".venv", "Lib", "site-packages", "flybrain", "build.py")
    text = open(brain_py, encoding="utf-8").read() if os.path.exists(brain_py) else ""
    btext = open(build_py, encoding="utf-8").read() if os.path.exists(build_py) else ""
    def num(key):
        m = re.search(rf"^\s*{key}\s*=\s*([0-9.]+)", text, re.M)
        return float(m.group(1)) if m else None
    params = {k: num(k) for k in ("dt", "tau", "gain", "tonic", "noise_hz", "noise_amp", "eye_gain")}
    p10c1b = json.load(open(os.path.join(HERE, "p10c1b_report.json"), encoding="utf-8"))
    trunc = p10c1b["benchmark"]["verification"]["truncation"]
    ev = json.load(open(os.path.join(HERE, "p100_evidence_review.json"), encoding="utf-8"))
    r = ev.get("review", {})
    return {
        "model": "normalized LIF (flybrain.FlyBrain); dt=0.020 s; v threshold 1.0; reset 0; refractory default 0",
        "confirmed_model_params": params,
        "postsynaptic_normalization": ("w /= np.maximum(incoming[post], 1.0)" in btext),
        "threshold_source": ("v >= 1.0" in text), "reset_source": ("self.v.ravel()[fired] = 0.0" in text),
        "truncation": trunc,
        "lplc2_lc4_to_kc": {"direct_edges": r.get("LP_direct_to_KC", {}).get("edges"),
                            "two_hop_intermediates": r.get("LP_2hop_to_KC", {}).get("n_intermediate"),
                            "two_hop_sum_w_out": r.get("LP_2hop_to_KC", {}).get("sum_w_out")},
        "observed_high_baseline": "KC ~0.91, MBON ~0.88 per-step spike probability (not calibrated Hz)",
        "computational_assumptions_may_raise_baseline": [
            "gain=3.0 and tonic=0.14 (hand-calibrated, not measured)",
            "noise_hz=1.2 / noise_amp=0.22",
            "truncated subnet removes ~4.2M incoming edges / Σw~4238 (external inhibition/excitation)",
            "per-postsynaptic L1 normalization concentrates weights",
            "dt=0.020 s step itself caps rates (max 1 spike/step)",
        ],
        "undetermined": ["relative contribution of gain/tonic/noise vs truncation vs connectome",
                         "whether the high baseline is a model artifact or a plausible biological regime",
                         "causal root cause"],
        "root_cause": "NOT_ESTABLISHED",
    }


def verdicts(high, kc, elig, audit):
    kc_none = high["KC"]["none"]["stim"]
    frac85 = kc_none["frac_ge_0.85"]
    high_state = ("SUPPORTED" if frac85 >= 0.5 else ("LIMITED" if frac85 >= 0.2 else "INCONCLUSIVE"))
    het_state = ("OBSERVED" if (kc["positive"] >= 16 and kc["negative"] >= 16 and kc["cancellation_present"])
                 else ("NOT_OBSERVED" if (kc["positive"] < 16 and kc["negative"] < 16) else "INCONCLUSIVE"))
    return {
        "P10C2_A0_Entrance": "PASS",
        "P10C2_A0_KC_Main_Endpoint": "NOT_DETECTED",
        "P10C2_A0_Ceiling": "FAIL",
        "P10C2_A0_Canonical": "INCONCLUSIVE",
        "Level_A_Training": "NO-GO", "Level_B": "NO-GO", "Level_C": "NO-GO",
        "Biological": "NO-GO", "Training_Authorization": False,
        "HIGH_BASELINE_PHENOTYPE": high_state,
        "CELL_LEVEL_HETEROGENEITY": het_state,
        "MODEL_ROOT_CAUSE": "NOT_ESTABLISHED",
        "worth_new_preregistered_model_validity_experiment": bool(high_state == "SUPPORTED"),
        "note": "post-hoc descriptive diagnosis; does not rewrite the frozen A0; no responsive-subset reclassification",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    arc = load_archive()
    if not arc.get("ok"):
        json.dump({"status": arc.get("status"), "integrity": arc.get("integrity")},
                  open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("RAW_DATA_UNAVAILABLE / integrity:", arc)
        return 1
    data = arc["data"]
    high = high_baseline(data)
    kc = kc_cell_level(data)
    elig = eligibility_distribution(data)
    audit = source_audit()
    ver = verdicts(high, kc, elig, audit)
    report = {"archive": {"status": arc["status"], "dir": arc["archive_dir"],
                          "integrity": arc["integrity"]},
              "high_baseline": high, "kc_cell_level": kc,
              "eligibility_distribution": elig, "source_audit": audit,
              "verdicts": ver, "training_authorized": False}
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else bool(o) if isinstance(o, np.bool_) else str(o))
    print("[P11-0] archive:", arc["status"], "files:", arc["integrity"]["file_count"])
    print("  KC none mean:", high["KC"]["none"]["stim"]["population_mean"]["mean"],
          "frac>=0.85:", high["KC"]["none"]["stim"]["frac_ge_0.85"],
          "top10_share:", high["KC"]["none"]["concentration"]["top10_share_mean"])
    print("  KC cells pos/neg/near:", kc["positive"], kc["negative"], kc["near_zero"],
          "| BH sig:", kc["bh_fdr"]["n_significant"])
    print("  eligibility change mean:", elig["paired_change"]["mean"],
          "pos/neg:", elig["paired_change"]["positive"], elig["paired_change"]["negative"],
          "| BH sig:", elig["bh_fdr"]["n_significant"])
    print("  verdicts:", ver)
    print("wrote p11_0_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
