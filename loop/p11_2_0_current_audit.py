"""P11-2-0：KC 高发放的理论电流预算、W 只读统计、来源勘误与 A1 准入（静态/数学）。

**不运行 FlyBrain.step、不创建真实神经动力学、不做单神经元时间演化。**
仅读安装源码、官方 W（只读）、归档报告；W 不修改；ΔW=0。
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
REPORT = os.path.join(HERE, "p11_2_0_report.json")
EXP_REPORT = os.path.join(HERE, "p11_1c_experiment_report.json")
DT = 0.020
TAU = 0.100
GAIN = 3.0
TONIC = 0.14
NOISE_HZ = 1.2
NOISE_AMP = 0.22
THRESHOLD = 1.0


def source_audit():
    p = os.path.join(ROOT, ".venv", "Lib", "site-packages", "flybrain", "brain.py")
    t = open(p, encoding="utf-8").read()
    build = open(os.path.join(ROOT, ".venv", "Lib", "site-packages", "flybrain", "build.py"),
                 encoding="utf-8").read()
    return {
        "decay_source": "self.decay = np.float32(np.exp(-self.dt / self.tau))" in t,
        "gain_applied_to_synaptic_current": "current = self.synaptic_input(self.fired) * self.gain" in t,
        "decay_before_add": "self.v *= self.decay" in t,
        "tonic_added_with_current": "self.v += current + self.tonic" in t,
        "noise_bernoulli": "self.rng.random((self.n, B)) < self.noise_hz * self.dt" in t,
        "noise_amplitude_float32": "* np.float32(self.noise_amp)" in t,
        "threshold_ge_1": "fired = xp.flatnonzero(self.v >= 1.0)" in t,
        "reset_zero": "self.v.ravel()[fired] = 0.0" in t,
        "synaptic_uses_previous_fired": "self.synaptic_input(self.fired)" in t,
        "tonic_rescaled": "self.tonic = type(self).tonic * (1 - np.exp(-self.dt / self.tau))" in t,
        "postsynaptic_L1_normalization_source": "w /= np.maximum(incoming[post], 1.0)" in build,
        "params": {"dt": DT, "tau": TAU, "gain": GAIN, "tonic": TONIC,
                   "noise_hz": NOISE_HZ, "noise_amp": NOISE_AMP, "threshold": THRESHOLD},
        "update_order": "current=W@prev_fired*gain; v*=decay; v+=current+tonic; v+=noise; fire if v>=1; reset 0",
    }


def math_budget():
    decay = math.exp(-DT / TAU)
    tonic_fp = TONIC / (1 - decay)
    single_step_current = THRESHOLD - TONIC
    sustained_current = (1 - decay) - TONIC
    noise_p = NOISE_HZ * DT
    mean_noise = noise_p * NOISE_AMP
    tonic_noise_fp = (TONIC + mean_noise) / (1 - decay)
    return {
        "decay": round(decay, 9),
        "tonic_only_fixed_point": round(tonic_fp, 6),
        "tonic_alone_reaches_threshold": bool(tonic_fp >= THRESHOLD),
        "single_step_input_from_reset_post_gain": round(single_step_current, 6),
        "single_step_input_from_reset_raw_Wx": round(single_step_current / GAIN, 6),
        "sustained_subthreshold_current_post_gain": round(sustained_current, 6),
        "sustained_subthreshold_raw_Wx": round(sustained_current / GAIN, 6),
        "every_step_firing_current_post_gain": round(THRESHOLD - TONIC, 6),
        "every_step_firing_raw_Wx": round((THRESHOLD - TONIC) / GAIN, 6),
        "noise_probability_per_step": round(noise_p, 6),
        "noise_mean_voltage_per_step": round(mean_noise, 6),
        "tonic_noise_mean_fixed_point": round(tonic_noise_fp, 6),
        "tonic_noise_alone_reaches_threshold_in_expectation": bool(tonic_noise_fp >= THRESHOLD),
        "note": "mathematical necessary conditions under no recurrence/no noise (or mean noise); not causal results",
    }


def kc_budget():
    W = sp.load_npz(os.path.join(DATA_DIR, "weights.npz")).tocsr()
    b = np.load(os.path.join(DATA_DIR, "brain.npz"), allow_pickle=True)
    ct = b["cell_type"].astype(str)
    kc = np.flatnonzero(np.char.startswith(ct, "KC"))
    sub = W[kc]
    pos = np.empty(len(kc)); neg = np.empty(len(kc)); edges = np.empty(len(kc), dtype=int)
    for i in range(len(kc)):
        a = sub.data[sub.indptr[i]:sub.indptr[i + 1]]
        pos[i] = float(a[a > 0].sum()); neg[i] = float(-a[a < 0].sum()); edges[i] = len(a)
    l1 = pos + neg
    th = (THRESHOLD - TONIC) / GAIN
    coo = sub.tocoo(); ctypes = ct[coo.col]
    agg = {}
    for typ, v in zip(ctypes, coo.data):
        a = agg.setdefault(typ, [0.0, 0.0, 0]); a[0 if v > 0 else 1] += float(v); a[2] += 1
    top = sorted(agg.items(), key=lambda kv: -(abs(kv[1][0]) + abs(kv[1][1])))[:12]
    return {
        "n_KC": int(len(kc)), "n_edges_total": int(sub.nnz),
        "pos_sum": {"mean": round(float(pos.mean()), 6), "median": round(float(np.median(pos)), 6),
                    "min": round(float(pos.min()), 6), "max": round(float(pos.max()), 6)},
        "neg_abs": {"mean": round(float(neg.mean()), 6), "median": round(float(np.median(neg)), 6),
                    "min": round(float(neg.min()), 6), "max": round(float(neg.max()), 6)},
        "L1": {"mean": round(float(l1.mean()), 6), "median": round(float(np.median(l1)), 6)},
        "edges": {"mean": round(float(edges.mean()), 2), "median": int(np.median(edges)),
                  "min": int(edges.min()), "max": int(edges.max()), "zero_rows": int((edges == 0).sum())},
        "every_step_raw_threshold": round(th, 6),
        "n_pos_ge_threshold": int((pos >= th).sum()),
        "frac_pos_ge_threshold": round(float((pos >= th).mean()), 6),
        "max_postgain_current_from_static_budget": {"min": round(float(GAIN * pos.min()), 4),
                                                     "max": round(float(GAIN * pos.max()), 4)},
        "static_weights_can_reach_every_step_threshold": bool((pos >= th).all()),
        "actual_Wx_unknown": "full 166,700 step-wise spikes were not saved; cannot reconstruct W@spikes",
        "top_presynaptic_types": [{"type": str(k), "pos_sum": round(v[0], 3),
                                   "neg_sum": round(v[1], 3), "edges": v[2]} for k, v in top],
        "W_rows_L1_consistent": bool(abs(float(l1.mean()) - 1.0) < 1e-3),
    }


def source_errata():
    cur = os.path.join(HERE, "p11_1c_experiment.py")
    raw = open(cur, "rb").read()
    lf = hashlib.sha256(raw).hexdigest()
    crlf = hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()
    rep = json.load(open(EXP_REPORT, encoding="utf-8"))
    rh = rep.get("script_sha256")
    return {
        "current_LF_sha256": lf, "current_CRLF_sha256": crlf,
        "report_script_sha256": rh,
        "report_matches_current_LF": rh == lf,
        "report_matches_current_CRLF": rh == crlf,
        "report_hash_timing": "computed in compare-only post-processing, AFTER the experiment and AFTER parent-side patches",
        "run_commit": rep.get("run_commit"), "result_commit": "f4176df",
        "conclusion": ("the report hash equals the CRLF form of the CURRENT (post-patch) file; it does not prove the "
                       "run-time source bytes. The experiment child ran the pre-patch script; the hash was taken later. "
                       "SOURCE_PROVENANCE remains PARTIAL."),
        "errata_recorded": True, "historical_report_unchanged": True,
    }


def a1_proposal():
    return {
        "id": "P11-2-A1", "status": "PROPOSAL ONLY; NOT AUTHORIZED; NOT RUN",
        "goal": "no-recurrent single-neuron control to test whether tonic and/or noise can produce firing at the model parameters",
        "conditions": ["tonic only, noise off", "tonic + noise on (original noise_hz/noise_amp)"],
        "parameters": {"dt": DT, "tau": TAU, "tonic": TONIC, "noise_hz": NOISE_HZ,
                       "noise_amp": NOISE_AMP, "threshold": THRESHOLD, "gain": "irrelevant (no synaptic input)"},
        "sample": "pre-fixed: 100 independent realizations x 10,000 steps (1,000,000 steps total)",
        "criteria": {
            "tonic_only": "expected 0 spikes (fixed point 0.772 < 1)",
            "tonic_noise": "expected sparse, non-sustained firing; if observed rate is high/continuous -> single-neuron noise/tonic mechanism can explain saturation",
            "decision": "pre-registered: report firing probability per step and longest consecutive run; no parameter search",
        },
        "not_allowed": ["choosing gain/tonic/noise to make KC look better", "physiological calibration claim",
                        "A2/A3 zero-recurrent/shuffled W", "phase B sensitivity", "training/dW/Dijkstra/B/C"],
        "authorization": "requires a separate written approval for one A1 run; A2/A3/B remain NO-GO",
    }


def verdicts():
    return {
        "TONIC_ALONE_REACHES_THRESHOLD": "NO",
        "EVERY_STEP_RAW_Wx_THRESHOLD": round((THRESHOLD - TONIC) / GAIN, 6),
        "STATIC_WEIGHTS_SUFFICIENT_IN_PRINCIPLE": "YES (all 4,064 KC pos_sum >= 0.2867)",
        "ACTUAL_Wx_CONTRIBUTION": "UNKNOWN (no full spike history)",
        "MECHANISMS_MATHEMATICALLY_EXCLUDED": ["tonic alone", "tonic+mean noise alone (mean fixed point 0.8008 < 1)"],
        "CANDIDATE_MECHANISMS": ["recurrent KC->KC excitation (KCab/KCg-m) with gain 3.0",
                                 "noise as a trigger", "APL inhibition balance", "step cap 1 spike/step"],
        "MODEL_ROOT_CAUSE": "NOT_ESTABLISHED",
        "SOURCE_PROVENANCE": "PARTIAL",
        "A1_WORTH_APPLYING": "YES (separate one-run authorization; low cost, decisive for tonic/noise vs recurrent)",
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
    src = source_audit(); mb = math_budget(); kb = kc_budget()
    err = source_errata(); a1 = a1_proposal(); v = verdicts()
    hashes = w_readonly_check()
    report = {"protocol": "P11-2-0/v1", "source_audit": src, "math_budget": mb,
              "kc_weight_budget": kb, "source_errata": err, "a1_proposal": a1,
              "official_hashes": {"ok": hashes.get("ok"),
                                  "files": {k: x.get("sha256") for k, x in hashes.get("files", {}).items()}},
              "verdicts": v, "training_authorized": False}
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else bool(o) if isinstance(o, np.bool_) else str(o))
    print("[P11-2-0] source checks:", all(src[k] for k in ("decay_source", "gain_applied_to_synaptic_current",
                                                           "threshold_ge_1", "reset_zero", "noise_bernoulli")))
    print("  decay:", mb["decay"], "| tonic fp:", mb["tonic_only_fixed_point"],
          "| single-step raw:", mb["single_step_input_from_reset_raw_Wx"],
          "| every-step raw:", mb["every_step_firing_raw_Wx"])
    print("  KC pos mean/min/max:", kb["pos_sum"]["mean"], kb["pos_sum"]["min"], kb["pos_sum"]["max"],
          "| all >= threshold:", kb["static_weights_can_reach_every_step_threshold"])
    print("  top types:", [(x["type"], x["pos_sum"], x["neg_sum"]) for x in kb["top_presynaptic_types"][:5]])
    print("  hashes LF/CRLF/report:", err["current_LF_sha256"][:12], err["current_CRLF_sha256"][:12],
          (err["report_script_sha256"] or "")[:12], "| match CRLF:", err["report_matches_current_CRLF"])
    print("  verdicts:", v)
    print("wrote p11_2_0_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
