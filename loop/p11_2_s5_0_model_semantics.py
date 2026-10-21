"""P11-2-S5-0: static audit of FlyBrain LIF/drive semantics.

Reads source and connectome matrices; never calls FlyBrain.step.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp

import flybrain.brain as fb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
S3R = os.path.join(HERE, "p11_2_s3_report.json")
S3M = os.path.join(HERE, "p11_2_s3_run_meta.json")
S4R = os.path.join(HERE, "p11_2_s4_report.json")
S4MD = os.path.join(HERE, "P11_2_S4_OFFLINE_DIAGNOSIS.md")
OUT = os.path.join(HERE, "p11_2_s5_0_report.json")
DT = 0.020
TAU = 0.100
TONIC = 0.14
GAIN = 3.0
NOISE_HZ = 1.2
NOISE_AMP = 0.22
PRIMARY_AMOUNT = 0.03625384938440364
LOW_AMOUNT = 0.01812692469220182


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _line(src, needle, start=0):
    lines = src.splitlines()
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i + 1
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    src = open(fb.__file__, encoding="utf-8").read()
    s3 = _load(S3R); s3m = _load(S3M); s4 = _load(S4R)
    s4md = open(S4MD, encoding="utf-8").read()
    decay = math.exp(-DT / TAU)
    sham = TONIC / (1 - decay)
    low = (TONIC + LOW_AMOUNT) / (1 - decay)
    primary = (TONIC + PRIMARY_AMOUNT) / (1 - decay)
    # static connectome feedback
    W = sp.load_npz(os.path.join(DATA_DIR, "weights.npz")).tocsr()
    b = np.load(os.path.join(DATA_DIR, "brain.npz"), allow_pickle=True)
    ct = b["cell_type"].astype(str)
    kc = np.flatnonzero(np.char.startswith(ct, "KC"))
    is_kc = np.zeros(W.shape[0], bool); is_kc[kc] = True
    nonkc = ~is_kc
    Wb = W.copy(); Wb.data = np.ones_like(Wb.data)
    coo = Wb.tocoo(); keep = ~(is_kc[coo.row] & is_kc[coo.col])
    WbB = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=W.shape).tocsr()
    direct_kk = int(WbB[kc][:, kc].nnz)
    endpoints = 0; paths = 0.0
    for start in range(0, len(kc), 512):
        rows = kc[start:start + 512]
        Q = (WbB[rows][:, nonkc] @ WbB[nonkc][:, kc]).tocoo()
        endpoints += int(Q.nnz); paths += float(Q.data.sum())
    # Hamming-style output: all old API/line facts
    lines = {
        "decay": _line(src, "self.decay = np.float32"),
        "step": _line(src, "def step("),
        "current": _line(src, "current = self.synaptic_input(self.fired) * self.gain"),
        "leak": _line(src, "self.v *= self.decay"),
        "tonic": _line(src, "self.v += current + self.tonic"),
        "noise": _line(src, "self.rng.random((self.n, B))"),
        "inject": _line(src, "for idx, amount in inject:"),
        "threshold": _line(src, "fired = xp.flatnonzero(self.v >= 1.0)"),
        "reset": _line(src, "self.v.ravel()[fired] = 0.0"),
        "reset_seed": _line(src, "self.rng = xp.random.default_rng(seed)"),
        "step_return": _line(src, "return flat"),
        "synaptic_input": _line(src, "def synaptic_input"),
        "reset_method": _line(src, "def reset("),
        "propgate": _line(src, "def _propagate"),
    }
    s4_lines = s4md.splitlines()
    s4_corrected = {
        "full_kc_negative": 2737, "other_kc_negative": 2734,
        "full_kc_positive": 1146, "other_kc_positive": 835,
        "full_kc_zero": 181, "other_kc_zero": 175,
        "positive_sum_check": 311 + 835, "negative_sum_check": 3 + 2734,
        "zero_sum_check": 6 + 175,
        "s4_markdown_has_2737_full": any("full-4064 KC" in x and "2737" in x for x in s4_lines),
        "s4_markdown_has_2734_other": any("other-3744 KC" in x and "2734" in x for x in s4_lines),
        "s4_json_sha256": _sha(S4R),
    }
    rep = {
        "protocol": "P11-2-S5-0/v1",
        "baseline": "68dca94",
        "status": "S5_0_SOURCE_SEMANTICS_COMPLETE; NO_NEW_DYNAMICS",
        "source": {
            "file": fb.__file__, "sha256": _sha(fb.__file__),
            "model": "LIF; v dimensionless; threshold 1.0; reset 0.0",
            "version": getattr(fb, "__version__", "0.1.0"),
            "lines": lines,
        },
        "step_semantics": {
            "order": "synaptic current from previous fired * gain; v*=decay; v+=current+tonic; v+=noise; optional eye; inject; refractory; fire if v>=1; reset fired to 0; record last_spike; steps++",
            "v_units": "dimensionless model units, not mV; threshold exactly 1.0; no mV conversion in source",
            "leak": {"decay": decay, "formula": "exp(-dt/tau)", "dt": DT, "tau": TAU},
            "tonic": TONIC,
            "tonic_rescaling": "in __init__, tonic is rescaled by (1-exp(-dt/tau))/(1-exp(-0.020/tau)); at dt=0.020 it stays 0.14",
            "gain": GAIN,
            "noise": {"hz": NOISE_HZ, "amp": NOISE_AMP,
                      "mechanism": "per-step Bernoulli mask rng.random((n,batch)) < hz*dt, add amp; not Gaussian"},
            "inject": {"index_space": "flat global neuron row indices into v",
                       "timing": "after leak/tonic/noise/eye, before refractory/fire/reset",
                       "unit": "voltage increment in the same dimensionless v units; not nA",
                       "per_step_primary": PRIMARY_AMOUNT, "per_step_low": LOW_AMOUNT},
            "spike": {"threshold": 1.0, "reset": 0.0,
                      "order": "fire indices from v>=1, then v.ravel()[fired]=0; pre-reset v is not stored"},
            "refractory_default": 0,
            "return": "fired indices only, not voltages",
            "synaptic_weights": {
                "orientation": "W[post, pre]; CPU CSC columns are presynaptic; _propagate sums outgoing weights of fired neurons into postsynaptic rows",
                "source_lines": {"propagate": lines["propgate"], "synaptic_input": lines["synaptic_input"]},
                "sign": "signed float32 weights; no abs/clipping in step",
                "normalization": "none in step; weights loaded directly; aggregate multiplied by gain=3.0",
                "units": "dimensionless model units, not nA or conductance",
            },
            "rng_reset": "reset(seed) zeroes v and sets default_rng(seed); S3 run_condition calls reset(seed) for every condition, so sham/low/primary use the same noise stream across 400 steps",
            "indirect_feedback": {"direct_kc_kc_edges_in_B": direct_kk,
                                  "nonKC_2hop_kc_to_kc_endpoints": int(endpoints),
                                  "nonKC_2hop_kc_to_kc_paths": int(paths),
                                  "exists": bool(direct_kk == 0 and endpoints > 0),
                                  "note": "static reachability only; longer paths not exhaustively enumerated; no dynamics"},
        },
        "static_math": {
            "formula": "v_inf = (tonic + a)/(1-exp(-dt/tau)); no synaptic/no noise/no spike/reset",
            "decay": decay, "tonic": TONIC,
            "sham": round(sham, 6), "low": round(low, 6), "primary": round(primary, 6),
            "amount_primary_formula_match": abs(PRIMARY_AMOUNT - 0.20 * (1 - decay)) < 1e-15,
            "amount_low_formula_match": abs(LOW_AMOUNT - 0.10 * (1 - decay)) < 1e-15,
            "expected_sham": 0.7723, "expected_low": 0.8723, "expected_primary": 0.9723,
            "matches": all(abs(x - y) < 5e-5 for x, y in ((sham, 0.7723), (low, 0.8723), (primary, 0.9723))),
            "caveat": "This is a no-spike/no-noise/no-synaptic fixed point; it does not predict firing-rate increment. Threshold 1.0 is above all three fixed points; noise/spikes/network can change outcomes.",
        },
        "observed_s3_s4": {
            "kc_primary_rate_delta": s3["kc_primary"]["mean"],
            "kc_low_rate_delta": s3["kc_low"]["mean"],
            "kc_primary_p": s3["kc_primary"]["p"],
            "mbon_primary_rate_delta": s3["mbon_primary"]["mean"],
            "mbon_primary_p": s3["mbon_primary"]["p"],
            "low_primary_ratio": s4["model_semantics"]["low_to_primary_rate_ratio"],
            "observed_is_not_voltage": True,
        },
        "biological_evidence": [
            {"component": "dt=0.020 s, tau=0.100 s", "source_assumption": "LIF discretization / fly64-style model",
             "biological_basis": "not calibrated here; source cites ornata/fly64 comparison, not measured fly membrane",
             "status": "UNVALIDATED_FOR_THIS_MODEL", "supported": "model-internal time constants", "not_supported": "biological membrane time constant claim"},
            {"component": "threshold=1.0, reset=0.0", "source_assumption": "normalized LIF threshold/reset",
             "biological_basis": "no experimental calibration in source", "status": "ENGINEERING_NORMALIZATION",
             "supported": "model-internal spike definition", "not_supported": "mV threshold or biological spike threshold"},
            {"component": "tonic=0.14, gain=3.0, noise_hz=1.2, noise_amp=0.22", "source_assumption": "hand-calibrated not measured",
             "biological_basis": "source docstring explicitly says hand-calibrated, not measured", "status": "ENGINEERING_HAND_CALIBRATED",
             "supported": "model-internal baseline/noise behavior", "not_supported": "biological firing-rate/noise parameters"},
            {"component": "sparse W", "source_assumption": "official connectome weights, signed; no normalization in step",
             "biological_basis": "anatomical connectivity only; sign/magnitude not physiological calibration",
             "status": "CONNECTOME_STRUCTURE_ONLY", "supported": "model-internal network topology/weights",
             "not_supported": "synaptic current/conductance or biological sign/magnitude claim"},
            {"component": "inject amount", "source_assumption": "direct voltage increment, not nA",
             "biological_basis": "engineering stimulus, no current-clamp calibration", "status": "ENGINEERING_STIMULUS",
             "supported": "model-internal voltage perturbation", "not_supported": "biological current/optogenetic dose equivalence"},
        ],
        "api_access": {
            "v_after_step": "brain.v is readable after step (post-reset for fired cells)",
            "pre_reset_v": "not stored; lost for fired cells",
            "component_breakdown": "step does not store leak/synaptic/tonic/noise/inject components",
            "synaptic_input": "brain.synaptic_input(brain.fired) can recompute aggregate synaptic current for the last fired set; no component history",
            "noise_trace": "Bernoulli mask is not stored",
            "instrumentation_required": True,
            "package_modification_required": False,
            "proposed": "subclass/wrapper of FlyBrain.step or a small callback copy that logs target KC/MBON (and preselected relays): v before, v after leak/tonic, synaptic current, noise sample, inject, v pre-threshold, v post-reset, fired, threshold distance",
        },
        "s4_correction": s4_corrected,
        "answers": {
            "q1_lif_vs_prereg": "YES at code level: the S3 injection amount is a direct voltage increment, and the static formula matches the frozen amount; this does not validate biological current semantics.",
            "q2_biological_parameters": "NO validated biological calibration for threshold/reset/tonic/gain/noise/tau; W is anatomical structure only. Source says tonic/gain/noise are hand-calibrated, not measured.",
            "q3_s3_evidence": "S3 proves a significant model-internal KC rate increase (+0.022858) under direct voltage, and low/primary ratio 0.508873; it cannot prove natural sensory drive, biological current equivalence, single-synapse causality, or downstream MBON function.",
            "q4_reliable_voltage_logging": "The current API alone is insufficient for pre-reset/component traces; a small-memory instrumented step is feasible and should be validated before any new full run.",
            "q5_next": "B first: pause stronger stimulation and validate model/biological-drive semantics with voltage/current instrumentation; only then pre-register a new dose study.",
        },
        "decision": {"priority": "B",
                     "reason": "Direct voltage injection and hand-calibrated dynamics; observed weak rate increment cannot be interpreted as biological dose response without voltage/current traces.",
                     "do_not": "Do not change dose, rerun S3, or extrapolate stronger stimulation; no new full-brain authorization."},
        "gates": {"NEW_FULL_BRAIN_AUTHORIZATION": False, "TRAINING_AUTHORIZATION": False,
                  "REAL_SYNAPTIC_LEARNING_READY": "NO", "Level_A/B/C/Biological": "NO-GO",
                  "Robot": "NO-GO", "Maze": "NO-GO", "Dijkstra": "NO-GO", "DeltaW": "NO-GO"},
        "no_new_dynamics": True,
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({"status": rep["status"], "source_sha": rep["source"]["sha256"],
                      "math": [round(sham, 6), round(low, 6), round(primary, 6)],
                      "indirect_kc_feedback": {"endpoints": endpoints, "paths": paths, "direct": direct_kk},
                      "priority": rep["decision"]["priority"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
