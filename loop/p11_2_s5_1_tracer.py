"""P11-2-S5-1: small-memory FlyBrain step tracer with exact per-step logging.

No full brain is loaded here. The tracer is a subclass of the installed FlyBrain
and is validated only on tiny synthetic networks.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable

import numpy as np

import flybrain.brain as fb
from flybrain import FlyBrain

EXPECTED_SOURCE_SHA256 = "142ecfffd5d32e30655829e34b38b04024e49beae662bd078279e9610c789b0c"
if hashlib.sha256(open(fb.__file__, "rb").read()).hexdigest() != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("FLYBRAIN_SOURCE_SHA_MISMATCH")

TRACE_FIELDS = [
    ("step", "<i4"), ("neuron", "<i4"),
    ("v_start", "<f8"), ("v_leak", "<f8"), ("synaptic", "<f8"),
    ("tonic", "<f8"), ("noise", "<f8"), ("eye", "<f8"), ("inject", "<f8"),
    ("v_pre_threshold", "<f8"), ("v_post_reset", "<f8"), ("threshold_distance", "<f8"),
    ("fired", "u1"), ("refractory_zeroed", "u1"),
]
TRACE_DTYPE = np.dtype(TRACE_FIELDS)


class TraceWriter:
    """Append fixed records to a .bin stream; write a JSON header + SHA at close."""

    def __init__(self, path: str, targets: Iterable[int]):
        self.path = os.path.abspath(path)
        self.targets = [int(x) for x in targets]
        self.records = 0
        self._h = hashlib.sha256()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._f = open(self.path, "wb")

    def write_step(self, step: int, fields: dict):
        n = len(self.targets)
        rows = np.zeros(n, dtype=TRACE_DTYPE)
        rows["step"] = step
        rows["neuron"] = np.asarray(self.targets, np.int32)
        for name in ("v_start", "v_leak", "synaptic", "tonic", "noise", "eye",
                     "inject", "v_pre_threshold", "v_post_reset", "threshold_distance"):
            rows[name] = np.asarray(fields[name], np.float64)
        rows["fired"] = np.asarray(fields["fired"], np.uint8)
        rows["refractory_zeroed"] = np.asarray(fields["refractory_zeroed"], np.uint8)
        b = rows.tobytes()
        self._f.write(b)
        self._h.update(b)
        self.records += n

    def close(self):
        self._f.close()
        digest = self._h.hexdigest()
        header = {
            "protocol": "P11-2-S5-1/TRACE/v1",
            "binary": os.path.basename(self.path),
            "target_indices": self.targets,
            "index_order": "header target_indices order; one row per target per step",
            "fields": [{"name": n, "dtype": d} for n, d in TRACE_FIELDS],
            "units": "FlyBrain model units; v is dimensionless; not mV/nA",
            "records": self.records,
            "sha256": digest,
        }
        with open(self.path + ".json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(header, f, ensure_ascii=False, indent=1)
        return header


class InstrumentedFlyBrain(FlyBrain):
    """Exact copy of FlyBrain.step plus optional target-only trace logging.

    Supports CPU batch=1 only. The installed package is not modified.
    """

    def __init__(self, *args, targets=None, trace_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._targets = None if targets is None else np.asarray(sorted(set(int(x) for x in targets)), dtype=np.int64)
        if self._targets is not None and (self._targets.size == 0 or self._targets.min() < 0 or self._targets.max() >= self.n):
            raise ValueError("targets out of range")
        self._trace = self._targets is not None
        self._writer = TraceWriter(trace_path, self._targets) if (self._trace and trace_path) else None
        self.last_trace = None

    def step(self, eye_drive=None, inject=()):
        if self.batch != 1:
            raise NotImplementedError("tracer supports batch=1 only")
        xp, B = self.xp, self.batch
        target = self._targets
        log = self._trace
        if log:
            v_start = self.v[target, 0].copy()
        current = self.synaptic_input(self.fired) * self.gain
        self.v *= self.decay
        if log:
            v_leak = self.v[target, 0].copy()
            syn = current[target, 0].copy()
        self.v += current + self.tonic
        noise = (self.rng.random((self.n, B)) < self.noise_hz * self.dt)
        self.v += noise * np.float32(self.noise_amp)
        if log:
            noise_log = noise[target, 0].astype(np.float64) * float(np.float32(self.noise_amp))
            tonic_log = np.full(len(target), self.tonic, np.float64)
        if eye_drive is not None:
            drive = xp.asarray(eye_drive, dtype=xp.float32)
            eye_add = (drive[:, None] if drive.ndim == 1 else drive) * self.eye_gain
            self.v[self._visual] += eye_add
            if log:
                eye_log = np.zeros(len(target), np.float64)
                for j, t in enumerate(target):
                    pos = np.flatnonzero(self._visual == t)
                    if pos.size:
                        eye_log[j] = float(eye_add[pos[0], 0])
        elif log:
            eye_log = np.zeros(len(target), np.float64)
        if log:
            inject_log = np.zeros(len(target), np.float64)
        for idx, amount in inject:
            idx_arr = xp.asarray(idx)
            amt = self._amount(amount)
            self.v[idx_arr] += amt
            if log:
                val = float(np.asarray(amt).reshape(-1)[0])
                for j, t in enumerate(target):
                    if np.isin(t, idx_arr):
                        inject_log[j] += val
        if self.refractory_steps:
            ref = (self.steps - self.last_spike) <= self.refractory_steps
            self.v[ref] = 0.0
            if log:
                ref_log = ref[target, 0].astype(np.uint8)
        elif log:
            ref_log = np.zeros(len(target), np.uint8)
        if log:
            v_pre = self.v[target, 0].copy()
        fired = xp.flatnonzero(self.v >= 1.0)
        if log:
            fired_log = np.isin(target, fired).astype(np.uint8)
        self.v.ravel()[fired] = 0.0
        if log:
            v_post = self.v[target, 0].copy()
        if self.refractory_steps:
            self.last_spike.ravel()[fired] = self.steps
        self.fired = fired
        self.steps += 1
        if log:
            fields = {
                "v_start": v_start, "v_leak": v_leak, "synaptic": syn,
                "tonic": tonic_log, "noise": noise_log, "eye": eye_log,
                "inject": inject_log, "v_pre_threshold": v_pre,
                "v_post_reset": v_post, "threshold_distance": 1.0 - v_pre,
                "fired": fired_log, "refractory_zeroed": ref_log,
            }
            self.last_trace = {"step": self.steps - 1, "targets": target.copy(), **fields,
                               "fired_indices": np.asarray(fired)}
            if self._writer is not None:
                self._writer.write_step(self.steps - 1, fields)
        flat = fired if xp is np else fired.get()
        return flat

    def close_trace(self):
        if self._writer is not None:
            return self._writer.close()
        return None
