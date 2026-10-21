"""P11-2-S5-1: small-memory FlyBrain step tracer with exact per-step logging.

No full brain is loaded here. The tracer is a subclass of the installed FlyBrain
and is validated only on tiny synthetic networks.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import tracemalloc
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
        order_sha = hashlib.sha256(np.asarray(self.targets, "<i8").tobytes()).hexdigest()
        header = {
            "protocol": "P11-2-S5-1/TRACE/v1",
            "format_version": "S5TRACE/1",
            "binary": os.path.basename(self.path),
            "target_indices": self.targets,
            "target_order_sha256": order_sha,
            "index_order": "header target_indices order; one row per target per step",
            "fields": [{"name": n, "dtype": d} for n, d in TRACE_FIELDS],
            "dtype_itemsize": TRACE_DTYPE.itemsize,
            "units": "FlyBrain model units; v is dimensionless; not mV/nA",
            "records": self.records,
            "sha256": digest,
        }
        with open(self.path + ".json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(header, f, ensure_ascii=False, indent=1)
        return header


class TraceReadError(ValueError):
    pass


def read_trace(path: str):
    """Strict round-trip reader: header, SHA, size, target order, rows."""
    path = os.path.abspath(path)
    header_path = path + ".json"
    if not os.path.exists(path) or not os.path.exists(header_path):
        raise TraceReadError("TRACE_OR_HEADER_MISSING")
    with open(header_path, encoding="utf-8") as f:
        h = json.load(f)
    if h.get("protocol") != "P11-2-S5-1/TRACE/v1" or h.get("format_version") != "S5TRACE/1":
        raise TraceReadError("TRACE_FORMAT_MISMATCH")
    fields = [{"name": n, "dtype": d} for n, d in TRACE_FIELDS]
    if h.get("fields") != fields or h.get("dtype_itemsize") != TRACE_DTYPE.itemsize:
        raise TraceReadError("TRACE_SCHEMA_MISMATCH")
    targets = [int(x) for x in h.get("target_indices", [])]
    if not targets or len(set(targets)) != len(targets):
        raise TraceReadError("TRACE_TARGET_ORDER_INVALID")
    order_sha = hashlib.sha256(np.asarray(targets, "<i8").tobytes()).hexdigest()
    if h.get("target_order_sha256") != order_sha:
        raise TraceReadError("TRACE_TARGET_ORDER_HASH_MISMATCH")
    records = int(h.get("records", -1))
    if records < 0 or records % len(targets):
        raise TraceReadError("TRACE_RECORD_COUNT_INVALID")
    size = os.path.getsize(path)
    if size != records * TRACE_DTYPE.itemsize:
        raise TraceReadError("TRACE_SIZE_MISMATCH")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != h.get("sha256"):
        raise TraceReadError("TRACE_SHA256_MISMATCH")
    rows = np.fromfile(path, dtype=TRACE_DTYPE)
    if rows.size != records:
        raise TraceReadError("TRACE_TRUNCATED")
    steps = records // len(targets)
    for i in range(steps):
        block = rows[i * len(targets):(i + 1) * len(targets)]
        if not np.all(block["step"] == i):
            raise TraceReadError(f"TRACE_STEP_ORDER_MISMATCH:{i}")
        if not np.array_equal(block["neuron"], np.asarray(targets, np.int32)):
            raise TraceReadError(f"TRACE_NEURON_ORDER_MISMATCH:{i}")
    return h, rows.reshape(steps, len(targets))


def benchmark_writer(n_targets, steps=400, path=None):
    """Write one synthetic 400-step group; measure wall time and tracemalloc peak."""
    path = path or os.path.join(os.environ.get("TEMP", "."), f"s5_2_bench_{n_targets}.bin")
    targets = list(range(int(n_targets)))
    n = len(targets)
    zeros = np.zeros(n, np.float64)
    fired = np.zeros(n, np.uint8)
    ref = np.zeros(n, np.uint8)
    fields = {"v_start": zeros, "v_leak": zeros, "synaptic": zeros, "tonic": zeros,
              "noise": zeros, "eye": zeros, "inject": zeros,
              "v_pre_threshold": zeros, "v_post_reset": zeros, "threshold_distance": zeros,
              "fired": fired, "refractory_zeroed": ref}
    tracemalloc.start()
    t0 = time.perf_counter()
    w = TraceWriter(path, targets)
    for step in range(int(steps)):
        w.write_step(step, fields)
    hdr = w.close()
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"targets": n, "steps": int(steps), "records": hdr["records"],
            "bytes": os.path.getsize(path), "MiB": os.path.getsize(path) / 1024 ** 2,
            "wall_s": round(elapsed, 4), "tracemalloc_peak_MiB": round(peak / 1024 ** 2, 4),
            "sha256": hdr["sha256"], "path": path}


class InstrumentedFlyBrain(FlyBrain):
    """Exact copy of FlyBrain.step plus optional target-only trace logging.

    Supports CPU batch=1 only. The installed package is not modified.
    """

    def __init__(self, *args, targets=None, trace_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_supported()
        self._targets = None if targets is None else np.asarray(sorted(set(int(x) for x in targets)), dtype=np.int64)
        if self._targets is not None and (self._targets.size == 0 or self._targets.min() < 0 or self._targets.max() >= self.n):
            raise ValueError("targets out of range")
        self._trace = self._targets is not None
        self._writer = TraceWriter(trace_path, self._targets) if (self._trace and trace_path) else None
        self.last_trace = None

    def _check_supported(self):
        if self.device != "cpu" or self.xp is not np or self.batch != 1:
            raise NotImplementedError("tracer supports NumPy CPU batch=1 only")

    def step(self, eye_drive=None, inject=()):
        self._check_supported()
        # Validate every injection group before touching v; non-uniform per-neuron
        # vectors are not supported by FlyBrain batch=1 semantics.
        groups = []
        for idx, amount in inject:
            idx_arr = self.xp.asarray(idx)
            amt = self._amount(amount)
            if np.asarray(amt).size != 1:
                raise ValueError("UNSUPPORTED_INJECT_AMOUNT: batch=1 inject amount must be scalar/size-1")
            groups.append((idx_arr, amt))
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
        v_before_inject = self.v[target, 0].copy() if log else None
        for idx_arr, amt in groups:
            self.v[idx_arr] += amt
        inject_log = ((self.v[target, 0] - v_before_inject).astype(np.float64)
                      if log else None)
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
