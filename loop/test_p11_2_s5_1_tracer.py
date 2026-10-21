"""P11-2-S5-1 synthetic equivalence tests for the FlyBrain step tracer.

Only tiny synthetic networks (12 neurons) are used; official MaleCNS is never loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp

from flybrain import FlyBrain
import loop.p11_2_s5_1_tracer as tracer


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _make_tiny(root):
    n = 12
    rng = np.random.default_rng(20261021)
    W = sp.random(n, n, density=0.28, random_state=7,
                  data_rvs=lambda k: rng.normal(0, 0.35, k)).tocsr()
    W[1, 0] = 0.8; W[2, 1] = -0.7; W[3, 2] = 0.6; W[4, 3] = -0.5
    np.savez(os.path.join(root, "brain.npz"),
             ids=np.arange(1000, 1000 + n),
             visual=np.array([0], np.int64),
             azimuth=np.zeros(n, np.float32),
             cell_type=np.array(["LPLC2"] + ["KC"] * 6 + ["MBON"] * 5),
             side=np.array(["L", "R"] * 6))
    sp.save_npz(os.path.join(root, "weights.npz"), W)
    return W


class S51Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="s51 tiny ")
        cls.W = _make_tiny(cls.tmp)
        cls.targets = [0, 1, 2, 3]
        cls.decay = float(np.exp(-0.020 / 0.100))

    def _pair(self, refractory=0.0, with_trace=True, noise_hz=None):
        kw = dict(data=self.tmp, seed=7, device="cpu", batch=1, dt=0.020, refractory=refractory)
        orig = FlyBrain(**kw)
        tp = os.path.join(self.tmp, f"trace_{len(os.listdir(self.tmp))}.bin") if with_trace else None
        inst = tracer.InstrumentedFlyBrain(**kw, targets=self.targets if with_trace else None,
                                           trace_path=tp)
        if noise_hz is not None:
            orig.noise_hz = noise_hz; inst.noise_hz = noise_hz
        orig.reset(7); inst.reset(7)
        return orig, inst, tp

    def _state(self, b):
        return json.dumps(b.rng.bit_generator.state, sort_keys=True, default=str)

    def _compare(self, n_steps=40, amounts=(0.0, 0.01812692469220182, 0.03625384938440364),
                 refractory=0.0, eye=False, noise_hz=None):
        orig, inst, tp = self._pair(refractory=refractory, with_trace=True, noise_hz=noise_hz)
        try:
            for t in range(n_steps):
                amount = amounts[t % len(amounts)]
                inj = [(np.array(self.targets), amount)] if amount else []
                eye_drive = np.array([0.3], np.float32) if (eye and t < 15) else None
                fo = np.asarray(orig.step(eye_drive=eye_drive, inject=inj))
                fi = np.asarray(inst.step(eye_drive=eye_drive, inject=inj))
                self.assertTrue(np.array_equal(fo, fi), f"fired mismatch step {t}")
                self.assertTrue(np.array_equal(orig.v, inst.v), f"v mismatch step {t}")
                self.assertEqual(orig.steps, inst.steps)
                self.assertEqual(self._state(orig), self._state(inst))
                if orig.last_spike is not None:
                    self.assertTrue(np.array_equal(orig.last_spike, inst.last_spike))
                tr = inst.last_trace
                pre = tr["v_pre_threshold"]
                pred = (tr["v_start"] * self.decay + tr["synaptic"] + tr["tonic"]
                        + tr["noise"] + tr["eye"] + tr["inject"])
                keep = tr["refractory_zeroed"] == 0
                np.testing.assert_allclose(pre[keep], pred[keep], rtol=0, atol=2e-6)
                np.testing.assert_allclose(tr["v_post_reset"],
                                           np.where(tr["fired"] == 1, 0.0, pre), atol=1e-7)
                np.testing.assert_allclose(tr["threshold_distance"], 1.0 - pre, atol=1e-12)
                self.assertTrue(np.array_equal(tr["fired"].astype(bool), np.isin(self.targets, fi)))
        finally:
            hdr = inst.close_trace()
            if hdr is not None:
                self.assertEqual(hdr["records"], n_steps * len(self.targets))
                self.assertEqual(hdr["sha256"], _sha(tp))
                self.assertEqual(os.path.getsize(tp), hdr["records"] * tracer.TRACE_DTYPE.itemsize)
                with open(tp + ".json", encoding="utf-8") as f:
                    disk = json.load(f)
                self.assertEqual(disk["target_indices"], self.targets)
                self.assertIn("dimensionless", disk["units"])
        return orig, inst

    def test_source_hash(self):
        self.assertEqual(tracer.EXPECTED_SOURCE_SHA256,
                         "142ecfffd5d32e30655829e34b38b04024e49beae662bd078279e9610c789b0c")
        import flybrain.brain as fb
        self.assertEqual(_sha(fb.__file__), tracer.EXPECTED_SOURCE_SHA256)

    def test_injection_eye_and_weights(self):
        self.assertTrue((self.W.data > 0).any() and (self.W.data < 0).any())
        self._compare(40, eye=True)

    def test_no_firing(self):
        orig, inst, _ = self._pair(with_trace=False, noise_hz=0.0)
        # zero injection, no noise; fixed point below threshold
        for _ in range(20):
            fo = np.asarray(orig.step())
            fi = np.asarray(inst.step())
            self.assertTrue(np.array_equal(fo, fi))
            self.assertTrue(np.array_equal(orig.v, inst.v))
        self.assertEqual(orig.fired.size, 0)

    def test_sustained_firing(self):
        orig, inst, _ = self._pair(with_trace=False)
        any_fired = False
        for _ in range(40):
            inj = [(np.array([1, 2, 3]), 0.08)]
            fo = np.asarray(orig.step(inject=inj))
            fi = np.asarray(inst.step(inject=inj))
            self.assertTrue(np.array_equal(fo, fi))
            self.assertTrue(np.array_equal(orig.v, inst.v))
            any_fired |= bool(fi.size)
        self.assertTrue(any_fired)

    def test_refractory(self):
        orig, inst, _ = self._pair(refractory=0.040, with_trace=True)
        try:
            for t in range(30):
                inj = [(np.array([1, 2]), 0.08)] if t < 20 else []
                fo = np.asarray(orig.step(inject=inj))
                fi = np.asarray(inst.step(inject=inj))
                self.assertTrue(np.array_equal(fo, fi))
                self.assertTrue(np.array_equal(orig.v, inst.v))
                self.assertTrue(np.array_equal(orig.last_spike, inst.last_spike))
                tr = inst.last_trace
                ref = tr["refractory_zeroed"].astype(bool)
                if ref.any():
                    self.assertTrue(np.all(tr["v_pre_threshold"][ref] == 0.0))
                    self.assertTrue(np.all(tr["v_post_reset"][ref] == 0.0))
        finally:
            inst.close_trace()

    def test_logging_disabled_does_not_change(self):
        orig, inst, _ = self._pair(with_trace=False)
        for s in range(30):
            inj = [(np.array([1]), 0.03625384938440364)] if s < 15 else []
            fo = np.asarray(orig.step(inject=inj))
            fi = np.asarray(inst.step(inject=inj))
            self.assertTrue(np.array_equal(fo, fi))
            self.assertTrue(np.array_equal(orig.v, inst.v))
            self.assertEqual(self._state(orig), self._state(inst))


    def test_report(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "p11_2_s5_1_report.json"), encoding="utf-8") as f:
            r = json.load(f)
        self.assertEqual(r["status"], "IMPLEMENTATION_EQUIVALENCE_GO; MODEL_MECHANISM_OBSERVABILITY_GO; BIOLOGICAL_VALIDITY_UNVALIDATED")
        self.assertTrue(r["source"]["match"])
        self.assertEqual(r["prototype"]["record_bytes"], tracer.TRACE_DTYPE.itemsize)
        self.assertEqual(r["equivalence"]["tests_ran"], 6)
        self.assertEqual(r["equivalence"]["tests_passed"], 6)
        self.assertTrue(r["equivalence"]["v_bitwise_equal"])
        self.assertTrue(r["equivalence"]["rng_bit_generator_state_equal"])
        self.assertTrue(r["observability"]["pre_reset_v_captured"])
        self.assertFalse(r["observability"]["inference_from_post_step_v"])
        self.assertLess(r["resource_estimate"]["with_1415_targets_MiB"], 100)
        self.assertEqual(r["biological_validity"]["status"], "UNVALIDATED")
        self.assertFalse(r["gates"]["NEW_FULL_BRAIN_AUTHORIZATION"])
        self.assertEqual(r["gates"]["REAL_SYNAPTIC_LEARNING_READY"], "NO")


    def test_nonuniform_overlapping_injection(self):
        orig, inst, tp = self._pair(with_trace=True)
        try:
            for t in range(20):
                inj = [(np.array([1]), 0.030),
                       (np.array([1, 2]), 0.010),
                       (np.array([2, 3]), -0.005)]
                fo = np.asarray(orig.step(inject=inj))
                fi = np.asarray(inst.step(inject=inj))
                self.assertTrue(np.array_equal(fo, fi))
                self.assertTrue(np.array_equal(orig.v, inst.v))
                self.assertEqual(self._state(orig), self._state(inst))
                tr = inst.last_trace
                expected = np.array([0.0, 0.040, 0.005, -0.005])
                np.testing.assert_allclose(tr["inject"], expected, rtol=0, atol=1e-7)
                pred = (tr["v_start"] * self.decay + tr["synaptic"] + tr["tonic"]
                        + tr["noise"] + tr["eye"] + tr["inject"])
                keep = tr["refractory_zeroed"] == 0
                np.testing.assert_allclose(tr["v_pre_threshold"][keep], pred[keep], atol=2e-6)
        finally:
            inst.close_trace()

    def test_unsupported_vector_injection_rejected_without_state_change(self):
        orig, inst, tp = self._pair(with_trace=True)
        try:
            v0 = inst.v.copy(); st0 = self._state(inst); steps0 = inst.steps
            with self.assertRaises(ValueError) as cm:
                inst.step(inject=[(np.array([1, 2]), np.array([0.01, 0.02]))])
            self.assertIn("UNSUPPORTED_INJECT_AMOUNT", str(cm.exception))
            self.assertTrue(np.array_equal(inst.v, v0))
            self.assertEqual(self._state(inst), st0)
            self.assertEqual(inst.steps, steps0)
        finally:
            inst.close_trace()

    def test_backend_rejects_batch_and_cuda(self):
        kw = dict(data=self.tmp, seed=7, device="cpu", batch=2, dt=0.020)
        with self.assertRaises(NotImplementedError):
            tracer.InstrumentedFlyBrain(**kw, targets=[0])
        inst = tracer.InstrumentedFlyBrain(data=self.tmp, seed=7, device="cpu",
                                           batch=1, dt=0.020, targets=[0])
        inst.device = "cuda"
        with self.assertRaises(NotImplementedError):
            inst.step()

    def test_trace_roundtrip_content(self):
        orig, inst, tp = self._pair(refractory=0.040, with_trace=True)
        traces = []
        try:
            for t in range(15):
                inj = [(np.array([1, 2]), 0.03), (np.array([2, 3]), 0.01)] if t < 10 else []
                eye = np.array([0.2], np.float32) if t < 5 else None
                fo = np.asarray(orig.step(eye_drive=eye, inject=inj))
                fi = np.asarray(inst.step(eye_drive=eye, inject=inj))
                self.assertTrue(np.array_equal(fo, fi))
                self.assertTrue(np.array_equal(orig.v, inst.v))
                traces.append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                               for k, v in inst.last_trace.items()})
            inst.close_trace()
        except Exception:
            inst.close_trace()
            raise
        h, rows = tracer.read_trace(tp)
        self.assertEqual(h["records"], 15 * len(self.targets))
        self.assertEqual(rows.shape, (15, len(self.targets)))
        self.assertEqual(h["sha256"], _sha(tp))
        for i, tr in enumerate(traces):
            for j, target in enumerate(self.targets):
                row = rows[i, j]
                self.assertEqual(int(row["step"]), i)
                self.assertEqual(int(row["neuron"]), target)
                for name in ("v_start", "v_leak", "synaptic", "tonic", "noise", "eye",
                             "inject", "v_pre_threshold", "v_post_reset", "threshold_distance"):
                    self.assertEqual(float(row[name]), float(tr[name][j]))
                self.assertEqual(int(row["fired"]), int(tr["fired"][j]))
                self.assertEqual(int(row["refractory_zeroed"]), int(tr["refractory_zeroed"][j]))

    def test_trace_reader_rejects_truncation_hash_and_target_order(self):
        orig, inst, tp = self._pair(with_trace=True)
        for t in range(3):
            inst.step(inject=[(np.array([1]), 0.01)])
        hdr = inst.close_trace()
        with open(tp, "rb") as f:
            data = f.read()
        bad = tp + ".trunc.bin"
        with open(bad, "wb") as f:
            f.write(data[:-1])
        with open(bad + ".json", "w", encoding="utf-8") as f:
            json.dump(hdr, f)
        with self.assertRaises(tracer.TraceReadError):
            tracer.read_trace(bad)
        bad2 = tp + ".hash.bin"
        b = bytearray(data); b[0] ^= 0x01
        with open(bad2, "wb") as f:
            f.write(b)
        with open(bad2 + ".json", "w", encoding="utf-8") as f:
            json.dump(hdr, f)
        with self.assertRaises(tracer.TraceReadError):
            tracer.read_trace(bad2)
        bad3 = tp + ".order.bin"
        with open(bad3, "wb") as f:
            f.write(data)
        h2 = dict(hdr); h2["target_indices"] = list(reversed(hdr["target_indices"]))
        with open(bad3 + ".json", "w", encoding="utf-8") as f:
            json.dump(h2, f)
        with self.assertRaises(tracer.TraceReadError):
            tracer.read_trace(bad3)
        with self.assertRaises(tracer.TraceReadError):
            tracer.read_trace(tp + ".missing.bin")

    def test_writer_benchmark_415_and_1415(self):
        for n, expected_mib in ((415, 14.247894287109375), (1415, 48.580169677734375)):
            p = os.path.join(self.tmp, f"bench_{n}.bin")
            r = tracer.benchmark_writer(n, 400, p)
            self.assertEqual(r["records"], n * 400)
            self.assertAlmostEqual(r["MiB"], expected_mib, places=6)
            self.assertGreater(r["wall_s"], 0)
            self.assertGreaterEqual(r["tracemalloc_peak_MiB"], 0)
            self.assertEqual(r["sha256"], _sha(p))


    def test_s5_2_report(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "p11_2_s5_2_report.json"), encoding="utf-8") as f:
            r = json.load(f)
        self.assertEqual(r["status"], "TRACE_CORRECTNESS_GO; IMPLEMENTATION_EQUIVALENCE_GO; CPU_BATCH1_ONLY; BIOLOGICAL_VALIDITY_UNVALIDATED")
        self.assertEqual(r["source"]["sha256"], tracer.EXPECTED_SOURCE_SHA256)
        self.assertEqual(r["equivalence"]["tests_ran"], 14)
        self.assertEqual(r["equivalence"]["tests_passed"], 14)
        self.assertTrue(r["inject_fix"]["handles_overlap_and_duplicates"])
        self.assertTrue(r["inject_fix"]["per_target_isin_scan_removed"])
        self.assertTrue(r["backend_scope"]["cuda_rejected"])
        self.assertTrue(r["trace_io"]["rejects_truncated"])
        self.assertTrue(r["trace_io"]["rejects_sha_mismatch"])
        self.assertEqual(r["resource_measured"]["one_group_415"]["MiB"], 14.247894287109375)
        self.assertEqual(r["resource_measured"]["one_group_1415"]["MiB"], 48.580169677734375)
        self.assertFalse(r["resource_measured"]["full_brain_memory_verified"])
        self.assertFalse(r["gates"]["NEW_FULL_BRAIN_AUTHORIZATION"])
        self.assertEqual(r["biological_validity"], "UNVALIDATED")


if __name__ == "__main__":
    unittest.main()
