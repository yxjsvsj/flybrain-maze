"""Synthetic/offline acceptance for the D2 data->fit->held-out pipeline.

No full MaleCNS, no FlyBrain.step, no hardware, no real candidate training.
All tensors here are small synthetic arrays.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from demo.brain_turn_d2 import collect, evaluate, frames, gates, pipeline, protocol, readout, train


def _pack(path: Path, scene: str, distance_mm: int) -> None:
    n = protocol.TOF_FRAMES
    ranges = np.full((n, 5), np.nan)
    status = np.full((n, 5), "NO_TARGET", dtype="<U10")
    t = np.arange(n) / protocol.TOF_HZ
    seq = np.arange(n) + 1
    if scene == "B_FL_only":
        ranges[:, 1] = 400.0
        status[:, 1] = "VALID"
    elif scene == "C_FR_only":
        ranges[:, 3] = 400.0
        status[:, 3] = "VALID"
    elif scene == "D_FL_FR_symmetric":
        ranges[:, 1] = 400.0
        ranges[:, 3] = 400.0
        status[:, 1] = "VALID"
        status[:, 3] = "VALID"
    np.savez(path, ranges_mm=ranges, status=status, t=t, seq=seq,
             scene=np.asarray(scene), distance_mm=np.asarray(int(distance_mm)))


def _runs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for seed in protocol.TRAIN_SEEDS + protocol.VAL_SEEDS:
        for dist in protocol.DISTANCES_MM:
            for scene in protocol.SCENES:
                for rep in range(protocol.REPEATS):
                    target = {"A_open": 0.0, "B_FL_only": -0.2,
                              "C_FR_only": 0.2, "D_FL_FR_symmetric": 0.0}[scene]
                    x = rng.normal(0, 0.05, (protocol.STEPS, 6)).astype(np.float32)
                    x[:, 0] += target
                    x[:, 1] -= target
                    t = np.arange(protocol.STEPS) / protocol.BRAIN_HZ
                    teacher = np.full(protocol.STEPS, target, np.float32)
                    np.savez(root / protocol.run_name(seed, dist, scene, rep),
                             features=x, teacher=teacher, t=t,
                             seed=np.asarray(seed), distance_mm=np.asarray(dist),
                             scene=np.asarray(scene), repeat=np.asarray(rep),
                             side_window=np.asarray(protocol.SIDE_WINDOW),
                             recovery_window=np.asarray(protocol.RECOVERY_WINDOW),
                             brain_dt=np.asarray(1.0 / protocol.BRAIN_HZ),
                             trace_tau=np.asarray(protocol.TRACE_TAU))


class ProtocolTest(unittest.TestCase):
    def test_default_authorization_is_not_approved(self):
        with self.assertRaises(protocol.Refused):
            protocol.load_auth()
        self.assertTrue(protocol.AUTH.exists())

    def test_freeze_matches(self):
        z = protocol.load_freeze()
        self.assertEqual(z["protocol"], protocol.PROTOCOL)
        self.assertIn("collect.py", z["scripts"])

    def test_scope_size(self):
        self.assertEqual(protocol.RUNS, 120)
        self.assertEqual(protocol.TOTAL_STEPS, 60_000)
        self.assertEqual(protocol.STEPS, 500)
        self.assertEqual(protocol.TOF_FRAMES, 100)

    def test_pipeline_refuses_before_brain(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = pipeline.main(["--frames-root", tmp, "--out-dir", str(Path(tmp) / "runs"),
                                "--auth", str(protocol.AUTH)])
        self.assertEqual(rc, 2)


class FrameTest(unittest.TestCase):
    def test_pack_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "A_open_400mm.npz"
            _pack(p, "A_open", 400)
            z = frames.validate_pack(p)
            self.assertEqual(z["frames"], 100)
            f = frames.make_frame(z, 3)
            self.assertEqual(f.seq, 4)
            self.assertIsNone(f.ranges["L"])


class ReadoutPipelineTest(unittest.TestCase):
    def test_zero_bias_fit_and_prediction(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(80, 5))
        y = 0.4 * x[:, 0] - 0.2 * x[:, 1]
        m = readout.fit_zero_bias(x, y, lam=0.01)
        pred = readout.predict_turn(x[:3], m)
        self.assertEqual(pred.shape, (3,))
        self.assertTrue(np.all(np.abs(pred) <= 1.0))

    def test_synthetic_train_validate_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            _runs(root)
            fit = train.fit_from_runs(root)
            model = fit["model"]
            self.assertEqual(model["weights"].shape, (6,))
            p = Path(tmp) / "candidate.npz"
            readout.save_model(p, model)
            report = evaluate.evaluate(root, p)
            self.assertEqual(report["status"], "EVALUATED")
            self.assertIn(report["verdict"], ("PASS", "FAIL"))
            self.assertEqual(len(report["per_seed"]["open_means"]), len(protocol.VAL_SEEDS))
            self.assertFalse(report["motor"])
            self.assertFalse(report["planner"])

    def test_collect_refuses_before_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(protocol.Refused):
                collect.collect(tmp, Path(tmp) / "runs", protocol.AUTH)


class GateContractTest(unittest.TestCase):
    def test_good_and_bad_gate_contract(self):
        good = gates.evaluate_validation([0.0, 0.0], [-0.2, -0.2], [0.2, 0.2],
                                         [0.0, 0.0], 0.8, 0.5, True)
        bad = gates.evaluate_validation([0.2, 0.2], [-0.2, -0.2], [0.2, 0.2],
                                        [0.0, 0.0], 0.8, 0.5, True)
        self.assertTrue(good["pass_all"])
        self.assertFalse(bad["pass_all"])


if __name__ == "__main__":
    unittest.main()
