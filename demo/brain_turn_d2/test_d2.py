"""Synthetic/offline regression tests for DEMO-BRAIN-TURN-D2.

No full brain is constructed, no dynamics are stepped, no readout is trained,
and no hardware is opened.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from demo.brain_turn_d2 import gates, review

HERE = Path(__file__).resolve().parent


class D2ReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = review.build_review()

    def test_d1_is_permanently_no_go(self):
        self.assertEqual(self.r["old_route"]["d1_verdict"], "NO_GO")
        d1 = json.loads((review.D1_JSON).read_text(encoding="utf-8"))
        self.assertEqual(d1["verdict"], "NO_GO")

    def test_old_readout_is_frozen_and_not_reused(self):
        self.assertEqual(self.r["old_route"]["readout_sha256"], review.OLD_READOUT_SHA)
        self.assertIn("not a natural motor readout", self.r["old_route"]["readout_identity"])
        self.assertTrue(self.r["old_route"]["p2d29_closed"])
        self.assertEqual(self.r["old_route"]["p2d29_offline_verdict"], "FAIL_OFFLINE")

    def test_candidate_is_insufficient_not_candidate_go(self):
        c = self.r["candidate"]
        self.assertEqual(c["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(self.r["conclusion"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(c["sufficient_for_bilateral_decision"])
        self.assertTrue(c["not_a_candidate_go"])
        self.assertFalse(c["candidate_trained"])
        self.assertFalse(c["training_run"])
        self.assertFalse(c["full_brain_run"])

    def test_no_left_and_right_is_invalid_neural_stimulus(self):
        c = self.r["candidate"]
        self.assertEqual(c["left_only_archives"], [])
        self.assertEqual(c["right_only_valid_neural_archives"], [])
        self.assertGreaterEqual(len(c["both_sides_archives"]), 1)
        self.assertEqual(c["open_archive"], ["A_OPEN"])

    def test_recoverability_tool_is_incompatible(self):
        rec = self.r["candidate"]["recoverability_tool"]
        self.assertFalse(rec["compatible"])
        self.assertFalse(rec["run"])
        self.assertIn("static_A/B/C/D", rec["expected_inputs"])

    def test_prereg_sizes_and_forbidden_actions(self):
        p = self.r["prereg"]
        self.assertEqual(p["data_collection"]["runs"], 120)
        self.assertEqual(p["data_collection"]["brain_steps"], 60_000)
        self.assertEqual(p["data_collection"]["motor"], "OFF; static rig only")
        self.assertFalse(p["readout"]["fit_intercept"])
        self.assertFalse(p["fixed_biology"]["teacher_used_at_inference"])
        self.assertEqual(p["fixed_biology"]["synaptic_dW"], 0.0)
        self.assertEqual(p["fixed_biology"]["inference_inputs"], ["1314 DN Trace values"])

    def test_authorization_is_closed(self):
        a = self.r["authorization"]
        self.assertFalse(a["NEW_FULL_BRAIN_AUTHORIZATION"])
        self.assertFalse(a["READOUT_TRAINING_AUTHORIZATION"])
        self.assertFalse(a["ROBOT_MOTION_AUTHORIZATION"])
        self.assertFalse(a["D2_NEXT_EXPERIMENT_AUTHORIZED"])
        self.assertFalse(self.r["next_experiment"]["authorized_now"])

    def test_source_manifest_hashes_present(self):
        m = self.r["source_manifest"]
        self.assertGreaterEqual(len(m), 5)
        for item in m:
            self.assertEqual(len(item["sha256"]), 64, item["path"])

    def test_outputs_exist(self):
        self.assertTrue((HERE / "D2_REVIEW.json").is_file())
        self.assertTrue((HERE / "D2_REVIEW.md").is_file())
        self.assertTrue((HERE / "D2_PREREG.md").is_file())
        self.assertTrue((HERE / "STORYBOARD.md").is_file())


class GateTest(unittest.TestCase):
    def test_teacher_contract(self):
        self.assertAlmostEqual(gates.teacher_turn(0.0, 0.0), 0.0)
        self.assertGreater(gates.teacher_turn(0.1, 0.6), 0.0)   # right near -> left
        self.assertLess(gates.teacher_turn(0.6, 0.1), 0.0)      # left near -> right
        self.assertAlmostEqual(gates.teacher_turn(0.5, 0.5), 0.0)

    def test_good_synthetic_validation_passes(self):
        r = gates.evaluate_validation(
            open_means=[0.01, -0.02],
            fl_means=[-0.20, -0.17],
            fr_means=[+0.18, +0.22],
            recovery_means=[0.01, -0.02],
            original_accuracy=0.80,
            shuffled_accuracy=0.50,
            zero_directional=True,
        )
        self.assertTrue(r["pass_all"], r)
        self.assertTrue(all(r["checks"].values()))

    def test_open_bias_fails(self):
        r = gates.evaluate_validation([0.01, 0.06], [-0.2, -0.2], [0.2, 0.2],
                                      [0.0, 0.0], 0.8, 0.5, True)
        self.assertFalse(r["pass_all"])
        self.assertFalse(r["checks"]["open_zero_bias"])

    def test_same_sign_fails(self):
        r = gates.evaluate_validation([0.0, 0.0], [0.2, 0.2], [0.2, 0.2],
                                      [0.0, 0.0], 0.8, 0.5, True)
        self.assertFalse(r["pass_all"])
        self.assertFalse(r["checks"]["unilateral_opposite_sign"])

    def test_seed_repeatability_fails(self):
        r = gates.evaluate_validation([0.0, 0.0], [-0.2, 0.2], [0.2, -0.2],
                                      [0.0, 0.0], 0.8, 0.5, True)
        self.assertFalse(r["pass_all"])
        self.assertFalse(r["checks"]["seed_repeatability"])

    def test_recovery_zero_and_shuffle_fail(self):
        r = gates.evaluate_validation([0.0, 0.0], [-0.2, -0.2], [0.2, 0.2],
                                      [0.0, 0.08], 0.7, 0.6, False)
        self.assertFalse(r["checks"]["removal_recovery"])
        self.assertFalse(r["checks"]["zero_ablation"])
        self.assertFalse(r["checks"]["shuffle_impairs"])
        self.assertFalse(r["pass_all"])

    def test_dn_only_inference_contract(self):
        mean = np.zeros(4)
        scale = np.ones(4)
        w = np.array([1.0, -1.0, 0.0, 0.0])
        self.assertAlmostEqual(gates.turn_from_dn(np.zeros(4), mean, scale, w), 0.0)
        self.assertAlmostEqual(gates.turn_from_dn(np.array([1.0, 0, 0, 0]), mean, scale, w), 1.0)
        self.assertAlmostEqual(gates.turn_from_dn(np.array([-1.0, 0, 0, 0]), mean, scale, w), -1.0)
        with self.assertRaises(ValueError):
            gates.turn_from_dn(np.zeros(3), mean, scale, w)

    def test_storyboard_reuses_replay_and_has_failure_plan(self):
        text = (HERE / "STORYBOARD.md").read_text(encoding="utf-8")
        self.assertIn("demo.brain_turn_7d.replay", text)
        self.assertIn("REPLAY", text)
        self.assertIn("失败素材", text)
        self.assertIn("无 `/api/drive`", text)


class SourceContractTest(unittest.TestCase):
    def test_no_brain_training_or_hardware_imports(self):
        for name in ("review.py", "gates.py"):
            text = (HERE / name).read_text(encoding="utf-8")
            import_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith(("import ", "from "))]
            joined = "\n".join(import_lines)
            self.assertNotIn("flybrain", joined, name)
            self.assertNotIn("hardware", joined, name)
            self.assertNotIn("pikachu", joined, name)
            self.assertNotIn("serial", joined, name)

    def test_no_training_call(self):
        text = (HERE / "review.py").read_text(encoding="utf-8")
        self.assertNotIn("Readout.fit", text)
        self.assertNotIn("train_p2_readout.main", text)
        self.assertNotIn("brain.step", text)
        self.assertNotIn("FlyBrain", text)

    def test_dashboard_reuses_d1_and_has_no_hardware(self):
        text = (HERE / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn("from demo.brain_turn_7d.replay import", text)
        for forbidden in ("pikachu", "serial", "arduino", "/api/drive"):
            self.assertNotIn(forbidden, text.lower())


if __name__ == "__main__":
    unittest.main()
