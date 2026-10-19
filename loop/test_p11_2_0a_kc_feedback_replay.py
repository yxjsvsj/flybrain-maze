"""P11-2-0a 静态回归测试（不运行 FlyBrain.step / P11-1c / 时间演化 / A1/A2/A3/B）。"""
from __future__ import annotations

import json
import math
import os
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_0a_report.json")
A1 = os.path.join(HERE, "P11_2_A1_ONE_TEST_APPROVAL.md")
PROPOSAL = os.path.join(HERE, "P11_2_NETWORK_CAUSAL_PREREG_PROPOSAL.md")
SRC = os.path.join(HERE, "p11_2_0a_kc_feedback_replay.py")


class P110aTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPORT, encoding="utf-8") as f:
            cls.r = json.load(f)
        cls.m = cls.r["math_bounds"]
        cls.rec = cls.r["reconstruction"]
        cls.v = cls.r["verdicts"]

    # ---- protocol / archive -------------------------------------------------
    def test_protocol(self):
        self.assertEqual(self.r["protocol"], "P11-2-0a/v1")
        self.assertFalse(self.r["training_authorized"])
        self.assertFalse(self.v["Training_Authorization"])

    def test_archive_verified(self):
        a = self.r["archive"]
        self.assertEqual(a["status"], "VERIFIED")
        self.assertEqual(a["file_count"], 29)
        self.assertEqual(a["missing"], [])
        self.assertEqual(a["mismatch"], [])
        self.assertEqual(a["extra"], [])

    def test_official_hashes(self):
        h = self.r["official_hashes"]
        self.assertTrue(h["ok"])
        self.assertEqual(h["files"]["brain.npz"],
                         "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca")
        self.assertEqual(h["files"]["weights.npz"],
                         "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c")

    # ---- math ---------------------------------------------------------------
    def test_math_bounds(self):
        self.assertAlmostEqual(self.m["decay"], 0.818730753, places=8)
        self.assertAlmostEqual(self.m["tonic_only_fixed_point"], 0.772331779, places=8)
        self.assertFalse(self.m["tonic_alone_can_fire"])
        self.assertAlmostEqual(self.m["tonic_noise_mean_fixed_point"], 0.801459721, places=8)
        self.assertFalse(self.m["tonic_noise_mean_can_fire"])
        v = self.m["max_voltage_reset_steps_1_4"]
        self.assertEqual(len(v), 4)
        self.assertAlmostEqual(v[0], 0.36, places=9)
        self.assertAlmostEqual(v[1], 0.654743071, places=9)
        self.assertAlmostEqual(v[2], 0.896058288, places=9)
        self.assertAlmostEqual(v[3], 1.093630477, places=9)
        self.assertLess(v[2], 1.0)
        self.assertGreaterEqual(v[3], 1.0)
        self.assertEqual(self.m["earliest_refire_step_after_reset"], 4)
        self.assertAlmostEqual(self.m["long_run_firing_fraction_upper_bound"], 0.25, places=9)

    def test_thresholds(self):
        self.assertAlmostEqual(self.m["raw_threshold_no_noise"], 0.2866667, places=6)
        self.assertAlmostEqual(self.m["raw_threshold_with_max_noise"], 0.2133333, places=6)
        self.assertAlmostEqual((1.0 - 0.14) / 3.0, 0.2866666667, places=9)
        self.assertAlmostEqual((1.0 - 0.14 - 0.22) / 3.0, 0.2133333333, places=9)

    def test_independent_trajectory(self):
        decay = math.exp(-0.02 / 0.1)
        v = 0.0; vals = []
        for _ in range(4):
            v = decay * v + 0.14 + 0.22
            vals.append(v)
        for a, b in zip(vals, self.m["max_voltage_reset_steps_1_4"]):
            self.assertAlmostEqual(a, b, places=7)
        self.assertLess(vals[2], 1.0)
        self.assertGreaterEqual(vals[3], 1.0)

    # ---- data / mapping -----------------------------------------------------
    def test_data_checks(self):
        c = self.rec["data_checks"]
        self.assertEqual(c["brain_n"], 166700)
        self.assertEqual(c["W_shape"], [166700, 166700])
        self.assertEqual(c["kc_idx_count"], 4064)
        self.assertTrue(c["kc_idx_unique"])
        self.assertTrue(c["kc_idx_sorted"])
        self.assertEqual(c["matched_count"], 320)
        self.assertTrue(c["matched_subset"])
        self.assertTrue(c["kc_cell_type_all_start_KC"])
        self.assertTrue(c["W_KC_KC_direction_check"]["equal"])
        self.assertEqual(c["W_dtype"], "float32")
        self.assertEqual(c["W_KC_KC_nnz"], 642933)
        self.assertEqual(c["W_KC_KC_pos_nnz"], 642933)
        self.assertEqual(c["W_KC_KC_neg_nnz"], 0)

    def test_W_KC_KC(self):
        w = self.rec["W_KC_KC"]
        self.assertEqual(w["shape"], [4064, 4064])
        self.assertEqual(w["nnz"], 642933)
        self.assertEqual(w["nnz_pos"], 642933)
        self.assertEqual(w["nnz_neg"], 0)
        self.assertTrue(w["signed_preserved"])

    def test_main_window_zeros(self):
        z = self.rec["main_window_zero_spikes"]
        self.assertTrue(z["identical_except_listed"])
        got = {x["seed"]: x["positions"] for x in z["seeds_with_zeros"]}
        self.assertEqual(got, {104: [[232, 2650]], 114: [[213, 2650]], 115: [[234, 2650]]})
        self.assertEqual(len(z["seeds_without_zeros"]), 22)

    # ---- reconstruction -----------------------------------------------------
    def test_all_stats(self):
        s = self.rec["all_4064_KC"]
        self.assertAlmostEqual(s["net"]["mean"], 1.679843, places=5)
        self.assertAlmostEqual(s["net"]["median"], 1.665857, places=5)
        self.assertAlmostEqual(s["net"]["p5"], 1.454108, places=5)
        self.assertAlmostEqual(s["net"]["p95"], 1.963188, places=5)
        self.assertAlmostEqual(s["net"]["min"], 0.132743, places=5)
        self.assertAlmostEqual(s["net"]["max"], 2.703486, places=5)
        self.assertEqual(s["pos"], s["net"])
        self.assertEqual(s["neg"]["mean"], 0.0)

    def test_matched_stats(self):
        s = self.rec["matched_320_KC"]
        self.assertAlmostEqual(s["net"]["mean"], 1.74451, places=5)
        self.assertAlmostEqual(s["net"]["median"], 1.745553, places=5)
        self.assertAlmostEqual(s["net"]["p95"], 1.949037, places=5)
        self.assertAlmostEqual(s["net"]["min"], 1.303277, places=5)
        self.assertAlmostEqual(s["net"]["max"], 2.703486, places=5)
        self.assertEqual(s["neg"]["mean"], 0.0)

    def test_per_seed_and_cell(self):
        ps = self.rec["per_seed"]
        self.assertEqual(len(ps), 25)
        self.assertEqual(self.rec["per_seed_net_mean_range"], [1.679843, 1.679843])
        self.assertEqual([x["seed"] for x in ps], list(range(101, 126)))
        seed114 = [x for x in ps if x["seed"] == 114]
        self.assertEqual(len(seed114), 1)
        self.assertAlmostEqual(seed114[0]["net_mean"], 1.679843, places=5)
        pc = self.rec["per_cell_4064"]
        self.assertEqual(pc["n_cells"], 4064)
        self.assertAlmostEqual(pc["mean"], 1.679843, places=5)
        self.assertAlmostEqual(pc["min"], 0.132743, places=5)
        self.assertAlmostEqual(pc["max"], 2.703486, places=5)

    def test_consecutive(self):
        c = self.rec["consecutive_firing"]
        self.assertEqual(c["n_events"], 20319994)
        self.assertEqual(c["n_ge_0.86"], 20305000)
        self.assertEqual(c["n_ge_0.64"], 20305000)
        self.assertEqual(c["n_lt_0.86"], 14994)
        self.assertEqual(c["n_lt_0.64"], 14994)
        self.assertAlmostEqual(c["frac_input_ge_0.86"], 0.999262, places=6)
        self.assertAlmostEqual(c["frac_input_ge_0.64"], 0.999262, places=6)
        self.assertEqual(c["prev_fired_all"], 20319997)
        self.assertEqual(c["fired_not_again"], 3)
        self.assertAlmostEqual(c["fired_not_again_input"]["min"], 0.132743, places=5)
        self.assertAlmostEqual(c["fired_not_again_input"]["max"], 0.132743, places=5)
        req = c["required_nonKC_at_max_noise"]
        self.assertAlmostEqual(req["frac_gt_0"], 0.000738, places=6)
        self.assertAlmostEqual(req["max"], 0.507257, places=5)
        self.assertEqual(req["median"], 0.0)

    # ---- verdicts / guards --------------------------------------------------
    def test_verdicts(self):
        v = self.v
        self.assertEqual(v["OFFLINE_RECONSTRUCTION"], "VERIFIED")
        self.assertTrue(v["TONIC_ONLY_EXCLUDED"])
        self.assertTrue(v["TONIC_NOISE_EVERY_STEP_EXCLUDED"])
        self.assertEqual(v["EARLIEST_REFIRE_STEP"], 4)
        self.assertAlmostEqual(v["FIRING_FRACTION_UPPER_BOUND"], 0.25, places=9)
        self.assertTrue(v["KC_KC_NET_STATIC_CAPACITY_SUFFICIENT"])
        self.assertAlmostEqual(v["KC_KC_NET_MEAN"], 1.679843, places=5)
        self.assertAlmostEqual(v["KC_KC_POS_MEAN"], 1.679843, places=5)
        self.assertAlmostEqual(v["KC_KC_NEG_MEAN"], 0.0, places=9)
        self.assertAlmostEqual(v["CONSECUTIVE_INPUT_GE_086_FRAC"], 0.999262, places=6)
        self.assertAlmostEqual(v["CONSECUTIVE_REQUIRING_NON_KC"], 0.000738, places=6)
        self.assertIn("NOT_ESTABLISHED", v["KC_KC_CAUSAL_ROOT_CAUSE"])
        self.assertEqual(v["SOURCE_PROVENANCE"], "PARTIAL")
        self.assertIn("A1", v["A1_STILL_OPTIONAL"])
        self.assertIn("YES", v["NETWORK_CAUSAL_EXPERIMENT_WORTH_APPLYING"])
        self.assertEqual(v["A2_A3_B"], "NO-GO")
        self.assertEqual(v["REAL_SYNAPTIC_LEARNING_READY"], "NO")
        self.assertEqual(v["Level_A_Training"], "NO-GO")
        self.assertEqual(v["Level_B"], "NO-GO")
        self.assertEqual(v["Level_C"], "NO-GO")
        self.assertEqual(v["Biological"], "NO-GO")

    def test_no_new_dynamics_in_source(self):
        with open(SRC, encoding="utf-8") as f:
            src = f.read()
        for token in ["from flybrain", "FlyBrain.step(", ".step(", "run_steps", "simulate", "evolve", "train("]:
            self.assertNotIn(token, src)
        ns = {}
        exec(compile("DT=0.020; TAU=0.100; TONIC=0.14; GAIN=3.0; NOISE_P=1.2*DT; NOISE_AMP=0.22", "<p>", "exec"), ns)
        self.assertAlmostEqual(ns["NOISE_P"], 0.024, places=12)
        self.assertAlmostEqual(ns["GAIN"], 3.0, places=12)

    def test_a1_errata(self):
        with open(A1, encoding="utf-8") as f:
            s = f.read()
        self.assertIn("1,000,000", s)
        self.assertIn("2,000,000", s)
        self.assertIn("每条件", s)

    def test_proposal_guard(self):
        with open(PROPOSAL, encoding="utf-8") as f:
            s = f.read()
        self.assertIn("PROPOSAL ONLY", s)
        self.assertIn("NOT AUTHORIZED", s)
        self.assertIn("642,933", s)
        self.assertIn("30,000", s)
        self.assertIn("Training Authorization=False", s)
        self.assertIn("NO-GO", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
