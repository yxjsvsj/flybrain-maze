"""P11-2-S3-0 预注册静态回归；不运行 FlyBrain 或完整脑。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_s3_0_report.json")
RUN_META = os.path.join(HERE, "p11_2_s1r_run_meta.json")


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class S30Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _load(REPORT)
        cls.meta = _load(RUN_META)

    def test_status_and_no_s3_authorization(self):
        self.assertEqual(self.r["status"], "PREREGISTRATION_COMPLETE; NOT_AUTHORIZED; NO_RUN")
        self.assertEqual(self.r["authorization"]["s3"], "NOT_AUTHORIZED")
        self.assertFalse(self.r["authorization"]["approved"])
        self.assertFalse(self.r["new_full_brain_authorization"])
        self.assertFalse(self.r["training_authorization"])
        self.assertTrue(self.r["no_s3_run"])
        self.assertFalse(self.r["resource"]["verified"])

    def test_frozen_s1r(self):
        f = self.r["frozen_s1r"]
        self.assertEqual(f["status"], "RUN_COMPLETE")
        self.assertEqual(f["grading"], "SENSORY_TRANSMISSION_NOT_DETECTED")
        self.assertAlmostEqual(f["entry_diff"], 0.066634727, places=9)
        self.assertAlmostEqual(f["matched_kc_diff"], -0.001430625, places=9)
        self.assertAlmostEqual(f["matched_mbon_diff"], -0.00176, places=9)

    def test_targets_and_design(self):
        t = self.r["target"]
        self.assertEqual(t["count"], 320)
        self.assertEqual(t["unique"], 320)
        self.assertTrue(t["no_reselection"])
        ids = np.load(os.path.join(self.meta["out_dir"], "ids.npz"), allow_pickle=True)
        target = np.asarray(ids["kc_matched"], dtype=np.int64)
        self.assertEqual(len(set(target.tolist())), 320)
        self.assertEqual(hashlib.sha256(target.astype("<i8").tobytes()).hexdigest(), t["sha256_i64_le"])
        d = self.r["design"]
        self.assertEqual(d["seeds"], list(range(201, 226)))
        self.assertEqual(d["conditions"], ["sham", "kc_low", "kc_primary"])
        self.assertEqual(d["steps_per_seed_condition"], 400)
        self.assertEqual(d["total_steps"], 30000)
        self.assertEqual(d["window"], [100, 300])
        self.assertEqual(d["npz_estimate"], 75)

    def test_injection_formula_and_levels(self):
        i = self.r["injection_semantics"]
        decay = math.exp(-0.020 / 0.100)
        primary = 0.20 * (1 - decay)
        low = 0.10 * (1 - decay)
        self.assertAlmostEqual(i["primary"]["amount"], primary, places=15)
        self.assertAlmostEqual(i["low"]["amount"], low, places=15)
        self.assertAlmostEqual(i["reference_s1r_amount"], 0.20 * (1 - decay), places=15)
        self.assertAlmostEqual(i["primary"]["amount"] / (1 - decay), 0.20, places=12)
        self.assertAlmostEqual(i["low"]["amount"] / (1 - decay), 0.10, places=12)
        # synthetic one-cell LIF fixed point; no FlyBrain
        v = 0.0
        for _ in range(2000):
            v = decay * v + 0.14 + primary
        self.assertAlmostEqual(v, (0.14 + primary) / (1 - decay), places=10)

    def test_endpoints_and_decision_tree(self):
        p = self.r["primary_endpoints"]
        self.assertEqual(p["kc_intervention"]["min_positive_effect"], 0.10)
        self.assertEqual(p["mbon_downstream"]["min_positive_effect"], 0.05)
        self.assertIn("conditional", p["mbon_downstream"])
        results = [x["result"] for x in self.r["decision_tree"]]
        self.assertIn("INCONCLUSIVE_KC_NOT_DRIVEN", results)
        self.assertIn("INCONCLUSIVE_KC_SATURATED", results)
        self.assertIn("KC_MBON_FUNCTIONAL_UNDER_DIRECT_DRIVE", results)
        self.assertIn("INCONCLUSIVE_DOWNSTREAM", results)
        self.assertIn("KC_MBON_NEGATIVE", results)

    def test_resource_and_gates(self):
        r = self.r["resource"]
        self.assertEqual(r["status"], "RESOURCE_PLAUSIBLE_TIGHT")
        self.assertFalse(r["verified"])
        self.assertEqual(r["worth_applying_one_run"], "YES_CONDITIONAL_ON_PREFLIGHT")
        self.assertEqual(r["wall_cap_s"], 900.0)
        self.assertEqual(r["mem_cap_MiB"], 2048)
        self.assertGreater(r["time_estimate_s"], 0)
        self.assertLess(r["time_estimate_s"], 900)
        self.assertLess(r["peak_estimate_MiB"], 2048)
        g = self.r["gates"]
        self.assertFalse(g["NEW_FULL_BRAIN_AUTHORIZATION"])
        self.assertFalse(g["TRAINING_AUTHORIZATION"])
        self.assertEqual(g["REAL_SYNAPTIC_LEARNING_READY"], "NO")
        self.assertEqual(g["Level_A/B/C/Biological"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
