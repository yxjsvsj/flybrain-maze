"""P11-2-S4-OFFLINE 报告回归；不运行神经动力学。"""
from __future__ import annotations

import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_s4_report.json")


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class S4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _load(REPORT)

    def test_status_and_reproduction(self):
        self.assertEqual(self.r["status"], "S4_OFFLINE_COMPLETE; NO_NEW_DYNAMICS; EXPLORATORY")
        rep = self.r["reproduction"]
        self.assertTrue(rep["kc_match"] and rep["kc_p_match"])
        self.assertTrue(rep["mbon_match"] and rep["mbon_p_match"])
        self.assertAlmostEqual(rep["kc_primary"]["mean"], 0.022858125, places=9)
        self.assertAlmostEqual(rep["kc_primary"]["ci"]["ci_low"], 0.0199255, places=9)
        self.assertAlmostEqual(rep["kc_primary"]["ci"]["ci_high"], 0.025606891, places=9)
        self.assertAlmostEqual(rep["mbon_primary"]["mean"], -0.000111579, places=9)
        self.assertAlmostEqual(rep["mbon_primary"]["p"], 0.9374145269393921, places=12)

    def test_archive(self):
        a = self.r["archive"]
        self.assertEqual(a["npz_count"], 75)
        self.assertTrue(a["raw_hashes_ok"])
        self.assertTrue(a["backup_hashes_ok"])
        self.assertTrue(a["run_meta_sha256_match"])
        self.assertEqual(a["manifest_files"], 79)

    def test_dose_response(self):
        d = self.r["dose_response"]
        self.assertAlmostEqual(d["low_to_primary_aggregate_ratio"], 0.508872666, places=9)
        self.assertAlmostEqual(d["per_seed_low_to_primary_ratio_mean"], 0.529491977, places=9)
        self.assertEqual(d["primary_direction_positive"], 25)
        self.assertEqual(d["low_direction_positive"], 25)
        self.assertGreater(d["kc_primary"]["min"], 0)
        self.assertGreater(d["kc_low"]["min"], 0)

    def test_cell_level(self):
        c = self.r["cell_level"]
        k = c["matched_kc_primary"]
        self.assertEqual((k["n_positive"], k["n_negative"], k["n_zero"]), (311, 3, 6))
        self.assertAlmostEqual(k["mean"], 0.022858125, places=9)
        self.assertAlmostEqual(k["top_1pct_contribution_to_mean"], 0.0009075, places=12)
        self.assertEqual(c["primary_cells_ge095"], 0)
        self.assertLess(c["primary_matched_rate_max"], 0.85)
        self.assertLess(c["other_kc_primary"]["mean"], 0)
        self.assertLess(c["full_kc_primary"]["mean"], 0)
        self.assertEqual(c["matched_mbon_primary"]["n_positive"], 51)
        self.assertEqual(c["matched_mbon_primary"]["n_negative"], 43)

    def test_time_bins_not_transient_only(self):
        t = self.r["time_bins"]
        self.assertEqual(t["bins"], [0, 100, 150, 200, 250, 300, 400])
        self.assertEqual(t["kc_primary_minus_sham"][0], 0.0)
        self.assertGreater(t["kc_primary_minus_sham"][1], 0.02)
        self.assertGreater(t["kc_primary_minus_sham"][4], 0.02)
        self.assertLess(t["kc_primary_minus_sham"][5], 0.005)

    def test_model_semantics(self):
        m = self.r["model_semantics"]
        self.assertTrue(m["inject_is_voltage"])
        self.assertTrue(m["reset_to_zero"])
        self.assertTrue(m["threshold"])
        self.assertAlmostEqual(m["dv_inf_primary"], 0.2, places=9)
        self.assertAlmostEqual(m["dv_inf_low"], 0.1, places=9)
        self.assertAlmostEqual(m["low_to_primary_rate_ratio"], 0.508872666, places=9)
        self.assertAlmostEqual(m["low_to_primary_dv_ratio"], 0.5, places=9)

    def test_mechanism_and_decision(self):
        d = self.r["decision"]
        self.assertEqual(d["priority"], "B")
        self.assertIn("pause", d["B"].lower())
        self.assertIn("B-first", d["next_experiment_candidate"])
        m = self.r["mechanism"]
        self.assertTrue(m["OBSERVED"])
        self.assertTrue(m["SUPPORTED_HYPOTHESIS"])
        self.assertTrue(m["UNKNOWN"])
        self.assertTrue(any("not prove" in x or "not recorded" in x for x in m["SUPPORTED_HYPOTHESIS"] + m["UNKNOWN"]))
        self.assertFalse(self.r["gates"]["NEW_FULL_BRAIN_AUTHORIZATION"])
        self.assertFalse(self.r["gates"]["TRAINING_AUTHORIZATION"])
        self.assertEqual(self.r["gates"]["REAL_SYNAPTIC_LEARNING_READY"], "NO")


if __name__ == "__main__":
    unittest.main()
