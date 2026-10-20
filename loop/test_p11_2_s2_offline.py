"""P11-2-S2-OFFLINE 报告回归：只读 JSON，不运行神经动力学。"""
from __future__ import annotations

import hashlib
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_s2_report.json")
FORMAL = os.path.join(HERE, "p11_2_s1r_report.json")


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class S2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _load(REPORT)

    def test_status_and_formal_preserved(self):
        self.assertEqual(self.r["status"], "OFFLINE_ANALYSIS_COMPLETE; NO_NEW_DYNAMICS; EXPLORATORY")
        self.assertTrue(self.r["formal_result_preserved"])
        self.assertEqual(self.r["formal_result_preserved"], True)
        self.assertEqual(self.r["audit"]["formal_report_status"], "RUN_COMPLETE")

    def test_independent_reproduction(self):
        a = self.r["A_reproduction"]
        self.assertTrue(a["formal_negative_reproduced"])
        self.assertTrue(a["all_listed_hashes_ok"])
        self.assertTrue(a["run_meta_sha256_match"])
        self.assertEqual(a["npz_count"], 50)
        self.assertAlmostEqual(a["entry"]["mean"], 0.066634727, places=9)
        self.assertAlmostEqual(a["entry"]["p"], 5.960464477539063e-08, places=15)
        self.assertAlmostEqual(a["primary"]["mean"], -0.001430625, places=9)
        self.assertAlmostEqual(a["primary"]["p"], 0.3523610830307007, places=12)
        self.assertAlmostEqual(a["primary"]["ci"]["ci_low"], -0.004300078, places=9)
        self.assertAlmostEqual(a["primary"]["ci"]["ci_high"], 0.001516953, places=9)
        self.assertTrue(a["primary"]["ci_match"])
        self.assertAlmostEqual(a["mbon"]["mean"], -0.00176, places=9)
        self.assertAlmostEqual(a["mbon"]["p"], 0.43899786472320557, places=12)
        self.assertAlmostEqual(a["full_kc"]["mean"], -0.001455118, places=9)
        self.assertAlmostEqual(a["full_mbon"]["mean"], -0.001725773, places=9)

    def test_formal_sha_not_changed(self):
        self.assertEqual(self.r["audit"]["formal_report_sha256"], _sha(FORMAL))

    def test_static_paths(self):
        s = self.r["B_static_paths"]
        self.assertEqual(s["source_count"], 311)
        self.assertEqual(s["direct_src_to_kc"]["edges"], 0)
        self.assertEqual(s["direct_src_to_mbon"]["edges"], 1)
        self.assertEqual(s["two_hop_kc"]["endpoint_pairs"], 7883)
        self.assertEqual(s["two_hop_kc"]["independent_paths"], 8162)
        self.assertEqual(s["two_hop_kc"]["paths_per_endpoint_max"], 3)
        self.assertAlmostEqual(s["two_hop_kc"]["paths_per_endpoint_mean"], 1.035393, places=6)
        self.assertEqual(s["two_hop_mbon"]["endpoint_pairs"], 2528)
        self.assertEqual(s["two_hop_mbon"]["independent_paths"], 3265)
        self.assertEqual(s["three_hop_mbon_via_kc_B"]["endpoint_pairs"], 8247)
        self.assertEqual(s["three_hop_mbon_via_kc_B"]["independent_paths"], 123163)
        self.assertEqual(s["matched_kc"]["n"], 320)
        self.assertEqual(s["matched_kc"]["reachable"], 320)
        self.assertEqual(s["matched_kc"]["independent_paths"], 8162)
        self.assertEqual(s["matched_kc"]["positive_kc"], 296)
        self.assertEqual(s["matched_kc"]["negative_kc"], 24)
        self.assertEqual(s["other_kc"]["reachable"], 0)
        self.assertEqual(s["matched_mbon"]["n"], 95)
        self.assertEqual(s["matched_mbon"]["independent_paths"], 3260)
        self.assertTrue(s["feedforward_preserved_in_B"])
        self.assertIn("not dynamic", s["note"])

    def test_cancellation_and_time(self):
        c = self.r["C_dynamic_exploration"]
        k = c["cancellation"]["matched_kc"]
        self.assertEqual((k["positive_cells"], k["negative_cells"], k["zero_cells"]), (116, 177, 27))
        self.assertAlmostEqual(k["positive_sum"], 0.151, places=9)
        self.assertAlmostEqual(k["negative_sum"], -0.6088, places=9)
        self.assertAlmostEqual(k["net_sum"], -0.4578, places=9)
        m = c["cancellation"]["matched_mbon"]
        self.assertEqual((m["positive_cells"], m["negative_cells"]), (34, 61))
        self.assertEqual(len(c["time_bins"]["bins"]), 7)
        self.assertEqual(len(c["time_bins"]["entry"]), 6)
        self.assertEqual(c["background_matched_kc"]["matched_none_ge085"], 0)
        self.assertEqual(len(c["low_feedback"]["cells"]), 3)

    def test_static_groups_and_paired_correction(self):
        c = self.r["C_dynamic_exploration"]
        g = c["static_groups"]
        self.assertLess(g["path_low"]["mean_diff"], 0)
        self.assertLess(g["path_high"]["mean_diff"], 0)
        self.assertAlmostEqual(g["corr_path_count_vs_km_diff"], 0.028287594, places=9)
        self.assertAlmostEqual(g["corr_signed_strength_vs_km_diff"], 0.001577481, places=9)
        p = c["paired"]
        self.assertGreater(p["correlations"]["kc_vs_mbon"]["pearson_r"], 0.8)
        self.assertLess(p["correlations"]["entry_vs_kc"]["q_bh"], 1.0)
        self.assertEqual(p["correlations"]["kc_vs_mbon"]["q_bh"], 3.634e-08)
        self.assertEqual(p["sign"]["entry_positive"], 25)

    def test_mechanism_answers_and_gates(self):
        d = self.r["D_mechanism"]
        self.assertTrue(d["answers"]["q1_formal_negative_reproduced"])
        self.assertTrue(d["answers"]["q2_static_reachable_to_matched_kc"])
        self.assertIn("PARTIAL/EXPLORATORY", d["answers"]["q3_local_cancellation"])
        self.assertIn("SUPPORTED_HYPOTHESIS", d["answers"]["q4_most_likely_limiting_stage"])
        self.assertTrue(d["answers"]["q5_ready_for_next_prereg"])
        self.assertIn("Do not execute", d["answers"]["next_experiment_candidate"])
        g = self.r["gates"]
        self.assertFalse(g["NEW_FULL_BRAIN_AUTHORIZATION"])
        self.assertFalse(g["TRAINING_AUTHORIZATION"])
        self.assertEqual(g["REAL_SYNAPTIC_LEARNING_READY"], "NO")
        self.assertEqual(g["Level_A/B/C/Biological"], "NO-GO")
        self.assertEqual(g["Robot"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
