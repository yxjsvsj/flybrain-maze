"""P11-2-C1-0 静态/合成回归测试。不得运行完整脑或 P11-2-C1 干预。"""
from __future__ import annotations

import json
import os
import unittest

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_c1_preflight_report.json")
SRC = os.path.join(HERE, "p11_2_c1_preflight.py")


class P11C10Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPORT, encoding="utf-8") as f:
            cls.r = json.load(f)
        cls.m = cls.r["mechanism_localization"]
        cls.w = cls.r["W_B"]["verification"]
        cls.n = cls.r["noise"]
        cls.res = cls.r["resource"]
        cls.st = cls.r["stats"]
        cls.fz = cls.r["freeze"]
        cls.v = cls.r["verdicts"]

    def test_protocol(self):
        self.assertEqual(self.r["protocol"], "P11-2-C1-0/v1")
        self.assertEqual(self.r["status"], "PREFLIGHT_COMPLETE; NO_DYNAMICS_RUN")
        self.assertFalse(self.r["training_authorized"])

    def test_archive_and_hashes(self):
        self.assertEqual(self.r["archive"]["status"], "VERIFIED")
        self.assertEqual(self.r["archive"]["file_count"], 29)
        self.assertTrue(self.r["official_hashes"]["ok"])
        self.assertEqual(self.r["official_hashes"]["files"]["brain.npz"],
                         "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca")
        self.assertEqual(self.r["official_hashes"]["files"]["weights.npz"],
                         "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c")

    def test_mechanism_counts(self):
        self.assertEqual(self.m["W_KC_KC"]["nnz"], 642933)
        self.assertEqual(self.m["W_KC_KC"]["positive_nnz"], 642933)
        self.assertEqual(self.m["W_KC_KC"]["negative_nnz"], 0)
        self.assertEqual(self.m["insufficient_total_events"], 14994)
        self.assertEqual(self.m["unique_kc_step"], 600)
        self.assertEqual(self.m["unique_kcs"], 3)
        self.assertFalse(self.m["spikes_imputed"])

    def test_low_feedback_kcs(self):
        ks = self.m["low_feedback_kcs"]
        self.assertEqual([k["local_index"] for k in ks], [2460, 2650, 3971])
        self.assertEqual([k["body_id"] for k in ks], [109321, 117446, 545375])
        self.assertEqual([k["insufficient_events"] for k in ks], [5000, 4994, 5000])
        self.assertEqual([k["unique_steps"] for k in ks], [200, 200, 200])
        self.assertAlmostEqual(ks[0]["event_input"]["mean"], 0.27907, places=5)
        self.assertAlmostEqual(ks[1]["event_input"]["mean"], 0.132743, places=5)
        self.assertAlmostEqual(ks[2]["event_input"]["mean"], 0.182432, places=5)
        self.assertAlmostEqual(ks[0]["needed_nonKC_at_max_noise"], 0.36093, places=5)
        self.assertAlmostEqual(ks[1]["needed_nonKC_at_max_noise"], 0.507257, places=5)
        self.assertAlmostEqual(ks[2]["needed_nonKC_at_max_noise"], 0.457568, places=5)
        self.assertAlmostEqual(ks[0]["non_kc_weight_budget"]["pos_sum"], 0.761628, places=5)
        self.assertAlmostEqual(ks[0]["non_kc_weight_budget"]["neg_sum"], -0.145349, places=5)
        self.assertAlmostEqual(ks[1]["non_kc_weight_budget"]["pos_sum"], 0.80531, places=5)
        self.assertAlmostEqual(ks[1]["non_kc_weight_budget"]["neg_sum"], -0.150442, places=5)
        self.assertAlmostEqual(ks[2]["non_kc_weight_budget"]["pos_sum"], 0.763514, places=5)
        self.assertAlmostEqual(ks[2]["non_kc_weight_budget"]["neg_sum"], -0.175676, places=5)
        self.assertFalse(any(k["non_kc_weight_budget"]["activity_imputed"] for k in ks))

    def test_kc2650_zeros(self):
        z = self.m["kc_2650_zeros"]
        self.assertEqual([(x["seed"], x["step"]) for x in z], [(104, 232), (114, 213), (115, 234)])
        self.assertTrue(all(x["local_index"] == 2650 and x["body_id"] == 117446 for x in z))
        self.assertTrue(all(abs(x["kc_kc_input"] - 0.132743) < 1e-6 for x in z))

    def test_p110a_crosscheck(self):
        self.assertTrue(self.v["P11_2_0A_CONSISTENCY"])
        c = self.r["p11_2_0a_crosscheck"]["checks"]
        self.assertTrue(all(c.values()))

    def test_W_B_verification(self):
        w = self.w
        self.assertEqual(w["W_nnz"], 25582938)
        self.assertEqual(w["W_B_nnz"], 24940005)
        self.assertEqual(w["expected_W_B_nnz"], 24940005)
        self.assertEqual(w["removed_edges"], 642933)
        self.assertAlmostEqual(w["removed_sum"], 2275.627441, places=6)
        self.assertTrue(w["removed_all_positive"])
        self.assertEqual(w["diff_nnz"], 642933)
        self.assertTrue(w["diff_only_KC_KC"])
        self.assertEqual(w["reverse_diff_nnz"], 642933)
        self.assertTrue(w["reverse_diff_only_KC_KC"])
        self.assertTrue(w["no_new_edges"])
        self.assertEqual(w["W_B_KC_KC_nnz"], 0)
        self.assertTrue(w["KC_KC_zeroed"])
        self.assertTrue(w["removed_data_matches"])
        self.assertTrue(w["other_edges_bitwise_unchanged"])
        self.assertFalse(w["renormalized"])
        self.assertTrue(w["derived_in_memory"])
        self.assertFalse(w["derived_file_written"])
        self.assertTrue(w["official_file_unchanged"])
        self.assertEqual(w["W_B_csc_bytes"], 200186844)
        self.assertEqual(w["W_A_csc_bytes"], 205330308)

    def test_noise_source(self):
        a = self.n["audit"]
        self.assertEqual(a["step_rng_calls"], ["self.rng.random"])
        self.assertTrue(a["one_rng_random_per_step"])
        self.assertEqual(a["rng_random_shape"], "(self.n, B)")
        self.assertTrue(a["drawn_after_current"])
        self.assertTrue(a["drawn_before_firing"])
        self.assertTrue(a["reset_uses_default_rng_seed"])
        self.assertEqual(a["other_rng_calls_in_step"], 0)
        self.assertTrue(a["cpu_only"])
        self.assertTrue(a["strict_pairing"])

    def test_noise_synthetic(self):
        s = self.n["synthetic"]
        self.assertTrue(s["same_seed_raw_equal"])
        self.assertTrue(s["same_seed_events_equal"])
        self.assertTrue(s["same_seed_equal"])
        self.assertTrue(s["different_seed_raw_differs"])
        self.assertTrue(s["condition_independent"])

    def test_resource(self):
        r = self.res
        self.assertEqual(r["total_steps"], 20000)
        self.assertEqual(r["conditions"], ["intact", "kc_kc_zero"])
        self.assertAlmostEqual(r["estimated_wall_s"], 305.08, places=2)
        self.assertEqual(r["wall_cap_s"], 900.0)
        self.assertAlmostEqual(r["wall_margin_s"], 594.92, places=2)
        self.assertAlmostEqual(r["p11_1b_peak_MB"], 1497.87, places=2)
        self.assertEqual(r["mem_cap_MiB"], 2048)
        self.assertAlmostEqual(r["W_B_csc_MiB"], 190.91, places=2)
        self.assertAlmostEqual(r["estimated_peak_with_W_B_MiB"], 1688.78, places=2)
        self.assertAlmostEqual(r["estimated_mem_margin_MiB"], 359.22, places=2)
        self.assertEqual(r["status"], "PLAUSIBLE_UNVERIFIED")
        self.assertFalse(r["verified"])
        self.assertFalse(r["full_brain_run_this_round"])

    def test_stats_prereg(self):
        s = self.st
        self.assertIn("matched 320 KC", s["primary_endpoint"])
        self.assertEqual(s["primary_unit"], "seed")
        self.assertEqual(s["n_pairs"], 25)
        self.assertEqual(s["effect"], "B - A")
        self.assertIn("sign-flip", s["test"])
        self.assertAlmostEqual(s["min_two_sided_p"], 2 / (1 << 25), places=12)
        self.assertAlmostEqual(s["min_meaningful_effect_absolute"], 0.10, places=12)
        self.assertIn("0.10", s["support_criterion"])
        self.assertTrue(s["keep_seed_114"])
        self.assertFalse(s["drop_seeds"])
        self.assertFalse(s["select_seeds"])
        self.assertFalse(s["cell_step_pseudo_replication"])
        self.assertIn("no replacement", s["missing_rule"])
        self.assertIn("900", s["overrun_rule"])
        self.assertIn("2048", s["overrun_rule"])
        self.assertTrue(any("2460" in x for x in s["secondary_endpoints"]))

    def test_freeze(self):
        f = self.fz
        self.assertTrue(f["P1_mapfix_freeze_empty"])
        self.assertEqual(f["P1_mapfix"], "db401f571712140ed4c513c59d71088a7f3ad2bd")
        self.assertTrue(f["P1_baseline_diff_historical_nonempty"])
        self.assertFalse(f["BASE_tag_present"])
        self.assertFalse(f["readout_working_tree_modified"])
        self.assertFalse(f["tags_modified"])
        self.assertEqual(f["head"], "ff8466b614e5c4952d64785bb4d6682063f2e8a4")
        self.assertTrue(f["head_matches_baseline"])
        self.assertTrue(f["dirty_files_preserved"])
        self.assertEqual(f["dirty_files"], ["hardware/tof5_client.py", "loop/diagnose.py", "loop/p10c1b_decoder_freeze.json"])
        self.assertEqual(f["tags"]["P1-mapfix"], "db401f571712140ed4c513c59d71088a7f3ad2bd")

    def test_safety(self):
        s = self.r["safety"]
        self.assertTrue(s["selftest_all_ok"])
        self.assertTrue(s["crash"]["ok"])
        self.assertEqual(s["crash"]["exit_code"], 3)
        self.assertFalse(s["crash"]["timeout"])
        self.assertFalse(s["crash"]["mem_exceeded"])
        self.assertTrue(s["all_synthetic_ok"])
        self.assertTrue(s["selftest"]["no_job_alloc"]["ok"])
        self.assertTrue(s["selftest"]["job_alloc_limit"]["ok"])
        self.assertTrue(s["selftest"]["assign_failure"]["ok"])
        self.assertTrue(s["selftest"]["timeout"]["ok"])
        self.assertTrue(s["selftest"]["no_retry"]["ok"])

    def test_verdicts(self):
        v = self.v
        self.assertFalse(v["P11_2_C1_EXECUTION_AUTHORIZATION"])
        self.assertFalse(v["TRAINING_AUTHORIZATION"])
        self.assertEqual(v["REAL_SYNAPTIC_LEARNING_READY"], "NO")
        self.assertEqual(v["Level_A_Training"], "NO-GO")
        self.assertEqual(v["Level_B"], "NO-GO")
        self.assertEqual(v["Level_C"], "NO-GO")
        self.assertEqual(v["Biological"], "NO-GO")
        self.assertEqual(v["A1_A2_A3_B"], "NO-GO")
        self.assertFalse(v["FULL_BRAIN_RUN_THIS_ROUND"])
        self.assertFalse(v["KC_KC_REMOVAL_EXECUTED"])
        self.assertTrue(v["KC_KC_REMOVAL_STRICT_IMPLEMENTABLE"])
        self.assertTrue(v["NOISE_PAIRING_STRICT"])
        self.assertTrue(v["RESOURCE_PLAUSIBLE"])
        self.assertFalse(v["RESOURCE_VERIFIED"])
        self.assertTrue(v["MINIMAL_PREREG_FROZEN"])
        self.assertTrue(v["READY_TO_APPLY_FOR_ONE_RUN"])

    def test_no_dynamics_in_source(self):
        with open(SRC, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("FlyBrain(", src)
        self.assertNotIn(".step(", src)
        self.assertNotIn("from flybrain import", src)
        self.assertNotIn("--child", src)
        self.assertNotIn("child_full_brain", src)

    def test_tiny_W_B(self):
        from loop.p11_2_c1_preflight import derive_W_B, verify_W_B
        dense = np.zeros((4, 4), np.float32)
        dense[0, 1] = 0.5; dense[1, 0] = 0.25
        dense[0, 2] = 0.3; dense[2, 0] = 0.4
        dense[2, 3] = 0.6; dense[3, 2] = 0.7
        W = sp.csr_matrix(dense)
        WB, removed = derive_W_B(W, np.array([0, 1]))
        v = verify_W_B(W, WB, removed, np.array([0, 1]))
        self.assertEqual(v["removed_edges"], 2)
        self.assertEqual(v["W_B_nnz"], 4)
        self.assertTrue(v["no_new_edges"])
        self.assertEqual(v["W_B_KC_KC_nnz"], 0)
        self.assertTrue(v["other_edges_bitwise_unchanged"])
        expect = dense.copy(); expect[0, 1] = 0; expect[1, 0] = 0
        self.assertTrue(np.array_equal(WB.toarray(), expect))

    def test_exact_signflip_small(self):
        from loop.p11_2_c1_preflight import exact_signflip_p
        def brute(d):
            d = np.asarray(d, float); n = len(d); obs = abs(float(d.sum())); c = 0
            for bits in range(1 << n):
                s = sum(d[i] if (bits >> i) & 1 else -d[i] for i in range(n))
                if abs(s) >= obs - 1e-12:
                    c += 1
            return c / (1 << n)
        self.assertAlmostEqual(exact_signflip_p([1, 1, 1, 1, 1]), 2 / 32, places=12)
        rng = np.random.default_rng(7)
        for n in (5, 6):
            d = rng.normal(size=n)
            self.assertAlmostEqual(exact_signflip_p(d), brute(d), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
