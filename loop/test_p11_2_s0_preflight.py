"""P11-2-S0 静态/合成回归测试。不运行 FlyBrain。"""
from __future__ import annotations

import json
import os
import unittest

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_s0_report.json")
SRC = os.path.join(HERE, "p11_2_s0_preflight.py")


class P11S0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPORT, encoding="utf-8") as f:
            cls.r = json.load(f)
        cls.s = cls.r["sensory"]
        cls.p = cls.r["path_audit"]
        cls.st = cls.r["stats"]
        cls.res = cls.r["resource"]
        cls.fz = cls.r["freeze"]
        cls.v = cls.r["verdicts"]

    def test_official_and_no_dynamics(self):
        self.assertTrue(self.r["official_hashes"]["ok"])
        self.assertEqual(self.r["official_hashes"]["files"]["brain.npz"],
                         "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca")
        self.assertEqual(self.r["official_hashes"]["files"]["weights.npz"],
                         "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c")
        with open(SRC, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("FlyBrain(", src)
        self.assertNotIn("from flybrain", src)
        self.assertNotIn(".step(", src)
        self.assertNotIn("run_under_job", src)
        self.assertNotIn("p11_2_c1_experiment", src)

    def test_sensory_targets_and_protocol(self):
        s = self.s
        self.assertEqual(s["lplc2_count"], 185)
        self.assertEqual(s["lc4_exact_count"], 126)
        self.assertEqual(s["lc4_startswith_count"], 227)
        self.assertEqual(s["lc4_startswith_overcount"], 101)
        self.assertEqual(s["lc4_extra_types"], ["LC40", "LC41", "LC43", "LC44", "LC45", "LC46"])
        self.assertEqual(s["target_count_exact"], 311)
        self.assertEqual(s["source_body_ids_unique"], 311)
        self.assertIn("exact", s["target_rule"])
        pr = s["protocol"]
        self.assertAlmostEqual(pr["amount_exact"], 0.03625384938440364, places=12)
        self.assertAlmostEqual(pr["amount_rounded"], 0.036254, places=6)
        self.assertTrue(pr["amount_is_0_036254"])
        self.assertTrue(pr["amount_is_not_0_6"])
        self.assertEqual(pr["stim_window"], [100, 300])
        self.assertFalse(pr["tof_mapping_changed"])
        self.assertEqual(pr["dW"], 0)
        self.assertTrue(s["source_protocol_audit"]["p10c2_0b_DV_0_2"])
        self.assertTrue(s["source_protocol_audit"]["p10c2_0b_AMOUNT_0_036254"])

    def test_path_audit(self):
        p = self.p
        self.assertEqual(p["kc_count"], 4064)
        self.assertEqual(p["mbon_count"], 97)
        self.assertEqual(p["matched_kc"], 320)
        self.assertEqual(p["matched_mbon"], 95)
        self.assertEqual(p["source_count"], 311)
        self.assertEqual(p["direct_src_to_kc"]["edges"], 0)
        self.assertEqual(p["direct_src_to_mbon"]["edges"], 1)
        self.assertEqual(p["two_hop_src_to_kc_A"]["edges"], 7883)
        self.assertEqual(p["two_hop_src_to_kc_A"]["positive"], 7783)
        self.assertEqual(p["two_hop_src_to_kc_A"]["negative"], 100)
        self.assertAlmostEqual(p["two_hop_src_to_kc_A"]["signed_sum"], 0.003759783, places=9)
        self.assertEqual(p["two_hop_src_to_kc_A_bool_paths"], 7883)
        self.assertEqual(p["removed_via_kc_intermediate_bool_paths"], 0)
        self.assertEqual(p["two_hop_src_to_kc_B_bool_paths"], 7883)
        self.assertTrue(p["feedforward_paths_preserved_in_B"])
        self.assertFalse(p["sensory_feedforward_uses_kc_kc"])
        self.assertEqual(p["two_hop_src_to_mbon_A_bool_paths"], 2528)
        self.assertEqual(p["three_hop_via_kc_kc_removed_bool_paths"], 0)
        self.assertEqual(p["kc_to_mbon"]["edges"], 61210)
        self.assertEqual(p["kc_to_mbon"]["negative"], 0)
        self.assertAlmostEqual(p["kc_to_mbon"]["signed_sum"], 56.178184509, places=6)
        self.assertEqual(p["mbon_unique_types"], 36)
        self.assertEqual(p["recurrent_kc_kc_edges_removed"], 642933)
        self.assertTrue(p["anatomical_reachability_not_dynamical_detectability"])

    def test_synthetic_path_feedforward(self):
        # 0=source, 1=nonKC, 2=KC, 3=KC, 4=MBON
        d = np.zeros((5, 5), np.float32)
        d[1, 0] = 0.5; d[2, 1] = 0.4       # feedforward source->nonKC->KC (W[post,pre])
        d[3, 2] = 0.3                       # KC->KC
        d[4, 2] = 0.6                       # KC->MBON
        d[4, 0] = 0.1                       # direct source->MBON
        d[4, 1] = 0.2                       # source->nonKC->MBON
        W = sp.csr_matrix(d)
        kc = np.array([2, 3]); src = np.array([0])
        A = (W[kc][:, src]).tocoo()
        A2 = (W[kc] @ W[:, src]).tocoo()
        B_dense = d.copy(); B_dense[3, 2] = 0
        B = sp.csr_matrix(B_dense)
        B2 = (B[kc] @ B[:, src]).tocoo()
        self.assertEqual(A.nnz, 0)                    # no direct source->KC
        self.assertEqual(A2.nnz, 1)                   # source->nonKC->KC
        self.assertEqual(B2.nnz, 1)                   # preserved in B
        self.assertEqual((W[4:5][:, kc]).nnz, 1)      # KC->MBON
        self.assertEqual((B[4:5][:, kc]).nnz, 1)      # preserved
        self.assertEqual((W[4:5][:, src]).nnz, 1)     # direct source->MBON
        self.assertEqual((W[4:5] @ W[:, src]).nnz, 1) # source->nonKC->MBON
        self.assertEqual(d[3, 2], 0.3); self.assertEqual(B_dense[3, 2], 0.0)

    def test_c1a_statistical_note(self):
        n = self.r["c1a_statistical_note"]
        self.assertAlmostEqual(n["full_B_mean"], 0.351093307, places=9)
        self.assertAlmostEqual(n["low3_contribution"], 0.000646358, places=9)
        self.assertAlmostEqual(n["rest_4061_group_mean"], 0.350705836, places=9)
        self.assertAlmostEqual(n["rest_4061_contribution_to_full_4064_mean"], 0.350446949, places=9)
        self.assertAlmostEqual(n["sum_check"], 0.351093307, places=9)
        self.assertAlmostEqual(n["rest_4061_group_mean"] * 4061 / 4064,
                               n["rest_4061_contribution_to_full_4064_mean"], places=9)
        self.assertFalse(n["historical_c1a_report_overwritten"])

    def test_stats_prereg(self):
        s = self.st
        self.assertIn("311", s["entry_positive_control"]["targets"])
        self.assertEqual(s["primary"]["min_meaningful_effect"], 0.05)
        self.assertEqual(s["primary"]["n"], 25)
        self.assertIn("SENSORY_TRANSMISSION_NOT_DETECTED", s["failure_conditions"]["entry_ok_kc_no"])
        self.assertIn("SENSORY_ENTRY_FAILED", s["failure_conditions"]["entry_fail"])
        self.assertIn("KC_RESPONSE_WITHOUT_MBON_TRANSMISSION", s["failure_conditions"]["kc_ok_mbon_no"])
        self.assertFalse(s["reuse_c1b"]["reuse_historical_C1_B_none"])
        self.assertIn("same-experiment", s["reuse_c1b"]["reason"])
        self.assertFalse(s["unit_rules"]["cell_step_pseudo_replication"])
        self.assertFalse(s["unit_rules"]["endpoint_switching"])
        self.assertTrue(s["unit_rules"]["seed_114_kept"])
        self.assertIn("sign-flip", s["none_drift_false_positive"]["test"])

    def test_resource(self):
        r = self.res
        self.assertEqual(r["c1_peak_MiB"], 1942.7)
        self.assertAlmostEqual(r["c1_margin_MiB"], 105.3, places=1)
        self.assertAlmostEqual(r["record_MiB_per_seed_both_conditions"], 3.728, places=3)
        self.assertAlmostEqual(r["static_extra_peak_MiB"], 5.73, places=2)
        self.assertAlmostEqual(r["estimated_peak_MiB"], 1948.43, places=2)
        self.assertAlmostEqual(r["estimated_margin_MiB"], 99.57, places=2)
        self.assertEqual(r["status"], "RESOURCE_PLAUSIBLE_TIGHT")
        self.assertFalse(r["verified"])
        self.assertTrue(r["no_auto_increase"])
        self.assertIn("save each seed", r["low_memory"])
        self.assertIn("2048", r["no_go_if"])
        self.assertTrue(r["job_peak_persist_before_postprocess"])

    def test_s1_not_authorized(self):
        a = self.r["s1_approval"]
        self.assertEqual(a["status"], "NOT_AUTHORIZED; NOT_RUN")
        self.assertTrue(a["separate_authorization_required"])
        self.assertTrue(a["c1_one_shot_consumed"])
        self.assertFalse(a["self_approval_config"])
        self.assertFalse(a["training_authorized"])
        self.assertIn("LPLC2/LC4 exact 311", a["request"])
        self.assertIn("0.036254", a["request"])

    def test_freeze(self):
        f = self.fz
        self.assertTrue(f["P1_mapfix_freeze_empty"])
        self.assertEqual(f["P1_mapfix"], "db401f571712140ed4c513c59d71088a7f3ad2bd")
        self.assertFalse(f["BASE_tag_present"])
        self.assertFalse(f["readout_working_tree_modified"])
        self.assertEqual(f["tags"]["P1-mapfix"], "db401f571712140ed4c513c59d71088a7f3ad2bd")
        self.assertTrue(f["dirty_files_preserved"])
        self.assertEqual(f["dirty_files"], ["hardware/tof5_client.py", "loop/diagnose.py", "loop/p10c1b_decoder_freeze.json"])
        self.assertEqual(f["head"], "4f017c2b22d748555d50d4c6258d5c80bbe33347")

    def test_verdicts(self):
        v = self.v
        self.assertEqual(v["DIRECT_LPLC2_LC4_TO_KC"], "ABSENT (0 edges)")
        self.assertEqual(v["FEEDFORWARD_KC_PATH"], "2-hop via non-KC; preserved in B")
        self.assertEqual(v["KC_KC_RECURRENT"], "removed (642,933)")
        self.assertEqual(v["KC_TO_MBON"], "61,210 existing edges, all positive; preserved in B")
        self.assertEqual(v["ANATOMICAL_REACHABILITY"], "NOT EQUAL TO DYNAMICAL DETECTABILITY")
        self.assertTrue(v["PRIMARY_ENDPOINT_FROZEN"])
        self.assertEqual(v["RESOURCE"], "RESOURCE_PLAUSIBLE_TIGHT")
        self.assertFalse(v["RESOURCE_VERIFIED"])
        self.assertFalse(v["S1_EXECUTION_AUTHORIZATION"])
        self.assertTrue(v["S1_READY_TO_APPLY"])
        self.assertFalse(v["TRAINING_AUTHORIZATION"])
        self.assertEqual(v["REAL_SYNAPTIC_LEARNING_READY"], "NO")
        self.assertEqual(v["Level_A/B/C/Biological"], "NO-GO")


if __name__ == "__main__":
    unittest.main(verbosity=2)
