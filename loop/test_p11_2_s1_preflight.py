"""P11-2-S1 静态预检测试。不得运行 FlyBrain 或 S1。"""
from __future__ import annotations

import json
import os
import unittest

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_s1_preflight_report.json")
SRC = os.path.join(HERE, "p11_2_s1_preflight.py")


class P11S1StaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPORT, encoding="utf-8") as f:
            cls.r = json.load(f)
        cls.c = cls.r["two_hop_correction"]
        cls.s = cls.r["sensory"]
        cls.st = cls.r["stats"]
        cls.res = cls.r["resource"]
        cls.fz = cls.r["freeze"]

    def test_status_not_authorized(self):
        self.assertEqual(self.r["status"], "STATIC_PREFLIGHT_COMPLETE; NOT_AUTHORIZED; NO_RUN")
        a = self.r["authorization"]
        self.assertEqual(a["status"], "NOT_AUTHORIZED")
        self.assertFalse(a["s1_execution_authorization"])
        self.assertFalse(a["full_brain_run"])
        self.assertFalse(a["s1_run_generated"])
        self.assertTrue(a["c1_authorization_consumed"])
        self.assertFalse(a["self_approval_config"])
        self.assertTrue(a["no_fabricated_run"])
        self.assertFalse(self.r["full_brain_run"])
        self.assertFalse(self.r["s1_run"])

    def test_two_hop_semantics(self):
        c = self.c
        self.assertEqual(c["kc_endpoint_pairs"], 7883)
        self.assertEqual(c["kc_independent_two_hop_paths"], 8162)
        self.assertEqual(c["kc_paths_per_endpoint_max"], 3)
        self.assertAlmostEqual(c["kc_paths_per_endpoint_mean"], 1.035393, places=6)
        self.assertEqual(c["aggregate_positive_endpoints"], 7783)
        self.assertEqual(c["aggregate_negative_endpoints"], 100)
        self.assertAlmostEqual(c["aggregate_signed_sum"], 0.003759782761, places=12)
        self.assertIn("endpoint pairs", c["s0_reported_7883_is"])
        self.assertIn("NOT independent paths", c["s0_reported_7883_is"])
        self.assertIn("aggregate", c["sign_7783_100_is"])
        self.assertFalse(c["signed_sum_is_dynamic_stimulus_effect"])
        self.assertIn("not a neural response", c["signed_sum_unit"])
        self.assertTrue(c["independent_paths_are_distinct"])
        self.assertEqual(c["mbon_endpoint_pairs"], 2528)
        self.assertEqual(c["mbon_independent_two_hop_paths"], 3265)
        self.assertEqual(c["mbon_paths_per_endpoint_max"], 6)
        self.assertAlmostEqual(c["mbon_paths_per_endpoint_mean"], 1.291535, places=6)
        self.assertEqual(c["direct_src_to_kc"], 0)
        self.assertEqual(c["direct_src_to_mbon"], 1)
        self.assertEqual(c["kc_to_mbon_edges"], 61210)
        self.assertTrue(c["kc_to_mbon_all_positive"])
        self.assertEqual(c["source_count"], 311)
        self.assertEqual(c["kc_count"], 4064)
        self.assertEqual(c["mbon_count"], 97)

    def test_B_preserves_feedforward(self):
        c = self.c
        self.assertEqual(c["kc_B_endpoint_pairs"], 7883)
        self.assertEqual(c["kc_B_independent_paths"], 8162)
        self.assertEqual(c["kc_removed_via_kc_intermediate"], 0)
        self.assertTrue(c["feedforward_preserved_in_B"])
        self.assertEqual(c["mbon_B_endpoint_pairs"], 2528)
        self.assertEqual(c["mbon_B_independent_paths"], 3265)

    def test_sensory_freeze(self):
        s = self.s
        self.assertEqual(s["lplc2"], 185)
        self.assertEqual(s["lc4_exact"], 126)
        self.assertEqual(s["targets"], 311)
        self.assertEqual(s["lc4_startswith"], 227)
        self.assertEqual(s["lc4_startswith_overcount"], 101)
        self.assertIn("exact", s["target_rule"])
        self.assertAlmostEqual(s["amount"], 0.03625384938440364, places=12)
        self.assertAlmostEqual(s["amount_rounded"], 0.036254, places=6)
        self.assertTrue(s["amount_is_0_036254"])
        self.assertTrue(s["amount_is_not_0_6"])
        self.assertEqual(s["stim_window"], [100, 300])
        self.assertFalse(s["tof_mapping_changed"])
        self.assertEqual(s["dW"], 0)
        for k in ("no_kc_mbon_stim", "no_target_change", "no_strength_change", "no_new_connections", "no_teacher"):
            self.assertTrue(s[k])

    def test_stats_and_negative_rule(self):
        st = self.st
        self.assertEqual(st["primary"]["min_positive_effect"], 0.05)
        self.assertEqual(st["primary"]["n"], 25)
        self.assertIn("exact", st["primary"]["test"])
        self.assertIn("bootstrap", st["primary"]["ci"])
        self.assertIn("SENSORY_TRANSMISSION_NOT_DETECTED", st["failure"]["entry_ok_kc_no_positive"])
        self.assertIn("SENSORY_ENTRY_FAILED", st["failure"]["entry_fail"])
        self.assertIn("KC_RESPONSE_WITHOUT_MBON_TRANSMISSION", st["failure"]["kc_ok_mbon_no"])
        nr = st["negative_response_rule"]
        self.assertTrue(nr["negative_does_not_pass_primary"])
        self.assertTrue(nr["negative_does_not_mean_no_change"])
        self.assertTrue(nr["reported_separately"])
        self.assertTrue(st["rules"]["seed_114_kept"])
        self.assertFalse(st["rules"]["cell_step_pseudo_replication"])
        self.assertFalse(st["rules"]["endpoint_switching"])
        self.assertFalse(st["rules"]["min_effect_change"])

    def test_resource(self):
        r = self.res
        self.assertEqual(r["c1_peak_MiB"], 1942.7)
        self.assertAlmostEqual(r["c1_margin_MiB"], 105.3, places=1)
        self.assertAlmostEqual(r["extra_MiB_per_seed_both"], 3.728, places=3)
        self.assertAlmostEqual(r["static_extra_MiB"], 5.73, places=2)
        self.assertAlmostEqual(r["estimated_peak_MiB"], 1948.43, places=2)
        self.assertAlmostEqual(r["estimated_margin_MiB"], 99.57, places=2)
        self.assertEqual(r["wall_cap_s"], 900.0)
        self.assertEqual(r["mem_cap_MiB"], 2048)
        self.assertEqual(r["status"], "RESOURCE_PLAUSIBLE_TIGHT")
        self.assertFalse(r["verified"])
        self.assertTrue(r["no_auto_increase"])
        self.assertTrue(r["low_memory_per_seed"])
        self.assertTrue(r["job_peak_persist_before_postprocess"])
        self.assertTrue(r["create_suspended_job_resume"])
        self.assertTrue(r["w_b_under_job"])
        self.assertIn("no retry", r["stop_rule"])

    def test_freeze_and_official(self):
        f = self.fz
        self.assertTrue(f["P1_mapfix_freeze_empty"])
        self.assertEqual(f["P1_mapfix"], "db401f571712140ed4c513c59d71088a7f3ad2bd")
        self.assertFalse(f["BASE_tag_present"])
        self.assertFalse(f["readout_working_tree_modified"])
        self.assertEqual(f["tags"]["P1-mapfix"], "db401f571712140ed4c513c59d71088a7f3ad2bd")
        self.assertTrue(f["dirty_files_preserved"])
        self.assertEqual(f["dirty_files"], ["hardware/tof5_client.py", "loop/diagnose.py", "loop/p10c1b_decoder_freeze.json"])
        self.assertEqual(f["head"], "6fe7db926f73040fc54c8375f1406dba37f02dd1")
        self.assertTrue(self.r["official_hashes"]["ok"])
        self.assertEqual(self.r["official_hashes"]["files"]["brain.npz"],
                         "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca")
        self.assertEqual(self.r["official_hashes"]["files"]["weights.npz"],
                         "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c")

    def test_synthetic_endpoint_vs_paths(self):
        # 0=source, 1,2=intermediates, 3=KC, 4=MBON; W[post,pre]
        d = np.zeros((5, 5), np.float32)
        d[1, 0] = 0.5; d[2, 0] = 0.4       # source -> two intermediates
        d[3, 1] = 0.6; d[3, 2] = 0.7       # intermediates -> KC
        d[4, 3] = 0.8                       # KC -> MBON
        d[4, 0] = 0.1                       # direct source -> MBON
        W = sp.csr_matrix(d)
        Wb = W.copy(); Wb.data = np.ones_like(Wb.data)
        P = (Wb[3:4] @ Wb[:, 0:1]).tocoo()
        A = (W[3:4] @ W[:, 0:1]).tocoo()
        self.assertEqual(P.nnz, 1)                 # one aggregated endpoint pair
        self.assertEqual(int(P.data.sum()), 2)     # two independent paths
        self.assertAlmostEqual(float(A.data.sum()), 0.58, places=6)  # 0.5*0.6+0.4*0.7
        self.assertEqual(A.nnz, 1)
        # B has no KC->KC; feedforward and KC->MBON preserved
        self.assertEqual((W[4:5][:, 3:4]).nnz, 1)
        self.assertEqual((W[4:5][:, 0:1]).nnz, 1)

    def test_no_dynamics_source(self):
        with open(SRC, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("FlyBrain(", src)
        self.assertNotIn("from flybrain", src)
        self.assertNotIn(".step(", src)
        self.assertNotIn("run_under_job", src)
        self.assertNotIn("p11_2_c1_experiment", src)

    def test_verdicts(self):
        v = self.r["verdicts"]
        self.assertEqual(v["AUTHORIZATION"], "NOT_AUTHORIZED")
        self.assertFalse(v["S1_EXECUTION"])
        self.assertFalse(v["FULL_BRAIN_RUN"])
        self.assertTrue(v["S0_ENDPOINT_PAIRS_7883"])
        self.assertTrue(v["S0_INDEPENDENT_PATHS_8162"])
        self.assertTrue(v["S0_SIGNED_SUM_IS_NOT_DYNAMIC"])
        self.assertTrue(v["NEGATIVE_RESPONSES_EXPLORATORY"])
        self.assertTrue(v["PRIMARY_FROZEN"])
        self.assertEqual(v["RESOURCE"], "RESOURCE_PLAUSIBLE_TIGHT")
        self.assertFalse(v["RESOURCE_VERIFIED"])
        self.assertTrue(v["S1_READY_TO_APPLY"])
        self.assertFalse(v["TRAINING_AUTHORIZATION"])
        self.assertEqual(v["REAL_SYNAPTIC_LEARNING_READY"], "NO")
        self.assertEqual(v["Level_A/B/C/Biological"], "NO-GO")


if __name__ == "__main__":
    unittest.main(verbosity=2)
