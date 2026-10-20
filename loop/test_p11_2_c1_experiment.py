"""P11-2-C1 静态/合成回归测试（不运行完整脑）。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp

import loop.p11_2_c1_experiment as e


class P11C1Test(unittest.TestCase):
    def test_resource_mib_conversion(self):
        self.assertAlmostEqual(e.P11_1B_PEAK_MB, 1497.87, places=2)
        self.assertAlmostEqual(e.P11_1B_PEAK_MB / 1.048576, 1428.48, places=2)
        self.assertAlmostEqual(e.P11_1B_PEAK_MIB + e.W_B_MIB, 1619.39, places=2)
        self.assertAlmostEqual(2048 - (e.P11_1B_PEAK_MIB + e.W_B_MIB), 428.61, places=2)

    def test_synaptic_input_fields_source(self):
        import flybrain
        path = os.path.join(os.path.dirname(flybrain.__file__), "brain.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        step = src.split("def step(")[1].split("\n    def ")[0]
        syn = src.split("def synaptic_input(")[1].split("\n    def ")[0]
        self.assertIn("self.rng.random((self.n, B))", step)
        self.assertIn("self.synaptic_input(self.fired)", step)
        self.assertIn("_propagate(self.indptr, self.indices, self.weights", syn)
        self.assertIn("self.n", syn)
        self.assertIn("self._W", syn)  # CUDA branch only
        self.assertIn("if self.device == \"cuda\"", syn)

    def test_w_b_tiny(self):
        n = 4
        dense = np.zeros((n, n), np.float32)
        dense[0, 1] = 0.5; dense[1, 0] = 0.25
        dense[0, 2] = 0.3; dense[2, 0] = 0.4
        dense[2, 3] = 0.6; dense[3, 2] = 0.7
        W = sp.csr_matrix(dense)
        A = (W.indptr, W.indices, W.data)
        is_kc = np.array([True, True, False, False])
        B, removed, rsum = e.build_W_B_csc(*A, is_kc)
        v = e.verify_W_B_fields(A, B, is_kc, np.array([0, 1]))
        self.assertTrue(v["ok"])
        self.assertEqual(removed, 2)
        self.assertAlmostEqual(rsum, 0.75, places=6)
        self.assertEqual(v["B_KC_KC_nnz"], 0)
        self.assertTrue(v["bitwise_other_columns"])
        self.assertEqual(len(B[1]), 4)

    def test_synthetic_tiny_end_to_end(self):
        r = e.synthetic_tiny()
        self.assertTrue(r["ok"])
        c = r["child"]
        self.assertTrue(c["ok"])
        self.assertEqual(c["n_neurons"], 4)
        self.assertEqual(c["nnz_A"], 6)
        self.assertEqual(c["nnz_B"], 4)
        self.assertEqual(c["removed_edges"], 2)
        self.assertEqual(c["KC_KC_nnz"], 0)
        self.assertTrue(c["noise_pairing_all"])
        self.assertTrue(c["field_switch"]["A_restore_exact"])
        self.assertTrue(c["field_switch"]["non_KC_unchanged"])
        self.assertGreater(c["field_switch"]["KC_posts_changed"], 0)
        self.assertEqual(len(c["seeds"]), 2)
        for s in c["seeds"]:
            self.assertTrue(s["A_finite"] and s["B_finite"])
            self.assertTrue(s["noise_pair_ok"])

    def test_exact_signflip_zero_and_float(self):
        self.assertAlmostEqual(e._exact_signflip([0, 0, 0, 0, 0]), 1.0, places=12)
        self.assertAlmostEqual(e._exact_signflip([1, 1, 1, 1, 1]), 2 / 32, places=12)
        self.assertAlmostEqual(e._exact_signflip([-1, -1, -1, -1, -1]), 2 / 32, places=12)

    def test_bootstrap_constant(self):
        ci = e._bootstrap_ci([-1.0] * 25)
        self.assertAlmostEqual(ci["ci_low"], -1.0, places=12)
        self.assertAlmostEqual(ci["ci_high"], -1.0, places=12)
        self.assertEqual(ci["n"], 10000)
        self.assertEqual(ci["seed"], 20261019)

    def test_auth_payload(self):
        a = e.auth_file("test approval")
        self.assertEqual(a["protocol"], "P11-2-C1/v1")
        self.assertEqual(a["status"], "APPROVED")
        self.assertEqual(a["approved_by"], "user")
        self.assertEqual(a["scope"], e._scope())
        self.assertEqual(a["scope"]["seeds"], list(range(101, 126)))
        self.assertEqual(a["scope"]["total_steps"], 20000)
        self.assertEqual(a["scope"]["wall_cap_s"], 900.0)
        self.assertEqual(a["scope"]["mem_cap_MiB"], 2048)
        self.assertIn("checksum", a)

    def test_analyze_synthetic_effect(self):
        tmp = tempfile.mkdtemp(prefix="p11_2_c1_stats_")
        np.savez(os.path.join(tmp, "ids.npz"), ids=np.arange(4064), kc_idx=np.arange(4064),
                 mb_idx=np.arange(97), kc_matched=np.arange(320), mb_matched=np.arange(95))
        ones = np.ones((400, 4064), np.uint8); zeros = np.zeros((400, 4064), np.uint8)
        mb1 = np.ones((400, 97), np.uint8); mb0 = np.zeros((400, 97), np.uint8)
        for s in e.SEEDS:
            np.savez(os.path.join(tmp, f"{s}_A.npz"), kc=ones, mb=mb1, kc_matched=np.ones((400, 320), np.uint8),
                     mb_matched=np.ones((400, 95), np.uint8), rng_before="x", rng_after="y")
            np.savez(os.path.join(tmp, f"{s}_B.npz"), kc=zeros, mb=mb0, kc_matched=np.zeros((400, 320), np.uint8),
                     mb_matched=np.zeros((400, 95), np.uint8), rng_before="x", rng_after="y")
        rep = e.analyze(tmp)
        p = rep["primary"]
        self.assertEqual(rep["completeness"], "25/25")
        self.assertAlmostEqual(p["A_mean"], 1.0, places=6)
        self.assertAlmostEqual(p["B_mean"], 0.0, places=6)
        self.assertAlmostEqual(p["effect_mean"], -1.0, places=6)
        self.assertAlmostEqual(p["exact_signflip_p"], 2 / (1 << 25), places=12)
        self.assertAlmostEqual(p["ci"]["ci_low"], -1.0, places=6)
        self.assertAlmostEqual(p["ci"]["ci_high"], -1.0, places=6)
        self.assertEqual(p["verdict"], "CAUSAL_CONTRIBUTION_SUPPORTED")
        self.assertEqual(rep["sensory_readiness"], "YES_FOR_SEPARATE_PROPOSAL")
        self.assertEqual(rep["secondary"]["low_feedback_A"], {"2460": 1.0, "2650": 1.0, "3971": 1.0})
        self.assertEqual(rep["secondary"]["low_feedback_B"], {"2460": 0.0, "2650": 0.0, "3971": 0.0})
        self.assertFalse(rep["gate"]["Training_Authorization"])
        self.assertEqual(rep["gate"]["REAL_SYNAPTIC_LEARNING_READY"], "NO")

    def test_completed_run_report(self):
        rp = os.path.join(os.path.dirname(__file__), "p11_2_c1_report.json")
        mp = os.path.join(os.path.dirname(__file__), "p11_2_c1_run_meta.json")
        fp = os.path.join(os.path.dirname(__file__), "p11_2_c1_raw_manifest.json")
        with open(rp, encoding="utf-8") as f:
            r = json.load(f)
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        with open(fp, encoding="utf-8") as f:
            man = json.load(f)
        self.assertEqual(r["status"], "COMPLETED_25_OF_25")
        self.assertEqual(r["completeness"], "25/25")
        self.assertTrue(r["complete_25_pairs"])
        p = r["primary"]
        self.assertAlmostEqual(p["A_mean"], 0.956807, places=6)
        self.assertAlmostEqual(p["B_mean"], 0.194136, places=6)
        self.assertAlmostEqual(p["effect_mean"], -0.762671, places=6)
        self.assertAlmostEqual(p["exact_signflip_p"], 2 / (1 << 25), places=12)
        self.assertAlmostEqual(p["ci"]["ci_low"], -0.766145, places=6)
        self.assertAlmostEqual(p["ci"]["ci_high"], -0.759004, places=6)
        self.assertEqual(p["verdict"], "CAUSAL_CONTRIBUTION_SUPPORTED")
        self.assertEqual(r["sensory_readiness"], "YES_FOR_SEPARATE_PROPOSAL")
        self.assertEqual(r["secondary"]["full_4064_B_cells_ge085"], 2)
        self.assertAlmostEqual(r["secondary"]["low_feedback_B"]["2460"], 0.9545, places=4)
        self.assertAlmostEqual(r["secondary"]["low_feedback_B"]["2650"], 0.6016, places=4)
        self.assertAlmostEqual(r["secondary"]["low_feedback_B"]["3971"], 0.9516, places=4)
        ch = r["child"]
        self.assertTrue(ch["ok"])
        self.assertEqual(ch["nnz_A"], 25582938)
        self.assertEqual(ch["nnz_B"], 24940005)
        self.assertEqual(ch["removed_edges"], 642933)
        self.assertEqual(ch["KC_KC_nnz"], 0)
        self.assertTrue(ch["noise_pairing_all"])
        self.assertTrue(ch["W_B_verify"]["ok"])
        self.assertTrue(ch["field_switch"]["A_restore_exact"])
        self.assertTrue(ch["field_switch"]["non_KC_unchanged"])
        self.assertEqual(m["status"], "RUN_COMPLETE")
        self.assertEqual(m["exit_code"], 0)
        self.assertFalse(m["timeout"])
        self.assertFalse(m["mem_exceeded"])
        self.assertAlmostEqual(m["job_peak_MiB"], 1942.7, places=1)
        self.assertAlmostEqual(m["run_wall_s"], 290.38, places=2)
        self.assertTrue(m["official_hashes"]["ok"])
        self.assertEqual(m["script_sha256"], "136bb62890165af551e1a65ae3db9d24b8085755106de3d4b15b19c11eeb1543")
        self.assertEqual(m["authorization_sha256"], "57ac353a449edb8a6bdaa8f4da7039db34df24b8356bb559e4333503662b57b4")
        self.assertEqual(len(man["files"]), 57)
        self.assertFalse(r["gate"]["Training_Authorization"])
        self.assertEqual(r["gate"]["REAL_SYNAPTIC_LEARNING_READY"], "NO")


if __name__ == "__main__":
    unittest.main(verbosity=2)
