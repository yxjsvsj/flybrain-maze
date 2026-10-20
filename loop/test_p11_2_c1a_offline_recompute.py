"""P11-2-C1a 离线统计勘误/归档/来源审计测试。不运行任何动力学。"""
from __future__ import annotations

import json
import os
import unittest

import numpy as np

from loop.p11_2_c1a_offline_recompute import window_mean, window_cells, longest

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_c1a_report.json")
ARCHIVE = os.path.join(HERE, "p11_2_c1a_archive_manifest.json")
SRC = os.path.join(HERE, "p11_2_c1a_offline_recompute.py")


class P11C1aTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPORT, encoding="utf-8") as f:
            cls.r = json.load(f)
        with open(ARCHIVE, encoding="utf-8") as f:
            cls.a = json.load(f)
        cls.p = cls.r["primary_errata"]
        cls.s = cls.r["secondary"]

    def test_window_guard_not_full_400(self):
        kc = np.zeros((400, 10), np.uint8)
        kc[100:300] = 1
        self.assertAlmostEqual(float(kc.mean()), 0.5, places=12)
        self.assertAlmostEqual(window_mean(kc), 1.0, places=12)
        self.assertAlmostEqual(float(window_cells(kc).mean()), 1.0, places=12)
        self.assertEqual(int(longest(kc).max()), 200)
        self.assertEqual(int(longest(kc[100:300]).max()), 200)
        self.assertNotEqual(window_mean(kc), float(kc.mean()))

    def test_primary_corrected(self):
        self.assertEqual(self.p["window"], [100, 300])
        self.assertAlmostEqual(self.p["A_mean"], 1.0, places=9)
        self.assertAlmostEqual(self.p["B_mean"], 0.201393125, places=9)
        self.assertAlmostEqual(self.p["mean"], -0.798606875, places=9)
        self.assertAlmostEqual(self.p["sd"], 0.010585539, places=9)
        self.assertAlmostEqual(self.p["median"], -0.798671875, places=9)
        self.assertAlmostEqual(self.p["exact_signflip_p"], 2 / (1 << 25), places=12)
        self.assertAlmostEqual(self.p["ci"]["ci_low"], -0.80257, places=6)
        self.assertAlmostEqual(self.p["ci"]["ci_high"], -0.794391, places=6)
        self.assertEqual(self.p["min_meaningful_effect"], -0.10)
        self.assertTrue(self.p["support"])
        self.assertEqual(self.p["verdict"], "CAUSAL_CONTRIBUTION_SUPPORTED")
        self.assertEqual(len(self.p["A"]), 25)
        self.assertEqual(len(self.p["B"]), 25)
        self.assertEqual(len(self.p["diffs"]), 25)

    def test_historical_primary_is_flagged(self):
        h = self.r["historical_primary"]
        self.assertIn("full 400", h["window"])
        self.assertAlmostEqual(h["A"], 0.956807, places=6)
        self.assertAlmostEqual(h["B"], 0.194136, places=6)
        self.assertAlmostEqual(h["effect"], -0.762671, places=6)
        self.assertNotAlmostEqual(h["effect"], self.p["mean"], places=3)

    def test_secondary_full_and_matched(self):
        f = self.s["full_4064_main"]
        self.assertAlmostEqual(f["A_population"], 0.999999852, places=9)
        self.assertAlmostEqual(f["B_population"], 0.351093307, places=9)
        self.assertEqual(f["A_cells"]["n_cells_per_seed"], 4064)
        self.assertEqual(f["B_cells"]["n_cells_per_seed"], 4064)
        self.assertEqual(f["A_cells"]["cells_ge085"], 101600)
        self.assertEqual(f["B_cells"]["cells_ge085"], 63)
        self.assertEqual(f["B_cells"]["cells_ge095"], 52)
        self.assertAlmostEqual(f["B_cells"]["frac_cells_ge085_mean_per_seed"], 0.000620079, places=9)
        self.assertAlmostEqual(f["B_cells"]["cells_ge085_mean_per_seed"], 2.52, places=4)
        self.assertEqual(f["B_cells"]["cells_ge085_min_per_seed"], 2)
        self.assertEqual(f["B_cells"]["cells_ge085_max_per_seed"], 4)
        self.assertEqual(f["B_cells"]["cells_ge085_per_seed"],
                         [2, 2, 2, 3, 4, 2, 3, 2, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 4, 3, 2, 2])
        m = self.s["matched_320_main"]
        self.assertAlmostEqual(m["A_population"], 1.0, places=9)
        self.assertAlmostEqual(m["B_population"], 0.201393125, places=9)
        self.assertEqual(m["B_cells"]["cells_ge085"], 0)
        self.assertEqual(m["B_cells"]["cells_ge095"], 0)
        self.assertEqual(m["B_cells"]["n_cells_per_seed"], 320)

    def test_secondary_mbon(self):
        b = self.s["mbon_main"]
        self.assertAlmostEqual(b["full97_A"], 0.922560825, places=9)
        self.assertAlmostEqual(b["full97_B"], 0.460771134, places=9)
        self.assertAlmostEqual(b["matched95_A"], 0.932690526, places=9)
        self.assertAlmostEqual(b["matched95_B"], 0.465776842, places=9)
        self.assertEqual(b["full97_A_cells"]["n_cells_per_seed"], 97)
        self.assertEqual(b["full97_A_cells"]["cells_ge085"], 2075)
        self.assertEqual(b["full97_B_cells"]["cells_ge085"], 0)
        self.assertEqual(b["matched95_A_cells"]["n_cells_per_seed"], 95)
        self.assertEqual(b["matched95_A_cells"]["cells_ge085"], 2075)
        self.assertEqual(b["matched95_B_cells"]["cells_ge085"], 0)

    def test_secondary_startup_and_longest(self):
        st = self.s["startup_full_400"]
        self.assertAlmostEqual(st["A_first_100"], 0.829078, places=6)
        self.assertAlmostEqual(st["B_first_100"], 0.298447, places=6)
        self.assertAlmostEqual(st["A_main"], 0.999999852, places=9)
        self.assertAlmostEqual(st["B_main"], 0.351093307, places=9)
        self.assertEqual(len(st["A_bins_20"]), 20)
        self.assertAlmostEqual(st["matched_320_A_first_100"], 0.827226, places=6)
        self.assertAlmostEqual(st["matched_320_B_first_100"], 0.17473, places=6)
        lr = self.s["longest"]
        self.assertEqual(lr["A_full400_max"], 389)
        self.assertEqual(lr["A_main200_max"], 200)
        self.assertEqual(lr["B_full400_max"], 385)
        self.assertEqual(lr["B_main200_max"], 200)
        self.assertGreater(lr["A_full400_mean"], 300)
        self.assertLess(lr["B_full400_mean"], 3)
        self.assertEqual(lr["matched_320"]["B_main200_max"], 4)
        f = self.s["full_400_vs_main"]
        self.assertAlmostEqual(f["A_full400"], 0.957269464, places=9)
        self.assertAlmostEqual(f["B_full400"], 0.337384055, places=9)
        self.assertAlmostEqual(f["A_main"], 0.999999852, places=9)
        self.assertAlmostEqual(f["B_main"], 0.351093307, places=9)

    def test_low3_not_main_source(self):
        lf = self.s["low_feedback"]
        self.assertEqual([x["body_id"] for x in lf["kcs"]], [109321, 117446, 545375])
        self.assertAlmostEqual(lf["kcs"][0]["A"], 1.0, places=6)
        self.assertAlmostEqual(lf["kcs"][1]["A"], 0.9994, places=6)
        self.assertAlmostEqual(lf["kcs"][2]["A"], 1.0, places=6)
        self.assertAlmostEqual(lf["kcs"][0]["B"], 1.0, places=6)
        self.assertAlmostEqual(lf["kcs"][1]["B"], 0.6274, places=6)
        self.assertAlmostEqual(lf["kcs"][2]["B"], 0.9994, places=6)
        self.assertAlmostEqual(lf["low3_contribution_to_full_B"], 0.000646358, places=9)
        self.assertAlmostEqual(lf["rest_4061_mean"], 0.350705836, places=9)
        self.assertAlmostEqual(lf["full_B_mean"], 0.351093307, places=9)
        self.assertFalse(lf["low3_is_main_source_of_remaining_activity"])
        self.assertGreater(lf["rest_4061_mean"], 100 * lf["low3_contribution_to_full_B"])

    def test_a_vs_p11c(self):
        a = self.r["A_vs_P11_1c"]
        self.assertEqual(a["bitwise_equal_seeds"], 25)
        self.assertEqual(a["total_seeds"], 25)
        self.assertEqual(a["diff_cells_counts"], {"0": 25})
        self.assertAlmostEqual(a["matched_main_p11_1c_mean"], 1.0, places=9)
        self.assertAlmostEqual(a["matched_main_c1A_mean"], 1.0, places=9)
        self.assertAlmostEqual(a["full_main_p11_1c_mean"], 0.999999852, places=9)
        self.assertAlmostEqual(a["full_main_c1A_mean"], 0.999999852, places=9)
        self.assertAlmostEqual(a["full400_p11_1c_mean"], 0.957269464, places=9)
        self.assertAlmostEqual(a["full400_c1A_mean"], 0.957269464, places=9)
        self.assertFalse(a["noise_pairing_assumption"])
        self.assertTrue(a["observed_spike_arrays_identical"])

    def test_archive(self):
        a = self.a
        self.assertEqual(a["file_count"], 57)
        self.assertTrue(a["source_verified"])
        self.assertTrue(a["archive_verified"])
        self.assertTrue(a["backup_verified"])
        self.assertTrue(a["all_copies_verified"])
        self.assertFalse(a["raw_unverified"])
        self.assertFalse(a["originals_modified"])
        self.assertFalse(a["npz_committed_to_git"])
        self.assertEqual(len(a["files"]), 57)
        self.assertTrue(all(x["sha256"] == x["archive_sha256"] == x["backup_sha256"] for x in a["files"]))
        self.assertGreater(a["total_bytes"], 8_000_000)
        self.assertIn("flybrain-maze-archive", a["archive_dir"])
        self.assertIn("backup", a["backup_dir"])

    def test_provenance(self):
        p = self.r["provenance"]
        self.assertEqual(p["authorization_provenance"], "VERIFIED")
        self.assertTrue(p["auth_checksum_ok"])
        self.assertTrue(p["approval_matches_known_user_text"])
        self.assertTrue(p["auth_matches_run_meta"])
        self.assertTrue(p["auth_matches_child"])
        self.assertTrue(p["script_matches_run_meta"])
        self.assertFalse(p["independent_signature"])
        self.assertFalse(p["timestamp_fabricated"])
        self.assertIn("chat", p["evidence"])
        self.assertEqual(p["auth_file_checksum"], "57ac353a449edb8a6bdaa8f4da7039db34df24b8356bb559e4333503662b57b4")
        self.assertEqual(p["freeze_sha"], "136bb62890165af551e1a65ae3db9d24b8085755106de3d4b15b19c11eeb1543")

    def test_resource_and_run(self):
        q = self.r["resource"]
        self.assertEqual(q["run_record"], "COMPLETE")
        self.assertEqual(q["job_peak_MiB"], 1942.7)
        self.assertEqual(q["run_wall_s"], 290.38)
        self.assertEqual(q["exit_code"], 0)
        self.assertFalse(q["timeout"])
        self.assertFalse(q["mem_exceeded"])
        self.assertTrue(q["official_hashes_ok"])
        self.assertEqual(q["removed_edges"], 642933)
        self.assertEqual(q["nnz_A"], 25582938)
        self.assertEqual(q["nnz_B"], 24940005)
        self.assertEqual(q["KC_KC_nnz"], 0)
        self.assertTrue(q["field_switch"]["A_restore_exact"])
        self.assertTrue(q["field_switch"]["non_KC_unchanged"])
        self.assertTrue(q["noise_pairing_all"])
        self.assertEqual(q["resource_status"], "MEASURED_UNDER_LIMIT")

    def test_verdicts_and_no_dynamics(self):
        v = self.r["verdicts"]
        self.assertEqual(v["EXPERIMENT_RUN"], "COMPLETE")
        self.assertEqual(v["RESOURCE"], "MEASURED_UNDER_LIMIT")
        self.assertEqual(v["PRIMARY_ERRATA"], "CORRECTED")
        self.assertEqual(v["CAUSAL_STATISTICAL_GATE"], "CAUSAL_CONTRIBUTION_SUPPORTED")
        self.assertEqual(v["AUTHORIZATION_PROVENANCE"], "VERIFIED")
        self.assertEqual(v["MODEL_CAUSAL_CONTRIBUTION"], "SUPPORTED")
        self.assertEqual(v["BIOLOGICAL_CAUSATION"], "NOT_ESTABLISHED")
        self.assertEqual(v["REAL_SYNAPTIC_LEARNING_READY"], "NO")
        self.assertFalse(v["Training_Authorization"])
        self.assertEqual(v["Level_A/B/C/Biological"], "NO-GO")
        self.assertEqual(v["A1_A2_A3/B"], "NO-GO")
        with open(SRC, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("FlyBrain(", src)
        self.assertNotIn("from flybrain", src)
        self.assertNotIn(".step(", src)
        self.assertNotIn("run_under_job", src)
        self.assertNotIn("p11_2_c1_experiment", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
