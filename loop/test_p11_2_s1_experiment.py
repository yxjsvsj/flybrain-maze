"""P11-2-S1 回归测试：合成 B/none-stim、窗口守卫、授权、实际报告。

实际报告测试只在 S1 已执行后运行；不重新运行完整脑。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from loop.p11_2_s1_experiment import analyze, synthetic_tiny, _window

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_s1_report.json")
RUN_META = os.path.join(HERE, "p11_2_s1_run_meta.json")
RAW = os.path.join(HERE, "p11_2_s1_raw_manifest.json")
AUTH = os.path.join(HERE, "p11_2_s1_authorization.json")


def _make_raw(tmp, stim_on=True, kc_none=0.0, entry_on=True):
    np.savez(os.path.join(tmp, "ids.npz"), ids=np.arange(4064), kc_idx=np.arange(4064),
             mb_idx=np.arange(97), kc_matched=np.arange(320), mb_matched=np.arange(95),
             lp_idx=np.arange(311))
    for s in range(101, 126):
        kcn = np.full((400, 4064), kc_none, np.uint8)
        kcs = np.zeros((400, 4064), np.uint8)
        mbn = np.zeros((400, 97), np.uint8); mbs = np.zeros((400, 97), np.uint8)
        lpn = np.zeros((400, 311), np.uint8); lps = np.zeros((400, 311), np.uint8)
        kcmn = np.full((400, 320), kc_none, np.uint8); kcms = np.zeros((400, 320), np.uint8)
        mbmn = np.zeros((400, 95), np.uint8); mbms = np.zeros((400, 95), np.uint8)
        if stim_on:
            kcs[100:300] = 1; kcms[100:300] = 1; mbs[100:300] = 1; mbms[100:300] = 1
            if entry_on:
                lps[100:300] = 1
        np.savez_compressed(os.path.join(tmp, f"{s}_none.npz"), kc=kcn, mb=mbn, lp=lpn,
                            kc_matched=kcmn, mb_matched=mbmn, rng_before="x", rng_after="y")
        np.savez_compressed(os.path.join(tmp, f"{s}_stim.npz"), kc=kcs, mb=mbs, lp=lps,
                            kc_matched=kcms, mb_matched=mbms, rng_before="x", rng_after="y")


class P11S1Test(unittest.TestCase):
    def test_synthetic_tiny(self):
        r = synthetic_tiny()
        self.assertTrue(r["ok"])
        c = r["child"]
        self.assertTrue(c["ok"])
        self.assertEqual(c["n_neurons"], 5)
        self.assertEqual(c["n_lp"], 2)
        self.assertEqual(c["removed_edges"], 1)
        self.assertEqual(c["nnz_B"], 4)
        self.assertTrue(c["noise_pairing_all"])
        self.assertTrue(c["field_switch"]["A_restore_exact"])
        self.assertTrue(c["field_switch"]["non_KC_unchanged"])
        self.assertEqual(len(c["seeds"]), 2)
        for s in c["seeds"]:
            self.assertTrue(s["none"]["finite"] and s["stim"]["finite"])
            self.assertTrue(s["noise_pair_ok"])
            self.assertGreater(s["stim"]["lp_n"], s["none"]["lp_n"])

    def test_window_guard(self):
        a = np.zeros((400, 3), np.uint8); a[100:300] = 1
        self.assertAlmostEqual(float(a.mean()), 0.5)
        self.assertAlmostEqual(float(_window(a).mean()), 1.0)
        self.assertEqual(_window(a).shape, (200, 3))

    def test_analyze_window_and_grading(self):
        tmp = tempfile.mkdtemp(prefix="p11_2_s1_stats_")
        _make_raw(tmp, stim_on=True)
        rep = analyze(tmp)
        self.assertEqual(rep["completeness"], "25/25")
        self.assertAlmostEqual(rep["entry"]["diff"]["mean"], 1.0, places=6)
        self.assertAlmostEqual(rep["primary"]["diff"]["mean"], 1.0, places=6)
        self.assertAlmostEqual(rep["mbon"]["matched_95_diff"]["mean"], 1.0, places=6)
        self.assertTrue(rep["entry"]["pass"])
        self.assertTrue(rep["primary"]["pass"])
        self.assertTrue(rep["mbon"]["pass"])
        self.assertEqual(rep["grading"], "SENSORY_TRANSMISSION_DETECTED_IN_SYNTHETIC_B")
        self.assertAlmostEqual(rep["secondary"]["full_4064_none"], 0.0, places=6)
        self.assertAlmostEqual(rep["secondary"]["full_4064_stim"], 1.0, places=6)
        self.assertAlmostEqual(rep["secondary"]["full_97_mbon_none"], 0.0, places=6)
        self.assertAlmostEqual(rep["secondary"]["full_97_mbon_stim"], 1.0, places=6)
        self.assertFalse(rep["negative"]["kc_negative"])
        self.assertEqual(rep["none_drift"]["mean"], 0.0)
        # The startup bins are full 400 steps (0.5); the primary is [100,300) (1.0).
        self.assertAlmostEqual(float(np.mean(rep["secondary"]["startup_stim_bins"])), 0.5, places=6)
        self.assertAlmostEqual(float(np.mean(rep["secondary"]["startup_none_bins"])), 0.0, places=6)
        self.assertAlmostEqual(rep["primary"]["diff"]["mean"], 1.0, places=6)

    def test_analyze_negative_is_separate(self):
        tmp = tempfile.mkdtemp(prefix="p11_2_s1_neg_")
        _make_raw(tmp, stim_on=False, kc_none=1.0, entry_on=True)
        # stim arrays are zero for KC; entry is zero too; make entry positive by editing one small case is not needed:
        rep = analyze(tmp)
        self.assertLess(rep["primary"]["diff"]["mean"], 0)
        self.assertFalse(rep["primary"]["pass"])
        self.assertFalse(rep["negative"]["kc_negative"] is None)
        self.assertIn(rep["grading"], ("SENSORY_ENTRY_FAILED", "SENSORY_TRANSMISSION_NOT_DETECTED"))
        self.assertNotEqual(rep["grading"], "SENSORY_TRANSMISSION_DETECTED_IN_SYNTHETIC_B")

    def test_auth_scope(self):
        a = json.load(open(AUTH, encoding="utf-8"))
        self.assertEqual(a["protocol"], "P11-2-S1/v1")
        self.assertEqual(a["status"], "APPROVED")
        self.assertEqual(a["approved_by"], "user")
        self.assertEqual(a["scope"]["network"], "B_synthetic_kc_kc_zero")
        self.assertEqual(a["scope"]["remove_edges"], 642933)
        self.assertEqual(a["scope"]["conditions"], ["none", "stim"])
        self.assertEqual(a["scope"]["seeds"], list(range(101, 126)))
        self.assertEqual(a["scope"]["total_steps"], 20000)
        self.assertEqual(a["scope"]["window"], [100, 300])
        self.assertIn("311", a["scope"]["targets"])
        self.assertEqual(a["scope"]["dv"], 0.2)
        self.assertEqual(a["scope"]["amount"], 0.036254)
        self.assertEqual(a["scope"]["device"], "cpu")
        self.assertEqual(a["scope"]["wall_cap_s"], 900.0)
        self.assertEqual(a["scope"]["mem_cap_MiB"], 2048)
        self.assertTrue(a["scope"]["one_run"])
        self.assertFalse(a["scope"]["auto_retry"])
        self.assertFalse(a["scope"]["training"])
        self.assertEqual(a["scope"]["dW"], 0)
        self.assertIn("我批准 P11-2-S1", a["approval_text"])
        self.assertIn("不批准训练", a["approval_text"])
        import hashlib
        c = a.pop("checksum")
        self.assertEqual(c, hashlib.sha256(json.dumps(a, sort_keys=True, separators=(",", ":")).encode()).hexdigest())

    def test_actual_report_if_present(self):
        if not os.path.exists(REPORT):
            self.skipTest("S1 not executed yet")
        r = json.load(open(REPORT, encoding="utf-8"))
        if not r.get("complete_25_pairs"):
            self.skipTest("S1 run did not complete; INCOMPLETE_NO_GO (no retry)")
        self.assertTrue(r.get("complete_25_pairs", False))
        self.assertEqual(r.get("completeness"), "25/25")
        self.assertIn("entry", r)
        self.assertIn("primary", r)
        self.assertIn("mbon", r)
        self.assertIn("negative", r)
        self.assertIn("grading", r)
        self.assertIn(r["grading"], ("SENSORY_ENTRY_FAILED", "SENSORY_TRANSMISSION_NOT_DETECTED",
                                     "KC_RESPONSE_WITHOUT_MBON_TRANSMISSION",
                                     "SENSORY_TRANSMISSION_DETECTED_IN_SYNTHETIC_B"))
        m = json.load(open(RUN_META, encoding="utf-8"))
        self.assertEqual(m["status"], "RUN_COMPLETE")
        self.assertEqual(m["exit_code"], 0)
        self.assertFalse(m["timeout"])
        self.assertFalse(m["mem_exceeded"])
        self.assertLessEqual(m["job_peak_MiB"], 2048)
        self.assertLessEqual(m["run_wall_s"], 900)
        man = json.load(open(RAW, encoding="utf-8"))
        self.assertGreaterEqual(len(man["files"]), 50)
        self.assertFalse(r["gate"]["Training_Authorization"])
        self.assertEqual(r["gate"]["REAL_SYNAPTIC_LEARNING_READY"], "NO")


if __name__ == "__main__":
    unittest.main(verbosity=2)
