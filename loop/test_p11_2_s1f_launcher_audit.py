"""P11-2-S1F 安全启动器/失败场景回归测试：只跑合成子进程，不跑 S1/完整脑。"""
from __future__ import annotations

import json
import os
import unittest

import loop.p11_2_s1f_launcher_audit as a

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_s1f_report.json")


class S1FTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs = a.failure_scene()
        cls.src = a.source_audit()
        cls.imp = a.import_audit()
        cls.safe = a.launcher_tests()
        cls.fz = a.repo_freeze()
        if os.path.exists(REPORT):
            with open(REPORT, encoding="utf-8") as f:
                cls.rep = json.load(f)
        else:
            cls.rep = None

    def test_failure_scene_preserved(self):
        fs = self.fs
        self.assertTrue(fs["all_hashes_ok"])
        self.assertEqual(fs["file_count"], 6)
        self.assertEqual(fs["experiment_log_bytes"], 0)
        self.assertFalse(fs["child_result_present"])
        self.assertEqual(fs["seed_files"], 0)
        self.assertEqual(fs["s1_report_status"], "INCOMPLETE_NO_GO")
        self.assertTrue(fs["script_freeze_ok"])
        self.assertEqual(fs["windows_event_evidence"], "NOT_AVAILABLE")

    def test_script_import_before_path_setup(self):
        s = self.src
        self.assertTrue(s["s1_top_level_import"])
        self.assertFalse(s["s1_root_insert_present"])
        self.assertTrue(s["s1_import_before_root"])
        self.assertEqual(s["historical_script_execution"], "absolute script path")

    def test_historical_log_handle_not_inheritable(self):
        s = self.src
        self.assertTrue(s["historical_CreateFileW_null_sa"])
        self.assertTrue(s["fixed_explicit_inheritable_sa"])
        self.assertTrue(s["fixed_handle_list"])
        self.assertEqual(self.imp["historical_launcher_A"]["log_bytes"], 0)

    def test_import_defect_reproduced(self):
        i = self.imp
        self.assertTrue(i["A_direct"]["module_not_found"])
        self.assertEqual(i["A_direct"]["rc"], 1)
        self.assertNotIn(i["historical_launcher_A"]["exit_code"], (None, 0))
        self.assertTrue(i["historical_launcher_A"]["ok"])
        self.assertEqual(i["historical_launcher_A"]["reason"], "completed")
        self.assertEqual(i["safe_launcher_A"]["child_status"], "CHILD_EXIT_NONZERO")
        self.assertTrue(i["safe_launcher_A"]["contains_ModuleNotFoundError"])
        self.assertTrue(i["B_module"]["success"])
        self.assertTrue(i["C_explicit_path"]["success"])
        self.assertEqual(i["import_defect"], "CONFIRMED_LAUNCHER_IMPORT_DEFECT")
        self.assertIn("does not by itself prove", i["caveat"])

    def test_safe_launcher_statuses(self):
        s = self.safe
        self.assertEqual(s["stdout"]["status"], "CHILD_EXIT_OK")
        self.assertEqual(s["stderr"]["status"], "CHILD_EXIT_OK")
        self.assertEqual(s["importerr"]["status"], "CHILD_EXIT_NONZERO")
        self.assertTrue(s["importerr"]["contains_traceback"])
        self.assertEqual(s["exit1"]["status"], "CHILD_EXIT_NONZERO")
        self.assertEqual(s["exit1"]["exit_code"], 1)
        self.assertEqual(s["invalid_log"]["status"], "CHILD_LOG_UNAVAILABLE")
        self.assertFalse(s["invalid_log"]["artifact_present"])
        self.assertEqual(s["assign_fail"]["status"], "CHILD_ASSIGN_FAILED")
        self.assertFalse(s["assign_fail"]["artifact_present"])
        self.assertEqual(s["timeout"]["status"], "CHILD_TIMEOUT")
        self.assertEqual(s["memory"]["status"], "CHILD_MEMORY_LIMIT")
        self.assertEqual(s["crash"]["status"], "CHILD_EXIT_NONZERO")
        self.assertEqual(s["crash"]["exit_code"], 7)
        self.assertEqual(s["artifact"]["status"], "CHILD_EXIT_OK")
        self.assertTrue(s["artifact"]["artifact_present"])
        self.assertTrue(s["artifact"]["experiment_succeeded"])
        self.assertEqual(s["missing_artifact"]["status"], "CHILD_OUTPUT_MISSING")
        self.assertFalse(s["missing_artifact"]["artifact_present"])
        self.assertEqual(s["no_retry"]["status"], "CHILD_EXIT_OK")
        self.assertEqual(s["no_retry"]["invocations"], 1)

    def test_tree_contained_without_false_parent_memory_limit(self):
        t = self.safe["tree"]
        self.assertEqual(t["status"], "CHILD_EXIT_OK")
        self.assertIn("TREE_DONE", t["log"])
        self.assertIn("MemoryError", t["log"])
        self.assertLess(t["job_peaks"]["PeakJobMemoryUsed_MiB"], 256)

    def test_repo_freeze(self):
        f = self.fz
        self.assertTrue(f["head_is_b801317"])
        self.assertTrue(f["P1_mapfix_is_db401f5"])
        self.assertTrue(f["readout_and_frozen_working_tree_clean"])
        self.assertEqual(f["dirty_three_count"], 3)
        self.assertTrue(f["official_hashes_match"])
        self.assertEqual(f["tags_at_head"], [])

    def test_status_semantics(self):
        self.assertTrue(self.safe["status_semantics"]["process_finished_is_not_success"])
        self.assertTrue(self.safe["status_semantics"]["nonzer_exit_no_success"])
        self.assertTrue(self.safe["fail_closed_on_invalid_log"])
        self.assertTrue(self.safe["job_assign_failure_terminates"])
        self.assertTrue(self.safe["no_retry_single_call"])
        self.assertTrue(self.safe["safe_launcher_reliably_logs_traceback"])

    def test_report_gates(self):
        if self.rep is None:
            self.skipTest("report not generated")
        v = self.rep["verdicts"]
        self.assertEqual(v["S1_STATUS"], "INCOMPLETE_NO_GO")
        self.assertEqual(v["NEW_S1_EXECUTION_AUTHORIZATION"], False)
        self.assertEqual(v["TRAINING_AUTHORIZATION"], False)
        self.assertEqual(v["REAL_SYNAPTIC_LEARNING_READY"], "NO")
        self.assertEqual(v["Level_A/B/C/Biological"], "NO-GO")
        self.assertEqual(v["A1_A2_A3/B"], "NO-GO")
        self.assertTrue(v["SAFE_LAUNCHER_LOGS_TRACEBACK"])
        self.assertTrue(v["JOB_SAFETY_PRESERVED"])
        self.assertIn("NOT_CONFIRMED", v["HISTORICAL_EXACT_CAUSE"])
        self.assertEqual(self.rep["status"], "DIAGNOSIS_COMPLETE; REPAIR_READY_FOR_NEW_APPROVAL")
        self.assertTrue(self.rep["historic_s1"]["child_result"] is False)
        self.assertEqual(self.rep["historic_s1"]["seed_files"], 0)
        self.assertEqual(self.rep["historic_s1"]["flybrain_step_count"], "UNKNOWN / NOT INFERRED")
        self.assertEqual(self.rep["historic_s1"]["neuron_response"], "NOT MEASURED / NOT INFERRED")


if __name__ == "__main__":
    unittest.main()
