"""P11-2-S1R-0 端到端合成回归；不加载完整脑，不运行 S1/S1R 生产实验。"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import tempfile
import unittest

import loop.p11_2_s1r_launcher as launcher

HERE = os.path.dirname(os.path.abspath(__file__))
BACKUP = r"C:\AA_project\guoying_archive\p11_2_s1_failed_20261020T104624"
S1_RAW = r"C:\Users\yang\AppData\Local\Temp\opencode\p11_2_s1_full\20261020T104624"

CHILD = r'''
import argparse, json, os, subprocess, sys, time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", type=int, default=25)
    p.add_argument("--exit-code", type=int, default=0)
    p.add_argument("--write-child", action="store_true")
    p.add_argument("--noise-bad", action="store_true")
    p.add_argument("--counter")
    p.add_argument("--stdout", action="store_true")
    p.add_argument("--stderr", action="store_true")
    p.add_argument("--traceback", action="store_true")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--mem", type=int, default=0)
    p.add_argument("--tree", action="store_true")
    a = p.parse_args()
    if a.counter:
        with open(a.counter, "a", encoding="utf-8") as f:
            f.write("1\n")
    if a.stdout:
        print("SYNTH_STDOUT")
    if a.stderr:
        print("SYNTH_STDERR", file=sys.stderr)
    if a.traceback:
        raise RuntimeError("SYNTH_TRACEBACK")
    if a.sleep:
        time.sleep(a.sleep)
    if a.mem:
        b = bytearray(a.mem * 1024 ** 2)
        print("MEM_OK", len(b))
    if a.tree:
        subprocess.run([sys.executable, "-c", "b=bytearray(512*1024**2);print(len(b))"])
    if a.write_child:
        seeds = [101 + i for i in range(a.seeds)]
        obj = {"ok": True, "seeds": [{"seed": s} for s in seeds],
               "noise_pairing_all": not a.noise_bad}
        with open(os.path.join(a.out_dir, "child_result.json"), "w", encoding="utf-8") as f:
            json.dump(obj, f)
    return a.exit_code

if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class S1R0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="p11 s1r synth ")
        pkg = os.path.join(cls.tmp, "s1r_synth")
        os.makedirs(pkg)
        _write(os.path.join(pkg, "__init__.py"), "")
        _write(os.path.join(pkg, "child.py"), CHILD)
        cls.pkg = "s1r_synth.child"
        cls.counter = os.path.join(cls.tmp, "counter.txt")

    def out(self, name):
        return os.path.join(self.tmp, "out dir with spaces", name)

    def _run(self, name, extra=(), expected=25, limit=256, wall=30.0, stats=None, inherit=True):
        return launcher.run_synthetic_chain(
            self.pkg, self.tmp, self.out(name), ["--out-dir", self.out(name), *extra],
            expected_seeds=expected, limit_mib=limit, wall_s=wall,
            stats_fn=stats, inherit_log=inherit)

    def test_production_uses_safe_launcher_not_old(self):
        with open(os.path.join(HERE, "p11_2_s1r_launcher.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("from loop.p11_2_s1f_safe_job_launcher import run_under_job_safe", src)
        self.assertIn("run_under_job_safe(", src)
        self.assertNotIn("from loop.p11_1c_full_brain_bench import run_under_job", src)
        self.assertNotRegex(src, r"(?<!_safe)run_under_job\(")
        self.assertIn("_run_child_module(EXPERIMENT_MODULE, [\"--child\", out_dir], ROOT", src)
        self.assertIn("module=module", src)
        self.assertIn("cwd=cwd", src)
        self.assertIn("[\"--child\", out_dir]", src)

    def test_production_helper_delegates_to_safe(self):
        seen = {}
        old = launcher.run_under_job_safe

        def fake(*args, **kwargs):
            seen.update(kwargs)
            return {"child_status": "CHILD_OUTPUT_MISSING", "process_finished": False,
                    "experiment_succeeded": False, "exit_code": 0, "timeout": False,
                    "mem_exceeded": False, "job_peaks": None, "log_bytes": 0,
                    "log_tail": "", "peak_MiB": None}

        launcher.run_under_job_safe = fake
        try:
            launcher._run_child_module(launcher.EXPERIMENT_MODULE,
                                       ["--child", r"C:\tmp path with spaces"],
                                       launcher.ROOT, self.tmp, limit_mib=2048,
                                       wall_s=900.0, required_artifact="x")
        finally:
            launcher.run_under_job_safe = old
        self.assertEqual(seen["module"], launcher.EXPERIMENT_MODULE)
        self.assertEqual(seen["cwd"], launcher.ROOT)
        self.assertEqual(seen["args"], ["--child", r"C:\tmp path with spaces"])
        self.assertEqual(seen["wall_cap_s"], 900.0)
        self.assertEqual(seen["mem_cap_mib"], 2048)

    def test_production_module_import_and_frozen_scope(self):
        mod = importlib.import_module("loop.p11_2_s1r_experiment")
        self.assertEqual(mod._scope(), launcher.SCOPE)
        self.assertEqual(mod.SEEDS, list(range(101, 126)))
        self.assertEqual(mod.STEPS, 400)
        self.assertEqual(mod.STIM, (100, 300))
        self.assertEqual(mod.AMOUNT, 0.036254)
        self.assertEqual(mod.DV, 0.2)
        self.assertEqual(mod.MEM_CAP_MIB, 2048)
        self.assertEqual(mod.WALL_CAP_S, 900.0)
        self.assertFalse(launcher.SCOPE["training"])
        self.assertEqual(launcher.SCOPE["dW"], 0)
        self.assertFalse(launcher.SCOPE["auto_retry"])

    def test_auth_and_freeze_are_independent_not_authorized(self):
        a = _load(launcher.AUTH)
        c = a.pop("checksum")
        self.assertEqual(c, hashlib.sha256(launcher._canon(a)).hexdigest())
        self.assertEqual(a["status"], "NOT_AUTHORIZED")
        self.assertIsNone(a["approved_by"])
        self.assertIsNone(a["approval_text"])
        self.assertNotEqual(c, "d9087107c485cca57f983725a11423d65c3f2da0770b5cff7d1212424a948ab5")
        z = _load(launcher.FREEZE)
        self.assertEqual(z["module"], "loop.p11_2_s1r_experiment")
        self.assertEqual(os.path.abspath(z["script"]), os.path.abspath(launcher.EXPERIMENT_PATH))
        self.assertEqual(z["sha256"], _sha(launcher.EXPERIMENT_PATH))

    def test_production_refuses_without_new_authorization(self):
        base = os.path.join(tempfile.gettempdir(), "opencode", "p11_2_s1r_full")
        before = set(os.listdir(base)) if os.path.isdir(base) else None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = launcher.main(["--execute"])
        after = set(os.listdir(base)) if os.path.isdir(base) else None
        self.assertEqual(rc, 2)
        self.assertIn("NOT_AUTHORIZED", buf.getvalue())
        self.assertIn('"new_s1r_execution_authorization": false', buf.getvalue())
        self.assertEqual(after, before)

    def test_full_chain_module_import_stdout_stderr_traceback(self):
        r = self._run("full", ["--write-child", "--stdout", "--stderr", "--counter", self.counter])
        self.assertEqual(r["run"]["child_status"], "CHILD_EXIT_OK")
        self.assertTrue(r["meta"]["experiment_succeeded"])
        self.assertIn("SYNTH_STDOUT", r["run"]["log_tail"])
        self.assertIn("SYNTH_STDERR", r["run"]["log_tail"])
        self.assertEqual(r["child"]["seeds_observed"], 25)
        self.assertTrue(r["child"]["complete"])
        with open(self.counter, encoding="utf-8") as f:
            self.assertEqual(len(f.read().splitlines()), 1)

    def test_traceback_and_exit1(self):
        r = self._run("traceback", ["--traceback"])
        self.assertEqual(r["run"]["child_status"], "CHILD_EXIT_NONZERO")
        self.assertEqual(r["run"]["exit_code"], 1)
        self.assertIn("SYNTH_TRACEBACK", r["run"]["log_tail"])
        self.assertFalse(r["meta"]["experiment_succeeded"])

    def test_exit0_missing_artifact(self):
        r = self._run("missing", ["--seeds", "25"])
        self.assertEqual(r["run"]["child_status"], "CHILD_OUTPUT_MISSING")
        self.assertFalse(r["child"]["exists"])
        self.assertFalse(r["meta"]["experiment_succeeded"])

    def test_exit0_short_seed_count_is_not_success(self):
        r = self._run("short", ["--write-child", "--seeds", "24"], expected=25)
        self.assertEqual(r["run"]["child_status"], "CHILD_EXIT_OK")
        self.assertEqual(r["child"]["seeds_observed"], 24)
        self.assertFalse(r["child"]["complete"])
        self.assertFalse(r["meta"]["experiment_succeeded"])
        self.assertEqual(r["meta"]["status"], "INCOMPLETE_NO_GO")

    def test_metadata_written_before_stats(self):
        seen = {}

        def stats(out_dir, meta):
            seen["meta"] = os.path.exists(os.path.join(out_dir, "run_meta.json"))
            seen["child"] = os.path.exists(os.path.join(out_dir, "child_result.json"))
            seen["status"] = meta["status"]

        r = self._run("order", ["--write-child"], stats=stats)
        self.assertTrue(seen["meta"])
        self.assertTrue(seen["child"])
        self.assertTrue(r["meta"]["experiment_succeeded"])
        self.assertEqual(seen["status"], "RUN_COMPLETE")

    def test_job_assignment_order_and_tree(self):
        with open(os.path.join(HERE, "p11_2_s1f_safe_job_launcher.py"), encoding="utf-8") as f:
            safe = f.read()
        self.assertIn("CREATE_SUSPENDED", safe)
        self.assertIn("assign_resume_or_terminate", safe)
        r = self._run("tree", ["--tree", "--write-child"])
        self.assertEqual(r["run"]["child_status"], "CHILD_EXIT_OK")
        self.assertIn("MemoryError", r["run"]["log_tail"])
        self.assertIsNotNone(r["run"]["job_peaks"])
        self.assertLess(r["run"]["job_peaks"]["PeakJobMemoryUsed_MiB"], 256)
        self.assertTrue(r["meta"]["experiment_succeeded"])

    def test_timeout_memory_and_log_fail_closed(self):
        t = self._run("timeout", ["--sleep", "10"], wall=1.0)
        self.assertEqual(t["run"]["child_status"], "CHILD_TIMEOUT")
        self.assertFalse(t["meta"]["experiment_succeeded"])

        m = self._run("mem", ["--mem", "512"], limit=256)
        self.assertEqual(m["run"]["child_status"], "CHILD_MEMORY_LIMIT")
        self.assertFalse(m["meta"]["experiment_succeeded"])

        log = self._run("badlog", ["--write-child"], inherit=False)
        self.assertEqual(log["run"]["child_status"], "CHILD_LOG_UNAVAILABLE")
        self.assertFalse(os.path.exists(os.path.join(log["out_dir"], "child_result.json")))

    def test_historical_s1_preserved_and_backup(self):
        rep = _load(os.path.join(HERE, "p11_2_s1_report.json"))
        self.assertEqual(rep["status"], "INCOMPLETE_NO_GO")
        meta = _load(os.path.join(HERE, "p11_2_s1_run_meta.json"))
        self.assertEqual(meta["exit_code"], 1)
        self.assertEqual(meta["run_wall_s"], 0.82)
        self.assertEqual(meta["job_peak_MiB"], 521.05)
        man = _load(os.path.join(HERE, "p11_2_s1_raw_manifest.json"))
        self.assertEqual(len(man["files"]), 6)
        for row in man["files"]:
            self.assertEqual(_sha(os.path.join(S1_RAW, row["name"])), row["sha256"])
        self.assertEqual(os.path.getsize(os.path.join(S1_RAW, "experiment.log")), 0)
        self.assertTrue(os.path.isdir(BACKUP))
        for row in man["files"]:
            self.assertEqual(_sha(os.path.join(BACKUP, row["name"])), row["sha256"])

    def test_official_and_repo_freeze(self):
        from loop.p11_2_s1f_launcher_audit import repo_freeze
        f = repo_freeze()
        self.assertTrue(f["P1_mapfix_is_db401f5"])
        self.assertTrue(f["readout_and_frozen_working_tree_clean"])
        self.assertEqual(f["dirty_three_count"], 3)
        self.assertTrue(f["official_hashes_match"])

    def test_report_gates(self):
        r = _load(os.path.join(HERE, "p11_2_s1r0_report.json"))
        self.assertEqual(r["status"], "S1R0_INTEGRATION_COMPLETE; NOT_AUTHORIZED; NO_RUN")
        self.assertFalse(r["new_s1r_execution_authorization"])
        self.assertFalse(r["training_authorization"])
        self.assertEqual(r["real_synaptic_learning_ready"], "NO")
        self.assertEqual(r["levels"], "NO-GO")
        self.assertEqual(r["authorization"]["status"], "NOT_AUTHORIZED")
        self.assertFalse(r["authorization"]["old_s1_status_reused"])
        self.assertTrue(r["freeze"]["match"])
        self.assertTrue(r["production"]["uses_safe_launcher"])
        self.assertFalse(r["production"]["uses_old_run_under_job"])
        self.assertTrue(r["production"]["module_import_ok"])
        self.assertEqual(r["synthetic_tests"]["passed"], 16)
        self.assertEqual(r["historical_s1"]["report_status"], "INCOMPLETE_NO_GO")
        self.assertTrue(r["historical_s1"]["all_raw_hashes_ok"])
        self.assertTrue(r["historical_s1"]["all_backup_hashes_ok"])
        self.assertTrue(r["freeze_checks"]["P1_mapfix_is_db401f5"])
        self.assertTrue(r["freeze_checks"]["head_is_4823ba8"])
        self.assertTrue(r["freeze_checks"]["readout_and_frozen_working_tree_clean"])
        self.assertFalse(r["resources"]["resource_verified"])
        self.assertTrue(r["no_full_brain_run"])

    def test_approval_file_not_authorized(self):
        p = os.path.join(HERE, "P11_2_S1R_ONE_RUN_APPROVAL.md")
        with open(p, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("NOT_AUTHORIZED", text)
        self.assertNotIn("status: APPROVED", text)
        self.assertIn("NEW_S1R_EXECUTION_AUTHORIZATION=False", text)
        self.assertIn("2 GiB", text)
        self.assertIn("900", text)


if __name__ == "__main__":
    unittest.main()
