"""P11-2-S5-3 synthetic acceptance; no official MaleCNS."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import time
import unittest

import numpy as np

import loop.p11_2_s5_3_experiment as experiment
import loop.p11_2_s5_3_launcher as launcher
from loop.p11_2_s5_1_tracer import TraceWriter, read_trace, TRACE_DTYPE

HERE = os.path.dirname(os.path.abspath(__file__))
AUTH = "a" * 64
W_HASH = "b" * 64


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class S53Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="s53 synth ")
        s1 = _load(os.path.join(HERE, "p11_2_s1r_run_meta.json"))
        z = np.load(os.path.join(s1["out_dir"], "ids.npz"), allow_pickle=True)
        cls.kc = np.sort(np.asarray(z["kc_matched"], np.int64))
        cls.mb = np.sort(np.asarray(z["mb_matched"], np.int64))
        cls.joint = np.concatenate([cls.kc, cls.mb])

    def _fixture(self, name):
        out = os.path.join(self.tmp, name); os.makedirs(out)
        zeros = np.zeros(415, np.float64)
        fired = np.zeros(415, np.uint8)
        ref = np.zeros(415, np.uint8)
        fields = {"v_start": zeros, "v_leak": zeros, "synaptic": zeros, "tonic": zeros,
                  "noise": zeros, "eye": zeros, "inject": zeros,
                  "v_pre_threshold": zeros, "v_post_reset": zeros,
                  "threshold_distance": np.ones(415), "fired": fired,
                  "refractory_zeroed": ref}
        child_rows = []
        for seed in experiment.SEEDS:
            groups = {}
            for cond in experiment.CONDITIONS:
                path = os.path.join(out, f"{seed}_{cond}.bin")
                w = TraceWriter(path, self.joint)
                for step in range(400):
                    w.write_step(step, fields)
                h = w.close()
                groups[cond] = {"seed": seed, "condition": cond,
                                "amount": experiment.AMOUNTS[cond],
                                "trace_file": os.path.basename(path),
                                "trace_sha256": h["sha256"], "trace_records": h["records"],
                                "finite": True, "rng_before": f"rng-{seed}", "rng_after": f"rng-{seed}",
                                "kc_rate_window": 0.0, "mbon_rate_window": 0.0}
            child_rows.append({"seed": seed, "groups": groups, "noise_pair_ok": True})
        child = {
            "ok": True, "n_neurons": 166700, "n_kc": 4064, "n_mb": 97,
            "target_sha256": experiment.TARGET_SHA, "mbon_sha256": experiment.MBON_SHA,
            "joint_target_sha256": experiment.JOINT_SHA, "target_count": 415,
            "target_kc": 320, "target_mbon": 95, "removed_edges": 642933,
            "nnz_B": 24940005, "KC_KC_nnz": 0, "B_hash": W_HASH,
            "script_sha256": _sha(experiment.__file__), "auth_sha256": AUTH,
            "seeds": child_rows, "conditions": experiment.CONDITIONS, "steps": 400,
            "groups": 15, "noise_pairing_all": True, "device": "cpu", "batch": 1,
            "training": False, "dW": 0, "amount_primary": experiment.AMOUNT_PRIMARY,
            "amount_low": experiment.AMOUNT_LOW, "stim_window": [100, 300],
        }
        child_path = os.path.join(out, "child_result.json")
        with open(child_path, "w", encoding="utf-8") as f:
            json.dump(child, f)
        run = {"child_status": "CHILD_EXIT_OK", "process_finished": True, "exit_code": 0,
               "timeout": False, "mem_exceeded": False, "log_bytes": 0, "log_tail": "",
               "job_peaks": {"PeakJobMemoryUsed_MiB": 1.0}}
        pre = {"auth": {"checksum": AUTH}, "freeze": {"sha256": "f" * 64},
               "official": {"ok": True, "files": {}}, "joint": self.joint}
        meta = launcher._base_meta(pre, run, out, 1.0, child_path, "test-lock")
        return {"out": out, "child_path": child_path, "run": run, "pre": pre, "meta": meta,
                "report": os.path.join(self.tmp, name + "_report.json"),
                "manifest": os.path.join(self.tmp, name + "_manifest.json"),
                "run_meta": os.path.join(self.tmp, name + "_run_meta.json")}

    def _post(self, fx, report=None, manifest=None, run_meta=None):
        return launcher.postprocess(
            fx["out"], fx["run"], fx["meta"], fx["pre"], fx["child_path"],
            report_path=report or fx["report"], manifest_path=manifest or fx["manifest"],
            run_meta_path=run_meta or fx["run_meta"], exp=experiment)

    def test_unauthorized_entry(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = launcher.main(["--execute"])
        self.assertEqual(rc, 2)
        self.assertIn("NOT_AUTHORIZED", buf.getvalue())
        self.assertFalse(os.path.isdir(launcher.LOCK_DIR))

    def test_lock(self):
        p = os.path.join(self.tmp, "locks", "s5_3.consumed.json")
        launcher.acquire_one_shot("x", lock_path=p)
        with self.assertRaises(launcher.Refused):
            launcher.acquire_one_shot("x", lock_path=p)
        self.assertTrue(os.path.exists(p))

    def test_synthetic_success_and_integrity(self):
        fx = self._fixture("success")
        meta = self._post(fx)
        self.assertEqual(meta["status"], "RUN_COMPLETE")
        self.assertEqual(meta["analysis_status"], "DESCRIPTIVE_COMPLETE")
        rep = _load(fx["report"]); man = _load(fx["manifest"]); rm = _load(fx["run_meta"])
        self.assertEqual(rep["status"], "RUN_COMPLETE")
        self.assertEqual(rep["trace_count"], 15)
        self.assertEqual(man["status"], "RUN_COMPLETE")
        self.assertEqual(rm["status"], "RUN_COMPLETE")
        self.assertTrue(rm["experiment_succeeded"])
        names = [x["name"] for x in man["files"]]
        self.assertEqual(len([n for n in names if n.endswith(".bin")]), 15)
        self.assertEqual(len([n for n in names if n.endswith(".bin.json")]), 15)
        self.assertNotIn("run_meta.json", names)
        self.assertEqual(man["run_meta_sha256"], _sha(fx["run_meta"]))
        for row in man["files"]:
            self.assertEqual(_sha(os.path.join(fx["out"], row["name"])), row["sha256"])

    def test_missing_truncated_hash_order_rejected(self):
        fx = self._fixture("tamper")
        p = os.path.join(fx["out"], "301_sham.bin")
        with open(p, "rb") as f:
            data = f.read()
        # read_trace strict checks
        bad = os.path.join(fx["out"], "bad.bin")
        with open(bad, "wb") as f:
            f.write(data[:-1])
        with open(bad + ".json", "w", encoding="utf-8") as f:
            json.dump(_load(p + ".json"), f)
        with self.assertRaises(Exception):
            read_trace(bad)
        bad2 = os.path.join(fx["out"], "bad2.bin")
        b = bytearray(data); b[0] ^= 1
        with open(bad2, "wb") as f:
            f.write(b)
        with open(bad2 + ".json", "w", encoding="utf-8") as f:
            json.dump(_load(p + ".json"), f)
        with self.assertRaises(Exception):
            read_trace(bad2)
        bad3 = os.path.join(fx["out"], "bad3.bin")
        with open(bad3, "wb") as f:
            f.write(data)
        h = _load(p + ".json"); h["target_indices"] = list(reversed(h["target_indices"]))
        with open(bad3 + ".json", "w", encoding="utf-8") as f:
            json.dump(h, f)
        with self.assertRaises(Exception):
            read_trace(bad3)
        os.remove(p)
        meta = self._post(fx, report=os.path.join(self.tmp, "missing_report.json"),
                          manifest=os.path.join(self.tmp, "missing_manifest.json"),
                          run_meta=os.path.join(self.tmp, "missing_run_meta.json"))
        self.assertEqual(meta["status"], "INCOMPLETE_NO_GO")
        self.assertFalse(meta["experiment_succeeded"])

    def test_persistence_failure(self):
        fx = self._fixture("persist")
        orig = launcher._write_json
        state = {"failed": False}
        def flaky(path, obj):
            if os.path.abspath(path) == os.path.abspath(fx["report"]) and not state["failed"]:
                state["failed"] = True
                raise OSError("simulated report failure")
            return orig(path, obj)
        launcher._write_json = flaky
        try:
            meta = self._post(fx)
        finally:
            launcher._write_json = orig
        self.assertEqual(meta["status"], "PERSISTENCE_FAILED")
        self.assertFalse(meta["scientific_go"])
        self.assertNotEqual(_load(fx["run_meta"])["status"], "RUN_COMPLETE")

    def test_synthetic_job_failures(self):
        pkg = os.path.join(self.tmp, "s53_synth"); os.makedirs(pkg, exist_ok=True)
        with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as f:
            f.write("")
        with open(os.path.join(pkg, "child.py"), "w", encoding="utf-8") as f:
            f.write("import argparse,os,time\n"
                    "p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--mode');a=p.parse_args()\n"
                    "if a.mode=='timeout': time.sleep(10)\n"
                    "if a.mode=='mem': b=bytearray(512*1024**2)\n"
                    "if a.mode=='exit': raise SystemExit(1)\n")
        from loop.p11_2_s1f_safe_job_launcher import run_under_job_safe
        for mode, want in (("timeout", "CHILD_TIMEOUT"), ("mem", "CHILD_MEMORY_LIMIT"),
                           ("exit", "CHILD_EXIT_NONZERO")):
            out = os.path.join(self.tmp, "job_" + mode); os.makedirs(out)
            name = f"Local\\P11_2_S5_3_test_{os.getpid()}_{time.time_ns()}"
            r = run_under_job_safe(256 * 1024 ** 2, module="s53_synth.child",
                                   args=["--out", out, "--mode", mode], cwd=self.tmp,
                                   wall_cap_s=1 if mode == "timeout" else 30,
                                   mem_cap_mib=256, log_path=os.path.join(out, "log"),
                                   job_name=name)
            self.assertEqual(r["child_status"], want)


if __name__ == "__main__":
    unittest.main()
