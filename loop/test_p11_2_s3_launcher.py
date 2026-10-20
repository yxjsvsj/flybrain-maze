"""P11-2-S3-1 static/synthetic acceptance for the S3 production chain.

No FlyBrain import/initialization; all dynamics-like fixtures are synthetic NPZ.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

import loop.p11_2_s3_experiment as experiment
import loop.p11_2_s3_launcher as launcher

HERE = os.path.dirname(os.path.abspath(__file__))
AUTH = "a" * 64
W_HASH = "b" * 64
SCRIPT_SHA = experiment._sha256(experiment.__file__)
TARGET_SHA = experiment.TARGET_SHA
TARGET = None


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _set_rate(arr, rate):
    n = int(round(float(rate) * 200))
    arr[100:100 + n, :] = 1


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _rng(seed, phase):
    return hashlib.sha256(f"{seed}:{phase}".encode()).hexdigest()


class S31Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global TARGET
        meta = _load(os.path.join(HERE, "p11_2_s1r_run_meta.json"))
        z = np.load(os.path.join(meta["out_dir"], "ids.npz"), allow_pickle=True)
        TARGET = np.asarray(z["kc_matched"], np.int64)
        cls.tmp = tempfile.mkdtemp(prefix="p11 s3 synth ")

    def _fixture(self, name, kc_effect=0.15, mb_effect=0.10, low_kc=0.05,
                 low_mb=0.02, mb_deltas=None, kc_deltas=None, child_bad=None):
        out = os.path.join(self.tmp, name)
        os.makedirs(out)
        child_rows = []
        for i, seed in enumerate(experiment.SEEDS):
            kc_d = kc_effect if kc_deltas is None else kc_deltas[i]
            mb_d = mb_effect if mb_deltas is None else mb_deltas[i]
            rows = {}
            for cond in experiment.CONDITIONS:
                kc = np.zeros((experiment.STEPS, 4064), np.uint8)
                mb = np.zeros((experiment.STEPS, 97), np.uint8)
                lp = np.zeros((experiment.STEPS, 311), np.uint8)
                kcm = np.zeros((experiment.STEPS, 320), np.uint8)
                mbm = np.zeros((experiment.STEPS, 95), np.uint8)
                kc_rate = 0.20 + (low_kc if cond == "kc_low" else kc_effect if cond == "kc_primary" else 0.0)
                mb_rate = 0.40 + (low_mb if cond == "kc_low" else mb_effect if cond == "kc_primary" else 0.0)
                if cond == "kc_low":
                    kc_rate = 0.20 + low_kc
                    mb_rate = 0.40 + low_mb
                elif cond == "kc_primary":
                    kc_rate = 0.20 + kc_d
                    mb_rate = 0.40 + mb_d
                else:
                    kc_rate = 0.20
                    mb_rate = 0.40
                _set_rate(kcm, kc_rate); _set_rate(mbm, mb_rate)
                np.savez_compressed(os.path.join(out, f"{seed}_{cond}.npz"),
                                    kc=kc, mb=mb, lp=lp, kc_matched=kcm, mb_matched=mbm,
                                    seed=seed, condition=cond,
                                    injection_amount=experiment.AMOUNTS[cond],
                                    injection_window=np.array([100, 300]),
                                    target_idx=TARGET, target_sha=TARGET_SHA,
                                    script_sha=SCRIPT_SHA, auth_sha=AUTH, w_hash=W_HASH,
                                    rng_before=_rng(seed, "before"), rng_after=_rng(seed, "after"),
                                    finite=True, no_sensory_injection=True, no_mbon_injection=True)
                rows[cond] = {"seed": seed, "condition": cond, "finite": True,
                              "amount": experiment.AMOUNTS[cond],
                              "rng_before": _rng(seed, "before"), "rng_after": _rng(seed, "after")}
            child_rows.append({"seed": seed, "sham": rows["sham"], "kc_low": rows["kc_low"],
                               "kc_primary": rows["kc_primary"], "noise_pair_ok": True})
        child = {"ok": True, "n_neurons": 166700, "n_kc": 4064, "n_mb": 97,
                 "n_kc_matched": 320, "n_mb_matched": 95, "n_lp": 311,
                 "removed_edges": 642933, "nnz_B": 24940005, "KC_KC_nnz": 0,
                 "target_sha": TARGET_SHA, "B_hash": W_HASH, "script_sha256": SCRIPT_SHA,
                 "auth_sha256": AUTH, "seeds": child_rows, "noise_pairing_all": True,
                 "conditions": experiment.CONDITIONS, "steps": 400, "device": "cpu",
                 "batch": 1, "training": False, "dW": 0,
                 "amount_primary": experiment.AMOUNT_PRIMARY, "amount_low": experiment.AMOUNT_LOW,
                 "stim_window": [100, 300]}
        if child_bad:
            child_bad(child)
        child_path = os.path.join(out, "child_result.json")
        with open(child_path, "w", encoding="utf-8") as f:
            json.dump(child, f)
        run = {"child_status": "CHILD_EXIT_OK", "process_finished": True, "exit_code": 0,
               "timeout": False, "mem_exceeded": False, "log_bytes": 0, "log_tail": "",
               "job_peaks": {"PeakProcessMemoryUsed_MiB": 1.0, "PeakJobMemoryUsed_MiB": 1.0}}
        pre = {"auth": {"checksum": AUTH}, "freeze": {"sha256": "f" * 64},
               "official": {"ok": True, "files": {}}}
        meta = launcher._base_meta(pre, run, out, 1.0, child_path, "test-lock")
        return {"out": out, "child_path": child_path, "run": run, "pre": pre, "meta": meta,
                "report": os.path.join(self.tmp, name + "_report.json"),
                "manifest": os.path.join(self.tmp, name + "_manifest.json"),
                "run_meta": os.path.join(self.tmp, name + "_run_meta.json")}

    def _post(self, fx):
        return launcher.postprocess_and_persist(
            fx["out"], fx["run"], fx["meta"], fx["pre"], fx["child_path"],
            report_path=fx["report"], manifest_path=fx["manifest"],
            run_meta_path=fx["run_meta"], exp=experiment)

    def _fast_analyze(self):
        orig = experiment.exact_signflip
        experiment.exact_signflip = lambda d: 0.01 if abs(float(np.mean(d))) > 1e-12 else 1.0
        return orig

    def test_unauthorized_production_entry(self):
        before = set(os.listdir(launcher.LOCK_DIR)) if os.path.isdir(launcher.LOCK_DIR) else None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = launcher.main(["--execute"])
        after = set(os.listdir(launcher.LOCK_DIR)) if os.path.isdir(launcher.LOCK_DIR) else None
        self.assertEqual(rc, 2)
        self.assertIn("NOT_AUTHORIZED", buf.getvalue())
        self.assertEqual(after, before)
        self.assertEqual(_load(launcher.AUTH)["status"], "NOT_AUTHORIZED")

    def test_lock_second_execution_refused(self):
        p = os.path.join(self.tmp, "locks", "s3.consumed.json")
        launcher.acquire_one_shot("x", lock_path=p, out_dir="o", freeze_sha="f")
        with self.assertRaises(launcher.Refused) as cm:
            launcher.acquire_one_shot("x", lock_path=p, out_dir="o", freeze_sha="f")
        self.assertIn("AUTHORIZATION_ALREADY_CONSUMED", str(cm.exception))
        self.assertTrue(os.path.exists(p))

    def test_full_success_and_integrity(self):
        fx = self._fixture("success")
        meta = self._post(fx)
        self.assertEqual(meta["status"], "RUN_COMPLETE")
        self.assertTrue(meta["experiment_succeeded"])
        self.assertTrue(meta["scientific_go"])
        rep = _load(fx["report"]); man = _load(fx["manifest"]); rm = _load(fx["run_meta"])
        self.assertEqual(rep["status"], "RUN_COMPLETE")
        self.assertEqual(rep["decision"], "KC_MBON_FUNCTIONAL_UNDER_DIRECT_DRIVE")
        self.assertEqual(rm["decision"], rep["decision"])
        self.assertEqual(man["decision"], rep["decision"])
        names = [x["name"] for x in man["files"]]
        self.assertEqual(len([n for n in names if n.endswith(".npz")]), 75)
        self.assertNotIn("run_meta.json", names)
        for row in man["files"]:
            p = os.path.join(fx["out"], row["name"])
            self.assertEqual(_sha(p), row["sha256"])
            self.assertEqual(os.path.getsize(p), row["size"])
        self.assertEqual(man["run_meta_sha256"], _sha(fx["run_meta"]))
        json.dumps(rep, ensure_ascii=False)

    def test_decision_paths(self):
        cases = [
            ("kc_fail", dict(kc_effect=0.05, mb_effect=0.10), "INCONCLUSIVE_KC_NOT_DRIVEN"),
            ("sat", dict(kc_effect=0.70, mb_effect=0.10), "INCONCLUSIVE_KC_SATURATED"),
            ("mbon_neg", dict(kc_effect=0.15, mb_effect=-0.05), "KC_MBON_NEGATIVE"),
            ("mbon_small", dict(kc_effect=0.15, mb_effect=0.04), "INCONCLUSIVE_DOWNSTREAM"),
            ("mbon_ns", dict(kc_effect=0.15, mb_effect=0.0,
                             mb_deltas=[0.01 if i % 2 == 0 else -0.01 for i in range(25)]),
             "INCONCLUSIVE_DOWNSTREAM"),
        ]
        orig = self._fast_analyze()
        try:
            for name, kwargs, want in cases:
                fx = self._fixture(name, **kwargs)
                meta = self._post(fx)
                rep = _load(fx["report"])
                self.assertEqual(meta["status"], "RUN_COMPLETE")
                self.assertEqual(rep["decision"], want)
                self.assertFalse(rep["scientific_go"])
                self.assertFalse(meta["scientific_go"])
        finally:
            experiment.exact_signflip = orig

    def test_missing_duplicate_hash(self):
        fx = self._fixture("bad")
        os.remove(os.path.join(fx["out"], "201_sham.npz"))
        r = experiment.validate_raw_75(fx["out"], auth_sha=AUTH, script_sha=SCRIPT_SHA)
        self.assertFalse(r["ok"]); self.assertTrue(any("MISSING_FILES" in x for x in r["errors"]))
        fx2 = self._fixture("bad_hash")
        p = os.path.join(fx2["out"], "201_sham.npz")
        np.savez_compressed(p, kc=np.zeros((1, 1), np.uint8))
        r2 = experiment.validate_raw_75(fx2["out"], auth_sha=AUTH, script_sha=SCRIPT_SHA)
        self.assertFalse(r2["ok"])
        bad = _load(fx2["child_path"])
        bad["seeds"][1]["seed"] = bad["seeds"][0]["seed"]
        with open(fx2["child_path"], "w", encoding="utf-8") as f:
            json.dump(bad, f)
        rc = experiment.validate_child_result(fx2["child_path"], raw_w_hashes=[W_HASH])
        self.assertFalse(rc["complete"])
        self.assertTrue(any("SEED_DUPLICATE" in x for x in rc["errors"]))

    def test_abnormal_run_statuses_are_incomplete(self):
        orig = self._fast_analyze()
        try:
            for status in ("CHILD_EXIT_NONZERO", "CHILD_TIMEOUT", "CHILD_MEMORY_LIMIT"):
                fx = self._fixture("run_" + status)
                fx["run"]["child_status"] = status
                meta = self._post(fx)
                self.assertEqual(meta["status"], "INCOMPLETE_NO_GO")
                self.assertFalse(meta["scientific_go"])
        finally:
            experiment.exact_signflip = orig

    def test_persistence_failures(self):
        orig_analyze = self._fast_analyze()
        try:
            for target in ("report", "manifest"):
                fx = self._fixture("fail_" + target)
                orig = launcher._write_json
                state = {"failed": False}
                def flaky(path, obj, target=target, fx=fx, state=state, orig=orig):
                    if os.path.abspath(path) == os.path.abspath(fx[target]) and not state["failed"]:
                        state["failed"] = True
                        raise OSError("simulated " + target + " failure")
                    return orig(path, obj)
                launcher._write_json = flaky
                try:
                    meta = self._post(fx)
                finally:
                    launcher._write_json = orig
                rep = _load(fx["report"]); rm = _load(fx["run_meta"])
                self.assertEqual(meta["status"], "PERSISTENCE_FAILED")
                self.assertEqual(rep["status"], "PERSISTENCE_FAILED")
                self.assertFalse(meta["scientific_go"])
                self.assertNotEqual(rm["status"], "RUN_COMPLETE")
        finally:
            experiment.exact_signflip = orig_analyze

    def test_post_stats_hash_tamper_rejects_run_complete(self):
        fx = self._fixture("post_hash_tamper")
        orig_exact = self._fast_analyze()
        orig_analyze = experiment.analyze

        def tamper(out_dir):
            rep = orig_analyze(out_dir)
            p = os.path.join(out_dir, "201_sham.npz")
            np.savez_compressed(p, kc=np.zeros((1, 1), np.uint8))
            return rep

        experiment.analyze = tamper
        try:
            meta = self._post(fx)
        finally:
            experiment.analyze = orig_analyze
            experiment.exact_signflip = orig_exact
        self.assertEqual(meta["status"], "POSTPROCESS_FAILED")
        self.assertFalse(meta["experiment_succeeded"])
        self.assertFalse(meta["scientific_go"])
        self.assertFalse(meta["post_stats_hashes_ok"])
        self.assertIn("201_sham.npz", meta["post_stats_hash_mismatches"])
        rep = _load(fx["report"]); rm = _load(fx["run_meta"])
        self.assertEqual(rep["status"], "POSTPROCESS_FAILED")
        self.assertIn("201_sham.npz", rep["run_meta"]["post_stats_hash_mismatches"])
        self.assertNotEqual(rm["status"], "RUN_COMPLETE")
        self.assertFalse(rm["experiment_succeeded"])

    def test_synthetic_job_timeout_memory_exit(self):
        pkg = os.path.join(self.tmp, "s3_synth"); os.makedirs(pkg, exist_ok=True)
        _write(os.path.join(pkg, "__init__.py"), "")
        _write(os.path.join(pkg, "child.py"),
               "import argparse, os, sys, time\n"
               "p=argparse.ArgumentParser(); p.add_argument('--out'); p.add_argument('--mode'); a=p.parse_args()\n"
               "if a.mode=='timeout': time.sleep(10)\n"
               "if a.mode=='mem': b=bytearray(512*1024**2)\n"
               "if a.mode=='exit': raise SystemExit(1)\n"
               "open(os.path.join(a.out,'artifact.txt'),'w').write('ok')\n")
        from loop.p11_2_s1f_safe_job_launcher import run_under_job_safe
        for mode, want in (("timeout", "CHILD_TIMEOUT"), ("mem", "CHILD_MEMORY_LIMIT"), ("exit", "CHILD_EXIT_NONZERO")):
            out = os.path.join(self.tmp, "job_" + mode); os.makedirs(out)
            name = f"Local\\P11_2_S3_test_{os.getpid()}_{time.time_ns()}"
            r = run_under_job_safe(256 * 1024 ** 2, module="s3_synth.child",
                                   args=["--out", out, "--mode", mode], cwd=self.tmp,
                                   wall_cap_s=1 if mode == "timeout" else 30,
                                   mem_cap_mib=256, log_path=os.path.join(out, "log"),
                                   required_artifact=os.path.join(out, "artifact.txt"), job_name=name)
            self.assertEqual(r["child_status"], want)


if __name__ == "__main__":
    unittest.main()
