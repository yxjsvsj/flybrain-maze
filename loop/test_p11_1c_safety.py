"""P11-1c 回归：Job 竞态修复、分配失败终止、超时/内存限制、全量 SHA、单位、无重试、未授权。"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from loop.p11_1c_full_brain_bench import safety_selftest

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p11_1c safety ===========")
    s = safety_selftest()
    check("no-job alloc succeeds (control)", s["no_job_alloc"]["ok"] is True)
    check("job 256 MiB alloc fails (isolation)", s["job_alloc_limit"]["ok"] is True)
    check("assignment failure terminates child", s["assign_failure"]["ok"] is True)
    check("timeout terminates child", s["timeout"]["ok"] is True)
    check("no auto-retry (calls=1)", s["no_retry"]["ok"] is True)
    check("all safety tests pass", s["all_ok"] is True)

    d = json.load(open(os.path.join(HERE, "p11_1c_report.json"), encoding="utf-8"))
    check("experiment NOT_AUTHORIZED", d["status"] == "NOT_AUTHORIZED")
    check("authorization not granted", d["authorization"]["granted"] is False)
    check("full brain NOT started", d["full_brain_started"] is False)
    check("official hashes full-equal", d["hashes_ok"] is True)
    check("brain full SHA", d["hashes"]["brain.npz"]["full_equal"] is True)
    check("weights full SHA", d["hashes"]["weights.npz"]["full_equal"] is True)
    v = d["verdicts"]
    check("RESOURCE NOT_AUTHORIZED", v["RESOURCE"] == "NOT_AUTHORIZED")
    check("FULL_BRAIN_BASELINE INCONCLUSIVE", v["FULL_BRAIN_BASELINE"] == "INCONCLUSIVE")
    check("FULL_VS_SUBNET_DIFFERENCE INCONCLUSIVE", v["FULL_VS_SUBNET_DIFFERENCE"] == "INCONCLUSIVE")
    check("MODEL_ROOT_CAUSE NOT_ESTABLISHED", v["MODEL_ROOT_CAUSE"] == "NOT_ESTABLISHED")
    check("Level A/B/C/Biological NO-GO", v["Level_A_Training"] == "NO-GO" and v["Level_B"] == "NO-GO"
          and v["Level_C"] == "NO-GO" and v["Biological"] == "NO-GO")
    check("training not authorized", d["training_authorized"] is False)

    src = open(os.path.join(HERE, "p11_1c_full_brain_bench.py"), encoding="utf-8").read()
    check("CREATE_SUSPENDED before assign", "CREATE_SUSPENDED" in src)
    check("AssignProcessToJobObject + ResumeThread", "AssignProcessToJobObject" in src and "ResumeThread" in src)
    check("assignment failure terminates", "assign_resume_or_terminate" in src and "TerminateProcess" in src)
    check("per-process AND job-total limits",
          "JOB_OBJECT_LIMIT_PROCESS_MEMORY" in src and "JOB_OBJECT_LIMIT_JOB_MEMORY" in src)
    check("full SHA equality (no prefix compare)", "actual == known" in src and "startswith" not in src)
    check("MiB/GiB units", "_mib" in src and "GiB" in src)
    check("no auto-retry helper", "def run_once" in src)
    check("no full-brain run in this task", d["full_brain_started"] is False)

    # the three pre-existing dirty files must be preserved
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        out = r.stdout
        check("pre-existing dirty files preserved", all(x in out for x in
              ("hardware/tof5_client.py", "loop/diagnose.py", "loop/p10c1b_decoder_freeze.json")))
    except Exception:                            # noqa: BLE001
        check("git status available", False)
    # P1 freeze
    try:
        r = subprocess.run(["git", "diff", "--stat", "P1-mapfix", "--", "nav", "config.py",
                            "loop/decode.py", "loop/encode.py", "loop/run.py"],
                           cwd=ROOT, capture_output=True, text=True, timeout=30)
        check("P1 freeze empty", r.returncode == 0 and r.stdout.strip() == "")
    except Exception:                            # noqa: BLE001
        check("P1 freeze check available", False)
    # historical P11-1b retained
    h = json.load(open(os.path.join(HERE, "p11_1b_report.json"), encoding="utf-8"))
    check("P11-1b historical RESOURCE_OK retained", h["verdict"] == "RESOURCE_OK")
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
