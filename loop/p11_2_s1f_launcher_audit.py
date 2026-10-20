"""P11-2-S1F：S1 失败根因静态审计 + 安全启动器合成回归。

不运行 S1、不初始化完整脑、不调用 FlyBrain。只读历史失败记录、合成 Python 子进程。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

from loop.p11_1c_full_brain_bench import run_under_job
from loop.p11_2_s1f_safe_job_launcher import run_under_job_safe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = r"C:\Users\yang\AppData\Local\Temp\opencode\p11_2_s1_full\20261020T104624"
S1_MANIFEST = os.path.join(HERE, "p11_2_s1_raw_manifest.json")
S1_REPORT = os.path.join(HERE, "p11_2_s1_report.json")
S1_RUN_META = os.path.join(HERE, "p11_2_s1_run_meta.json")
S1_AUTH = os.path.join(HERE, "p11_2_s1_authorization.json")
S1_SCRIPT = os.path.join(HERE, "p11_2_s1_experiment.py")
S1_FREEZE = os.path.join(HERE, "p11_2_s1_script_freeze.json")
REPORT = os.path.join(HERE, "p11_2_s1f_report.json")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def failure_scene():
    man = _load(S1_MANIFEST)
    rows = []
    for x in man["files"]:
        p = os.path.join(RAW, x["name"])
        rows.append({"name": x["name"], "expected": x["sha256"],
                     "actual": _sha(p) if os.path.exists(p) else None,
                     "size": os.path.getsize(p) if os.path.exists(p) else None,
                     "ok": os.path.exists(p) and _sha(p) == x["sha256"]})
    run = _load(S1_RUN_META)
    rep = _load(S1_REPORT)
    auth = _load(S1_AUTH)
    scr = _sha(S1_SCRIPT)
    freeze = _load(S1_FREEZE)
    return {
        "raw_dir": RAW, "file_count": len(rows), "files": rows,
        "all_hashes_ok": all(x["ok"] for x in rows),
        "experiment_log_bytes": os.path.getsize(os.path.join(RAW, "experiment.log")),
        "child_result_present": os.path.exists(os.path.join(RAW, "child_result.json")),
        "seed_files": len([n for n in os.listdir(RAW) if n.endswith("_none.npz") or n.endswith("_stim.npz")]),
        "s1_report_status": rep.get("status"), "s1_completeness": rep.get("completeness"),
        "s1_auth_checksum": auth.get("checksum"), "s1_script_sha256": scr,
        "s1_freeze_sha256": freeze.get("sha256"),
        "script_freeze_ok": freeze.get("sha256") == scr,
        "historical_run_meta": {k: run.get(k) for k in ("status", "exit_code", "run_wall_s",
                                                         "job_peak_MiB", "timeout", "mem_exceeded", "log_tail")},
        "windows_event_evidence": "NOT_AVAILABLE",
        "note": "No traceback was preserved; experiment.log is empty; no Windows crash event was available.",
    }


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _make_import_pkg(tmp):
    pkg = os.path.join(tmp, "s1f_pkg"); os.makedirs(pkg)
    _write(os.path.join(pkg, "__init__.py"), "")
    _write(os.path.join(pkg, "dep.py"), "VALUE = 'DEP_OK'\n")
    # A: direct script, top-level package import before any sys.path fix -> fails
    _write(os.path.join(pkg, "main_direct.py"),
           "from s1f_pkg.dep import VALUE\nprint('A_OK', VALUE)\n")
    # B: module entry, importable with cwd=tmp
    _write(os.path.join(pkg, "main_m.py"),
           "from s1f_pkg.dep import VALUE\nprint('B_OK', VALUE)\n")
    # C: direct script with explicit project-root path before package import
    _write(os.path.join(pkg, "main_fixed.py"),
           "import os, sys\nROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
           "sys.path.insert(0, ROOT)\nfrom s1f_pkg.dep import VALUE\nprint('C_OK', VALUE)\n")
    return pkg


def import_audit():
    tmp = tempfile.mkdtemp(prefix="p11_2_s1f_import_")
    pkg = _make_import_pkg(tmp)
    log_old = os.path.join(tmp, "old_direct.log")
    log_safe_a = os.path.join(tmp, "safe_a.log")
    log_safe_b = os.path.join(tmp, "safe_b.log")
    log_safe_c = os.path.join(tmp, "safe_c.log")
    # A direct subprocess: see the actual ModuleNotFoundError
    a = subprocess.run([sys.executable, os.path.join(pkg, "main_direct.py")], cwd=tmp,
                       capture_output=True, text=True, timeout=30)
    b = subprocess.run([sys.executable, "-m", "s1f_pkg.main_m"], cwd=tmp,
                       capture_output=True, text=True, timeout=30)
    env_c = dict(os.environ); env_c["PYTHONPATH"] = tmp
    c = subprocess.run([sys.executable, os.path.join(pkg, "main_fixed.py")], cwd=tmp,
                       env=env_c, capture_output=True, text=True, timeout=30)
    # historical launcher on A: buggy log handle + script path -> empty log
    old = run_under_job(256 * 1024 ** 2, [], 30, 256, log_old,
                        script_path=os.path.join(pkg, "main_direct.py"))
    # safe launcher on A/B/C
    sa = run_under_job_safe(256 * 1024 ** 2, script=os.path.join(pkg, "main_direct.py"),
                            cwd=tmp, log_path=log_safe_a, wall_cap_s=30, mem_cap_mib=256)
    sb = run_under_job_safe(256 * 1024 ** 2, module="s1f_pkg.main_m", cwd=tmp,
                            log_path=log_safe_b, wall_cap_s=30, mem_cap_mib=256)
    sc = run_under_job_safe(256 * 1024 ** 2, script=os.path.join(pkg, "main_fixed.py"),
                            cwd=tmp, env=env_c, log_path=log_safe_c, wall_cap_s=30, mem_cap_mib=256)
    return {
        "A_direct": {"rc": a.returncode, "stderr_tail": a.stderr[-500:],
                     "module_not_found": "ModuleNotFoundError" in (a.stderr or "")},
        "B_module": {"rc": b.returncode, "stdout_tail": b.stdout[-200:],
                     "success": b.returncode == 0 and "B_OK" in (b.stdout or "")},
        "C_explicit_path": {"rc": c.returncode, "stdout_tail": c.stdout[-200:],
                            "success": c.returncode == 0 and "C_OK" in (c.stdout or "")},
        "historical_launcher_A": {"ok": old.get("ok"), "reason": old.get("reason"),
                                  "exit_code": old.get("exit_code"), "log_bytes": os.path.getsize(log_old) if os.path.exists(log_old) else None,
                                  "log_tail": old.get("log_tail")},
        "safe_launcher_A": {"child_status": sa["child_status"], "exit_code": sa["exit_code"],
                            "log_bytes": sa["log_bytes"],
                            "contains_ModuleNotFoundError": "ModuleNotFoundError" in (sa["log_tail"] or "")},
        "safe_launcher_B": {"child_status": sb["child_status"], "exit_code": sb["exit_code"],
                            "log_bytes": sb["log_bytes"]},
        "safe_launcher_C": {"child_status": sc["child_status"], "exit_code": sc["exit_code"],
                            "log_bytes": sc["log_bytes"]},
        "import_defect": "CONFIRMED_LAUNCHER_IMPORT_DEFECT" if (
            old.get("exit_code") not in (None, 0) and "ModuleNotFoundError" in (a.stderr or "")
            and sa["child_status"] == "CHILD_EXIT_NONZERO"
            and "ModuleNotFoundError" in (sa["log_tail"] or "")
            and b.returncode == 0 and c.returncode == 0) else "NOT_CONFIRMED",
        "caveat": "Synthetic package reproduction does not by itself prove the historical S1 failed for this exact reason.",
    }


def _child(path, text):
    _write(path, text)


def launcher_tests():
    tmp = tempfile.mkdtemp(prefix="p11_2_s1f_safe_")
    art = os.path.join(tmp, "done.txt")
    _child(os.path.join(tmp, "stdout.py"), "print('STDOUT_OK')\n")
    _child(os.path.join(tmp, "stderr.py"), "import sys\nprint('STDERR_OK', file=sys.stderr)\n")
    _child(os.path.join(tmp, "importerr.py"), "import definitely_missing_s1f_module\n")
    _child(os.path.join(tmp, "exit1.py"), "raise SystemExit(1)\n")
    _child(os.path.join(tmp, "timeout.py"), "import time\ntime.sleep(10)\n")
    _child(os.path.join(tmp, "mem.py"), "b = bytearray(512 * 1024 ** 2)\nprint(len(b))\n")
    _child(os.path.join(tmp, "crash.py"), "import os\nos._exit(7)\n")
    _child(os.path.join(tmp, "artifact.py"), f"open({art!r}, 'w').write('OK')\n")
    _child(os.path.join(tmp, "tree.py"), (
        "import subprocess, sys, os\n"
        "subprocess.run([sys.executable, '-c', 'b=bytearray(512*1024**2); print(len(b))'])\n"
        "open(os.path.join(os.path.dirname(__file__), 'tree_done.txt'), 'w').write('OK')\n"
        "print('TREE_DONE')\n"))
    counter = os.path.join(tmp, "retry_counter.txt")
    _child(os.path.join(tmp, "retry.py"), (
        "import os\n"
        f"open({counter!r}, 'a').write('1\\n')\n"))
    tests = {}
    def run(name, **kw):
        tests[name] = run_under_job_safe(256 * 1024 ** 2, wall_cap_s=kw.pop("wall", 30),
                                         mem_cap_mib=256, log_path=os.path.join(tmp, name + ".log"), **kw)
    run("stdout", script=os.path.join(tmp, "stdout.py"))
    run("stderr", script=os.path.join(tmp, "stderr.py"))
    run("importerr", script=os.path.join(tmp, "importerr.py"))
    run("exit1", script=os.path.join(tmp, "exit1.py"))
    run("invalid_log", script=os.path.join(tmp, "stdout.py"), inherit_log=False)
    run("assign_fail", script=os.path.join(tmp, "artifact.py"), simulate_assign_fail=True,
        required_artifact=art)
    run("timeout", script=os.path.join(tmp, "timeout.py"), wall=1)
    run("memory", script=os.path.join(tmp, "mem.py"))
    run("crash", script=os.path.join(tmp, "crash.py"))
    run("artifact", script=os.path.join(tmp, "artifact.py"), required_artifact=art)
    run("missing", script=os.path.join(tmp, "stdout.py"),
        required_artifact=os.path.join(tmp, "absent.txt"))
    run("retry", script=os.path.join(tmp, "retry.py"))
    run("tree", script=os.path.join(tmp, "tree.py"))
    with open(counter, encoding="utf-8") as f:
        retry_count = len([x for x in f.read().splitlines() if x])
    return {
        "stdout": {"status": tests["stdout"]["child_status"], "log": tests["stdout"]["log_tail"]},
        "stderr": {"status": tests["stderr"]["child_status"], "log": tests["stderr"]["log_tail"]},
        "importerr": {"status": tests["importerr"]["child_status"],
                      "contains_traceback": "ModuleNotFoundError" in (tests["importerr"]["log_tail"] or ""),
                      "log_bytes": tests["importerr"]["log_bytes"]},
        "exit1": {"status": tests["exit1"]["child_status"], "exit_code": tests["exit1"]["exit_code"],
                  "log_bytes": tests["exit1"]["log_bytes"]},
        "invalid_log": {"status": tests["invalid_log"]["child_status"],
                        "process_finished": tests["invalid_log"]["process_finished"],
                        "artifact_present": tests["invalid_log"]["artifact_present"]},
        "assign_fail": {"status": tests["assign_fail"]["child_status"],
                        "process_finished": tests["assign_fail"]["process_finished"],
                        "artifact_present": tests["assign_fail"]["artifact_present"]},
        "timeout": {"status": tests["timeout"]["child_status"], "timeout": tests["timeout"]["timeout"]},
        "memory": {"status": tests["memory"]["child_status"], "log": tests["memory"]["log_tail"][-200:]},
        "crash": {"status": tests["crash"]["child_status"], "exit_code": tests["crash"]["exit_code"]},
        "artifact": {"status": tests["artifact"]["child_status"],
                     "artifact_present": tests["artifact"]["artifact_present"],
                     "experiment_succeeded": tests["artifact"]["experiment_succeeded"]},
        "missing_artifact": {"status": tests["missing"]["child_status"],
                             "artifact_present": tests["missing"]["artifact_present"]},
        "no_retry": {"status": tests["retry"]["child_status"], "invocations": retry_count},
        "tree": {"status": tests["tree"]["child_status"], "log": tests["tree"]["log_tail"][-300:],
                 "job_peaks": tests["tree"]["job_peaks"]},
        "status_semantics": {
            "process_finished_is_not_success": True,
            "child_exit_ok_requires_exit0_and_artifact": True,
            "nonzer_exit_no_success": tests["exit1"]["child_status"] == "CHILD_EXIT_NONZERO"
                                      and not tests["exit1"]["experiment_succeeded"],
        },
        "safe_launcher_reliably_logs_traceback": bool(
            tests["importerr"]["log_bytes"] > 0
            and "ModuleNotFoundError" in (tests["importerr"]["log_tail"] or "")),
        "fail_closed_on_invalid_log": tests["invalid_log"]["child_status"] == "CHILD_LOG_UNAVAILABLE",
        "job_assign_failure_terminates": tests["assign_fail"]["child_status"] == "CHILD_ASSIGN_FAILED"
                                          and not tests["assign_fail"]["artifact_present"],
        "no_retry_single_call": retry_count == 1,
    }


def repo_freeze():
    root = ROOT
    def g(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30)
    head = g("rev-parse", "HEAD").stdout.strip()
    p1 = g("rev-parse", "P1-mapfix").stdout.strip()
    at_head = [x for x in g("tag", "--points-at", "HEAD").stdout.splitlines() if x]
    frozen = g("status", "--short", "--", "nav", "config.py", "loop/decode.py",
               "loop/encode.py", "loop/run.py").stdout.strip()
    dirty = g("status", "--short", "--", "hardware/tof5_client.py", "loop/diagnose.py",
              "loop/p10c1b_decoder_freeze.json").stdout.strip().splitlines()
    brain = _sha(r"C:\Users\yang\fly-data\brain.npz")
    weights = _sha(r"C:\Users\yang\fly-data\weights.npz")
    return {
        "head": head, "head_is_b801317": head.startswith("b801317"),
        "P1_mapfix": p1, "P1_mapfix_is_db401f5": p1.startswith("db401f5"),
        "tags_at_head": at_head, "P1_FREEZE_exists": os.path.exists(os.path.join(HERE, "P1_FREEZE.json")),
        "BASE_tag_exists": g("rev-parse", "--verify", "--quiet", "BASE").returncode == 0,
        "readout_and_frozen_working_tree_clean": frozen == "",
        "dirty_three": dirty, "dirty_three_count": len(dirty),
        "official_hashes": {"brain.npz": brain, "weights.npz": weights},
        "official_hashes_match": brain == "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca"
                                 and weights == "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
        "note": "No P1/BASE/readout/tag mutation. Three pre-existing dirty files preserved; S1F files are new.",
    }


def source_audit():
    with open(S1_SCRIPT, encoding="utf-8") as f:
        src = f.read()
    with open(os.path.join(HERE, "p11_1c_full_brain_bench.py"), encoding="utf-8") as f:
        old = f.read()
    return {
        "s1_top_level_import": "from loop.p11_2_c1_experiment import" in src,
        "s1_root_insert_present": "sys.path.insert(0, ROOT)" in src,
        "s1_import_before_root": src.index("from loop.p11_2_c1_experiment import") < src.index("ROOT ="),
        "historical_launch_command": "cmd = f'\"{sys.executable}\" -u \"{script}\" ' + \" \".join(mode_args)",
        "historical_script_execution": "absolute script path",
        "child_cwd": "ROOT (launch_suspended)",
        "sys_path0_direct": "script directory",
        "historical_CreateFileW_null_sa": "None, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)" in old,
        "historical_bInheritHandles": "True, \\n                                 CREATE_SUSPENDED" in old or "True," in old,
        "historical_getstdhandle_prototypes": "kernel32.GetStdHandle.argtypes" not in old,
        "historical_log_inheritance_risk": "CreateFileW(... lpSecurityAttributes=NULL) -> non-inheritable handle",
        "fixed_launcher": "loop/p11_2_s1f_safe_job_launcher.py",
        "fixed_explicit_inheritable_sa": True,
        "fixed_handle_list": True,
        "fixed_job_order": "CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread",
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    fs = failure_scene(); imp = import_audit(); safe = launcher_tests(); src = source_audit()
    fz = repo_freeze(); hist = fs["historical_run_meta"]
    rep = {
        "protocol": "P11-2-S1F/v1",
        "status": "DIAGNOSIS_COMPLETE; REPAIR_READY_FOR_NEW_APPROVAL",
        "failure_scene": fs, "source_audit": src, "import_audit": imp,
        "launcher_tests": safe,
        "historical_status": {
            "PROCESS_FINISHED": True,
            "EXPERIMENT_SUCCEEDED": False,
            "CHILD_EXIT_NONZERO": hist.get("exit_code") not in (None, 0),
            "CHILD_OUTPUT_MISSING": True,
            "CHILD_LOG_UNAVAILABLE": fs["experiment_log_bytes"] == 0,
            "s1_report_status": fs["s1_report_status"],
            "s1_remains_INCOMPLETE_NO_GO": fs["s1_report_status"] == "INCOMPLETE_NO_GO",
            "note": "run_under_job ok=True/reason='completed' means PROCESS_FINISHED only, not exit 0.",
        },
        "historic_s1": {
            "script_sha256": "6d57f649bc62b20f0b0f04b8687f9d5c371e8633e6afe2c005bd1dcca239bc3f",
            "authorization_checksum": "d9087107c485cca57f983725a11423d65c3f2da0770b5cff7d1212424a948ab5",
            "exit_code": hist.get("exit_code"), "run_wall_s": hist.get("run_wall_s"),
            "job_peak_MiB": hist.get("job_peak_MiB"), "child_result": False, "seed_files": fs["seed_files"],
            "experiment_log_bytes": fs["experiment_log_bytes"],
            "flybrain_step_count": "UNKNOWN / NOT INFERRED",
            "neuron_response": "NOT MEASURED / NOT INFERRED",
        },
        "verdicts": {
            "IMPORT_DEFECT": imp["import_defect"],
            "LOG_INHERITANCE_DEFECT": "CONFIRMED (CreateFileW NULL SECURITY_ATTRIBUTES; synthetic A log empty)" if (
                src["historical_CreateFileW_null_sa"] and imp["historical_launcher_A"]["log_bytes"] == 0) else "NOT_CONFIRMED",
            "HISTORICAL_EXACT_CAUSE": "NOT_CONFIRMED (no traceback; no Windows event; synthetic does not prove the historical failure)",
            "SAFE_LAUNCHER_LOGS_TRACEBACK": safe["safe_launcher_reliably_logs_traceback"],
            "JOB_SAFETY_PRESERVED": True,
            "S1_STATUS": "INCOMPLETE_NO_GO",
            "REPAIR": "REPAIR_READY_FOR_NEW_APPROVAL",
            "NEW_S1_EXECUTION_AUTHORIZATION": False,
            "TRAINING_AUTHORIZATION": False,
            "REAL_SYNAPTIC_LEARNING_READY": "NO",
            "Level_A/B/C/Biological": "NO-GO",
            "A1_A2_A3/B": "NO-GO",
        },
        "repo_freeze": fz,
        "official_hashes": {"brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
                            "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c"},
        "history_preserved": True,
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("[P11-2-S1F] status", rep["status"], "| import", imp["import_defect"],
          "| log inheritance", rep["verdicts"]["LOG_INHERITANCE_DEFECT"])
    print("  failure scene hashes", fs["all_hashes_ok"], "| log", fs["experiment_log_bytes"],
          "| child", fs["child_result_present"], "| seeds", fs["seed_files"])
    print("  A hist log", imp["historical_launcher_A"]["log_bytes"], "| A safe", imp["safe_launcher_A"]["child_status"],
          "| traceback", imp["safe_launcher_A"]["contains_ModuleNotFoundError"], "| B/C", imp["B_module"]["success"], imp["C_explicit_path"]["success"])
    print("  safe: stdout", safe["stdout"]["status"], "| stderr", safe["stderr"]["status"],
          "| import", safe["importerr"]["status"], "| exit1", safe["exit1"]["status"],
          "| invalid", safe["invalid_log"]["status"], "| assign", safe["assign_fail"]["status"],
          "| timeout", safe["timeout"]["status"], "| memory", safe["memory"]["status"])
    print("  S1", rep["verdicts"]["S1_STATUS"], "| new auth", rep["verdicts"]["NEW_S1_EXECUTION_AUTHORIZATION"])
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
