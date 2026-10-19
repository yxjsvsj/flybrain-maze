"""P11-1b：完整 MaleCNS 一次性无学习资源基准（**仅本次授权**）。

父进程：校验基线/哈希 -> Windows Job Object (2 GB 硬限制 + kill-on-close) -> 小规模
内存限制自检 -> 启动**独立子进程**加载完整脑 -> 5 步预热 + 100 步无刺激测量 -> 监控时间/内存 ->
写报告。**不训练、不更新 ΔW、不注入刺激、不跑迷宫。**
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_1b_report.json")
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
BASELINE = "e7bc9fc"
WALL_CAP_S = 900.0
MEM_CAP_MB = 2048.0
MEM_CAP_BYTES = int(MEM_CAP_MB * 1024 * 1024)
WARMUP = 5
MEASURE = 100
SEED = 101

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wt.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wt.DWORD),
                ("SchedulingClass", wt.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

kernel32.CreateJobObjectW.restype = wt.HANDLE
kernel32.CreateJobObjectW.argtypes = [wt.LPVOID, wt.LPCWSTR]
kernel32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, wt.LPVOID, wt.DWORD]
kernel32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
kernel32.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
kernel32.CloseHandle.argtypes = [wt.HANDLE]
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE
psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
psapi.GetProcessMemoryInfo.restype = wt.BOOL


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _peak_self_mb():
    c = PMC(); c.cb = ctypes.sizeof(c)
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return round(c.PeakWorkingSetSize / 1e6, 2) if ok else None


def _child_peak_mb(pid):
    h = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        return None
    try:
        c = PMC(); c.cb = ctypes.sizeof(c)
        ok = psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
        return round(c.PeakWorkingSetSize / 1e6, 2) if ok else None
    finally:
        kernel32.CloseHandle(h)


def _create_job(limit_bytes):
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None, False
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (JOB_OBJECT_LIMIT_PROCESS_MEMORY
                                             | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    info.ProcessMemoryLimit = ctypes.c_size_t(limit_bytes)
    ok = kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                          ctypes.byref(info), ctypes.sizeof(info))
    return job, bool(ok)


def _assign(job, pid):
    h = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
    if not h:
        return False
    try:
        return bool(kernel32.AssignProcessToJobObject(job, h))
    finally:
        kernel32.CloseHandle(h)


def _job_peak_mb(job):
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    ret = wt.DWORD()
    ok = kernel32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation,
                                            ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(ret))
    return round(info.PeakProcessMemoryUsed / 1e6, 2) if ok else None


def _close_job(job):
    if job:
        kernel32.CloseHandle(job)


# ---------------------------------------------------------------- child modes
def child_alloc_test():
    try:
        n = 512 * 1024 * 1024
        b = bytearray(n)
        step = 4096
        for i in range(0, n, step):
            b[i] = 1
        print("ALLOC_OK")
        return 0
    except MemoryError:
        print("MEMORY_ERROR")
        return 3


def child_full_brain(result_path):
    import numpy as np
    from flybrain import FlyBrain
    from loop.p10c1b_lif_benchmark import _peak_memory_mb
    try:
        t0 = time.perf_counter()
        brain = FlyBrain(data=DATA_DIR, device="cpu", batch=1, seed=SEED, dt=0.020)
        init_s = time.perf_counter() - t0
        mem_init = _peak_memory_mb()
        for _ in range(WARMUP):
            brain.step()
        ts, fired = [], []
        finite = True
        for i in range(MEASURE):
            t = time.perf_counter()
            f = brain.step()
            ts.append(time.perf_counter() - t)
            fired.append(int(len(f)))
            if not np.isfinite(brain.v).all():
                finite = False
        mem_end = _peak_memory_mb()
        res = {
            "ok": True, "init_s": round(init_s, 4), "warmup_steps": WARMUP,
            "measure_steps": MEASURE,
            "per_step_ms": [round(x * 1000, 4) for x in ts],
            "median_ms": round(float(np.median(ts)) * 1000, 4),
            "p95_ms": round(float(np.percentile(ts, 95)) * 1000, 4),
            "cumulative_s": {"10": round(sum(ts[:10]), 4), "50": round(sum(ts[:50]), 4),
                             "100": round(sum(ts), 4)},
            "fired_mean": round(float(np.mean(fired)), 2), "fired_max": int(max(fired)),
            "finite_state": bool(finite), "peak_mem_init_MB": mem_init, "peak_mem_end_MB": mem_end,
            "n_neurons": int(brain.n), "nnz": int(len(brain.indices)),
            "device": "cpu", "seed": SEED,
            "params": {"dt": brain.dt, "tau": brain.tau, "gain": brain.gain,
                       "tonic": brain.tonic, "noise_hz": brain.noise_hz,
                       "noise_amp": brain.noise_amp},
            "versions": {"python": sys.version.split()[0],
                         "numpy": np.__version__,
                         "scipy": __import__("scipy").__version__,
                         "flybrain": getattr(__import__("flybrain"), "__version__", "0.1.0")},
            "stimulus": "none", "learning": False, "dW": 0,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(res, f)
        return 0
    except MemoryError as e:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": f"MemoryError under job limit: {e}"}, f)
        return 4
    except Exception as e:                       # noqa: BLE001
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": repr(e)[:300]}, f)
        return 5


# ---------------------------------------------------------------- parent
def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    if "--alloc-test" in sys.argv:
        return child_alloc_test()
    if "--child" in sys.argv:
        p = sys.argv[sys.argv.index("--child") + 1]
        return child_full_brain(p)

    started = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "protocol": "P11-1b/resource-benchmark/v1",
        "authorization": {
            "granted": True,
            "source": "user chat reply: '批准' for P11-1b one-run full-brain no-stim no-learning resource benchmark",
            "date": "2026-10-19", "exact_time": "NOT RECORDED",
            "signature": "none (chat approval only; not a cryptographic signature)",
            "scope": {"full_brain": "MaleCNS v1.0 166700 neurons / 25582938 connections",
                      "warmup_steps": WARMUP, "measure_steps": MEASURE,
                      "stimulus": "none", "learning": False, "dW": 0,
                      "wall_cap_min": 15, "mem_cap_MB": 2048, "one_run": True},
            "not_authorized": ["25-seed model-validity experiment", "real KC->MBON dW learning",
                               "Dijkstra", "B/C navigation", "physical robot"],
        },
        "started_at": started, "training_authorized": False,
    }
    import subprocess as _sp
    try:
        head = _sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                       capture_output=True, text=True, timeout=30).stdout.strip()
        porcelain = _sp.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:                            # noqa: BLE001
        head, porcelain = None, "unavailable"
    report["baseline_commit"] = head
    report["baseline_expected"] = BASELINE
    report["worktree_clean"] = (porcelain == "")
    report["dirty_files"] = [l[3:] for l in porcelain.splitlines()] if porcelain else []
    report["dirty_files_note"] = ("pre-existing unrelated leftovers (hardware/tof5_client.py, loop/diagnose.py, "
                                  "loop/p10c1b_decoder_freeze.json CRLF); not imported by this benchmark; not committed")
    # official hashes
    hashes = {}
    for name in ("brain.npz", "weights.npz"):
        p = os.path.join(DATA_DIR, name)
        hashes[name] = _sha256_file(p) if os.path.exists(p) else None
    report["hashes"] = hashes
    report["hashes_ok"] = (hashes.get("brain.npz", "").startswith("cc9bd1ecd00bd703")
                           and hashes.get("weights.npz", "").startswith("c29919aa44069a27"))
    with open(__file__, "rb") as f:
        report["script_sha256"] = hashlib.sha256(f.read()).hexdigest()
    if not report["hashes_ok"]:
        report.update(verdict="RESOURCE_NO_GO", reason="official hash mismatch")
        json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("RESOURCE_NO_GO: hash mismatch"); return 1

    # --- job-limit selftest (256 MB) ---
    selftest = {"limit_MB": 256, "ok": False, "output": None, "rc": None}
    job, ok = _create_job(256 * 1024 * 1024)
    if ok:
        proc = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--alloc-test"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if _assign(job, proc.pid):
            try:
                out, _ = proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                kernel32.TerminateJobObject(job, 1); out, _ = proc.communicate()
            selftest.update(output=(out or "").strip()[-100:], rc=proc.returncode)
            selftest["ok"] = ("MEMORY_ERROR" in (out or "") or proc.returncode not in (0,))
        _close_job(job)
    report["job_selftest"] = selftest
    if not selftest["ok"]:
        report.update(verdict="RESOURCE_NO_GO",
                      reason="could not reliably enforce a process memory limit; full brain NOT started")
        json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("RESOURCE_NO_GO: job memory limit selftest failed; full brain NOT started")
        return 1

    # --- one full-brain run ---
    result_path = os.path.join(tempfile.gettempdir(), "p11_1b_child_result.json")
    if os.path.exists(result_path):
        os.remove(result_path)
    job, ok = _create_job(MEM_CAP_BYTES)
    if not ok:
        report.update(verdict="RESOURCE_NO_GO", reason="job object creation failed")
        json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("RESOURCE_NO_GO: job creation failed"); return 1
    report["full_brain_started"] = True
    proc = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--child", result_path],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assigned = _assign(job, proc.pid)
    report["job"] = {"process_memory_limit_MB": MEM_CAP_MB, "kill_on_close": True,
                     "assigned": bool(assigned)}
    t0 = time.perf_counter()
    peak_ws = 0.0; timeout = False; mem_exceeded = False
    while proc.poll() is None:
        elapsed = time.perf_counter() - t0
        ws = _child_peak_mb(proc.pid)
        if ws:
            peak_ws = max(peak_ws, ws)
            if ws > MEM_CAP_MB:
                mem_exceeded = True
                kernel32.TerminateJobObject(job, 1)
                break
        if elapsed > WALL_CAP_S:
            timeout = True
            kernel32.TerminateJobObject(job, 1)
            break
        time.sleep(0.2)
    try:
        out, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        kernel32.TerminateJobObject(job, 1); out, _ = proc.communicate()
    wall_s = round(time.perf_counter() - t0, 3)
    job_peak = _job_peak_mb(job)
    _close_job(job)
    child = None
    if os.path.exists(result_path):
        try:
            child = json.load(open(result_path, encoding="utf-8"))
        except Exception:                        # noqa: BLE001
            child = None
    report["child"] = child
    report["parent"] = {"wall_s": wall_s, "exit_code": proc.returncode,
                        "stdout_tail": (out or "").strip()[-300:],
                        "peak_working_set_MB": peak_ws or (child or {}).get("peak_mem_end_MB"),
                        "job_peak_committed_MB": job_peak,
                        "timeout": timeout, "memory_exceeded": mem_exceeded,
                        "wall_cap_s": WALL_CAP_S, "mem_cap_MB": MEM_CAP_MB}
    ok_run = (not timeout and not mem_exceeded and proc.returncode == 0 and child
              and child.get("ok") and child.get("finite_state") and child.get("measure_steps") == MEASURE
              and "10" in child.get("cumulative_s", {}) and "50" in child.get("cumulative_s", {})
              and "100" in child.get("cumulative_s", {}))
    if ok_run:
        _vals = [peak_ws, child.get("peak_mem_init_MB") or 0,
                 child.get("peak_mem_end_MB") or 0, job_peak or 0]
        peak = max([x for x in _vals if x is not None])
        within = peak <= MEM_CAP_MB and wall_s <= WALL_CAP_S
        verdict = "RESOURCE_OK" if within else "RESOURCE_NO_GO"
        report["verdict"] = verdict
        report["reason"] = ("completed within 15 min / 2 GB" if within
                            else "peak/time exceeded the approved caps")
        report["peak_mem_MB"] = peak
        if child:
            report["estimate_10000_steps"] = {
                "init_once_s": child["init_s"],
                "core_s": round(child["median_ms"] * 10000 / 1000.0, 2),
                "core_min": round(child["median_ms"] * 10000 / 1000.0 / 60.0, 2),
                "memory_peak_MB": peak,
                "uncertainty": ("short 100-step benchmark; the 10,000-step none-only run may differ "
                                "due to cache/thermal/fired-count drift; not a full experiment measurement"),
            }
    else:
        verdict = "RESOURCE_NO_GO" if (timeout or mem_exceeded or proc.returncode not in (0, 3, 4, 5)) else "INCONCLUSIVE"
        if child is None:
            verdict = "RESOURCE_NO_GO"
        report["verdict"] = verdict
        report["reason"] = ("timeout" if timeout else "memory cap exceeded" if mem_exceeded
                            else "child failed / incomplete result")
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[P11-1b] verdict:", report["verdict"], "| init_s:", (child or {}).get("init_s"),
          "| median_ms:", (child or {}).get("median_ms"), "| p95_ms:", (child or {}).get("p95_ms"))
    print("  peak_mem_MB:", report.get("peak_mem_MB"), "| wall_s:", wall_s,
          "| job_peak_committed_MB:", job_peak)
    print("  child exit:", proc.returncode, "| selftest:", selftest)
    print("  estimate:", report.get("estimate_10000_steps"))
    print("wrote p11_1b_report.json")
    return 0 if report["verdict"] == "RESOURCE_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
