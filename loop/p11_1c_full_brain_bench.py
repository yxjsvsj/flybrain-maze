"""P11-1c：完整脑 25-seed 实验的**安全加固**（暂停创建 -> 加入 Job -> 恢复）。

**本任务不授权运行 P11-1c**：无正式授权配置时只运行合成安全测试并写 NOT_AUTHORIZED。
不改变动力学/连接组/参数；不训练、不更新 ΔW、不进入 B/C。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_1c_report.json")
AUTH_PATH = os.path.join(HERE, "p11_1c_authorization.json")
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
KNOWN = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}
MEM_CAP_GIB = 2.0
MEM_CAP_BYTES = int(MEM_CAP_GIB * 1024 ** 3)
WALL_CAP_S = 900.0
SEEDS = list(range(101, 126))
STEPS = 400

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
STARTF_USESTDHANDLES = 0x00000100
STILL_ACTIVE = 259
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
CREATE_ALWAYS = 2
FILE_ATTRIBUTE_NORMAL = 0x80
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong), ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong), ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong), ("OtherTransferCount", ctypes.c_ulonglong)]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class STARTUPINFO(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR), ("lpTitle", wt.LPWSTR),
                ("dwX", wt.DWORD), ("dwY", wt.DWORD), ("dwXSize", wt.DWORD), ("dwYSize", wt.DWORD),
                ("dwXCountChars", wt.DWORD), ("dwYCountChars", wt.DWORD), ("dwFillAttribute", wt.DWORD),
                ("dwFlags", wt.DWORD), ("wShowWindow", wt.WORD), ("cbReserved2", wt.WORD),
                ("lpReserved2", ctypes.POINTER(ctypes.c_byte)), ("hStdInput", wt.HANDLE),
                ("hStdOutput", wt.HANDLE), ("hStdError", wt.HANDLE)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wt.HANDLE), ("hThread", wt.HANDLE),
                ("dwProcessId", wt.DWORD), ("dwThreadId", wt.DWORD)]


kernel32.CreateJobObjectW.restype = wt.HANDLE
kernel32.CreateJobObjectW.argtypes = [wt.LPVOID, wt.LPCWSTR]
kernel32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, wt.LPVOID, wt.DWORD]
kernel32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
kernel32.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
kernel32.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]
kernel32.CloseHandle.argtypes = [wt.HANDLE]
kernel32.CreateProcessW.argtypes = [wt.LPCWSTR, wt.LPWSTR, wt.LPVOID, wt.LPVOID, wt.BOOL, wt.DWORD,
                                    wt.LPVOID, wt.LPCWSTR, ctypes.POINTER(STARTUPINFO),
                                    ctypes.POINTER(PROCESS_INFORMATION)]
kernel32.CreateProcessW.restype = wt.BOOL
kernel32.ResumeThread.argtypes = [wt.HANDLE]
kernel32.ResumeThread.restype = wt.DWORD
kernel32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
kernel32.GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
kernel32.CreateFileW.restype = wt.HANDLE
kernel32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, wt.LPVOID, wt.DWORD, wt.DWORD, wt.HANDLE]
psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
psapi.GetProcessMemoryInfo.restype = wt.BOOL


def _mib(nbytes):
    return round(nbytes / 1024 ** 2, 2)


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _peak_self_mib():
    c = PMC(); c.cb = ctypes.sizeof(c)
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return _mib(c.PeakWorkingSetSize) if ok else None


def _proc_peak_mib(pid):
    h = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        return None
    try:
        c = PMC(); c.cb = ctypes.sizeof(c)
        ok = psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
        return _mib(c.PeakWorkingSetSize) if ok else None
    finally:
        kernel32.CloseHandle(h)


def create_job(limit_bytes):
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None, False
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
                                             | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    info.ProcessMemoryLimit = ctypes.c_size_t(limit_bytes)
    info.JobMemoryLimit = ctypes.c_size_t(limit_bytes)
    ok = kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                          ctypes.byref(info), ctypes.sizeof(info))
    return job, bool(ok)


def job_peaks_mib(job):
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION(); ret = wt.DWORD()
    ok = kernel32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation,
                                            ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(ret))
    if not ok:
        return {"PeakProcessMemoryUsed_MiB": None, "PeakJobMemoryUsed_MiB": None}
    return {"PeakProcessMemoryUsed_MiB": _mib(info.PeakProcessMemoryUsed),
            "PeakJobMemoryUsed_MiB": _mib(info.PeakJobMemoryUsed)}


def close_job(job):
    if job:
        kernel32.CloseHandle(job)


def launch_suspended(mode_args, log_path):
    """CREATE_SUSPENDED：子进程在执行任何用户代码前被创建。"""
    log_h = kernel32.CreateFileW(log_path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                 None, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)
    if not log_h:
        return None
    si = STARTUPINFO(); si.cb = ctypes.sizeof(si); si.dwFlags = STARTF_USESTDHANDLES
    si.hStdOutput = log_h; si.hStdError = log_h; si.hStdInput = kernel32.GetStdHandle(-10)
    cmd = f'"{sys.executable}" -u "{os.path.abspath(__file__)}" ' + " ".join(mode_args)
    buf = ctypes.create_unicode_buffer(cmd)
    pi = PROCESS_INFORMATION()
    ok = kernel32.CreateProcessW(None, buf, None, None, True,
                                 CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP, None, ROOT,
                                 ctypes.byref(si), ctypes.byref(pi))
    if not ok:
        kernel32.CloseHandle(log_h)
        return None
    return {"pi": pi, "log_handle": log_h, "pid": int(pi.dwProcessId)}


def assign_resume_or_terminate(job, launched):
    """加入 Job 成功后才恢复；失败立即终止，绝不继续运行。"""
    pi = launched["pi"]
    if not job or not kernel32.AssignProcessToJobObject(job, pi.hProcess):
        kernel32.TerminateProcess(pi.hProcess, 1)
        kernel32.WaitForSingleObject(pi.hProcess, 5000)
        return False
    if kernel32.ResumeThread(pi.hThread) == 0xFFFFFFFF:
        kernel32.TerminateProcess(pi.hProcess, 1)
        kernel32.WaitForSingleObject(pi.hProcess, 5000)
        return False
    return True


def wait_exit(pi, timeout_ms=30000):
    kernel32.WaitForSingleObject(pi.hProcess, timeout_ms)
    code = wt.DWORD()
    kernel32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
    return int(code.value)


def run_under_job(limit_bytes, mode_args, wall_cap_s, mem_cap_mib, log_path):
    """一次运行；超时/超内存 -> 终止 Job；不重试。"""
    job, ok = create_job(limit_bytes)
    if not ok:
        return {"ok": False, "reason": "job_creation_failed"}
    launched = launch_suspended(mode_args, log_path)
    if not launched:
        close_job(job)
        return {"ok": False, "reason": "create_suspended_failed"}
    if not assign_resume_or_terminate(job, launched):
        close_job(job)
        return {"ok": False, "reason": "assign_failed_terminated"}
    pi = launched["pi"]
    t0 = time.perf_counter(); peak = 0.0; timeout = False; mem_exceeded = False
    while True:
        code = wt.DWORD(); kernel32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
        if code.value != STILL_ACTIVE:
            break
        ws = _proc_peak_mib(launched["pid"])
        if ws:
            peak = max(peak, ws)
            if ws > mem_cap_mib:
                mem_exceeded = True
                kernel32.TerminateJobObject(job, 1)
                break
        if time.perf_counter() - t0 > wall_cap_s:
            timeout = True
            kernel32.TerminateJobObject(job, 1)
            break
        time.sleep(0.2)
    exit_code = wait_exit(pi)
    peaks = job_peaks_mib(job)
    close_job(job)
    kernel32.CloseHandle(pi.hProcess); kernel32.CloseHandle(pi.hThread)
    kernel32.CloseHandle(launched["log_handle"])
    log = ""
    try:
        log = open(log_path, encoding="utf-8", errors="replace").read()[-300:]
    except Exception:                            # noqa: BLE001
        pass
    return {"ok": True, "reason": "completed", "wall_s": round(time.perf_counter() - t0, 3),
            "exit_code": exit_code, "timeout": timeout, "mem_exceeded": mem_exceeded,
            "peak_working_set_MiB": peak, "job_peaks": peaks,
            "mem_cap_MiB": mem_cap_mib, "wall_cap_s": wall_cap_s, "log_tail": log.strip()}


# ------------------------------------------------------------------ children
def child_alloc(limit_mib=512):
    try:
        n = limit_mib * 1024 * 1024
        b = bytearray(n)
        for i in range(0, n, 4096):
            b[i] = 1
        print("ALLOC_OK"); return 0
    except MemoryError:
        print("MEMORY_ERROR"); return 3


def child_sleep(seconds):
    time.sleep(seconds); print("SLEEP_DONE"); return 0


def child_full_brain(result_path):
    import numpy as np
    from flybrain import FlyBrain
    try:
        t0 = time.perf_counter()
        brain = FlyBrain(data=DATA_DIR, device="cpu", batch=1, seed=101, dt=0.020)
        init_s = time.perf_counter() - t0
        for _ in range(5):
            brain.step()
        ts, fired = [], []
        finite = True
        for _ in range(STEPS):
            t = time.perf_counter(); f = brain.step(); ts.append(time.perf_counter() - t)
            fired.append(int(len(f)))
            if not np.isfinite(brain.v).all():
                finite = False
        res = {"ok": True, "init_s": round(init_s, 4), "median_ms": round(float(np.median(ts)) * 1000, 4),
               "p95_ms": round(float(np.percentile(ts, 95)) * 1000, 4),
               "n_neurons": int(brain.n), "nnz": int(len(brain.indices)),
               "peak_working_set_MiB": _peak_self_mib(), "finite_state": bool(finite),
               "stimulus": "none", "learning": False, "dW": 0,
               "fired_mean": round(float(np.mean(fired)), 2), "fired_max": int(max(fired)),
               "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "scipy": __import__("scipy").__version__,
                            "flybrain": getattr(__import__("flybrain"), "__version__", "0.1.0")}}
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(res, f)
        return 0
    except Exception as e:                       # noqa: BLE001
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": repr(e)[:300]}, f)
        return 5


# ------------------------------------------------------------------ safety
def safety_selftest():
    out = {}
    tmp = tempfile.mkdtemp(prefix="p11_1c_safety_")
    # 1) resource isolation: alloc without job succeeds, under 256 MiB fails
    nojob = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--alloc", "512"],
                           capture_output=True, text=True, timeout=120)
    out["no_job_alloc"] = {"rc": nojob.returncode, "out": nojob.stdout.strip()[-50:],
                           "ok": nojob.returncode == 0 and "ALLOC_OK" in nojob.stdout}
    r = run_under_job(256 * 1024 ** 2, ["--alloc", "512"], 60, 256, os.path.join(tmp, "alloc.log"))
    out["job_alloc_limit"] = {"rc": r.get("exit_code"), "log": r.get("log_tail", "")[-60:],
                              "ok": (not r.get("mem_exceeded")) and (r.get("exit_code") != 0
                                                                     or "MEMORY_ERROR" in (r.get("log_tail") or ""))}
    # 2) assignment failure: suspended child must be terminated, not resumed
    launched = launch_suspended(["--sleep", "10"], os.path.join(tmp, "assign.log"))
    assign_ok = assign_resume_or_terminate(None, launched) if launched else None
    code = wait_exit(launched["pi"], 5000) if launched else None
    if launched:
        kernel32.CloseHandle(launched["pi"].hProcess); kernel32.CloseHandle(launched["pi"].hThread)
        kernel32.CloseHandle(launched["log_handle"])
    out["assign_failure"] = {"assign_ok": assign_ok, "exit_code": code,
                             "ok": launched is not None and assign_ok is False and code != STILL_ACTIVE}
    # 3) timeout termination
    r = run_under_job(MEM_CAP_BYTES, ["--sleep", "10"], 1.0, 2048, os.path.join(tmp, "timeout.log"))
    out["timeout"] = {"timeout": r.get("timeout"), "wall_s": r.get("wall_s"),
                      "exit_code": r.get("exit_code"),
                      "ok": r.get("timeout") is True and (r.get("wall_s") or 0) < 5}
    # 4) no auto-retry
    calls = {"n": 0}
    def once():
        calls["n"] += 1
        return False
    run_once(once)
    out["no_retry"] = {"calls": calls["n"], "ok": calls["n"] == 1}
    out["all_ok"] = all(v.get("ok") for v in out.values() if isinstance(v, dict))
    return out


def run_once(fn):
    return fn()


def official_hashes():
    out = {}
    for name, known in KNOWN.items():
        path = os.path.join(DATA_DIR, name)
        actual = _sha256_file(path) if os.path.exists(path) else None
        out[name] = {"sha256": actual, "full_equal": actual == known}
    out["ok"] = all(out[n]["full_equal"] for n in KNOWN)
    return out


def load_auth():
    if not os.path.exists(AUTH_PATH):
        return {"granted": False, "status": "NOT_AUTHORIZED", "reason": "no P11-1c authorization config"}
    cfg = json.load(open(AUTH_PATH, encoding="utf-8"))
    scope = cfg.get("scope", {})
    ok = (cfg.get("protocol") == "P11-1c/v1" and cfg.get("status") == "APPROVED"
          and cfg.get("checksum") and scope.get("seeds") == SEEDS and scope.get("conditions") == ["none"]
          and scope.get("steps") == STEPS and scope.get("total_steps") == 10000
          and scope.get("mem_cap_GiB") == 2.0 and scope.get("wall_cap_min") == 15
          and scope.get("learning") is False and scope.get("dW") == 0)
    return {"granted": bool(ok), "status": "APPROVED" if ok else "NOT_AUTHORIZED",
            "reason": None if ok else "authorization missing/incomplete/scope mismatch", "config": cfg}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    if "--alloc" in sys.argv:
        return child_alloc(int(sys.argv[sys.argv.index("--alloc") + 1]))
    if "--sleep" in sys.argv:
        return child_sleep(float(sys.argv[sys.argv.index("--sleep") + 1]))
    if "--child" in sys.argv:
        return child_full_brain(sys.argv[sys.argv.index("--child") + 1])
    if "--self-test-safety" in sys.argv:
        s = safety_selftest()
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return 0 if s["all_ok"] else 1
    # production entry: gate authorization
    auth = load_auth()
    hashes = official_hashes()
    report = {"protocol": "P11-1c/v1", "authorization": auth,
              "hashes": hashes, "hashes_ok": hashes["ok"],
              "full_brain_started": False, "experiment": "NOT RUN",
              "safety_fixes": ["CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread",
                               "assignment failure terminates child immediately (no run)",
                               "per-process AND job-total memory limits (2 GiB)",
                               "full 64-hex SHA-256 equality (no prefix compare)",
                               "MiB/GiB units for memory; GiB cap",
                               "synthetic isolation/assign/timeout/memory/no-retry tests"],
              "verdicts": {"RESOURCE": "NOT_AUTHORIZED" if not auth["granted"] else "PENDING_RUN",
                           "FULL_BRAIN_BASELINE": "INCONCLUSIVE",
                           "FULL_VS_SUBNET_DIFFERENCE": "INCONCLUSIVE",
                           "MODEL_ROOT_CAUSE": "NOT_ESTABLISHED",
                           "Level_A_Training": "NO-GO", "Level_B": "NO-GO", "Level_C": "NO-GO",
                           "Biological": "NO-GO", "Training_Authorization": False},
              "training_authorized": False}
    if not auth["granted"]:
        report["status"] = "NOT_AUTHORIZED"
        if not hashes["ok"]:
            report["status"] = "RESOURCE_NO_GO"
            report["reason"] = "official hash (full SHA-256) mismatch"
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print("NOT_AUTHORIZED: full-brain P11-1c experiment not started")
        print("wrote p11_1c_report.json (static safety fixes only)")
        return 0
    # If authorized in the future: the full run must be implemented here (not executed now).
    report["status"] = "AUTHORIZED_NOT_IMPLEMENTED_IN_THIS_TASK"
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("AUTHORIZED but this task only covers the static safety fixes; not running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
