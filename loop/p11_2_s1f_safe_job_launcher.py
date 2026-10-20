"""P11-2-S1F：独立的安全 Windows Job 启动器（合成测试用；不运行完整脑）。

修复点：日志句柄显式可继承 + 句柄继承列表；CREATE_SUSPENDED -> Job -> Resume；
区分 PROCESS_FINISHED 与 EXPERIMENT_SUCCEEDED；失败 fail-closed、不重试。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import os
import sys
import time

from loop.p11_1c_full_brain_bench import (kernel32, psapi, STARTUPINFO, PROCESS_INFORMATION, PMC,
                                          ROOT, CREATE_SUSPENDED, CREATE_NEW_PROCESS_GROUP,
                                          STARTF_USESTDHANDLES, STILL_ACTIVE, GENERIC_WRITE,
                                          FILE_SHARE_READ, FILE_SHARE_WRITE, CREATE_ALWAYS,
                                          FILE_ATTRIBUTE_NORMAL, create_job, close_job,
                                          assign_resume_or_terminate, wait_exit, job_peaks_mib, _mib)

EXTENDED_STARTUPINFO_PRESENT = 0x00080000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
HANDLE_FLAG_INHERIT = 0x00000001


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("nLength", wt.DWORD), ("lpSecurityDescriptor", wt.LPVOID), ("bInheritHandle", wt.BOOL)]


class STARTUPINFOEX(ctypes.Structure):
    _fields_ = [("StartupInfo", STARTUPINFO), ("lpAttributeList", wt.LPVOID)]


kernel32.GetStdHandle.argtypes = [wt.DWORD]
kernel32.GetStdHandle.restype = wt.HANDLE
kernel32.GetHandleInformation.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
kernel32.GetHandleInformation.restype = wt.BOOL
kernel32.InitializeProcThreadAttributeList.argtypes = [wt.LPVOID, wt.DWORD, wt.DWORD,
                                                       ctypes.POINTER(ctypes.c_size_t)]
kernel32.InitializeProcThreadAttributeList.restype = wt.BOOL
kernel32.UpdateProcThreadAttribute.argtypes = [wt.LPVOID, wt.DWORD, ctypes.c_size_t, wt.LPVOID,
                                               ctypes.c_size_t, wt.LPVOID, wt.LPVOID]
kernel32.UpdateProcThreadAttribute.restype = wt.BOOL
kernel32.DeleteProcThreadAttributeList.argtypes = [wt.LPVOID]


def _inherit_ok(h):
    flags = wt.DWORD(0)
    if not h or not kernel32.GetHandleInformation(h, ctypes.byref(flags)):
        return False
    return bool(flags.value & HANDLE_FLAG_INHERIT)


def _command(script=None, module=None, args=()):
    py = sys.executable
    if module:
        cmd = f'"{py}" -u -m {module}'
    else:
        cmd = f'"{py}" -u "{os.path.abspath(script)}"'
    if args:
        cmd += " " + " ".join(str(a) for a in args)
    return cmd


def _env_block(env):
    items = [f"{k}={v}" for k, v in sorted(env.items())] + [""]
    return ctypes.create_unicode_buffer("\0".join(items) + "\0")


def _open_log(path, inherit_log=True):
    if inherit_log:
        sa = SECURITY_ATTRIBUTES(); sa.nLength = ctypes.sizeof(sa); sa.bInheritHandle = True
        h = kernel32.CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                 ctypes.byref(sa), CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)
    else:
        h = kernel32.CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                 None, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)
    if not h:
        return None
    if inherit_log and not _inherit_ok(h):
        kernel32.CloseHandle(h)
        return None
    return h


def launch_suspended_safe(script=None, module=None, args=(), log_path=None, cwd=None,
                          env=None, inherit_log=True):
    """CreateProcessW with inheritable log handles + restricted handle list."""
    if not log_path:
        return None, "CHILD_LOG_UNAVAILABLE"
    log_h = _open_log(log_path, inherit_log)
    if not log_h:
        return None, "CHILD_LOG_UNAVAILABLE"
    if not _inherit_ok(log_h):
        kernel32.CloseHandle(log_h)
        return None, "CHILD_LOG_UNAVAILABLE"
    stdin_h = kernel32.GetStdHandle(-10)
    stdin_ok = bool(stdin_h) and _inherit_ok(stdin_h)
    handles = [log_h] + ([stdin_h] if stdin_ok else [])
    h_array = (wt.HANDLE * len(handles))(*handles)
    size = ctypes.c_size_t(0)
    kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
    if not size.value:
        kernel32.CloseHandle(log_h)
        return None, "CHILD_LOG_UNAVAILABLE"
    attr_buf = ctypes.create_string_buffer(size.value)
    attr_ptr = ctypes.cast(attr_buf, wt.LPVOID)
    if not kernel32.InitializeProcThreadAttributeList(attr_ptr, 1, 0, ctypes.byref(size)):
        kernel32.CloseHandle(log_h)
        return None, "CHILD_LOG_UNAVAILABLE"
    if not kernel32.UpdateProcThreadAttribute(attr_ptr, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                              ctypes.cast(h_array, wt.LPVOID),
                                              ctypes.sizeof(h_array), None, None):
        kernel32.DeleteProcThreadAttributeList(attr_ptr); kernel32.CloseHandle(log_h)
        return None, "CHILD_LOG_UNAVAILABLE"
    si = STARTUPINFOEX()
    si.StartupInfo.cb = ctypes.sizeof(si)
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES
    si.StartupInfo.hStdInput = stdin_h if stdin_ok else None
    si.StartupInfo.hStdOutput = log_h
    si.StartupInfo.hStdError = log_h
    si.lpAttributeList = attr_ptr
    pi = PROCESS_INFORMATION()
    cmd = _command(script, module, args)
    buf = ctypes.create_unicode_buffer(cmd)
    env_buf = None
    flags = EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP
    if env is not None:
        env_buf = _env_block(env); flags |= CREATE_UNICODE_ENVIRONMENT
    ok = kernel32.CreateProcessW(None, buf, None, None, True, flags,
                                 ctypes.cast(env_buf, wt.LPVOID) if env_buf is not None else None,
                                 os.path.abspath(cwd or ROOT),
                                 ctypes.cast(ctypes.byref(si), ctypes.POINTER(STARTUPINFO)),
                                 ctypes.byref(pi))
    kernel32.DeleteProcThreadAttributeList(attr_ptr)
    kernel32.CloseHandle(log_h)
    if not ok:
        return None, "CHILD_ASSIGN_FAILED"
    return {"pi": pi, "pid": int(pi.dwProcessId), "cmd": cmd}, None


def run_under_job_safe(limit_bytes, script=None, module=None, args=(), wall_cap_s=15.0,
                       mem_cap_mib=2048, log_path=None, cwd=None, env=None, inherit_log=True,
                       simulate_assign_fail=False, required_artifact=None):
    """One run; explicit child status; no retry."""
    job, job_ok = create_job(limit_bytes)
    if not job_ok:
        return {"child_status": "CHILD_ASSIGN_FAILED", "process_finished": False,
                "experiment_succeeded": False, "log_available": False, "log_bytes": 0,
                "log_tail": "", "exit_code": None, "timeout": False, "mem_exceeded": False,
                "job_peaks": None, "artifact_present": False, "required_artifact": required_artifact}
    launched, err = launch_suspended_safe(script, module, args, log_path, cwd, env, inherit_log)
    if not launched:
        close_job(job)
        return {"child_status": err, "process_finished": False, "experiment_succeeded": False,
                "log_available": False, "log_bytes": 0, "log_tail": "", "exit_code": None,
                "timeout": False, "mem_exceeded": False, "job_peaks": None,
                "artifact_present": False, "required_artifact": required_artifact}
    if simulate_assign_fail or not assign_resume_or_terminate(job, launched):
        try:
            kernel32.TerminateProcess(launched["pi"].hProcess, 1)
            kernel32.WaitForSingleObject(launched["pi"].hProcess, 5000)
        finally:
            close_job(job)
        log = _read_log(log_path)
        return {"child_status": "CHILD_ASSIGN_FAILED", "process_finished": False,
                "experiment_succeeded": False, "log_available": log["available"],
                "log_bytes": log["bytes"], "log_tail": log["tail"], "exit_code": None,
                "timeout": False, "mem_exceeded": False, "job_peaks": None,
                "child_pid": launched["pid"], "artifact_present": False,
                "required_artifact": required_artifact}
    pi = launched["pi"]; t0 = time.perf_counter(); peak = 0.0; timeout = False; mem_exceeded = False
    while True:
        code = wt.DWORD(); kernel32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
        if code.value != STILL_ACTIVE:
            break
        p = PMC(); p.cb = ctypes.sizeof(p)
        if psapi.GetProcessMemoryInfo(pi.hProcess, ctypes.byref(p), ctypes.sizeof(p)):
            ws = p.WorkingSetSize / 1024 ** 2
            peak = max(peak, ws)
            if ws > mem_cap_mib:
                mem_exceeded = True; kernel32.TerminateJobObject(job, 1); break
        if time.perf_counter() - t0 > wall_cap_s:
            timeout = True; kernel32.TerminateJobObject(job, 1); break
        time.sleep(0.2)
    exit_code = wait_exit(pi)
    peaks = job_peaks_mib(job)
    close_job(job)
    log = _read_log(log_path)
    artifact = bool(required_artifact and os.path.exists(required_artifact))
    if not mem_exceeded and exit_code != 0 and ("MemoryError" in log["tail"] or "MEMORY_ERROR" in log["tail"]):
        mem_exceeded = True
    if mem_exceeded:
        status = "CHILD_MEMORY_LIMIT"
    elif timeout:
        status = "CHILD_TIMEOUT"
    elif exit_code != 0:
        status = "CHILD_EXIT_NONZERO"
    elif required_artifact and not artifact:
        status = "CHILD_OUTPUT_MISSING"
    else:
        status = "CHILD_EXIT_OK"
    return {"child_status": status, "process_finished": True,
            "experiment_succeeded": status == "CHILD_EXIT_OK", "log_available": log["available"],
            "log_bytes": log["bytes"], "log_tail": log["tail"], "exit_code": exit_code,
            "timeout": timeout, "mem_exceeded": mem_exceeded, "job_peaks": peaks,
            "peak_MiB": round(peak, 2), "child_pid": launched["pid"],
            "required_artifact": required_artifact, "artifact_present": artifact}


def _read_log(path):
    try:
        with open(path, "rb") as f:
            b = f.read()
        return {"available": True, "bytes": len(b), "tail": b.decode("utf-8", "replace")[-4000:]}
    except OSError:
        return {"available": False, "bytes": 0, "tail": ""}
