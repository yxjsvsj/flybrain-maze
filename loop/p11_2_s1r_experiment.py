"""P11-2-S1R：完整 MaleCNS 合成 B 网络 none/stim 一次性感觉传递实验。

只有 `loop.p11_2_s1r_launcher.py` 在独立的 S1R 授权/冻结校验通过后，才允许以
`python -u -m loop.p11_2_s1r_experiment --child <out_dir>` 启动本模块。
当前授权状态必须为 NOT_AUTHORIZED 时 fail-closed。**不改 W、不训练、不更新 ΔW、不跑 Dijkstra/迷宫/B/C/车。**
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import gc
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp

from loop.p11_2_c1_experiment import (build_W_B_csc, verify_W_B_fields, install_weights,
                                      field_switch_probe, rng_hash, _sha256, load_archive,
                                      DATA_DIR, OFFICIAL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
AUTH = os.path.join(HERE, "p11_2_s1r_authorization.json")
FREEZE = os.path.join(HERE, "p11_2_s1r_script_freeze.json")
REPORT = os.path.join(HERE, "p11_2_s1r_report.json")
RUN_META = os.path.join(HERE, "p11_2_s1r_run_meta.json")
RAW_MANIFEST = os.path.join(HERE, "p11_2_s1r_raw_manifest.json")
SEEDS = list(range(101, 126))
STEPS = 400
STIM = (100, 300)
DV = 0.2
DT = 0.020
TAU = 0.100
AMOUNT = 0.036254
MIN_KC = 0.05
MIN_MBON = 0.05
WALL_CAP_S = 900.0
MEM_CAP_MIB = 2048
MEM_CAP_BYTES = MEM_CAP_MIB * 1024 ** 2
BOOT_N = 10000
BOOT_SEED = 20261019

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_ALL_ACCESS = 0x1F001F
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetCurrentProcess.restype = wt.HANDLE
_kernel32.GetCurrentProcessId.restype = wt.DWORD
_kernel32.IsProcessInJob.argtypes = [wt.HANDLE, wt.HANDLE, ctypes.POINTER(wt.BOOL)]
_kernel32.IsProcessInJob.restype = wt.BOOL
_kernel32.QueryInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, wt.LPVOID,
                                                wt.DWORD, ctypes.POINTER(wt.DWORD)]
_kernel32.QueryInformationJobObject.restype = wt.BOOL
_kernel32.OpenJobObjectW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
_kernel32.OpenJobObjectW.restype = wt.HANDLE
_kernel32.CloseHandle.argtypes = [wt.HANDLE]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong), ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong), ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong), ("OtherTransferCount", ctypes.c_ulonglong)]


class _JOB_BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]


class _JOB_EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _JOB_BASIC_LIMIT), ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _JOB_PID_LIST(ctypes.Structure):
    _fields_ = [("NumberOfAssignedProcesses", wt.DWORD), ("NumberOfProcessIdsInList", wt.DWORD),
                ("ProcessIdList", ctypes.c_size_t * 2048)]


def _query_extended(job):
    info = _JOB_EXTENDED_LIMIT()
    ret = wt.DWORD(0)
    if not _kernel32.QueryInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                               ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(ret)):
        return None
    return info


def _pid_in_job(job, pid):
    rows = _JOB_PID_LIST()
    ret = wt.DWORD(0)
    if not _kernel32.QueryInformationJobObject(job, _JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                                               ctypes.byref(rows), ctypes.sizeof(rows), ctypes.byref(ret)):
        return False
    n = min(int(rows.NumberOfProcessIdsInList), 2048)
    return int(pid) in [int(rows.ProcessIdList[i]) for i in range(n)]


def assert_in_controlled_job(expected_bytes=MEM_CAP_BYTES):
    """强制命名 Job；环境变量只是名称提示，必须打开真实句柄并核验限制/PID。"""
    name = (os.environ.get("P11_2_S1R_JOB_NAME") or "").strip()
    if not name:
        return {"ok": False, "reason": "JOB_NAME_MISSING"}
    in_job = wt.BOOL(0)
    if not _kernel32.IsProcessInJob(_kernel32.GetCurrentProcess(), None, ctypes.byref(in_job)):
        return {"ok": False, "reason": "IS_PROCESS_IN_JOB_FAILED", "job_name": name}
    if not in_job.value:
        return {"ok": False, "reason": "NOT_IN_JOB", "job_name": name}
    pid = int(_kernel32.GetCurrentProcessId())
    job = _kernel32.OpenJobObjectW(_JOB_OBJECT_ALL_ACCESS, False, name)
    if not job:
        return {"ok": False, "reason": "NAMED_JOB_OPEN_FAILED", "job_name": name}
    try:
        info = _query_extended(job)
        if info is None:
            return {"ok": False, "reason": "QUERY_JOB_FAILED", "job_name": name}
        flags = int(info.BasicLimitInformation.LimitFlags)
        if not (flags & _JOB_OBJECT_LIMIT_PROCESS_MEMORY and flags & _JOB_OBJECT_LIMIT_JOB_MEMORY):
            return {"ok": False, "reason": "JOB_MEMORY_LIMIT_FLAGS_MISSING",
                    "limit_flags": flags, "job_name": name}
        if not (flags & _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE):
            return {"ok": False, "reason": "KILL_ON_JOB_CLOSE_MISSING",
                    "limit_flags": flags, "job_name": name}
        if int(info.ProcessMemoryLimit) != expected_bytes or int(info.JobMemoryLimit) != expected_bytes:
            return {"ok": False, "reason": "JOB_MEMORY_LIMIT_MISMATCH",
                    "process_mem_limit": int(info.ProcessMemoryLimit),
                    "job_mem_limit": int(info.JobMemoryLimit), "limit_flags": flags,
                    "job_name": name}
        if not _pid_in_job(job, pid):
            return {"ok": False, "reason": "PID_NOT_IN_NAMED_JOB", "job_name": name}
        return {"ok": True, "reason": "CONTROLLED_JOB_OK", "job_name": name,
                "limit_flags": flags, "process_mem_limit": int(info.ProcessMemoryLimit),
                "job_mem_limit": int(info.JobMemoryLimit),
                "peak_process_MiB": round(info.PeakProcessMemoryUsed / 1024 ** 2, 2),
                "peak_job_MiB": round(info.PeakJobMemoryUsed / 1024 ** 2, 2)}
    finally:
        _kernel32.CloseHandle(job)


def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def _scope():
    return {"n_neurons": 166700, "network": "B_synthetic_kc_kc_zero",
            "remove_edges": 642933, "conditions": ["none", "stim"], "seeds": SEEDS,
            "steps": STEPS, "total_steps": 2 * len(SEEDS) * STEPS, "window": list(STIM),
            "targets": "exact LPLC2 + LC4 (311)", "dv": DV, "amount": AMOUNT,
            "device": "cpu", "wall_cap_s": WALL_CAP_S, "mem_cap_MiB": MEM_CAP_MIB,
            "one_run": True, "auto_retry": False, "training": False, "dW": 0}


def auth_payload(text):
    return {"protocol": "P11-2-S1R/v1", "status": "APPROVED", "approved_by": "user",
            "approval_text": text, "scope": _scope()}


def auth_file(text):
    p = auth_payload(text); p["checksum"] = hashlib.sha256(_canon(p)).hexdigest(); return p


def load_auth():
    a = json.load(open(AUTH, encoding="utf-8")); c = a.pop("checksum", None)
    if c != hashlib.sha256(_canon(a)).hexdigest():
        raise RuntimeError("AUTHORIZATION_CHECKSUM_MISMATCH")
    a["checksum"] = c
    if a.get("protocol") != "P11-2-S1R/v1" or a.get("status") != "APPROVED" or a.get("approved_by") != "user":
        raise RuntimeError("AUTHORIZATION_NOT_APPROVED")
    if a.get("scope") != _scope():
        raise RuntimeError("AUTHORIZATION_SCOPE_MISMATCH")
    return a


def load_freeze():
    f = json.load(open(FREEZE, encoding="utf-8"))
    if os.path.abspath(f.get("script", "")) != os.path.abspath(__file__):
        raise RuntimeError("FREEZE_SCRIPT_PATH_MISMATCH")
    if f.get("sha256") != _sha256(__file__):
        raise RuntimeError("SCRIPT_FREEZE_MISMATCH")
    return f


def official_hashes():
    files = {k: _sha256(os.path.join(DATA_DIR, k)) for k in OFFICIAL}
    return {"ok": all(files[k] == v for k, v in OFFICIAL.items()), "files": files}


def _entry(lp, seed):
    return float(lp[STIM[0]:STIM[1]].mean())


def run_condition(brain, seed, kc_idx, mb_idx, lp_pos, kc_map, mb_map, fields, condition,
                  out_dir, script_sha, auth_sha, w_hash, kc_m, mb_m):
    install_weights(brain, fields)
    brain.reset(seed)
    before = rng_hash(brain)
    kc = np.zeros((STEPS, len(kc_idx)), np.uint8)
    mb = np.zeros((STEPS, len(mb_idx)), np.uint8)
    lp = np.zeros((STEPS, len(lp_pos)), np.uint8)
    finite = True
    t0 = time.perf_counter()
    for t in range(STEPS):
        inj = [(lp_pos, AMOUNT)] if (condition == "stim" and STIM[0] <= t < STIM[1]) else []
        fired = np.asarray(brain.step(inject=inj))
        if not np.isfinite(brain.v).all():
            finite = False
        lm = np.isin(fired, lp_pos)
        if lm.any():
            lp[t, np.searchsorted(lp_pos, fired[lm])] = 1
        fk = fired[kc_map[fired] >= 0]
        fm = fired[mb_map[fired] >= 0]
        kc[t, kc_map[fk]] = 1
        mb[t, mb_map[fm]] = 1
    wall = time.perf_counter() - t0
    after = rng_hash(brain)
    np.savez_compressed(os.path.join(out_dir, f"{seed}_{condition}.npz"),
                        kc=kc, mb=mb, lp=lp, kc_matched=kc[:, kc_m], mb_matched=mb[:, mb_m],
                        seed=seed, condition=condition, script_sha=script_sha, auth_sha=auth_sha,
                        w_hash=w_hash, rng_before=before, rng_after=after,
                        amount=AMOUNT, stim_window=np.array(STIM))
    out = {"seed": seed, "condition": condition, "wall_s": round(wall, 4), "finite": finite,
           "rng_before": before, "rng_after": after, "lp_n": int(lp.sum()), "kc_n": int(kc.sum()),
           "mb_n": int(mb.sum())}
    del kc, mb, lp
    gc.collect()
    return out


def run_child(data_dir, ids_path, matched_path, out_dir, seeds, steps, validate=True,
              expected=(4064, 97, 320, 95), lp_expected=311):
    from flybrain import FlyBrain
    script_sha = _sha256(__file__)
    auth_sha = "SYNTHETIC"
    if validate:
        auth = load_auth(); freeze = load_freeze(); h = official_hashes()
        if not h["ok"]:
            raise RuntimeError("OFFICIAL_HASH_MISMATCH")
        if freeze.get("sha256") != script_sha:
            raise RuntimeError("SCRIPT_FREEZE_MISMATCH")
        auth_sha = auth["checksum"]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    brain = FlyBrain(data=data_dir, device="cpu", batch=1, seed=101, dt=0.020)
    init_s = time.perf_counter() - t0
    meta = np.load(os.path.join(data_dir, "brain.npz"), allow_pickle=True)
    ids = meta["ids"]; ct = brain.cell_type.astype(str)
    kc_idx = np.flatnonzero(np.char.startswith(ct, "KC"))
    mb_idx = np.flatnonzero(np.char.startswith(ct, "MBON"))
    lp_pos = np.asarray(brain.cells(["LPLC2", "LC4"]))
    z = np.load(ids_path, allow_pickle=True)
    kc_ref = z["kc_idx"]; mb_ref = z["mb_idx"]
    kc_m_global = z["kc_matched"]; mb_m_global = z["mb_matched"]
    if not (np.array_equal(kc_idx, kc_ref) and np.array_equal(mb_idx, mb_ref)):
        raise RuntimeError("KC/MBON INDEX_MISMATCH")
    kc_m = np.searchsorted(kc_idx, kc_m_global); mb_m = np.searchsorted(mb_idx, mb_m_global)
    if not (np.array_equal(kc_idx[kc_m], kc_m_global) and np.array_equal(mb_idx[mb_m], mb_m_global)):
        raise RuntimeError("MATCHED_MAPPING_MISMATCH")
    n_kc, n_mb, n_km, n_mm = expected
    if not (len(kc_idx) == n_kc and len(mb_idx) == n_mb and len(kc_m) == n_km and len(mb_m) == n_mm):
        raise RuntimeError(f"EXPECTED_COUNTS_MISMATCH {len(kc_idx)}/{len(mb_idx)}/{len(kc_m)}/{len(mb_m)}")
    if len(lp_pos) != lp_expected:
        raise RuntimeError(f"LP_COUNT_MISMATCH {len(lp_pos)} != {lp_expected}")
    np.savez(os.path.join(out_dir, "ids.npz"), ids=ids, kc_idx=kc_idx, mb_idx=mb_idx,
             kc_matched=kc_m_global, mb_matched=mb_m_global, lp_idx=lp_pos)
    if os.path.exists(matched_path):
        with open(matched_path, "rb") as f:
            b = f.read()
        with open(os.path.join(out_dir, "matched_ids.json"), "wb") as f:
            f.write(b)
    A = (brain.indptr, brain.indices, brain.weights)
    is_kc = np.zeros(brain.n, bool); is_kc[kc_idx] = True
    B, removed, removed_sum = build_W_B_csc(*A, is_kc)
    vf = verify_W_B_fields(A, B, is_kc, kc_idx)
    if not vf["ok"]:
        raise RuntimeError(f"W_B_VERIFY_FAIL {vf}")
    if n_kc == 4064 and (removed != 642933 or len(B[1]) != 24940005):
        raise RuntimeError(f"W_B_FULL_COUNT_MISMATCH {removed}/{len(B[1])}")
    probe = np.concatenate([kc_idx[:10], mb_idx[:2]])
    fs = field_switch_probe(brain, A, B, probe, kc_idx, is_kc)
    if not (fs["A_restore_exact"] and fs["non_KC_unchanged"]):
        raise RuntimeError(f"CODE_INTEGRATION_NO_GO {fs}")
    A_hash = hashlib.sha256(A[0].tobytes() + A[1].tobytes() + A[2].tobytes()).hexdigest()
    B_hash = hashlib.sha256(B[0].tobytes() + B[1].tobytes() + B[2].tobytes()).hexdigest()
    kc_map = np.full(brain.n, -1, np.int64); kc_map[kc_idx] = np.arange(len(kc_idx))
    mb_map = np.full(brain.n, -1, np.int64); mb_map[mb_idx] = np.arange(len(mb_idx))
    seeds_out = []
    for seed in seeds:
        none = run_condition(brain, seed, kc_idx, mb_idx, lp_pos, kc_map, mb_map, B, "none",
                             out_dir, script_sha, auth_sha, B_hash, kc_m, mb_m)
        stim = run_condition(brain, seed, kc_idx, mb_idx, lp_pos, kc_map, mb_map, B, "stim",
                             out_dir, script_sha, auth_sha, B_hash, kc_m, mb_m)
        pair = bool(none["rng_before"] == stim["rng_before"] and none["rng_after"] == stim["rng_after"])
        seeds_out.append({"seed": seed, "none": none, "stim": stim, "noise_pair_ok": pair})
    res = {"ok": True, "init_s": round(init_s, 4), "n_neurons": int(brain.n),
           "nnz_A": int(len(A[1])), "nnz_B": int(len(B[1])), "removed_edges": int(removed),
           "removed_sum": round(float(removed_sum), 6), "KC_KC_nnz": int(vf["B_KC_KC_nnz"]),
           "A_hash": A_hash, "B_hash": B_hash, "script_sha256": script_sha, "auth_sha256": auth_sha,
           "field_switch": fs, "W_B_verify": vf, "n_kc": n_kc, "n_mb": n_mb,
           "n_kc_matched": n_km, "n_mb_matched": n_mm, "n_lp": int(len(lp_pos)),
           "amount": AMOUNT, "stim_window": list(STIM), "seeds": seeds_out,
           "noise_pairing_all": all(x["noise_pair_ok"] for x in seeds_out),
           "device": "cpu", "batch": 1, "steps": steps, "dW": 0, "training": False,
           "versions": _versions()}
    with open(os.path.join(out_dir, "child_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def _versions():
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "flybrain": getattr(__import__("flybrain"), "__version__", "0.1.0")}


# ---------------------------------------------------------------- stats
def _st(x):
    x = np.asarray(x, float)
    return {"mean": round(float(x.mean()), 9), "sd": round(float(x.std(ddof=1)), 9),
            "median": round(float(np.median(x)), 9), "n": int(len(x))}


def _exact(d):
    from loop.p11_2_c1_preflight import exact_signflip_p
    return float(exact_signflip_p(np.asarray(d, float)))


def _boot(d):
    rng = np.random.default_rng(BOOT_SEED); d = np.asarray(d, float)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT_N)]
    return {"ci_low": round(float(np.percentile(m, 2.5)), 9),
            "ci_high": round(float(np.percentile(m, 97.5)), 9), "n": BOOT_N, "seed": BOOT_SEED}


def _window(arr):
    return arr[STIM[0]:STIM[1]]


def _cell_summary(cell):
    c85 = (cell >= 0.85).sum(axis=1); c95 = (cell >= 0.95).sum(axis=1)
    return {"mean_rate": round(float(cell.mean()), 9), "rate_min": round(float(cell.min()), 9),
            "rate_median": round(float(np.median(cell)), 9), "rate_max": round(float(cell.max()), 9),
            "n_cells_per_seed": int(cell.shape[1]), "n_cell_seed": int(cell.size),
            "cells_ge085": int(c85.sum()), "cells_ge095": int(c95.sum()),
            "cells_ge085_per_seed": [int(x) for x in c85],
            "cells_ge085_mean_per_seed": round(float(c85.mean()), 4),
            "frac_ge085_mean_per_seed": round(float((c85 / cell.shape[1]).mean()), 9),
            "note": "per-cell mean rate in [100,300); not (spikes>=threshold).mean()"}


def _bins(arr, nb=20):
    return [round(float(x.mean()), 6) for x in np.array_split(arr, nb, axis=0)]


def _longest(arr):
    a = np.asarray(arr, dtype=bool)
    if a.ndim != 2 or a.shape[0] == 0:
        return np.zeros(a.shape[1] if a.ndim == 2 else 0, np.int32)
    runs = np.zeros(a.shape, np.int32)
    runs[0] = a[0]
    for i in range(1, a.shape[0]):
        runs[i] = (runs[i - 1] + 1) * a[i]
    return runs.max(axis=0).astype(np.int32)


_RAW_SHAPES = {"kc": (STEPS, 4064), "mb": (STEPS, 97), "lp": (STEPS, 311),
               "kc_matched": (STEPS, 320), "mb_matched": (STEPS, 95)}


def _scalar_str(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return str(v)


def entry_means(rows):
    """真实的 none/stim 绝对均值；diff = stim_mean - none_mean。"""
    if not rows:
        return {"none": None, "stim": None, "diff": None, "n": 0}
    n = np.asarray([float(x["entry_none"]) for x in rows], float)
    s = np.asarray([float(x["entry_stim"]) for x in rows], float)
    return {"none": float(n.mean()), "stim": float(s.mean()),
            "diff": float(s.mean() - n.mean()), "n": len(rows)}


def validate_raw_dataset(out_dir, auth_sha=None, script_sha=None):
    """严格核验 25 seeds × none/stim = 50 个原始 NPZ；返回文件哈希。"""
    errors = []
    files = []
    expected = {f"{s}_{c}.npz" for s in SEEDS for c in ("none", "stim")}
    actual = {n for n in os.listdir(out_dir)
              if n.endswith("_none.npz") or n.endswith("_stim.npz")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"MISSING_FILES:{','.join(missing)}")
    if extra:
        errors.append(f"EXTRA_FILES:{','.join(extra)}")
    script_expect = script_sha or _sha256(__file__)
    w_hashes = set()
    rng_pairs = {}
    seeds_seen = set()
    conditions_seen = set()
    for seed in SEEDS:
        for cond in ("none", "stim"):
            name = f"{seed}_{cond}.npz"
            path = os.path.join(out_dir, name)
            if not os.path.exists(path):
                continue
            try:
                with np.load(path, allow_pickle=True) as z:
                    for key in ("seed", "condition", "amount", "stim_window", "script_sha",
                                "auth_sha", "w_hash", "rng_before", "rng_after"):
                        if key not in z.files:
                            errors.append(f"{name}:MISSING_KEY:{key}")
                    for key, shape in _RAW_SHAPES.items():
                        if key not in z.files:
                            errors.append(f"{name}:MISSING_ARRAY:{key}")
                            continue
                        arr = z[key]
                        if tuple(arr.shape) != shape:
                            errors.append(f"{name}:SHAPE:{key}:{arr.shape}!={shape}")
                        if not np.issubdtype(arr.dtype, np.number):
                            errors.append(f"{name}:DTYPE:{key}:{arr.dtype}")
                        elif not bool(np.isfinite(arr).all()):
                            errors.append(f"{name}:NONFINITE:{key}")
                    if "seed" in z.files:
                        got_seed = int(z["seed"].item() if hasattr(z["seed"], "item") else z["seed"])
                        if got_seed != seed:
                            errors.append(f"{name}:SEED_LABEL:{got_seed}!={seed}")
                        seeds_seen.add(got_seed)
                    if "condition" in z.files:
                        got_cond = _scalar_str(z["condition"])
                        conditions_seen.add(got_cond)
                        if got_cond != cond:
                            errors.append(f"{name}:CONDITION_LABEL:{got_cond}!={cond}")
                    if "amount" in z.files:
                        amt = float(z["amount"].item() if hasattr(z["amount"], "item") else z["amount"])
                        if abs(amt - AMOUNT) > 1e-9:
                            errors.append(f"{name}:AMOUNT:{amt}")
                    if "stim_window" in z.files:
                        win = [int(x) for x in np.asarray(z["stim_window"]).reshape(-1).tolist()]
                        if win != list(STIM):
                            errors.append(f"{name}:STIM_WINDOW:{win}")
                    if "script_sha" in z.files:
                        got = _scalar_str(z["script_sha"])
                        if got != script_expect:
                            errors.append(f"{name}:SCRIPT_SHA:{got}")
                    if "auth_sha" in z.files and auth_sha is not None:
                        got = _scalar_str(z["auth_sha"])
                        if got != auth_sha:
                            errors.append(f"{name}:AUTH_SHA:{got}")
                    if "w_hash" in z.files:
                        wh = _scalar_str(z["w_hash"])
                        if not re.fullmatch(r"[0-9a-fA-F]{64}", wh):
                            errors.append(f"{name}:W_HASH_INVALID:{wh}")
                        w_hashes.add(wh)
                    if "rng_before" in z.files and "rng_after" in z.files:
                        rb = _scalar_str(z["rng_before"])
                        ra = _scalar_str(z["rng_after"])
                        if not re.fullmatch(r"[0-9a-fA-F]{64}", rb):
                            errors.append(f"{name}:RNG_BEFORE_INVALID:{rb}")
                        if not re.fullmatch(r"[0-9a-fA-F]{64}", ra):
                            errors.append(f"{name}:RNG_AFTER_INVALID:{ra}")
                        rng_pairs.setdefault(seed, {})[cond] = {"before": rb, "after": ra}
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}:LOAD_FAILED:{type(e).__name__}:{e}")
            files.append({"name": name, "sha256": _sha256(path), "size": os.path.getsize(path)})
    noise_ok = True
    for seed in SEEDS:
        pair = rng_pairs.get(seed, {})
        n = pair.get("none")
        s = pair.get("stim")
        if not n or not s or n.get("before") != s.get("before") or n.get("after") != s.get("after"):
            noise_ok = False
            errors.append(f"NOISE_PAIR_MISMATCH:{seed}")
    if len(w_hashes) != 1:
        errors.append(f"W_HASH_COUNT:{len(w_hashes)}")
    if set(seeds_seen) != set(SEEDS):
        errors.append(f"SEED_COVERAGE:{len(seeds_seen)}/{len(SEEDS)}")
    if set(conditions_seen) != {"none", "stim"}:
        errors.append(f"CONDITION_COVERAGE:{sorted(conditions_seen)}")
    h = hashlib.sha256()
    for row in sorted(files, key=lambda x: x["name"]):
        h.update(f"{row['name']}:{row['sha256']}\n".encode())
    return {"ok": not errors, "errors": errors, "files": files, "file_count": len(files),
            "seeds_observed": sorted(seeds_seen), "conditions_observed": sorted(conditions_seen),
            "noise_pairing_all": noise_ok, "w_hashes": sorted(w_hashes),
            "dataset_sha256": h.hexdigest(), "expected_files": len(expected)}


def verify_raw_hashes(out_dir, files):
    bad = []
    for row in files:
        p = os.path.join(out_dir, row["name"])
        if not os.path.exists(p) or _sha256(p) != row["sha256"]:
            bad.append(row["name"])
    return {"ok": not bad, "mismatches": bad}


def analyze(out_dir, validate=True):
    if validate:
        raw = validate_raw_dataset(out_dir)
        if not raw["ok"]:
            raise RuntimeError("RAW_DATASET_INVALID:" + "|".join(raw["errors"][:8]))
    ids = np.load(os.path.join(out_dir, "ids.npz"), allow_pickle=True)
    rows = []
    celln_rows = []; cells_rows = []; mbn_rows = []; mbs_rows = []
    for s in SEEDS:
        zn = np.load(os.path.join(out_dir, f"{s}_none.npz"), allow_pickle=True)
        zs = np.load(os.path.join(out_dir, f"{s}_stim.npz"), allow_pickle=True)
        kcn = zn["kc"]; kcs = zs["kc"]; mbn = zn["mb"]; mbs = zs["mb"]
        lpn = zn["lp"]; lps = zs["lp"]
        kcmn = zn["kc_matched"]; kcms = zs["kc_matched"]
        mbmn = zn["mb_matched"]; mbms = zs["mb_matched"]
        entry_n = float(_window(lpn).mean()); entry_s = float(_window(lps).mean())
        kc_n = float(_window(kcmn).mean()); kc_s = float(_window(kcms).mean())
        mb_n = float(_window(mbmn).mean()); mb_s = float(_window(mbms).mean())
        # 细胞发放数组只保留在 50 个 NPZ；JSON 只存标量和必要的短列表。
        celln_rows.append(_window(kcn).mean(axis=0))
        cells_rows.append(_window(kcs).mean(axis=0))
        mbn_rows.append(_window(mbn).mean(axis=0))
        mbs_rows.append(_window(mbs).mean(axis=0))
        low_n = [float(x) for x in _window(kcn)[:, [2460, 2650, 3971]].mean(axis=0)]
        low_s = [float(x) for x in _window(kcs)[:, [2460, 2650, 3971]].mean(axis=0)]
        rows.append({"seed": s, "entry_none": entry_n, "entry_stim": entry_s,
                     "entry_diff": entry_s - entry_n,
                     "kc_none": kc_n, "kc_stim": kc_s, "kc_diff": kc_s - kc_n,
                     "mb_none": mb_n, "mb_stim": mb_s, "mb_diff": mb_s - mb_n,
                     "full_kc_none": float(_window(kcn).mean()), "full_kc_stim": float(_window(kcs).mean()),
                     "full_mb_none": float(_window(mbn).mean()), "full_mb_stim": float(_window(mbs).mean()),
                     "low_none": low_n, "low_stim": low_s,
                     "none_drift": float(_window(kcmn)[100:].mean() - _window(kcmn)[:100].mean()),
                     "startup_none": _bins(kcn), "startup_stim": _bins(kcs),
                     "long_full_none": int(_longest(kcn).max()), "long_main_none": int(_longest(_window(kcn)).max()),
                     "long_full_stim": int(_longest(kcs).max()), "long_main_stim": int(_longest(_window(kcs)).max())})
    d = np.array([x["kc_diff"] for x in rows]); de = np.array([x["entry_diff"] for x in rows])
    dm = np.array([x["mb_diff"] for x in rows])
    pe = _exact(de); pk = _exact(d); pm = _exact(dm)
    entry_pass = bool(de.mean() > 0 and pe < 0.05)
    kc_pass = bool(d.mean() >= MIN_KC and pk < 0.05 and _boot(d)["ci_low"] > 0)
    mb_pass = bool(dm.mean() >= MIN_MBON and pm < 0.05 and _boot(dm)["ci_low"] > 0)
    if not entry_pass:
        grade = "SENSORY_ENTRY_FAILED"
    elif not kc_pass:
        grade = "SENSORY_TRANSMISSION_NOT_DETECTED"
    elif not mb_pass:
        grade = "KC_RESPONSE_WITHOUT_MBON_TRANSMISSION"
    else:
        grade = "SENSORY_TRANSMISSION_DETECTED_IN_SYNTHETIC_B"
    neg = {"kc_negative": bool(d.mean() < 0 and pk < 0.05), "mbon_negative": bool(dm.mean() < 0 and pm < 0.05),
           "entry_negative": bool(de.mean() < 0 and pe < 0.05),
           "note": "negative effects are exploratory real observations; they do not pass the positive primary and do not mean no sensory change"}
    celln = np.array(celln_rows); cells = np.array(cells_rows)
    mbn = np.array(mbn_rows); mbs = np.array(mbs_rows)
    em = entry_means(rows)
    return {
        "protocol": "P11-2-S1R/v1", "status": "ANALYSIS_COMPLETE",
        "completeness": f"{len(rows)}/25",
        "per_seed": rows,
        "entry": {"A_none": round(float(em["none"]), 9),
                  "A_stim": round(float(em["stim"]), 9),
                  "diff": {**_st(de), "mean": round(float(em["diff"]), 9)},
                  "exact_p": pe, "pass": entry_pass,
                  "note": "311 exact LPLC2/LC4; stim-none; mean>0 and p<0.05"},
        "primary": {"endpoint": "matched-320 KC [100,300) stim-none", "diff": _st(d),
                    "exact_p": pk, "ci": _boot(d), "min_positive_effect": MIN_KC,
                    "pass": kc_pass, "note": "mean>=+0.05, p<0.05, 95% CI excludes 0"},
        "mbon": {"matched_95_diff": _st(dm), "exact_p": pm, "ci": _boot(dm),
                 "min_positive_effect": MIN_MBON, "pass": mb_pass},
        "negative": neg, "grading": grade,
        "none_drift": {"mean": round(float(np.mean([x['none_drift'] for x in rows])), 9),
                       "exact_p": _exact(np.array([x['none_drift'] for x in rows]))},
        "secondary": {
            "full_4064_none": round(float(np.mean([x['full_kc_none'] for x in rows])), 9),
            "full_4064_stim": round(float(np.mean([x['full_kc_stim'] for x in rows])), 9),
            "full_97_mbon_none": round(float(np.mean([x['full_mb_none'] for x in rows])), 9),
            "full_97_mbon_stim": round(float(np.mean([x['full_mb_stim'] for x in rows])), 9),
            "kc_cell_none": _cell_summary(celln), "kc_cell_stim": _cell_summary(cells),
            "mbon_cell_none": _cell_summary(mbn), "mbon_cell_stim": _cell_summary(mbs),
            "low_feedback": {"2460": [round(float(np.mean([x['low_none'][0] for x in rows])), 9),
                                      round(float(np.mean([x['low_stim'][0] for x in rows])), 9)],
                             "2650": [round(float(np.mean([x['low_none'][1] for x in rows])), 9),
                                      round(float(np.mean([x['low_stim'][1] for x in rows])), 9)],
                             "3971": [round(float(np.mean([x['low_none'][2] for x in rows])), 9),
                                      round(float(np.mean([x['low_stim'][2] for x in rows])), 9)]},
            "startup_none_bins": [round(float(x), 6) for x in np.mean([x['startup_none'] for x in rows], axis=0)],
            "startup_stim_bins": [round(float(x), 6) for x in np.mean([x['startup_stim'] for x in rows], axis=0)],
            "longest_full_none": int(np.mean([x['long_full_none'] for x in rows])),
            "longest_main_none": int(np.mean([x['long_main_none'] for x in rows])),
            "longest_full_stim": int(np.mean([x['long_full_stim'] for x in rows])),
            "longest_main_stim": int(np.mean([x['long_main_stim'] for x in rows])),
        },
        "interpretation": "model-internal synthetic B only; no biological function or learning claim",
        "gate": {"Training_Authorization": False, "REAL_SYNAPTIC_LEARNING_READY": "NO",
                 "Level_A/B/C/Biological": "NO-GO", "A1_A2_A3/B": "NO-GO",
                 "DeltaW": 0, "Dijkstra": "NO", "Maze": "NO", "B/C": "NO", "Robot": "NO"},
    }


def _manifest(out_dir, extra):
    rows = []
    for n in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, n)
        if os.path.isfile(p):
            rows.append({"name": n, "sha256": _sha256(p), "size": os.path.getsize(p)})
    return {"protocol": "P11-2-S1R/v1", "out_dir": out_dir, "files": rows, **extra}


def synthetic_tiny():
    from flybrain import FlyBrain
    tmp = tempfile.mkdtemp(prefix="p11_2_s1r_tiny_")
    data = os.path.join(tmp, "data"); os.makedirs(data)
    n = 5
    d = np.zeros((n, n), np.float32)
    # 0 LPLC2, 1 LC4, 2 KC, 3 KC, 4 MBON; W[post,pre]
    d[2, 0] = 0.4; d[2, 1] = 0.5; d[3, 2] = 0.3; d[4, 2] = 0.6; d[4, 3] = 0.2
    ids = np.array([100, 101, 102, 103, 104])
    np.savez(os.path.join(data, "brain.npz"), ids=ids, visual=np.array([], np.int64),
             azimuth=np.zeros(n, np.float32), cell_type=np.array(["LPLC2", "LC4", "KC", "KC", "MBON"]),
             side=np.array(["L", "R", "L", "R", "L"]))
    sp.save_npz(os.path.join(data, "weights.npz"), sp.csr_matrix(d))
    np.savez(os.path.join(tmp, "ids.npz"), ids=ids, kc_idx=np.array([2, 3]), mb_idx=np.array([4]),
             kc_matched=np.array([2]), mb_matched=np.array([4]), lp_idx=np.array([0, 1]))
    with open(os.path.join(tmp, "matched_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"kc": [102], "mbon": [104]}, f)
    out = os.path.join(tmp, "out"); os.makedirs(out)
    res = run_child(data, os.path.join(tmp, "ids.npz"), os.path.join(tmp, "matched_ids.json"),
                    out, [101, 102], 300, validate=False, expected=(2, 1, 1, 1), lp_expected=2)
    return {"tmp": tmp, "out": out, "child": res,
            "ok": bool(res["ok"] and res["noise_pairing_all"] and res["removed_edges"] == 1
                       and res["n_lp"] == 2 and res["field_switch"]["A_restore_exact"])}


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--synthetic" in argv:
        r = synthetic_tiny(); print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
        return 0 if r["ok"] else 2
    if "--child" in argv:
        i = argv.index("--child"); out_dir = os.path.abspath(argv[i + 1])
        guard = assert_in_controlled_job()
        if not guard["ok"]:
            print("[child] S1R_CHILD_JOB_GUARD_FAILED", json.dumps(guard, ensure_ascii=False))
            return 9
        auth = load_auth(); freeze = load_freeze(); repo = load_archive()
        res = run_child(DATA_DIR, os.path.join(repo["raw_dir"], "ids.npz"),
                        os.path.join(repo["raw_dir"], "matched_ids.json"), out_dir,
                        SEEDS, STEPS, validate=True)
        print("[child] ok", res["ok"], "| seeds", len(res["seeds"]), "| removed", res["removed_edges"],
              "| noise", res["noise_pairing_all"], "| lp", res["n_lp"])
        return 0 if res["ok"] else 3
    if "--postprocess" in argv:
        i = argv.index("--postprocess"); out_dir = os.path.abspath(argv[i + 1])
        meta = json.load(open(os.path.join(out_dir, "run_meta.json"), encoding="utf-8"))
        child = json.load(open(os.path.join(out_dir, "child_result.json"), encoding="utf-8"))
        rep = analyze(out_dir); rep["run_meta"] = meta; rep["child"] = child
        json.dump(rep, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("wrote", REPORT); return 0
    print("S1R_EXPERIMENT_MUST_BE_LAUNCHED_BY_loop.p11_2_s1r_launcher")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
