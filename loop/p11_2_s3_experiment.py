"""P11-2-S3 child experiment: direct matched-KC drive, three conditions.

Production module is fail-closed until an independent S3 authorization exists.
No full-brain code is executed by imports or static tests.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

from loop.p11_2_c1_experiment import (build_W_B_csc, verify_W_B_fields, install_weights,
                                      field_switch_probe, rng_hash, _sha256, load_archive,
                                      DATA_DIR, OFFICIAL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
AUTH = os.path.join(HERE, "p11_2_s3_authorization.json")
FREEZE = os.path.join(HERE, "p11_2_s3_script_freeze.json")
SEEDS = list(range(201, 226))
CONDITIONS = ["sham", "kc_low", "kc_primary"]
STEPS = 400
STIM = (100, 300)
DT = 0.020
TAU = 0.100
DV_PRIMARY = 0.20
DV_LOW = 0.10
AMOUNT_PRIMARY = 0.03625384938440364
AMOUNT_LOW = 0.01812692469220182
AMOUNTS = {"sham": 0.0, "kc_low": AMOUNT_LOW, "kc_primary": AMOUNT_PRIMARY}
TARGET_SHA = "6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385"
WALL_CAP_S = 900.0
MEM_CAP_MIB = 2048
MEM_CAP_BYTES = MEM_CAP_MIB * 1024 ** 2
MIN_KC_EFFECT = 0.10
MIN_MBON_EFFECT = 0.05
BOOT_N = 10000
BOOT_SEED = 20261019
KC_SAT_MEAN = 0.85
KC_SAT_FRAC = 0.10
KC_CELL_CEIL = 0.95

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
    info = _JOB_EXTENDED_LIMIT(); ret = wt.DWORD(0)
    if not _kernel32.QueryInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                               ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(ret)):
        return None
    return info


def _pid_in_job(job, pid):
    rows = _JOB_PID_LIST(); ret = wt.DWORD(0)
    if not _kernel32.QueryInformationJobObject(job, _JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                                               ctypes.byref(rows), ctypes.sizeof(rows), ctypes.byref(ret)):
        return False
    return int(pid) in [int(rows.ProcessIdList[i]) for i in range(min(int(rows.NumberOfProcessIdsInList), 2048))]


def assert_in_controlled_job(expected_bytes=MEM_CAP_BYTES):
    name = (os.environ.get("P11_2_S3_JOB_NAME") or "").strip()
    if not name:
        return {"ok": False, "reason": "JOB_NAME_MISSING"}
    in_job = wt.BOOL(0)
    if not _kernel32.IsProcessInJob(_kernel32.GetCurrentProcess(), None, ctypes.byref(in_job)):
        return {"ok": False, "reason": "IS_PROCESS_IN_JOB_FAILED", "job_name": name}
    if not in_job.value:
        return {"ok": False, "reason": "NOT_IN_JOB", "job_name": name}
    job = _kernel32.OpenJobObjectW(_JOB_OBJECT_ALL_ACCESS, False, name)
    if not job:
        return {"ok": False, "reason": "NAMED_JOB_OPEN_FAILED", "job_name": name}
    try:
        info = _query_extended(job)
        if info is None:
            return {"ok": False, "reason": "QUERY_JOB_FAILED", "job_name": name}
        flags = int(info.BasicLimitInformation.LimitFlags)
        if not (flags & _JOB_OBJECT_LIMIT_PROCESS_MEMORY and flags & _JOB_OBJECT_LIMIT_JOB_MEMORY):
            return {"ok": False, "reason": "JOB_MEMORY_LIMIT_FLAGS_MISSING", "limit_flags": flags}
        if not (flags & _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE):
            return {"ok": False, "reason": "KILL_ON_JOB_CLOSE_MISSING", "limit_flags": flags}
        if int(info.ProcessMemoryLimit) != expected_bytes or int(info.JobMemoryLimit) != expected_bytes:
            return {"ok": False, "reason": "JOB_MEMORY_LIMIT_MISMATCH",
                    "process_mem_limit": int(info.ProcessMemoryLimit),
                    "job_mem_limit": int(info.JobMemoryLimit), "limit_flags": flags}
        if not _pid_in_job(job, int(_kernel32.GetCurrentProcessId())):
            return {"ok": False, "reason": "PID_NOT_IN_NAMED_JOB", "job_name": name}
        return {"ok": True, "reason": "CONTROLLED_JOB_OK", "job_name": name,
                "limit_flags": flags, "process_mem_limit": int(info.ProcessMemoryLimit),
                "job_mem_limit": int(info.JobMemoryLimit)}
    finally:
        _kernel32.CloseHandle(job)


def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def _scope():
    return {"n_neurons": 166700, "network": "B_synthetic_kc_kc_zero",
            "remove_edges": 642933, "conditions": CONDITIONS, "seeds": SEEDS,
            "steps": STEPS, "total_steps": len(CONDITIONS) * len(SEEDS) * STEPS,
            "window": list(STIM), "targets": "S1R matched-320 KC", "target_sha256": TARGET_SHA,
            "dv_primary": DV_PRIMARY, "dv_low": DV_LOW,
            "amount_primary": AMOUNT_PRIMARY, "amount_low": AMOUNT_LOW,
            "device": "cpu", "wall_cap_s": WALL_CAP_S, "mem_cap_MiB": MEM_CAP_MIB,
            "one_run": True, "auto_retry": False, "training": False, "dW": 0}


def load_auth():
    if not os.path.exists(AUTH):
        raise RuntimeError("AUTHORIZATION_MISSING")
    with open(AUTH, encoding="utf-8") as f:
        a = json.load(f)
    c = a.pop("checksum", None)
    if c != hashlib.sha256(_canon(a)).hexdigest():
        raise RuntimeError("AUTHORIZATION_CHECKSUM_MISMATCH")
    a["checksum"] = c
    if a.get("protocol") != "P11-2-S3/v1" or a.get("status") != "APPROVED" or a.get("approved_by") != "user":
        raise RuntimeError("AUTHORIZATION_NOT_APPROVED")
    if a.get("scope") != _scope():
        raise RuntimeError("AUTHORIZATION_SCOPE_MISMATCH")
    return a


def load_freeze():
    with open(FREEZE, encoding="utf-8") as f:
        z = json.load(f)
    if z.get("module") != "loop.p11_2_s3_experiment" or os.path.abspath(z.get("script", "")) != os.path.abspath(__file__):
        raise RuntimeError("FREEZE_PATH_MISMATCH")
    if z.get("sha256") != _sha256(__file__):
        raise RuntimeError("SCRIPT_FREEZE_MISMATCH")
    return z


def official_hashes():
    files = {k: _sha256(os.path.join(DATA_DIR, k)) for k in OFFICIAL}
    return {"ok": all(files[k] == v for k, v in OFFICIAL.items()), "files": files}


def target_from_ids(ids):
    target = np.asarray(ids["kc_matched"], dtype=np.int64)
    if len(target) != 320 or len(set(target.tolist())) != 320:
        raise RuntimeError("TARGET_SET_NOT_320")
    if hashlib.sha256(target.astype("<i8").tobytes()).hexdigest() != TARGET_SHA:
        raise RuntimeError("TARGET_SHA_MISMATCH")
    return target


def _tag(seed, cond):
    return f"{seed}_{cond}.npz"


def run_condition(brain, seed, cond, target, kc_idx, mb_idx, lp_pos, kc_map, mb_map,
                  kc_m, mb_m, fields, out_dir, script_sha, auth_sha, w_hash):
    install_weights(brain, fields)
    brain.reset(seed)
    before = rng_hash(brain)
    kc = np.zeros((STEPS, len(kc_idx)), np.uint8)
    mb = np.zeros((STEPS, len(mb_idx)), np.uint8)
    lp = np.zeros((STEPS, len(lp_pos)), np.uint8)
    finite = True
    amount = AMOUNTS[cond]
    for t in range(STEPS):
        inj = [(target, amount)] if (amount > 0 and STIM[0] <= t < STIM[1]) else []
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
    after = rng_hash(brain)
    np.savez_compressed(os.path.join(out_dir, _tag(seed, cond)),
                        kc=kc, mb=mb, lp=lp, kc_matched=kc[:, kc_m], mb_matched=mb[:, mb_m],
                        seed=seed, condition=cond, injection_amount=amount,
                        injection_window=np.array(STIM), target_idx=target,
                        target_sha=TARGET_SHA, script_sha=script_sha, auth_sha=auth_sha,
                        w_hash=w_hash, rng_before=before, rng_after=after,
                        finite=finite, no_sensory_injection=True, no_mbon_injection=True)
    return {"seed": seed, "condition": cond, "finite": bool(finite),
            "rng_before": before, "rng_after": after, "amount": amount}


def run_child(out_dir, validate=True):
    from flybrain import FlyBrain
    if validate:
        auth = load_auth(); freeze = load_freeze(); h = official_hashes()
        if not h["ok"]:
            raise RuntimeError("OFFICIAL_HASH_MISMATCH")
        auth_sha = auth["checksum"]
    else:
        auth_sha = "SYNTHETIC"
    repo = load_archive()
    if not repo.get("ok"):
        raise RuntimeError(f"ARCHIVE_UNVERIFIED:{repo}")
    z = np.load(os.path.join(repo["raw_dir"], "ids.npz"), allow_pickle=True)
    target = target_from_ids(z)
    brain = FlyBrain(data=DATA_DIR, device="cpu", batch=1, seed=201, dt=0.020)
    ct = brain.cell_type.astype(str)
    kc_idx = np.flatnonzero(np.char.startswith(ct, "KC"))
    mb_idx = np.flatnonzero(np.char.startswith(ct, "MBON"))
    lp_pos = np.asarray(brain.cells(["LPLC2", "LC4"]))
    if not (np.array_equal(kc_idx, z["kc_idx"]) and np.array_equal(mb_idx, z["mb_idx"])):
        raise RuntimeError("KC/MBON_INDEX_MISMATCH")
    if len(kc_idx) != 4064 or len(mb_idx) != 97 or len(lp_pos) != 311:
        raise RuntimeError("COUNT_MISMATCH")
    kc_m = np.searchsorted(kc_idx, np.asarray(z["kc_matched"]))
    mb_m = np.searchsorted(mb_idx, np.asarray(z["mb_matched"]))
    if not (np.array_equal(kc_idx[kc_m], z["kc_matched"]) and np.array_equal(mb_idx[mb_m], z["mb_matched"])):
        raise RuntimeError("MATCHED_MAPPING_MISMATCH")
    A = (brain.indptr, brain.indices, brain.weights)
    is_kc = np.zeros(brain.n, bool); is_kc[kc_idx] = True
    B, removed, removed_sum = build_W_B_csc(*A, is_kc)
    vf = verify_W_B_fields(A, B, is_kc, kc_idx)
    if not vf["ok"]:
        raise RuntimeError(f"W_B_VERIFY_FAIL:{vf}")
    if removed != 642933 or len(B[1]) != 24940005:
        raise RuntimeError(f"W_B_COUNT_MISMATCH:{removed}/{len(B[1])}")
    probe = np.concatenate([kc_idx[:10], mb_idx[:2]])
    fs = field_switch_probe(brain, A, B, probe, kc_idx, is_kc)
    if not (fs["A_restore_exact"] and fs["non_KC_unchanged"]):
        raise RuntimeError(f"CODE_INTEGRATION_NO_GO:{fs}")
    B_hash = hashlib.sha256(B[0].tobytes() + B[1].tobytes() + B[2].tobytes()).hexdigest()
    kc_map = np.full(brain.n, -1, np.int64); kc_map[kc_idx] = np.arange(len(kc_idx))
    mb_map = np.full(brain.n, -1, np.int64); mb_map[mb_idx] = np.arange(len(mb_idx))
    script_sha = _sha256(__file__)
    rows = []
    for seed in SEEDS:
        outs = {}
        for cond in CONDITIONS:
            outs[cond] = run_condition(brain, seed, cond, target, kc_idx, mb_idx, lp_pos,
                                       kc_map, mb_map, kc_m, mb_m, B, out_dir, script_sha, auth_sha, B_hash)
        paired = all(outs[c]["rng_before"] == outs["sham"]["rng_before"] and
                     outs[c]["rng_after"] == outs["sham"]["rng_after"] for c in CONDITIONS)
        rows.append({"seed": seed, "sham": outs["sham"], "kc_low": outs["kc_low"],
                     "kc_primary": outs["kc_primary"], "noise_pair_ok": bool(paired)})
    res = {"ok": True, "n_neurons": int(brain.n), "n_kc": 4064, "n_mb": 97,
           "n_kc_matched": 320, "n_mb_matched": 95, "n_lp": 311,
           "removed_edges": int(removed), "nnz_B": int(len(B[1])),
           "KC_KC_nnz": int(vf["B_KC_KC_nnz"]), "target_sha": TARGET_SHA,
           "B_hash": B_hash, "script_sha256": script_sha, "auth_sha256": auth_sha,
           "seeds": rows, "noise_pairing_all": all(x["noise_pair_ok"] for x in rows),
           "conditions": CONDITIONS, "steps": STEPS, "device": "cpu", "batch": 1,
           "training": False, "dW": 0, "amount_primary": AMOUNT_PRIMARY,
           "amount_low": AMOUNT_LOW, "stim_window": list(STIM)}
    with open(os.path.join(out_dir, "child_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def _shape_checks(z, name):
    shapes = {"kc": (STEPS, 4064), "mb": (STEPS, 97), "lp": (STEPS, 311),
              "kc_matched": (STEPS, 320), "mb_matched": (STEPS, 95)}
    errs = []
    for k, shape in shapes.items():
        if k not in z.files:
            errs.append(f"{name}:MISSING:{k}")
        elif tuple(z[k].shape) != shape:
            errs.append(f"{name}:SHAPE:{k}:{z[k].shape}!={shape}")
        elif not np.issubdtype(z[k].dtype, np.number) or not bool(np.isfinite(z[k]).all()):
            errs.append(f"{name}:NONFINITE:{k}")
    return errs


def validate_raw_75(out_dir, auth_sha=None, script_sha=None):
    errors = []; files = []
    expected = {_tag(s, c) for s in SEEDS for c in CONDITIONS}
    actual = {n for n in os.listdir(out_dir) if n.endswith(".npz") and n != "ids.npz"}
    if expected - actual:
        errors.append(f"MISSING_FILES:{sorted(expected - actual)}")
    if actual - expected:
        errors.append(f"EXTRA_FILES:{sorted(actual - expected)}")
    script_sha = script_sha or _sha256(__file__)
    target_seen = set(); w_hashes = set(); rng = {}
    for seed in SEEDS:
        for cond in CONDITIONS:
            name = _tag(seed, cond); p = os.path.join(out_dir, name)
            if not os.path.exists(p):
                continue
            try:
                with np.load(p, allow_pickle=True) as z:
                    errors.extend(_shape_checks(z, name))
                    for key in ("seed", "condition", "injection_amount", "injection_window",
                                "target_idx", "target_sha", "script_sha", "auth_sha",
                                "w_hash", "rng_before", "rng_after", "finite",
                                "no_sensory_injection", "no_mbon_injection"):
                        if key not in z.files:
                            errors.append(f"{name}:MISSING_KEY:{key}")
                    if "seed" in z.files and int(z["seed"]) != seed:
                        errors.append(f"{name}:SEED_LABEL")
                    if "condition" in z.files and str(z["condition"]) != cond:
                        errors.append(f"{name}:CONDITION_LABEL")
                    if "injection_amount" in z.files and abs(float(z["injection_amount"]) - AMOUNTS[cond]) > 1e-15:
                        errors.append(f"{name}:AMOUNT:{float(z['injection_amount'])}")
                    if "injection_window" in z.files:
                        win = [int(x) for x in np.asarray(z["injection_window"]).reshape(-1)]
                        if win != list(STIM):
                            errors.append(f"{name}:WINDOW:{win}")
                    if "target_sha" in z.files and str(z["target_sha"]) != TARGET_SHA:
                        errors.append(f"{name}:TARGET_SHA")
                    if "target_idx" in z.files:
                        t = np.asarray(z["target_idx"], np.int64)
                        target_seen.add(hashlib.sha256(t.astype("<i8").tobytes()).hexdigest())
                    if "script_sha" in z.files and str(z["script_sha"]) != script_sha:
                        errors.append(f"{name}:SCRIPT_SHA")
                    if "auth_sha" in z.files and auth_sha is not None and str(z["auth_sha"]) != auth_sha:
                        errors.append(f"{name}:AUTH_SHA")
                    if "w_hash" in z.files:
                        wh = str(z["w_hash"]); w_hashes.add(wh)
                        if len(wh) != 64 or any(c not in "0123456789abcdefABCDEF" for c in wh):
                            errors.append(f"{name}:W_HASH_INVALID")
                    if "rng_before" in z.files and "rng_after" in z.files:
                        rng.setdefault(seed, {})[cond] = (str(z["rng_before"]), str(z["rng_after"]))
                    if "finite" in z.files and not bool(z["finite"]):
                        errors.append(f"{name}:FINITE_FALSE")
                    if "no_sensory_injection" in z.files and not bool(z["no_sensory_injection"]):
                        errors.append(f"{name}:SENSORY_INJECTION")
                    if "no_mbon_injection" in z.files and not bool(z["no_mbon_injection"]):
                        errors.append(f"{name}:MBON_INJECTION")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}:LOAD_FAILED:{type(e).__name__}:{e}")
            files.append({"name": name, "sha256": _sha256(p), "size": os.path.getsize(p)})
    for seed in SEEDS:
        vals = rng.get(seed, {})
        if len(vals) != 3 or len({v[0] for v in vals.values()}) != 1 or len({v[1] for v in vals.values()}) != 1:
            errors.append(f"RNG_PAIR_MISMATCH:{seed}")
    if len(w_hashes) != 1:
        errors.append(f"W_HASH_COUNT:{len(w_hashes)}")
    if target_seen != {TARGET_SHA}:
        errors.append(f"TARGET_VECTOR_HASH:{sorted(target_seen)}")
    h = hashlib.sha256()
    for row in sorted(files, key=lambda x: x["name"]):
        h.update(f"{row['name']}:{row['sha256']}\n".encode())
    return {"ok": not errors, "errors": errors, "files": files, "file_count": len(files),
            "npz_count": len([x for x in files if x["name"].endswith(".npz")]),
            "dataset_sha256": h.hexdigest(), "w_hashes": sorted(w_hashes),
            "target_hashes": sorted(target_seen)}


def validate_child_result(path, raw_w_hashes=None):
    if not os.path.exists(path):
        return {"exists": False, "complete": False, "errors": ["CHILD_OUTPUT_MISSING"]}
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    errors = []
    if obj.get("ok") is not True:
        errors.append("CHILD_NOT_OK")
    expected = {"n_neurons": 166700, "n_kc": 4064, "n_mb": 97, "n_kc_matched": 320,
                "n_mb_matched": 95, "n_lp": 311, "steps": STEPS, "device": "cpu",
                "batch": 1, "training": False, "dW": 0, "removed_edges": 642933,
                "nnz_B": 24940005, "KC_KC_nnz": 0, "target_sha": TARGET_SHA}
    for k, v in expected.items():
        if obj.get(k) != v:
            errors.append(f"PROTOCOL:{k}")
    if obj.get("conditions") != CONDITIONS:
        errors.append("CONDITIONS")
    if obj.get("noise_pairing_all") is not True:
        errors.append("NOISE_PAIRING_ALL")
    seeds = obj.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 25:
        errors.append("SEED_COUNT")
        seeds = seeds if isinstance(seeds, list) else []
    seen = set()
    for row in seeds:
        try:
            s = int(row["seed"])
        except Exception:  # noqa: BLE001
            errors.append("SEED_INVALID"); continue
        if s not in SEEDS:
            errors.append(f"SEED_RANGE:{s}")
        if s in seen:
            errors.append(f"SEED_DUPLICATE:{s}")
        seen.add(s)
        for cond in CONDITIONS:
            part = row.get(cond)
            if not isinstance(part, dict):
                errors.append(f"CONDITION_MISSING:{s}:{cond}"); continue
            if part.get("finite") is not True:
                errors.append(f"NONFINITE:{s}:{cond}")
            try:
                if abs(float(part.get("amount")) - AMOUNTS[cond]) > 1e-15:
                    errors.append(f"AMOUNT:{s}:{cond}")
            except Exception:  # noqa: BLE001
                errors.append(f"AMOUNT_INVALID:{s}:{cond}")
        if row.get("noise_pair_ok") is not True:
            errors.append(f"NOISE_PAIR:{s}")
    if seen != set(SEEDS):
        errors.append(f"SEED_SET:{sorted(set(SEEDS) - seen)}")
    bhash = str(obj.get("B_hash", ""))
    if len(bhash) != 64:
        errors.append("B_HASH_INVALID")
    if raw_w_hashes is not None:
        hs = {str(x) for x in raw_w_hashes}
        if len(hs) != 1 or bhash not in hs:
            errors.append(f"B_HASH_RAW_MISMATCH:{bhash}!={sorted(hs)}")
    return {"exists": True, "complete": not errors, "errors": errors,
            "seeds_observed": len(seeds), "B_hash": bhash}


def exact_signflip(d):
    d = np.asarray(d, float); n = len(d); total = 1 << n; obs = abs(float(d.sum())); count = 0
    for start in range(0, total, 1 << 20):
        idx = np.arange(start, min(start + (1 << 20), total), dtype=np.int64)
        s = np.zeros(len(idx), float)
        for k in range(n):
            s += np.where((idx >> k) & 1, d[k], -d[k])
        count += int((np.abs(s) >= obs - 1e-12).sum())
    return count / total


def boot_ci(d):
    rng = np.random.default_rng(BOOT_SEED); d = np.asarray(d, float)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT_N)]
    return {"ci_low": round(float(np.percentile(m, 2.5)), 9),
            "ci_high": round(float(np.percentile(m, 97.5)), 9), "n": BOOT_N, "seed": BOOT_SEED}


def _bh(p):
    p = np.asarray(p, float); order = np.argsort(p); q = np.empty_like(p); prev = 1.0
    for rank in range(len(p) - 1, -1, -1):
        i = order[rank]; val = min(prev, p[i] * len(p) / (rank + 1)); q[i] = val; prev = val
    return q


def analyze(out_dir):
    means = {}
    primary_cells = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            with np.load(os.path.join(out_dir, _tag(seed, cond)), allow_pickle=True) as z:
                means.setdefault((seed, cond), {})["kc"] = float(z["kc_matched"][100:300].mean())
                means[(seed, cond)]["mb"] = float(z["mb_matched"][100:300].mean())
                if cond == "kc_primary":
                    primary_cells.append(z["kc_matched"][100:300].mean(axis=0))
    def diff(cond, endpoint):
        return np.asarray([means[(s, cond)][endpoint] - means[(s, "sham")][endpoint] for s in SEEDS], float)
    def pack(d):
        return {"mean": round(float(d.mean()), 9), "sd": round(float(d.std(ddof=1)), 9),
                "p": exact_signflip(d), "ci": boot_ci(d), "n": len(d)}
    kc_p = pack(diff("kc_primary", "kc")); mb_p = pack(diff("kc_primary", "mb"))
    kc_l = pack(diff("kc_low", "kc")); mb_l = pack(diff("kc_low", "mb"))
    low_q = _bh([kc_l["p"], mb_l["p"]])
    cell_mean = np.asarray(primary_cells).mean(axis=0)
    sat_mean = float(cell_mean.mean())
    sat_frac = float((cell_mean >= KC_CELL_CEIL).mean())
    kc_pass = bool(kc_p["mean"] >= MIN_KC_EFFECT and kc_p["p"] < 0.05 and kc_p["ci"]["ci_low"] > 0)
    not_sat = bool(sat_mean < KC_SAT_MEAN and sat_frac <= KC_SAT_FRAC)
    mb_pos = bool(mb_p["mean"] >= MIN_MBON_EFFECT and mb_p["p"] < 0.05 and mb_p["ci"]["ci_low"] > 0)
    mb_neg = bool(mb_p["mean"] <= -MIN_MBON_EFFECT and mb_p["p"] < 0.05 and mb_p["ci"]["ci_high"] < 0)
    if not kc_pass:
        decision = "INCONCLUSIVE_KC_NOT_DRIVEN"
    elif not not_sat:
        decision = "INCONCLUSIVE_KC_SATURATED"
    elif mb_pos:
        decision = "KC_MBON_FUNCTIONAL_UNDER_DIRECT_DRIVE"
    elif mb_neg:
        decision = "KC_MBON_NEGATIVE"
    else:
        decision = "INCONCLUSIVE_DOWNSTREAM"
    return {
        "protocol": "P11-2-S3/v1", "status": "ANALYSIS_COMPLETE",
        "completeness": f"{len(SEEDS)}/25 seeds; {len(SEEDS) * 3}/75 NPZ",
        "kc_primary": {**kc_p, "pass": kc_pass, "min_effect": MIN_KC_EFFECT},
        "mbon_primary": {**mb_p, "positive": mb_pos, "negative": mb_neg,
                         "min_positive_effect": MIN_MBON_EFFECT},
        "kc_low": {**kc_l, "q_bh": round(float(low_q[0]), 12)},
        "mbon_low": {**mb_l, "q_bh": round(float(low_q[1]), 12)},
        "saturation": {"mean_kc_primary": round(sat_mean, 9),
                       "fraction_cell_mean_ge095": round(sat_frac, 9),
                       "mean_lt_085": bool(sat_mean < KC_SAT_MEAN),
                       "frac_le_010": bool(sat_frac <= KC_SAT_FRAC),
                       "pass": not_sat},
        "decision": decision,
        "scientific_go": bool(decision == "KC_MBON_FUNCTIONAL_UNDER_DIRECT_DRIVE"),
        "downstream_interpretable": bool(kc_pass and not_sat),
        "interpretation_note": "Artificial synchronous KC drive only; no natural pathway, single-synapse, or learning claim.",
        "training": False, "dW": 0,
        "gate": {"Training_Authorization": False, "REAL_SYNAPTIC_LEARNING_READY": "NO",
                 "Level_A/B/C/Biological": "NO-GO", "A1_A2_A3/B": "NO-GO",
                 "DeltaW": 0, "Dijkstra": "NO", "Maze": "NO", "B/C": "NO", "Robot": "NO"},
        "per_seed": [{"seed": s, "kc_primary_diff": round(float(means[(s, "kc_primary")]["kc"] - means[(s, "sham")]["kc"]), 9),
                      "mbon_primary_diff": round(float(means[(s, "kc_primary")]["mb"] - means[(s, "sham")]["mb"]), 9)}
                     for s in SEEDS],
    }


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--child" in argv:
        i = argv.index("--child"); out_dir = os.path.abspath(argv[i + 1])
        g = assert_in_controlled_job()
        if not g["ok"]:
            print("[child] S3_CHILD_JOB_GUARD_FAILED", json.dumps(g, ensure_ascii=False)); return 9
        try:
            res = run_child(out_dir, validate=True)
        except Exception as e:  # noqa: BLE001
            print("[child] S3_CHILD_FAILED", type(e).__name__, e); return 3
        print("[child] ok", res["ok"], "| seeds", len(res["seeds"]),
              "| noise", res["noise_pairing_all"], "| target", res["target_sha"][:12])
        return 0
    print("S3_EXPERIMENT_MUST_BE_LAUNCHED_BY_p11_2_s3_launcher")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
