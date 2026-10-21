"""P11-2-S5-3 child: 415-target mechanism observation with S5TRACE/1 logs.

Not run in S5-3; default requires independent authorization and a named Job.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import sys

import numpy as np

from loop.p11_2_c1_experiment import (build_W_B_csc, verify_W_B_fields, install_weights,
                                      field_switch_probe, rng_hash, _sha256, load_archive,
                                      DATA_DIR, OFFICIAL)
from loop.p11_2_s5_1_tracer import InstrumentedFlyBrain, read_trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
AUTH = os.path.join(HERE, "p11_2_s5_3_authorization.json")
FREEZE = os.path.join(HERE, "p11_2_s5_3_script_freeze.json")
SEEDS = list(range(301, 306))
CONDITIONS = ["sham", "kc_low", "kc_primary"]
STEPS = 400
STIM = (100, 300)
AMOUNT_PRIMARY = 0.03625384938440364
AMOUNT_LOW = 0.01812692469220182
AMOUNTS = {"sham": 0.0, "kc_low": AMOUNT_LOW, "kc_primary": AMOUNT_PRIMARY}
TARGET_SHA = "6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385"
MBON_SHA = "2248dccc680981af16270f12f6365ca5832fab2a12bb5ace3b80cc73580b61a6"
JOINT_SHA = "942d87d267711edf943b0804ab9089ab23cec7532f017d10ebf81c3ec4c12567"
TARGET_COUNT = 415
TARGET_KC = 320
TARGET_MBON = 95
WALL_CAP_S = 900.0
MEM_CAP_MIB = 2048
MEM_CAP_BYTES = MEM_CAP_MIB * 1024 ** 2

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
    n = min(int(rows.NumberOfProcessIdsInList), 2048)
    return int(pid) in [int(rows.ProcessIdList[i]) for i in range(n)]


def assert_in_controlled_job(expected_bytes=MEM_CAP_BYTES):
    name = (os.environ.get("P11_2_S5_3_JOB_NAME") or "").strip()
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
            return {"ok": False, "reason": "JOB_MEMORY_LIMIT_FLAGS_MISSING", "job_name": name}
        if not (flags & _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE):
            return {"ok": False, "reason": "KILL_ON_JOB_CLOSE_MISSING", "job_name": name}
        if int(info.ProcessMemoryLimit) != expected_bytes or int(info.JobMemoryLimit) != expected_bytes:
            return {"ok": False, "reason": "JOB_MEMORY_LIMIT_MISMATCH", "job_name": name}
        if not _pid_in_job(job, int(_kernel32.GetCurrentProcessId())):
            return {"ok": False, "reason": "PID_NOT_IN_NAMED_JOB", "job_name": name}
        return {"ok": True, "reason": "CONTROLLED_JOB_OK", "job_name": name}
    finally:
        _kernel32.CloseHandle(job)


def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def _scope():
    return {"n_neurons": 166700, "network": "B_synthetic_kc_kc_zero",
            "remove_edges": 642933, "conditions": CONDITIONS, "seeds": SEEDS,
            "steps": STEPS, "total_steps": len(CONDITIONS) * len(SEEDS) * STEPS,
            "groups": len(CONDITIONS) * len(SEEDS), "window": list(STIM),
            "targets": "matched-320 KC + matched-95 MBON", "target_sha256": TARGET_SHA,
            "mbon_sha256": MBON_SHA, "joint_target_sha256": JOINT_SHA,
            "target_count": TARGET_COUNT, "amount_primary": AMOUNT_PRIMARY,
            "amount_low": AMOUNT_LOW, "device": "cpu", "wall_cap_s": WALL_CAP_S,
            "mem_cap_MiB": MEM_CAP_MIB, "one_run": True, "auto_retry": False,
            "training": False, "dW": 0}


def load_auth():
    if not os.path.exists(AUTH):
        raise RuntimeError("AUTHORIZATION_MISSING")
    with open(AUTH, encoding="utf-8") as f:
        a = json.load(f)
    c = a.pop("checksum", None)
    if c != hashlib.sha256(_canon(a)).hexdigest():
        raise RuntimeError("AUTHORIZATION_CHECKSUM_MISMATCH")
    a["checksum"] = c
    if a.get("protocol") != "P11-2-S5-3/v1" or a.get("status") != "APPROVED" or a.get("approved_by") != "user":
        raise RuntimeError("AUTHORIZATION_NOT_APPROVED")
    if a.get("scope") != _scope():
        raise RuntimeError("AUTHORIZATION_SCOPE_MISMATCH")
    return a


def load_freeze():
    with open(FREEZE, encoding="utf-8") as f:
        z = json.load(f)
    if z.get("experiment_sha256") != _sha256(__file__):
        raise RuntimeError("EXPERIMENT_FREEZE_MISMATCH")
    from loop.p11_2_s5_1_tracer import EXPECTED_SOURCE_SHA256
    if z.get("tracer_sha256") != _sha256(os.path.join(HERE, "p11_2_s5_1_tracer.py")):
        raise RuntimeError("TRACER_FREEZE_MISMATCH")
    if z.get("flybrain_source_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("FLYBRAIN_SOURCE_SHA_MISMATCH")
    return z


def official_hashes():
    files = {k: _sha256(os.path.join(DATA_DIR, k)) for k in OFFICIAL}
    return {"ok": all(files[k] == v for k, v in OFFICIAL.items()), "files": files}


def targets_from_ids(ids):
    kc = np.sort(np.asarray(ids["kc_matched"], np.int64))
    mb = np.sort(np.asarray(ids["mb_matched"], np.int64))
    joint = np.concatenate([kc, mb])
    ksha = hashlib.sha256(kc.astype("<i8").tobytes()).hexdigest()
    msha = hashlib.sha256(mb.astype("<i8").tobytes()).hexdigest()
    jsha = hashlib.sha256(joint.astype("<i8").tobytes()).hexdigest()
    if (len(kc), len(mb), len(joint)) != (TARGET_KC, TARGET_MBON, TARGET_COUNT):
        raise RuntimeError("TARGET_COUNT_MISMATCH")
    if (ksha, msha, jsha) != (TARGET_SHA, MBON_SHA, JOINT_SHA):
        raise RuntimeError("TARGET_SHA_MISMATCH")
    return kc, mb, joint


def run_group(brain, seed, cond, trace_path, kc_target, script_sha, auth_sha, b_hash):
    brain.reset(seed)
    before = rng_hash(brain)
    brain.start_trace(trace_path)
    finite = True
    amount = AMOUNTS[cond]
    for t in range(STEPS):
        inj = [(kc_target, amount)] if (amount > 0 and STIM[0] <= t < STIM[1]) else []
        brain.step(inject=inj)
        if not np.isfinite(brain.v).all():
            finite = False
    after = rng_hash(brain)
    hdr = brain.close_trace()
    _h, rows = read_trace(trace_path)
    fired = rows["fired"].astype(bool)
    kc_rate = float(fired[STIM[0]:STIM[1], :TARGET_KC].mean())
    mb_rate = float(fired[STIM[0]:STIM[1], TARGET_KC:].mean())
    return {"seed": seed, "condition": cond, "amount": amount,
            "trace_file": os.path.basename(trace_path), "trace_sha256": hdr["sha256"],
            "trace_records": hdr["records"], "finite": bool(finite),
            "rng_before": before, "rng_after": after,
            "kc_rate_window": round(kc_rate, 9), "mbon_rate_window": round(mb_rate, 9),
            "script_sha256": script_sha, "auth_sha256": auth_sha, "w_hash": b_hash}


def run_child(out_dir, validate=True):
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
    kc_target, mb_target, joint = targets_from_ids(z)
    brain = InstrumentedFlyBrain(data=DATA_DIR, device="cpu", batch=1, seed=301,
                                 dt=0.020, targets=joint)
    ct = brain.cell_type.astype(str)
    kc_idx = np.flatnonzero(np.char.startswith(ct, "KC"))
    mb_idx = np.flatnonzero(np.char.startswith(ct, "MBON"))
    if not (np.array_equal(kc_idx, z["kc_idx"]) and np.array_equal(mb_idx, z["mb_idx"])):
        raise RuntimeError("KC/MBON_INDEX_MISMATCH")
    if len(kc_idx) != 4064 or len(mb_idx) != 97:
        raise RuntimeError("COUNT_MISMATCH")
    A = (brain.indptr, brain.indices, brain.weights)
    is_kc = np.zeros(brain.n, bool); is_kc[kc_idx] = True
    B, removed, _ = build_W_B_csc(*A, is_kc)
    vf = verify_W_B_fields(A, B, is_kc, kc_idx)
    if not vf["ok"]:
        raise RuntimeError(f"W_B_VERIFY_FAIL:{vf}")
    if removed != 642933 or len(B[1]) != 24940005:
        raise RuntimeError("W_B_COUNT_MISMATCH")
    probe = np.concatenate([kc_idx[:10], mb_idx[:2]])
    fs = field_switch_probe(brain, A, B, probe, kc_idx, is_kc)
    if not (fs["A_restore_exact"] and fs["non_KC_unchanged"]):
        raise RuntimeError("CODE_INTEGRATION_NO_GO")
    b_hash = hashlib.sha256(B[0].tobytes() + B[1].tobytes() + B[2].tobytes()).hexdigest()
    script_sha = _sha256(__file__)
    rows = []
    for seed in SEEDS:
        outs = {}
        for cond in CONDITIONS:
            path = os.path.join(out_dir, f"{seed}_{cond}.bin")
            outs[cond] = run_group(brain, seed, cond, path, kc_target, script_sha, auth_sha, b_hash)
        paired = all(outs[c]["rng_before"] == outs["sham"]["rng_before"] and
                     outs[c]["rng_after"] == outs["sham"]["rng_after"] for c in CONDITIONS)
        rows.append({"seed": seed, "groups": outs, "noise_pair_ok": bool(paired)})
    res = {"ok": True, "n_neurons": int(brain.n), "n_kc": 4064, "n_mb": 97,
           "target_sha256": TARGET_SHA, "mbon_sha256": MBON_SHA,
           "joint_target_sha256": JOINT_SHA, "target_count": TARGET_COUNT,
           "target_kc": TARGET_KC, "target_mbon": TARGET_MBON,
           "removed_edges": int(removed), "nnz_B": int(len(B[1])), "KC_KC_nnz": int(vf["B_KC_KC_nnz"]),
           "B_hash": b_hash, "script_sha256": script_sha, "auth_sha256": auth_sha,
           "seeds": rows, "conditions": CONDITIONS, "steps": STEPS,
           "groups": len(rows) * len(CONDITIONS), "noise_pairing_all": all(x["noise_pair_ok"] for x in rows),
           "device": "cpu", "batch": 1, "training": False, "dW": 0, "amount_primary": AMOUNT_PRIMARY,
           "amount_low": AMOUNT_LOW, "stim_window": list(STIM)}
    with open(os.path.join(out_dir, "child_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


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
            print("[child] S5_3_CHILD_JOB_GUARD_FAILED", json.dumps(g, ensure_ascii=False)); return 9
        try:
            res = run_child(out_dir, validate=True)
        except Exception as e:  # noqa: BLE001
            print("[child] S5_3_CHILD_FAILED", type(e).__name__, e); return 3
        print("[child] ok", res["ok"], "| seeds", len(res["seeds"]), "| groups", res["groups"],
              "| noise", res["noise_pairing_all"], "| target", res["joint_target_sha256"][:12])
        return 0
    print("S5_3_EXPERIMENT_MUST_BE_LAUNCHED_BY_p11_2_s5_3_launcher")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
