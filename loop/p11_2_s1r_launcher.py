"""P11-2-S1R-0：未来一次性 S1R 的唯一生产启动入口。

只做静态集成/合成测试；未获独立 S1R 授权时 fail-closed，启动完整脑之前即拒绝。
子进程固定使用 `python -u -m loop.p11_2_s1r_experiment --child <out_dir>`，
cwd=仓库根，日志/产物/退出状态均由 S1F 安全 Job 启动器负责。
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import time

from loop.p11_2_s1f_safe_job_launcher import run_under_job_safe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_MODULE = "loop.p11_2_s1r_experiment"
EXPERIMENT_PATH = os.path.join(HERE, "p11_2_s1r_experiment.py")
AUTH = os.path.join(HERE, "p11_2_s1r_authorization.json")
FREEZE = os.path.join(HERE, "p11_2_s1r_script_freeze.json")
REPORT = os.path.join(HERE, "p11_2_s1r_report.json")
RUN_META = os.path.join(HERE, "p11_2_s1r_run_meta.json")
RAW_MANIFEST = os.path.join(HERE, "p11_2_s1r_raw_manifest.json")
LOCK_DIR = os.path.join(HERE, "p11_2_s1r_authorization_locks")
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
SEEDS = list(range(101, 126))
STEPS = 400
STIM = (100, 300)
AMOUNT = 0.036254
DV = 0.2
EXPECTED_N_NEURONS = 166700
EXPECTED_N_KC = 4064
EXPECTED_N_MBON = 97
EXPECTED_N_KC_MATCHED = 320
EXPECTED_N_MBON_MATCHED = 95
EXPECTED_N_LP = 311
EXPECTED_REMOVED_EDGES = 642933
EXPECTED_NNZ_B = 24940005
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
WALL_CAP_S = 900.0
MEM_CAP_MIB = 2048
MEM_CAP_BYTES = MEM_CAP_MIB * 1024 ** 2
OFFICIAL = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}
SCOPE = {
    "n_neurons": 166700, "network": "B_synthetic_kc_kc_zero",
    "remove_edges": 642933, "conditions": ["none", "stim"], "seeds": SEEDS,
    "steps": STEPS, "total_steps": 2 * len(SEEDS) * STEPS, "window": list(STIM),
    "targets": "exact LPLC2 + LC4 (311)", "dv": DV, "amount": AMOUNT,
    "device": "cpu", "wall_cap_s": WALL_CAP_S, "mem_cap_MiB": MEM_CAP_MIB,
    "one_run": True, "auto_retry": False, "training": False, "dW": 0,
}


class Refused(RuntimeError):
    pass


def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def lock_path_for(auth_checksum):
    return os.path.join(LOCK_DIR, f"{auth_checksum}.consumed.json")


def acquire_one_shot(auth_checksum, lock_path=None, out_dir=None, freeze_sha=None):
    """原子创建持久锁；O_EXCL 成功才算获得唯一执行权；任何失败都不删除。"""
    path = os.path.abspath(lock_path or lock_path_for(auth_checksum))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as e:
        raise Refused("AUTHORIZATION_ALREADY_CONSUMED") from e
    except OSError as e:
        raise Refused(f"LOCK_CREATE_FAILED:{type(e).__name__}:{e}") from e
    payload = {"protocol": "P11-2-S1R/v1", "status": "CONSUMED_STARTED",
               "auth_checksum": auth_checksum, "freeze_sha256": freeze_sha,
               "scope_sha256": hashlib.sha256(_canon(SCOPE)).hexdigest(),
               "out_dir": out_dir, "pid": os.getpid(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:             # noqa: BLE001
        # 文件已原子创建，失败也不自动删除：该授权仍视为已消耗。
        raise Refused(f"LOCK_WRITE_FAILED:{type(e).__name__}:{e}") from e
    return path


def _json_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")).hexdigest()


def _write_json(path, obj):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def auth_payload(approval_text=None, approved_by="user"):
    return {"protocol": "P11-2-S1R/v1", "status": "APPROVED" if approval_text else "NOT_AUTHORIZED",
            "approved_by": approved_by if approval_text else None,
            "approval_text": approval_text, "scope": SCOPE}


def auth_file(approval_text, approved_by="user"):
    p = auth_payload(approval_text, approved_by)
    p["checksum"] = hashlib.sha256(_canon(p)).hexdigest()
    return p


def load_auth():
    if not os.path.exists(AUTH):
        raise Refused("AUTH_MISSING")
    with open(AUTH, encoding="utf-8") as f:
        a = json.load(f)
    c = a.pop("checksum", None)
    if c != hashlib.sha256(_canon(a)).hexdigest():
        raise Refused("AUTH_CHECKSUM_MISMATCH")
    a["checksum"] = c
    if a.get("protocol") != "P11-2-S1R/v1":
        raise Refused("AUTH_PROTOCOL_MISMATCH")
    if a.get("status") != "APPROVED" or a.get("approved_by") != "user":
        raise Refused("NOT_AUTHORIZED")
    if a.get("scope") != SCOPE:
        raise Refused("AUTH_SCOPE_MISMATCH")
    return a


def load_freeze():
    if not os.path.exists(FREEZE):
        raise Refused("FREEZE_MISSING")
    with open(FREEZE, encoding="utf-8") as f:
        z = json.load(f)
    if z.get("module") != EXPERIMENT_MODULE:
        raise Refused("FREEZE_MODULE_MISMATCH")
    if os.path.abspath(z.get("script", "")) != os.path.abspath(EXPERIMENT_PATH):
        raise Refused("FREEZE_SCRIPT_PATH_MISMATCH")
    if z.get("sha256") != _sha(EXPERIMENT_PATH):
        raise Refused("FREEZE_SHA256_MISMATCH")
    return z


def official_hashes():
    files = {}
    for name, expected in OFFICIAL.items():
        p = os.path.join(DATA_DIR, name)
        files[name] = _sha(p) if os.path.exists(p) else None
    return {"ok": all(files[k] == v for k, v in OFFICIAL.items()), "files": files}


def preflight():
    auth = load_auth()                 # 未授权时绝不 import/launch
    freeze = load_freeze()
    try:
        mod = importlib.import_module(EXPERIMENT_MODULE)
    except Exception as e:             # noqa: BLE001
        raise Refused(f"MODULE_IMPORT_FAILED:{type(e).__name__}:{e}") from e
    h = official_hashes()
    if not h["ok"]:
        raise Refused("OFFICIAL_HASH_MISMATCH")
    if getattr(mod, "AUTH", None) != AUTH or getattr(mod, "FREEZE", None) != FREEZE:
        raise Refused("EXPERIMENT_AUTH_FREEZE_PATH_MISMATCH")
    return {"auth": auth, "freeze": freeze, "module": EXPERIMENT_MODULE,
            "module_path": getattr(mod, "__file__", None), "official": h}


def _as_str(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return str(v)


def validate_child_result(path, expected_seeds=25, raw_w_hashes=None):
    """严格核验 child_result；25 条记录本身不等于实验成功。"""
    if not os.path.exists(path):
        return {"path": path, "exists": False, "complete": False, "seeds_observed": 0,
                "noise_pairing_all": None, "reason": "CHILD_OUTPUT_MISSING", "errors": ["CHILD_OUTPUT_MISSING"]}
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:             # noqa: BLE001
        return {"path": path, "exists": True, "complete": False, "seeds_observed": 0,
                "noise_pairing_all": None, "reason": f"CHILD_RESULT_INVALID:{type(e).__name__}",
                "errors": [f"CHILD_RESULT_INVALID:{type(e).__name__}"]}
    errors = []
    if not isinstance(obj, dict):
        return {"path": path, "exists": True, "complete": False, "seeds_observed": 0,
                "noise_pairing_all": None, "reason": "CHILD_RESULT_NOT_OBJECT",
                "errors": ["CHILD_RESULT_NOT_OBJECT"]}
    seeds = obj.get("seeds")
    if not isinstance(seeds, list):
        errors.append("SEEDS_NOT_LIST")
        seeds = []
    if len(seeds) != expected_seeds:
        errors.append(f"SEED_COUNT:{len(seeds)}/{expected_seeds}")
    by = {}
    for i, row in enumerate(seeds):
        if not isinstance(row, dict):
            errors.append(f"SEED_ROW_NOT_OBJECT:{i}")
            continue
        try:
            seed = int(row.get("seed"))
        except Exception:              # noqa: BLE001
            errors.append(f"SEED_INVALID:{i}")
            continue
        if seed not in SEEDS:
            errors.append(f"SEED_OUT_OF_RANGE:{seed}")
        if seed in by:
            errors.append(f"SEED_DUPLICATE:{seed}")
        by[seed] = row
        for cond in ("none", "stim"):
            part = row.get(cond)
            if not isinstance(part, dict):
                errors.append(f"CONDITION_MISSING:{seed}:{cond}")
                continue
            if part.get("finite") is not True:
                errors.append(f"NONFINITE:{seed}:{cond}")
            if "condition" in part and _as_str(part.get("condition")) != cond:
                errors.append(f"CONDITION_LABEL:{seed}:{cond}")
            if "seed" in part:
                try:
                    if int(part.get("seed")) != seed:
                        errors.append(f"INNER_SEED:{seed}:{cond}")
                except Exception:      # noqa: BLE001
                    errors.append(f"INNER_SEED_INVALID:{seed}:{cond}")
        if row.get("noise_pair_ok") is not True:
            errors.append(f"NOISE_PAIR:{seed}")
    if set(by) != set(SEEDS):
        errors.append(f"SEED_SET:{sorted(set(SEEDS) - set(by))}")
    if obj.get("noise_pairing_all") is not True:
        errors.append("NOISE_PAIRING_ALL_NOT_TRUE")
    if obj.get("ok") is not True:
        errors.append("CHILD_NOT_OK")
    expected = {
        "n_neurons": EXPECTED_N_NEURONS, "n_kc": EXPECTED_N_KC, "n_mb": EXPECTED_N_MBON,
        "n_kc_matched": EXPECTED_N_KC_MATCHED, "n_mb_matched": EXPECTED_N_MBON_MATCHED,
        "n_lp": EXPECTED_N_LP, "steps": STEPS, "device": "cpu", "batch": 1,
        "training": False, "dW": 0, "removed_edges": EXPECTED_REMOVED_EDGES,
        "nnz_B": EXPECTED_NNZ_B, "KC_KC_nnz": 0,
    }
    for key, want in expected.items():
        if obj.get(key) != want:
            errors.append(f"PROTOCOL_MISMATCH:{key}:{obj.get(key)}!={want}")
    try:
        if abs(float(obj.get("amount")) - AMOUNT) > 1e-9:
            errors.append(f"AMOUNT:{obj.get('amount')}")
    except Exception:                  # noqa: BLE001
        errors.append("AMOUNT_INVALID")
    try:
        raw_win = obj.get("stim_window")
        win = [int(x) for x in raw_win] if isinstance(raw_win, (list, tuple)) else []
        if win != list(STIM):
            errors.append(f"STIM_WINDOW:{win}")
    except Exception:                  # noqa: BLE001
        errors.append("STIM_WINDOW_INVALID")
    bhash = _as_str(obj.get("B_hash", ""))
    if not SHA256_RE.fullmatch(bhash):
        errors.append(f"B_HASH_INVALID:{bhash}")
    raw_match = None
    if raw_w_hashes is not None:
        hs = {_as_str(x) for x in raw_w_hashes}
        raw_match = (len(hs) == 1 and bhash in hs)
        if not raw_match:
            errors.append(f"B_HASH_RAW_MISMATCH:{bhash}!={sorted(hs)}")
    complete = not errors
    return {"path": path, "exists": True, "complete": bool(complete),
            "seeds_observed": len(seeds), "noise_pairing_all": obj.get("noise_pairing_all"),
            "reason": None if complete else errors[0], "errors": errors,
            "B_hash": bhash, "raw_w_hashes_match": raw_match}


def evaluate_run(run, child):
    exp = bool(run.get("child_status") == "CHILD_EXIT_OK" and child.get("complete"))
    return {
        "process_finished": bool(run.get("process_finished")),
        "experiment_succeeded": exp,
        "child_status": run.get("child_status"),
        "exit_code": run.get("exit_code"),
        "timeout": bool(run.get("timeout")),
        "mem_exceeded": bool(run.get("mem_exceeded")),
        "child_complete": bool(child.get("complete")),
        "seeds_observed": int(child.get("seeds_observed", 0)),
        "child_reason": child.get("reason"),
    }


def _base_meta(pre, run, out_dir, wall_s, child_path, lock_path=None):
    peaks = run.get("job_peaks") or {}
    return {
        "protocol": "P11-2-S1R/v1", "out_dir": out_dir, "created": time.strftime("%Y%m%dT%H%M%S"),
        "lock_path": lock_path,
        "status": "INCOMPLETE_NO_GO", "experiment_succeeded": False,
        "process_finished": bool(run.get("process_finished")),
        "child_status": run.get("child_status"), "exit_code": run.get("exit_code"),
        "timeout": bool(run.get("timeout")), "mem_exceeded": bool(run.get("mem_exceeded")),
        "run_wall_s": round(wall_s, 3), "job_peak_MiB": peaks.get("PeakJobMemoryUsed_MiB"),
        "peak_working_set_MiB": run.get("peak_MiB"), "job_peaks": peaks,
        "log_bytes": run.get("log_bytes"), "log_tail": (run.get("log_tail") or "")[-2000:],
        "auth_checksum": pre["auth"]["checksum"], "freeze_sha256": pre["freeze"]["sha256"],
        "script_sha256": _sha(EXPERIMENT_PATH), "official_hashes": pre["official"],
        "seeds_expected": len(SEEDS), "child_result": child_path,
        "wall_cap_s": WALL_CAP_S, "mem_cap_MiB": MEM_CAP_MIB,
        "resource_verified": False,
        "resource_note": "C1 measured 1942.7 MiB / 105.3 MiB margin; S1/S1R not resource-measured.",
        "training_authorized": False, "dW": 0, "auto_retry": False,
    }


def _persist_run_meta(out_dir, meta, run_meta_path=None):
    # 必须先于任何统计后处理持久化
    _write_json(os.path.join(out_dir, "run_meta.json"), meta)
    _write_json(run_meta_path or RUN_META, meta)


def _copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)


def _manifest(out_dir, extra):
    rows = []
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            rows.append({"name": name, "sha256": _sha(p), "size": os.path.getsize(p)})
    return {"protocol": "P11-2-S1R/v1", "out_dir": out_dir, "files": rows, **extra}


def _env():
    e = os.environ.copy()
    e["PYTHONPATH"] = ROOT + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def _run_child_module(module, args, cwd, out_dir, limit_mib=256, wall_s=30.0,
                      required_artifact=None, inherit_log=True, job_name=None):
    env = _env()
    if job_name:
        env["P11_2_S1R_JOB_NAME"] = job_name       # hint only; child must open/verify real Job
    return run_under_job_safe(limit_mib * 1024 ** 2, module=module, args=args, cwd=cwd,
                              env=env, wall_cap_s=wall_s, mem_cap_mib=limit_mib,
                              log_path=os.path.join(out_dir, "experiment.log"),
                              inherit_log=inherit_log, required_artifact=required_artifact,
                              job_name=job_name)


def _failure_report(status, reason, meta, child, raw, child_raw):
    return {"protocol": "P11-2-S1R/v1", "status": status,
            "s1r_status": "INCOMPLETE_NO_GO",
            "completeness": f"{child.get('seeds_observed', 0)}/{len(SEEDS)}",
            "complete_25_pairs": False, "run_meta": meta, "child": child_raw,
            "raw": {"ok": raw.get("ok"), "file_count": raw.get("file_count"),
                    "dataset_sha256": raw.get("dataset_sha256"),
                    "errors": raw.get("errors", [])[:20]},
            "reason": reason, "training_authorized": False}


def _try_write(path, obj):
    try:
        _write_json(path, obj)
        return True
    except Exception:                  # noqa: BLE001
        return False


def _write_failure_files(out_dir, report_path, manifest_path, status, reason, meta, child, raw, child_raw):
    rep = _failure_report(status, reason, meta, child, raw, child_raw)
    man = _manifest(out_dir, {"status": status, "s1r_status": "INCOMPLETE_NO_GO",
                              "run_meta_sha256": _json_sha(meta),
                              "raw_ok": raw.get("ok"),
                              "dataset_sha256": raw.get("dataset_sha256")})
    ok_r = _try_write(report_path, rep)
    ok_m = _try_write(manifest_path, man)
    return ok_r and ok_m


def postprocess_and_persist(out_dir, run, meta, pre, child_path, report_path=None,
                            manifest_path=None, run_meta_path=None, exp=None):
    """生产后处理：验证 → analyze → JSON report → manifest → 最终 run_meta。

    任何统计失败或持久化失败都保持 INCOMPLETE_NO_GO；最终 run_meta 的 RUN_COMPLETE
    只在 report 与 manifest 都成功写入之后才落盘。
    """
    report_path = report_path or REPORT
    manifest_path = manifest_path or RAW_MANIFEST
    exp = exp or importlib.import_module(EXPERIMENT_MODULE)
    raw = exp.validate_raw_dataset(out_dir, auth_sha=pre["auth"]["checksum"],
                                   script_sha=meta["script_sha256"])
    child = validate_child_result(child_path, raw_w_hashes=raw["w_hashes"])
    ev = evaluate_run(run, child)
    meta.update(ev)
    meta["child_errors"] = child.get("errors", [])[:20]
    meta["raw_ok"] = bool(raw["ok"])
    meta["raw_file_count"] = raw["file_count"]
    meta["raw_dataset_sha256"] = raw["dataset_sha256"]
    meta["raw_noise_pairing_all"] = raw["noise_pairing_all"]
    meta["raw_errors"] = raw["errors"][:20]
    meta["raw_files"] = raw["files"]
    meta["experiment_succeeded"] = False
    meta["status"] = "INCOMPLETE_NO_GO"
    meta["s1r_status"] = "INCOMPLETE_NO_GO"
    _persist_run_meta(out_dir, meta, run_meta_path)
    child_raw = None
    if child["exists"]:
        try:
            with open(child_path, encoding="utf-8") as f:
                child_raw = json.load(f)
        except Exception:              # noqa: BLE001
            child_raw = None

    if not (ev["experiment_succeeded"] and raw["ok"]):
        reason = (raw["errors"][0] if not raw["ok"] else
                  child.get("reason") or meta.get("child_status") or "INCOMPLETE")
        if not _write_failure_files(out_dir, report_path, manifest_path, "INCOMPLETE_NO_GO",
                                    reason, meta, child, raw, child_raw):
            meta["status"] = "PERSISTENCE_FAILED"
            meta["s1r_status"] = "INCOMPLETE_NO_GO"
            _persist_run_meta(out_dir, meta, run_meta_path)
        return meta

    try:
        rep = exp.analyze(out_dir, validate=False)
        if rep.get("status") != "ANALYSIS_COMPLETE":
            raise RuntimeError(f"ANALYZE_BAD_STATUS:{rep.get('status')}")
        after = exp.verify_raw_hashes(out_dir, raw["files"])
        if not after["ok"]:
            raise RuntimeError(f"POST_STATS_HASH_MISMATCH:{after['mismatches']}")
    except Exception as e:             # noqa: BLE001
        meta["stats_error"] = f"{type(e).__name__}:{e}"
        meta["status"] = "POSTPROCESS_FAILED"
        meta["s1r_status"] = "INCOMPLETE_NO_GO"
        meta["experiment_succeeded"] = False
        _persist_run_meta(out_dir, meta, run_meta_path)
        _write_failure_files(out_dir, report_path, manifest_path, "POSTPROCESS_FAILED",
                             meta["stats_error"], meta, child, raw, child_raw)
        return meta

    meta_final = copy.deepcopy(meta)
    meta_final["experiment_succeeded"] = True
    meta_final["status"] = "RUN_COMPLETE"
    meta_final["s1r_status"] = "RUN_COMPLETE"
    meta_final["post_stats_hashes_ok"] = True
    rep["status"] = "RUN_COMPLETE"
    rep["s1r_status"] = "RUN_COMPLETE"
    rep["run_meta"] = meta_final
    rep["child"] = child_raw
    rep["complete_25_pairs"] = True
    rep["gate"] = {"Training_Authorization": False,
                   "REAL_SYNAPTIC_LEARNING_READY": "NO",
                   "Level_A/B/C/Biological": "NO-GO", "A1_A2_A3/B": "NO-GO",
                   "DeltaW": 0, "Dijkstra": "NO", "Maze": "NO", "B/C": "NO", "Robot": "NO"}
    manifest = _manifest(out_dir, {
        "status": "RUN_COMPLETE", "s1r_status": "RUN_COMPLETE",
        "script_sha256": meta_final["script_sha256"],
        "authorization_sha256": meta_final["auth_checksum"],
        "run_meta_sha256": _json_sha(meta_final),
        "raw_ok": raw["ok"], "dataset_sha256": raw["dataset_sha256"],
        "official_hashes": pre["official"],
    })
    try:
        _write_json(report_path, rep)
        _write_json(manifest_path, manifest)
    except Exception as e:             # noqa: BLE001
        meta["status"] = "PERSISTENCE_FAILED"
        meta["s1r_status"] = "INCOMPLETE_NO_GO"
        meta["experiment_succeeded"] = False
        meta["persistence_error"] = f"{type(e).__name__}:{e}"
        _persist_run_meta(out_dir, meta, run_meta_path)
        _write_failure_files(out_dir, report_path, manifest_path, "PERSISTENCE_FAILED",
                             meta["persistence_error"], meta, child, raw, child_raw)
        return meta
    try:
        # 最后一次成功 run_meta；此前不能永久写成成功。
        _persist_run_meta(out_dir, meta_final, run_meta_path)
    except Exception as e:             # noqa: BLE001
        meta["status"] = "PERSISTENCE_FAILED"
        meta["s1r_status"] = "INCOMPLETE_NO_GO"
        meta["experiment_succeeded"] = False
        meta["persistence_error"] = f"{type(e).__name__}:{e}"
        try:
            _persist_run_meta(out_dir, meta, run_meta_path)
        except Exception:              # noqa: BLE001
            pass
        _write_failure_files(out_dir, report_path, manifest_path, "PERSISTENCE_FAILED",
                             meta["persistence_error"], meta, child, raw, child_raw)
        return meta
    return meta_final


def run_s1r_once(out_dir=None):
    """唯一生产运行路径；调用前必须存在独立 APPROVED S1R 授权。"""
    pre = preflight()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.abspath(out_dir or os.path.join(
        tempfile.gettempdir(), "opencode", "p11_2_s1r_full", stamp))
    if os.path.exists(out_dir):
        raise Refused("OUT_DIR_EXISTS")
    # 先原子消耗授权；失败/崩溃都不自动恢复，第二次执行必然拒绝。
    lock_path = acquire_one_shot(pre["auth"]["checksum"], out_dir=out_dir,
                                 freeze_sha=pre["freeze"]["sha256"])
    try:
        os.makedirs(out_dir, exist_ok=False)
    except Exception as e:             # noqa: BLE001
        raise Refused(f"OUT_DIR_CREATE_FAILED:{type(e).__name__}:{e}") from e
    _copy_if_exists(AUTH, os.path.join(out_dir, "authorization.json"))
    _copy_if_exists(FREEZE, os.path.join(out_dir, "script_freeze.json"))
    try:
        exp = importlib.import_module(EXPERIMENT_MODULE)
        repo = exp.load_archive()
        for name in ("ids.npz", "matched_ids.json"):
            _copy_if_exists(os.path.join(repo["raw_dir"], name), os.path.join(out_dir, name))
    except Exception as e:             # noqa: BLE001
        raise Refused(f"ARCHIVE_PREP_FAILED:{type(e).__name__}:{e}") from e
    child_path = os.path.join(out_dir, "child_result.json")
    job_name = f"Local\\P11_2_S1R_{os.getpid()}_{time.time_ns()}"
    t0 = time.perf_counter()
    run = _run_child_module(EXPERIMENT_MODULE, ["--child", out_dir], ROOT, out_dir,
                            limit_mib=MEM_CAP_MIB, wall_s=WALL_CAP_S,
                            required_artifact=child_path, job_name=job_name)
    wall = time.perf_counter() - t0
    meta = _base_meta(pre, run, out_dir, wall, child_path, lock_path)
    meta["job_name"] = job_name
    _persist_run_meta(out_dir, meta)
    return postprocess_and_persist(out_dir, run, meta, pre, child_path, exp=exp)


def run_synthetic_chain(module, cwd, out_dir, args=(), expected_seeds=25, limit_mib=256,
                        wall_s=30.0, stats_fn=None, inherit_log=True):
    """只用于合成回归；不触碰生产授权/完整脑。"""
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir):
        raise Refused("OUT_DIR_EXISTS")
    os.makedirs(out_dir, exist_ok=False)
    child_path = os.path.join(out_dir, "child_result.json")
    run = _run_child_module(module, list(args), os.path.abspath(cwd), out_dir,
                            limit_mib=limit_mib, wall_s=wall_s,
                            required_artifact=child_path, inherit_log=inherit_log)
    meta = {"protocol": "P11-2-S1R/SYNTHETIC", "out_dir": out_dir,
            "status": "INCOMPLETE_NO_GO", "experiment_succeeded": False,
            "process_finished": bool(run.get("process_finished")),
            "child_status": run.get("child_status"), "exit_code": run.get("exit_code"),
            "timeout": bool(run.get("timeout")), "mem_exceeded": bool(run.get("mem_exceeded")),
            "job_peaks": run.get("job_peaks"), "log_bytes": run.get("log_bytes"),
            "log_tail": (run.get("log_tail") or "")[-2000:], "seeds_expected": expected_seeds}
    _write_json(os.path.join(out_dir, "run_meta.json"), meta)   # metadata first
    child = validate_child_result(child_path, expected_seeds)
    ev = evaluate_run(run, child)
    meta.update(ev)
    meta["status"] = "RUN_COMPLETE" if ev["experiment_succeeded"] else "INCOMPLETE_NO_GO"
    _write_json(os.path.join(out_dir, "run_meta.json"), meta)
    if stats_fn is not None:
        stats_fn(out_dir, meta)         # 必须在 run_meta 之后
    return {"run": run, "child": child, "meta": meta, "out_dir": out_dir}


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--execute" in argv:
            meta = run_s1r_once()
            print("RUN_COMPLETE" if meta["experiment_succeeded"] else "INCOMPLETE_NO_GO")
            return 0 if meta["experiment_succeeded"] else 5
        pre = preflight()
        print(json.dumps({"status": "PREFLIGHT_OK_NOT_RUN", "module": pre["module"],
                          "official_hashes_ok": pre["official"]["ok"]}, ensure_ascii=False))
        return 0
    except Refused as e:
        print(json.dumps({"status": "REFUSED", "reason": str(e),
                          "new_s1r_execution_authorization": False},
                         ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
