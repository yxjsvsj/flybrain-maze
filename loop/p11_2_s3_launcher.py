"""P11-2-S3 production parent: fail-closed until independent S3 authorization."""
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
EXPERIMENT_MODULE = "loop.p11_2_s3_experiment"
EXPERIMENT_PATH = os.path.join(HERE, "p11_2_s3_experiment.py")
AUTH = os.path.join(HERE, "p11_2_s3_authorization.json")
FREEZE = os.path.join(HERE, "p11_2_s3_script_freeze.json")
REPORT = os.path.join(HERE, "p11_2_s3_report.json")
RUN_META = os.path.join(HERE, "p11_2_s3_run_meta.json")
RAW_MANIFEST = os.path.join(HERE, "p11_2_s3_raw_manifest.json")
LOCK_DIR = os.path.join(HERE, "p11_2_s3_authorization_locks")
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
SEEDS = list(range(201, 226))
CONDITIONS = ["sham", "kc_low", "kc_primary"]
STEPS = 400
STIM = (100, 300)
AMOUNT_PRIMARY = 0.03625384938440364
AMOUNT_LOW = 0.01812692469220182
TARGET_SHA = "6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385"
WALL_CAP_S = 900.0
MEM_CAP_MIB = 2048
MEM_CAP_BYTES = MEM_CAP_MIB * 1024 ** 2
OFFICIAL = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}
SCOPE = {
    "n_neurons": 166700, "network": "B_synthetic_kc_kc_zero", "remove_edges": 642933,
    "conditions": CONDITIONS, "seeds": SEEDS, "steps": STEPS,
    "total_steps": len(CONDITIONS) * len(SEEDS) * STEPS, "window": list(STIM),
    "targets": "S1R matched-320 KC", "target_sha256": TARGET_SHA,
    "dv_primary": 0.20, "dv_low": 0.10,
    "amount_primary": AMOUNT_PRIMARY, "amount_low": AMOUNT_LOW,
    "device": "cpu", "wall_cap_s": WALL_CAP_S, "mem_cap_MiB": MEM_CAP_MIB,
    "one_run": True, "auto_retry": False, "training": False, "dW": 0,
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


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


def lock_path_for(auth_checksum):
    return os.path.join(LOCK_DIR, f"{auth_checksum}.consumed.json")


def acquire_one_shot(auth_checksum, lock_path=None, out_dir=None, freeze_sha=None):
    path = os.path.abspath(lock_path or lock_path_for(auth_checksum))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as e:
        raise Refused("AUTHORIZATION_ALREADY_CONSUMED") from e
    except OSError as e:
        raise Refused(f"LOCK_CREATE_FAILED:{type(e).__name__}:{e}") from e
    payload = {"protocol": "P11-2-S3/v1", "status": "CONSUMED_STARTED",
               "auth_checksum": auth_checksum, "freeze_sha256": freeze_sha,
               "scope_sha256": hashlib.sha256(_canon(SCOPE)).hexdigest(),
               "out_dir": out_dir, "pid": os.getpid(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
            f.flush(); os.fsync(f.fileno())
    except Exception as e:  # noqa: BLE001
        raise Refused(f"LOCK_WRITE_FAILED:{type(e).__name__}:{e}") from e
    return path


def auth_payload(approval_text=None, approved_by="user"):
    return {"protocol": "P11-2-S3/v1", "status": "APPROVED" if approval_text else "NOT_AUTHORIZED",
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
    if a.get("protocol") != "P11-2-S3/v1":
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
        raise Refused("FREEZE_PATH_MISMATCH")
    if z.get("sha256") != _sha(EXPERIMENT_PATH):
        raise Refused("FREEZE_SHA_MISMATCH")
    return z


def official_hashes():
    files = {k: _sha(os.path.join(DATA_DIR, k)) if os.path.exists(os.path.join(DATA_DIR, k)) else None
             for k in OFFICIAL}
    return {"ok": all(files[k] == v for k, v in OFFICIAL.items()), "files": files}


def preflight():
    auth = load_auth()
    freeze = load_freeze()
    try:
        mod = importlib.import_module(EXPERIMENT_MODULE)
    except Exception as e:  # noqa: BLE001
        raise Refused(f"MODULE_IMPORT_FAILED:{type(e).__name__}:{e}") from e
    h = official_hashes()
    if not h["ok"]:
        raise Refused("OFFICIAL_HASH_MISMATCH")
    if getattr(mod, "AUTH", None) != AUTH or getattr(mod, "FREEZE", None) != FREEZE:
        raise Refused("EXPERIMENT_AUTH_FREEZE_PATH_MISMATCH")
    if getattr(mod, "TARGET_SHA", None) != TARGET_SHA:
        raise Refused("TARGET_SHA_MISMATCH")
    return {"auth": auth, "freeze": freeze, "official": h}


def _persist_run_meta(out_dir, meta, run_meta_path=None):
    _write_json(os.path.join(out_dir, "run_meta.json"), meta)
    _write_json(run_meta_path or RUN_META, meta)


def _manifest(out_dir, extra):
    rows = []
    for name in sorted(os.listdir(out_dir)):
        if name == "run_meta.json":
            continue
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            rows.append({"name": name, "sha256": _sha(p), "size": os.path.getsize(p)})
    return {"protocol": "P11-2-S3/v1", "out_dir": out_dir, "files": rows, **extra}


def _env(job_name=None):
    e = os.environ.copy()
    e["PYTHONPATH"] = ROOT + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e["PYTHONIOENCODING"] = "utf-8"
    if job_name:
        e["P11_2_S3_JOB_NAME"] = job_name
    return e


def _run_child_module(module, args, cwd, out_dir, limit_mib=256, wall_s=30.0,
                      required_artifact=None, job_name=None):
    return run_under_job_safe(limit_mib * 1024 ** 2, module=module, args=args, cwd=cwd,
                              env=_env(job_name), wall_cap_s=wall_s, mem_cap_mib=limit_mib,
                              log_path=os.path.join(out_dir, "experiment.log"),
                              required_artifact=required_artifact, job_name=job_name)


def _failure_report(status, reason, meta, raw, child):
    return {"protocol": "P11-2-S3/v1", "status": status, "run_status": "INCOMPLETE_NO_GO",
            "scientific_go": False, "decision": "NOT_INTERPRETED",
            "completeness": "INCOMPLETE", "run_meta": meta,
            "raw": {"ok": raw.get("ok"), "file_count": raw.get("file_count"),
                    "dataset_sha256": raw.get("dataset_sha256"), "errors": raw.get("errors", [])[:20]},
            "child": child, "reason": reason, "training_authorized": False}


def _try_write(path, obj):
    try:
        _write_json(path, obj); return True
    except Exception:  # noqa: BLE001
        return False


def _write_failure_files(out_dir, report_path, manifest_path, status, reason, meta, raw, child):
    rep = _failure_report(status, reason, meta, raw, child)
    man = _manifest(out_dir, {"status": status, "run_status": "INCOMPLETE_NO_GO",
                              "scientific_go": False, "run_meta_sha256": _json_sha(meta),
                              "raw_ok": raw.get("ok"), "dataset_sha256": raw.get("dataset_sha256")})
    return _try_write(report_path, rep) and _try_write(manifest_path, man)


def postprocess_and_persist(out_dir, run, meta, pre, child_path, report_path=None,
                            manifest_path=None, run_meta_path=None, exp=None):
    report_path = report_path or REPORT
    manifest_path = manifest_path or RAW_MANIFEST
    exp = exp or importlib.import_module(EXPERIMENT_MODULE)
    raw = exp.validate_raw_75(out_dir, auth_sha=pre["auth"]["checksum"],
                              script_sha=meta["script_sha256"])
    child = exp.validate_child_result(child_path, raw_w_hashes=raw["w_hashes"])
    meta.update({"process_finished": bool(run.get("process_finished")),
                 "child_status": run.get("child_status"), "exit_code": run.get("exit_code"),
                 "timeout": bool(run.get("timeout")), "mem_exceeded": bool(run.get("mem_exceeded")),
                 "raw_ok": bool(raw["ok"]), "raw_file_count": raw["file_count"],
                 "raw_dataset_sha256": raw["dataset_sha256"], "raw_errors": raw["errors"][:20],
                 "raw_files": raw["files"], "child_errors": child.get("errors", [])[:20],
                 "experiment_succeeded": False, "scientific_go": False,
                 "decision": "NOT_INTERPRETED", "status": "INCOMPLETE_NO_GO",
                 "run_status": "INCOMPLETE_NO_GO"})
    _persist_run_meta(out_dir, meta, run_meta_path)
    if not (run.get("child_status") == "CHILD_EXIT_OK" and raw["ok"] and child["complete"]):
        reason = (raw["errors"][0] if not raw["ok"] else
                  (child["errors"][0] if child.get("errors") else f"RUN_STATUS:{run.get('child_status')}"))
        if not _write_failure_files(out_dir, report_path, manifest_path, "INCOMPLETE_NO_GO",
                                    reason, meta, raw, child):
            meta["status"] = "PERSISTENCE_FAILED"; meta["run_status"] = "INCOMPLETE_NO_GO"
            _persist_run_meta(out_dir, meta, run_meta_path)
        return meta
    try:
        rep = exp.analyze(out_dir)
        if rep.get("status") != "ANALYSIS_COMPLETE":
            raise RuntimeError(f"ANALYZE_BAD_STATUS:{rep.get('status')}")
        post_hashes = exp.verify_raw_75_hashes(out_dir, raw["files"])
        meta["post_stats_hashes_ok"] = bool(post_hashes["ok"])
        meta["post_stats_hash_mismatches"] = post_hashes["mismatches"][:20]
        if not post_hashes["ok"]:
            raise RuntimeError("POST_STATS_HASH_MISMATCH:" + ",".join(post_hashes["mismatches"]))
    except Exception as e:  # noqa: BLE001
        meta["status"] = "POSTPROCESS_FAILED"; meta["run_status"] = "INCOMPLETE_NO_GO"
        meta["stats_error"] = f"{type(e).__name__}:{e}"; meta["experiment_succeeded"] = False
        meta["scientific_go"] = False
        meta.setdefault("post_stats_hashes_ok", False)
        _persist_run_meta(out_dir, meta, run_meta_path)
        _write_failure_files(out_dir, report_path, manifest_path, "POSTPROCESS_FAILED",
                             meta["stats_error"], meta, raw, child)
        return meta
    meta_final = copy.deepcopy(meta)
    meta_final.update({"status": "RUN_COMPLETE", "run_status": "RUN_COMPLETE",
                       "experiment_succeeded": True,
                       "decision": rep["decision"], "scientific_go": bool(rep["scientific_go"]),
                       "post_stats_hashes_ok": bool(post_hashes["ok"])})
    rep["status"] = "RUN_COMPLETE"; rep["run_status"] = "RUN_COMPLETE"
    rep["run_meta"] = meta_final
    rep["scientific_go"] = bool(rep["scientific_go"])
    manifest = _manifest(out_dir, {
        "status": "RUN_COMPLETE", "run_status": "RUN_COMPLETE",
        "scientific_go": bool(rep["scientific_go"]), "decision": rep["decision"],
        "script_sha256": meta_final["script_sha256"],
        "authorization_sha256": meta_final["auth_checksum"],
        "run_meta_sha256": _json_sha(meta_final),
        "raw_ok": raw["ok"], "dataset_sha256": raw["dataset_sha256"],
        "official_hashes": pre["official"],
    })
    try:
        _write_json(report_path, rep)
        _write_json(manifest_path, manifest)
    except Exception as e:  # noqa: BLE001
        meta["status"] = "PERSISTENCE_FAILED"; meta["run_status"] = "INCOMPLETE_NO_GO"
        meta["experiment_succeeded"] = False; meta["scientific_go"] = False
        meta["persistence_error"] = f"{type(e).__name__}:{e}"
        _persist_run_meta(out_dir, meta, run_meta_path)
        _write_failure_files(out_dir, report_path, manifest_path, "PERSISTENCE_FAILED",
                             meta["persistence_error"], meta, raw, child)
        return meta
    try:
        _persist_run_meta(out_dir, meta_final, run_meta_path)
    except Exception as e:  # noqa: BLE001
        meta["status"] = "PERSISTENCE_FAILED"; meta["run_status"] = "INCOMPLETE_NO_GO"
        meta["experiment_succeeded"] = False; meta["scientific_go"] = False
        meta["persistence_error"] = f"{type(e).__name__}:{e}"
        try:
            _persist_run_meta(out_dir, meta, run_meta_path)
        except Exception:  # noqa: BLE001
            pass
        _write_failure_files(out_dir, report_path, manifest_path, "PERSISTENCE_FAILED",
                             meta["persistence_error"], meta, raw, child)
        return meta
    return meta_final


def _base_meta(pre, run, out_dir, wall_s, child_path, lock_path):
    peaks = run.get("job_peaks") or {}
    return {"protocol": "P11-2-S3/v1", "out_dir": out_dir, "created": time.strftime("%Y%m%dT%H%M%S"),
            "lock_path": lock_path, "status": "INCOMPLETE_NO_GO", "run_status": "INCOMPLETE_NO_GO",
            "experiment_succeeded": False, "scientific_go": False, "decision": "NOT_INTERPRETED",
            "process_finished": bool(run.get("process_finished")),
            "child_status": run.get("child_status"), "exit_code": run.get("exit_code"),
            "timeout": bool(run.get("timeout")), "mem_exceeded": bool(run.get("mem_exceeded")),
            "run_wall_s": round(wall_s, 3), "job_peak_MiB": peaks.get("PeakJobMemoryUsed_MiB"),
            "job_peaks": peaks, "log_bytes": run.get("log_bytes"),
            "log_tail": (run.get("log_tail") or "")[-2000:],
            "auth_checksum": pre["auth"]["checksum"], "freeze_sha256": pre["freeze"]["sha256"],
            "script_sha256": _sha(EXPERIMENT_PATH), "target_sha256": TARGET_SHA,
            "official_hashes": pre["official"], "seeds_expected": len(SEEDS),
            "conditions": CONDITIONS, "steps": STEPS, "total_steps": len(CONDITIONS) * len(SEEDS) * STEPS,
            "child_result": child_path, "wall_cap_s": WALL_CAP_S, "mem_cap_MiB": MEM_CAP_MIB,
            "resource_verified": False, "resource_note": "S1R peak 1943.16 MiB; S3 static estimate 1948.16 MiB; unverified.",
            "training_authorized": False, "dW": 0}


def run_s3_once(out_dir=None):
    pre = preflight()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.abspath(out_dir or os.path.join(tempfile.gettempdir(), "opencode", "p11_2_s3_full", stamp))
    if os.path.exists(out_dir):
        raise Refused("OUT_DIR_EXISTS")
    lock_path = acquire_one_shot(pre["auth"]["checksum"], out_dir=out_dir, freeze_sha=pre["freeze"]["sha256"])
    try:
        os.makedirs(out_dir, exist_ok=False)
    except Exception as e:  # noqa: BLE001
        raise Refused(f"OUT_DIR_CREATE_FAILED:{type(e).__name__}:{e}") from e
    for src in (AUTH, FREEZE):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))
    exp = importlib.import_module(EXPERIMENT_MODULE)
    child_path = os.path.join(out_dir, "child_result.json")
    job_name = f"Local\\P11_2_S3_{os.getpid()}_{time.time_ns()}"
    t0 = time.perf_counter()
    run = _run_child_module(EXPERIMENT_MODULE, ["--child", out_dir], ROOT, out_dir,
                            limit_mib=MEM_CAP_MIB, wall_s=WALL_CAP_S,
                            required_artifact=child_path, job_name=job_name)
    wall = time.perf_counter() - t0
    meta = _base_meta(pre, run, out_dir, wall, child_path, lock_path)
    meta["job_name"] = job_name
    _persist_run_meta(out_dir, meta)
    return postprocess_and_persist(out_dir, run, meta, pre, child_path, exp=exp)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--execute" in argv:
            meta = run_s3_once()
            print("RUN_COMPLETE" if meta["status"] == "RUN_COMPLETE" else meta["status"])
            return 0 if meta["status"] == "RUN_COMPLETE" else 5
        pre = preflight()
        print(json.dumps({"status": "PREFLIGHT_OK_NOT_RUN", "module": EXPERIMENT_MODULE,
                          "target_sha": TARGET_SHA, "official_hashes_ok": pre["official"]["ok"]},
                         ensure_ascii=False))
        return 0
    except Refused as e:
        print(json.dumps({"status": "REFUSED", "reason": str(e),
                          "new_s3_execution_authorization": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
