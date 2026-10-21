"""P11-2-S5-3 parent: mechanism-observation launcher, default refuse."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

from loop.p11_2_s1f_safe_job_launcher import run_under_job_safe
from loop.p11_2_s5_1_tracer import read_trace, TRACE_DTYPE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = "loop.p11_2_s5_3_experiment"
EXPERIMENT = os.path.join(HERE, "p11_2_s5_3_experiment.py")
TRACER = os.path.join(HERE, "p11_2_s5_1_tracer.py")
AUTH = os.path.join(HERE, "p11_2_s5_3_authorization.json")
FREEZE = os.path.join(HERE, "p11_2_s5_3_script_freeze.json")
REPORT = os.path.join(HERE, "p11_2_s5_3_report.json")
RUN_META = os.path.join(HERE, "p11_2_s5_3_run_meta.json")
RAW_MANIFEST = os.path.join(HERE, "p11_2_s5_3_raw_manifest.json")
LOCK_DIR = os.path.join(HERE, "p11_2_s5_3_authorization_locks")
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
S1R_META = os.path.join(HERE, "p11_2_s1r_run_meta.json")
SEEDS = list(range(301, 306))
CONDITIONS = ["sham", "kc_low", "kc_primary"]
STEPS = 400
STIM = (100, 300)
TARGET_SHA = "6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385"
MBON_SHA = "2248dccc680981af16270f12f6365ca5832fab2a12bb5ace3b80cc73580b61a6"
JOINT_SHA = "942d87d267711edf943b0804ab9089ab23cec7532f017d10ebf81c3ec4c12567"
TARGET_COUNT = 415
AMOUNT_PRIMARY = 0.03625384938440364
AMOUNT_LOW = 0.01812692469220182
WALL_CAP_S = 900.0
MEM_CAP_MIB = 2048
MEM_CAP_BYTES = MEM_CAP_MIB * 1024 ** 2
EXPECTED_SOURCE_SHA = "142ecfffd5d32e30655829e34b38b04024e49beae662bd078279e9610c789b0c"
OFFICIAL = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}
SCOPE = {
    "n_neurons": 166700, "network": "B_synthetic_kc_kc_zero", "remove_edges": 642933,
    "conditions": CONDITIONS, "seeds": SEEDS, "steps": STEPS,
    "total_steps": len(CONDITIONS) * len(SEEDS) * STEPS,
    "groups": len(CONDITIONS) * len(SEEDS), "window": list(STIM),
    "targets": "matched-320 KC + matched-95 MBON", "target_sha256": TARGET_SHA,
    "mbon_sha256": MBON_SHA, "joint_target_sha256": JOINT_SHA, "target_count": TARGET_COUNT,
    "amount_primary": AMOUNT_PRIMARY, "amount_low": AMOUNT_LOW,
    "device": "cpu", "wall_cap_s": WALL_CAP_S, "mem_cap_MiB": MEM_CAP_MIB,
    "one_run": True, "auto_retry": False, "training": False, "dW": 0,
}


class Refused(RuntimeError):
    pass


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def _json_sha(o):
    return hashlib.sha256(json.dumps(o, ensure_ascii=False, indent=1).encode()).hexdigest()


def _write_json(path, obj):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def lock_path_for(checksum):
    return os.path.join(LOCK_DIR, f"{checksum}.consumed.json")


def acquire_one_shot(checksum, lock_path=None, out_dir=None, freeze_sha=None):
    path = os.path.abspath(lock_path or lock_path_for(checksum))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as e:
        raise Refused("AUTHORIZATION_ALREADY_CONSUMED") from e
    except OSError as e:
        raise Refused(f"LOCK_CREATE_FAILED:{type(e).__name__}:{e}") from e
    payload = {"protocol": "P11-2-S5-3/v1", "status": "CONSUMED_STARTED",
               "auth_checksum": checksum, "freeze_sha256": freeze_sha,
               "scope_sha256": hashlib.sha256(_canon(SCOPE)).hexdigest(),
               "out_dir": out_dir, "pid": os.getpid(),
               "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
            f.flush(); os.fsync(f.fileno())
    except Exception as e:  # noqa: BLE001
        raise Refused(f"LOCK_WRITE_FAILED:{type(e).__name__}:{e}") from e
    return path


def auth_payload(text=None, approved_by="user"):
    return {"protocol": "P11-2-S5-3/v1", "status": "APPROVED" if text else "NOT_AUTHORIZED",
            "approved_by": approved_by if text else None, "approval_text": text, "scope": SCOPE}


def auth_file(text, approved_by="user"):
    p = auth_payload(text, approved_by)
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
    if a.get("protocol") != "P11-2-S5-3/v1" or a.get("status") != "APPROVED" or a.get("approved_by") != "user":
        raise Refused("NOT_AUTHORIZED")
    if a.get("scope") != SCOPE:
        raise Refused("AUTH_SCOPE_MISMATCH")
    return a


def load_freeze():
    with open(FREEZE, encoding="utf-8") as f:
        z = json.load(f)
    if z.get("experiment_sha256") != _sha(EXPERIMENT):
        raise Refused("EXPERIMENT_FREEZE_MISMATCH")
    if z.get("tracer_sha256") != _sha(TRACER):
        raise Refused("TRACER_FREEZE_MISMATCH")
    if z.get("flybrain_source_sha256") != EXPECTED_SOURCE_SHA:
        raise Refused("FLYBRAIN_SOURCE_SHA_MISMATCH")
    if z.get("target_sha256") != TARGET_SHA or z.get("mbon_sha256") != MBON_SHA or z.get("joint_target_sha256") != JOINT_SHA:
        raise Refused("TARGET_FREEZE_MISMATCH")
    return z


def official_hashes():
    files = {k: _sha(os.path.join(DATA_DIR, k)) if os.path.exists(os.path.join(DATA_DIR, k)) else None
             for k in OFFICIAL}
    return {"ok": all(files[k] == v for k, v in OFFICIAL.items()), "files": files}


def preflight():
    auth = load_auth(); freeze = load_freeze()
    h = official_hashes()
    if not h["ok"]:
        raise Refused("OFFICIAL_HASH_MISMATCH")
    s1 = json.load(open(S1R_META, encoding="utf-8"))
    ids = np.load(os.path.join(s1["out_dir"], "ids.npz"), allow_pickle=True)
    kc = np.sort(np.asarray(ids["kc_matched"], np.int64))
    mb = np.sort(np.asarray(ids["mb_matched"], np.int64))
    joint = np.concatenate([kc, mb])
    for arr, want in ((kc, TARGET_SHA), (mb, MBON_SHA), (joint, JOINT_SHA)):
        if hashlib.sha256(arr.astype("<i8").tobytes()).hexdigest() != want:
            raise Refused("TARGET_HASH_MISMATCH")
    mod = importlib.import_module(MODULE)
    if getattr(mod, "AUTH", None) != AUTH or getattr(mod, "FREEZE", None) != FREEZE:
        raise Refused("EXPERIMENT_PATHS_MISMATCH")
    return {"auth": auth, "freeze": freeze, "official": h, "joint": joint}


def _persist_meta(out_dir, meta, path=None):
    _write_json(os.path.join(out_dir, "run_meta.json"), meta)
    _write_json(path or RUN_META, meta)


def _manifest(out_dir, extra):
    rows = []
    for name in sorted(os.listdir(out_dir)):
        if name == "run_meta.json":
            continue
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            rows.append({"name": name, "sha256": _sha(p), "size": os.path.getsize(p)})
    return {"protocol": "P11-2-S5-3/v1", "out_dir": out_dir, "files": rows, **extra}


def _base_meta(pre, run, out_dir, wall, child_path, lock_path):
    peaks = run.get("job_peaks") or {}
    return {"protocol": "P11-2-S5-3/v1", "out_dir": out_dir,
            "created": time.strftime("%Y%m%dT%H%M%S"), "lock_path": lock_path,
            "status": "INCOMPLETE_NO_GO", "run_status": "INCOMPLETE_NO_GO",
            "analysis_status": "NOT_RUN", "scientific_go": False,
            "decision": "NOT_INTERPRETED", "experiment_succeeded": False,
            "child_status": run.get("child_status"), "exit_code": run.get("exit_code"),
            "timeout": bool(run.get("timeout")), "mem_exceeded": bool(run.get("mem_exceeded")),
            "run_wall_s": round(wall, 3), "job_peak_MiB": peaks.get("PeakJobMemoryUsed_MiB"),
            "job_peaks": peaks, "log_bytes": run.get("log_bytes"),
            "log_tail": (run.get("log_tail") or "")[-2000:],
            "auth_checksum": pre["auth"]["checksum"], "freeze_sha256": pre["freeze"]["sha256"],
            "script_sha256": _sha(EXPERIMENT), "tracer_sha256": _sha(TRACER),
            "flybrain_source_sha256": EXPECTED_SOURCE_SHA,
            "target_sha256": TARGET_SHA, "mbon_sha256": MBON_SHA, "joint_target_sha256": JOINT_SHA,
            "official_hashes": pre["official"], "seeds_expected": len(SEEDS),
            "groups_expected": len(SEEDS) * len(CONDITIONS), "target_count": TARGET_COUNT,
            "steps": STEPS, "total_steps": len(SEEDS) * len(CONDITIONS) * STEPS,
            "wall_cap_s": WALL_CAP_S, "mem_cap_MiB": MEM_CAP_MIB,
            "resource_verified": False, "training_authorized": False, "dW": 0}


def _validate(out_dir, child_path, joint, auth_sha, tracer_sha):
    errors = []
    with open(child_path, encoding="utf-8") as f:
        child = json.load(f)
    expected = {
        "ok": True, "n_neurons": 166700, "n_kc": 4064, "n_mb": 97,
        "target_count": TARGET_COUNT, "target_sha256": TARGET_SHA,
        "mbon_sha256": MBON_SHA, "joint_target_sha256": JOINT_SHA,
        "groups": 15, "steps": STEPS, "device": "cpu", "batch": 1,
        "training": False, "dW": 0, "removed_edges": 642933,
        "nnz_B": 24940005, "KC_KC_nnz": 0,
    }
    for k, v in expected.items():
        if child.get(k) != v:
            errors.append(f"CHILD:{k}")
    if child.get("conditions") != CONDITIONS:
        errors.append("CHILD:CONDITIONS")
    if child.get("noise_pairing_all") is not True:
        errors.append("CHILD:NOISE_PAIRING")
    if child.get("script_sha256") != _sha(EXPERIMENT):
        errors.append("CHILD:SCRIPT_SHA")
    if child.get("auth_sha256") != auth_sha:
        errors.append("CHILD:AUTH_SHA")
    if child.get("source_sha256") not in (None, EXPECTED_SOURCE_SHA):
        errors.append("CHILD:SOURCE_SHA")
    groups = child.get("seeds", [])
    seen = set()
    rows = []
    expected_files = set()
    trace_rows = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            expected_files.add(f"{seed}_{cond}.bin")
            expected_files.add(f"{seed}_{cond}.bin.json")
    actual_files = {n for n in os.listdir(out_dir) if n.endswith(".bin") or n.endswith(".bin.json")}
    if expected_files - actual_files:
        errors.append(f"MISSING_TRACES:{sorted(expected_files - actual_files)}")
    if actual_files - expected_files:
        errors.append(f"EXTRA_TRACES:{sorted(actual_files - expected_files)}")
    for row in groups:
        try:
            seed = int(row["seed"])
        except Exception:  # noqa: BLE001
            errors.append("SEED_INVALID"); continue
        if seed not in SEEDS or seed in seen:
            errors.append(f"SEED_INVALID_OR_DUP:{seed}")
        seen.add(seed)
        for cond in CONDITIONS:
            g = row.get("groups", {}).get(cond)
            if not isinstance(g, dict):
                errors.append(f"GROUP_MISSING:{seed}:{cond}"); continue
            p = os.path.join(out_dir, f"{seed}_{cond}.bin")
            try:
                h, rows_t = read_trace(p)
                if h["target_indices"] != [int(x) for x in joint]:
                    errors.append(f"TARGET_ORDER:{seed}:{cond}")
                if rows_t.shape != (STEPS, TARGET_COUNT):
                    errors.append(f"TRACE_SHAPE:{seed}:{cond}")
                if not np.isfinite(rows_t["v_start"]).all() or not np.isfinite(rows_t["v_pre_threshold"]).all():
                    errors.append(f"NONFINITE:{seed}:{cond}")
                if not np.all((rows_t["fired"] == 0) | (rows_t["fired"] == 1)):
                    errors.append(f"FIRED_NOT_BOOL:{seed}:{cond}")
                if not np.allclose(rows_t["threshold_distance"], 1.0 - rows_t["v_pre_threshold"], atol=1e-9):
                    errors.append(f"THRESHOLD_DIST:{seed}:{cond}")
                pred = (rows_t["v_start"] * np.exp(-0.020 / 0.100) + rows_t["synaptic"]
                        + rows_t["tonic"] + rows_t["noise"] + rows_t["eye"] + rows_t["inject"])
                keep = rows_t["refractory_zeroed"] == 0
                if not np.allclose(rows_t["v_pre_threshold"][keep], pred[keep], atol=2e-6):
                    errors.append(f"COMPONENT_CLOSURE:{seed}:{cond}")
                post = rows_t["v_post_reset"]
                pre = rows_t["v_pre_threshold"]
                if not np.allclose(post, np.where(rows_t["fired"] == 1, 0.0, pre), atol=1e-7):
                    errors.append(f"POST_RESET:{seed}:{cond}")
                if g.get("trace_sha256") != h["sha256"] or g.get("trace_records") != h["records"]:
                    errors.append(f"TRACE_HASH:{seed}:{cond}")
                if g.get("finite") is not True:
                    errors.append(f"TRACE_FINITE:{seed}:{cond}")
                trace_rows.append(rows_t)
            except Exception as e:  # noqa: BLE001
                errors.append(f"TRACE_READ_FAILED:{seed}:{cond}:{type(e).__name__}:{e}")
    if seen != set(SEEDS):
        errors.append(f"SEED_SET:{sorted(set(SEEDS) - seen)}")
    # descriptive summaries; no inferential thresholds
    descriptive = {}
    for cond in CONDITIONS:
        rr = []
        for si, seed in enumerate(SEEDS):
            p = os.path.join(out_dir, f"{seed}_{cond}.bin")
            if os.path.exists(p):
                _h, rows_t = read_trace(p)
                rr.append(rows_t)
        if not rr:
            continue
        arr = np.concatenate(rr, axis=0)  # steps/seeds stacked; cells are not independent
        win = arr[STIM[0]:STIM[1]]
        fired = win["fired"].astype(bool)
        dist = win["threshold_distance"]
        descriptive[cond] = {
            "kc_mean_rate_window": round(float(fired[:, :320].mean()), 9),
            "mbon_mean_rate_window": round(float(fired[:, 320:].mean()), 9),
            "kc_pre_v_mean": round(float(win["v_pre_threshold"][:, :320].mean()), 9),
            "mbon_pre_v_mean": round(float(win["v_pre_threshold"][:, 320:].mean()), 9),
            "kc_dist_mean": round(float(dist[:, :320].mean()), 9),
            "mbon_dist_mean": round(float(dist[:, 320:].mean()), 9),
            "kc_frac_dist_lt_005": round(float((dist[:, :320] < .05).mean()), 9),
            "kc_frac_dist_lt_010": round(float((dist[:, :320] < .10).mean()), 9),
            "kc_frac_dist_lt_020": round(float((dist[:, :320] < .20).mean()), 9),
            "synaptic_mean": round(float(win["synaptic"].mean()), 9),
            "tonic_mean": round(float(win["tonic"].mean()), 9),
            "noise_mean": round(float(win["noise"].mean()), 9),
            "inject_mean": round(float(win["inject"].mean()), 9),
            "fired_count_window": int(fired.sum()),
            "reset_count_window": int((win["fired"] == 1).sum()),
        }
    return {"ok": not errors, "errors": errors, "child": child,
            "files": [{"name": n, "sha256": _sha(os.path.join(out_dir, n)),
                       "size": os.path.getsize(os.path.join(out_dir, n))}
                      for n in sorted(actual_files)],
            "descriptive": descriptive, "trace_count": len(trace_rows)}


def _failure_report(status, reason, meta, raw, child):
    return {"protocol": "P11-2-S5-3/v1", "status": status, "run_status": "INCOMPLETE_NO_GO",
            "analysis_status": "NOT_RUN", "scientific_go": False, "decision": "NOT_INTERPRETED",
            "run_meta": meta, "child": child, "raw": raw, "reason": reason,
            "training_authorized": False}


def _try_write(path, obj):
    try:
        _write_json(path, obj); return True
    except Exception:  # noqa: BLE001
        return False


def _write_failure(out_dir, report_path, manifest_path, status, reason, meta, raw, child):
    rep = _failure_report(status, reason, meta, raw, child)
    man = _manifest(out_dir, {"status": status, "run_status": "INCOMPLETE_NO_GO",
                              "scientific_go": False, "run_meta_sha256": _json_sha(meta)})
    return _try_write(report_path, rep) and _try_write(manifest_path, man)


def postprocess(out_dir, run, meta, pre, child_path, report_path=None, manifest_path=None,
                run_meta_path=None, exp=None):
    report_path = report_path or REPORT
    manifest_path = manifest_path or RAW_MANIFEST
    exp = exp or importlib.import_module(MODULE)
    raw = _validate(out_dir, child_path, pre["joint"], pre["auth"]["checksum"], _sha(TRACER))
    meta.update({"process_finished": bool(run.get("process_finished")),
                 "raw_ok": bool(raw["ok"]), "raw_errors": raw["errors"][:20],
                 "raw_files": raw["files"], "experiment_succeeded": False,
                 "scientific_go": False, "decision": "NOT_INTERPRETED",
                 "analysis_status": "NOT_RUN", "status": "INCOMPLETE_NO_GO",
                 "run_status": "INCOMPLETE_NO_GO"})
    _persist_meta(out_dir, meta, run_meta_path)
    if not (run.get("child_status") == "CHILD_EXIT_OK" and raw["ok"]):
        reason = (raw["errors"][0] if raw["errors"] else f"RUN_STATUS:{run.get('child_status')}")
        if not _write_failure(out_dir, report_path, manifest_path, "INCOMPLETE_NO_GO",
                              reason, meta, raw, raw["child"]):
            meta["status"] = "PERSISTENCE_FAILED"; _persist_meta(out_dir, meta, run_meta_path)
        return meta
    meta_final = copy.deepcopy(meta)
    meta_final.update({"status": "RUN_COMPLETE", "run_status": "RUN_COMPLETE",
                       "analysis_status": "DESCRIPTIVE_COMPLETE",
                       "decision": "MECHANISM_OBSERVATION_DESCRIPTIVE",
                       "scientific_go": False, "experiment_succeeded": True})
    rep = {"protocol": "P11-2-S5-3/v1", "status": "RUN_COMPLETE",
           "analysis_status": "DESCRIPTIVE_COMPLETE",
           "decision": "MECHANISM_OBSERVATION_DESCRIPTIVE",
           "scientific_go": False, "no_new_significance": True,
           "run_meta": meta_final, "child": raw["child"],
           "descriptive": raw["descriptive"], "trace_count": raw["trace_count"],
           "note": "5 seeds, 415 cells, 400 steps; cell/time points are not independent seeds; no biological inference."}
    manifest = _manifest(out_dir, {"status": "RUN_COMPLETE", "analysis_status": "DESCRIPTIVE_COMPLETE",
                                   "scientific_go": False, "run_meta_sha256": _json_sha(meta_final),
                                   "auth_checksum": meta_final["auth_checksum"],
                                   "tracer_sha256": meta_final["tracer_sha256"]})
    try:
        _write_json(report_path, rep); _write_json(manifest_path, manifest)
    except Exception as e:  # noqa: BLE001
        meta["status"] = "PERSISTENCE_FAILED"; meta["run_status"] = "INCOMPLETE_NO_GO"
        meta["persistence_error"] = f"{type(e).__name__}:{e}"
        _persist_meta(out_dir, meta, run_meta_path)
        _write_failure(out_dir, report_path, manifest_path, "PERSISTENCE_FAILED",
                       meta["persistence_error"], meta, raw, raw["child"])
        return meta
    try:
        _persist_meta(out_dir, meta_final, run_meta_path)
    except Exception as e:  # noqa: BLE001
        meta["status"] = "PERSISTENCE_FAILED"; meta["run_status"] = "INCOMPLETE_NO_GO"
        meta["persistence_error"] = f"{type(e).__name__}:{e}"
        _write_failure(out_dir, report_path, manifest_path, "PERSISTENCE_FAILED",
                       meta["persistence_error"], meta, raw, raw["child"])
        return meta
    return meta_final


def run_once(out_dir=None):
    pre = preflight()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.abspath(out_dir or os.path.join(tempfile.gettempdir(), "opencode", "p11_2_s5_3_full", stamp))
    if os.path.exists(out_dir):
        raise Refused("OUT_DIR_EXISTS")
    lock = acquire_one_shot(pre["auth"]["checksum"], out_dir=out_dir, freeze_sha=pre["freeze"]["sha256"])
    os.makedirs(out_dir, exist_ok=False)
    for src in (AUTH, FREEZE):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))
    job_name = f"Local\\P11_2_S5_3_{os.getpid()}_{time.time_ns()}"
    child_path = os.path.join(out_dir, "child_result.json")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["P11_2_S5_3_JOB_NAME"] = job_name
    t0 = time.perf_counter()
    run = run_under_job_safe(MEM_CAP_BYTES, module=MODULE, args=["--child", out_dir], cwd=ROOT,
                             env=env, wall_cap_s=WALL_CAP_S, mem_cap_mib=MEM_CAP_MIB,
                             log_path=os.path.join(out_dir, "experiment.log"),
                             required_artifact=child_path, job_name=job_name)
    wall = time.perf_counter() - t0
    meta = _base_meta(pre, run, out_dir, wall, child_path, lock)
    meta["job_name"] = job_name
    _persist_meta(out_dir, meta)
    return postprocess(out_dir, run, meta, pre, child_path)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--execute" in argv:
            meta = run_once()
            print("RUN_COMPLETE" if meta["status"] == "RUN_COMPLETE" else meta["status"])
            return 0 if meta["status"] == "RUN_COMPLETE" else 5
        pre = preflight()
        print(json.dumps({"status": "PREFLIGHT_OK_NOT_RUN", "targets": TARGET_COUNT,
                          "official_hashes_ok": pre["official"]["ok"]}, ensure_ascii=False))
        return 0
    except Refused as e:
        print(json.dumps({"status": "REFUSED", "reason": str(e),
                          "new_s5_3_execution_authorization": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
