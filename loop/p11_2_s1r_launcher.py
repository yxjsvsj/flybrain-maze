"""P11-2-S1R-0：未来一次性 S1R 的唯一生产启动入口。

只做静态集成/合成测试；未获独立 S1R 授权时 fail-closed，启动完整脑之前即拒绝。
子进程固定使用 `python -u -m loop.p11_2_s1r_experiment --child <out_dir>`，
cwd=仓库根，日志/产物/退出状态均由 S1F 安全 Job 启动器负责。
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
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
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
SEEDS = list(range(101, 126))
STEPS = 400
STIM = (100, 300)
AMOUNT = 0.036254
DV = 0.2
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


def validate_child_result(path, expected_seeds=25):
    if not os.path.exists(path):
        return {"path": path, "exists": False, "complete": False, "seeds_observed": 0,
                "noise_pairing_all": None, "reason": "CHILD_OUTPUT_MISSING"}
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:             # noqa: BLE001
        return {"path": path, "exists": True, "complete": False, "seeds_observed": 0,
                "noise_pairing_all": None, "reason": f"CHILD_RESULT_INVALID:{type(e).__name__}"}
    seeds = obj.get("seeds")
    n = len(seeds) if isinstance(seeds, list) else 0
    complete = n == expected_seeds and obj.get("noise_pairing_all") is True
    reason = None if complete else (
        "SEED_COUNT_SHORT" if n != expected_seeds else "NOISE_PAIRING_NOT_ALL_TRUE")
    return {"path": path, "exists": True, "complete": bool(complete), "seeds_observed": n,
            "noise_pairing_all": obj.get("noise_pairing_all"), "reason": reason}


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


def _base_meta(pre, run, out_dir, wall_s, child_path):
    peaks = run.get("job_peaks") or {}
    return {
        "protocol": "P11-2-S1R/v1", "out_dir": out_dir, "created": time.strftime("%Y%m%dT%H%M%S"),
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


def _persist_run_meta(out_dir, meta):
    # 必须先于任何统计后处理持久化
    _write_json(os.path.join(out_dir, "run_meta.json"), meta)
    _write_json(RUN_META, meta)


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
                      required_artifact=None, inherit_log=True):
    return run_under_job_safe(limit_mib * 1024 ** 2, module=module, args=args, cwd=cwd,
                              env=_env(), wall_cap_s=wall_s, mem_cap_mib=limit_mib,
                              log_path=os.path.join(out_dir, "experiment.log"),
                              inherit_log=inherit_log, required_artifact=required_artifact)


def run_s1r_once(out_dir=None):
    """唯一生产运行路径；调用前必须存在独立 APPROVED S1R 授权。"""
    pre = preflight()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.abspath(out_dir or os.path.join(
        tempfile.gettempdir(), "opencode", "p11_2_s1r_full", stamp))
    if os.path.exists(out_dir):
        raise Refused("OUT_DIR_EXISTS")
    os.makedirs(out_dir, exist_ok=False)
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
    t0 = time.perf_counter()
    run = _run_child_module(EXPERIMENT_MODULE, ["--child", out_dir], ROOT, out_dir,
                            limit_mib=MEM_CAP_MIB, wall_s=WALL_CAP_S,
                            required_artifact=child_path)
    wall = time.perf_counter() - t0
    meta = _base_meta(pre, run, out_dir, wall, child_path)
    _persist_run_meta(out_dir, meta)       # 先写入运行元数据
    child = validate_child_result(child_path)
    ev = evaluate_run(run, child)
    meta.update(ev)
    meta["status"] = "RUN_COMPLETE" if ev["experiment_succeeded"] else "INCOMPLETE_NO_GO"
    _persist_run_meta(out_dir, meta)       # 仍在统计之前
    child_raw = None
    if child["exists"]:
        try:
            with open(child_path, encoding="utf-8") as f:
                child_raw = json.load(f)
        except Exception:                  # noqa: BLE001
            child_raw = None
    if ev["experiment_succeeded"]:
        rep = exp.analyze(out_dir)
        rep["run_meta"] = meta
        rep["child"] = child_raw
        rep["complete_25_pairs"] = True
        rep["gate"] = {"Training_Authorization": False, "REAL_SYNAPTIC_LEARNING_READY": "NO",
                       "Level_A/B/C/Biological": "NO-GO", "A1_A2_A3/B": "NO-GO",
                       "DeltaW": 0, "Dijkstra": "NO", "Maze": "NO", "B/C": "NO", "Robot": "NO"}
    else:
        rep = {"protocol": "P11-2-S1R/v1", "status": "INCOMPLETE_NO_GO",
               "completeness": f"{child['seeds_observed']}/{len(SEEDS)}",
               "complete_25_pairs": False, "run_meta": meta, "child": child_raw,
               "reason": child.get("reason") or meta.get("child_status"),
               "training_authorized": False}
    _write_json(REPORT, rep)
    manifest = _manifest(out_dir, {"status": rep["status"], "script_sha256": meta["script_sha256"],
                                   "authorization_sha256": meta["auth_checksum"],
                                   "run_meta_sha256": _sha(os.path.join(out_dir, "run_meta.json")),
                                   "official_hashes": pre["official"]})
    _write_json(RAW_MANIFEST, manifest)
    return meta


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
