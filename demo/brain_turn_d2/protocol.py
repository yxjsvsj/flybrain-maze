"""D2 protocol constants, frozen scope, authorization, and run locks."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUTH = HERE / "D2_AUTHORIZATION.json"
FREEZE = HERE / "D2_SCRIPT_FREEZE.json"
LOCK_DIR = HERE / "D2_AUTHORIZATION_LOCKS"
RUNS_DIR = HERE / "D2_RUNS"
REPORT = HERE / "D2_EVAL.json"
TRAIN_SEEDS = (3001, 3002, 3003)
VAL_SEEDS = (3051, 3052)
REPEATS = 3
DISTANCES_MM = (400, 700)
TRAIN_DISTANCE_MM = 400
VAL_DISTANCE_MM = 700
SCENES = ("A_open", "B_FL_only", "C_FR_only", "D_FL_FR_symmetric")
SECONDS = 10.0
TOF_HZ = 10.0
BRAIN_HZ = 50.0
STEPS = int(round(SECONDS * BRAIN_HZ))
TOF_FRAMES = int(round(SECONDS * TOF_HZ))
SIDE_WINDOW = (2.0, 6.0)
RECOVERY_WINDOW = (6.0, 7.0)
RUNS = len(SCENES) * len(DISTANCES_MM) * REPEATS * (len(TRAIN_SEEDS) + len(VAL_SEEDS))
TOTAL_STEPS = RUNS * STEPS
TRACE_TAU = 0.2
RIDGE_LAMBDA = 0.01
RANDOM_SEED = 20261022
WALL_CAP_S = 1800.0
MEM_CAP_MIB = 2048
PROTOCOL = "DEMO-BRAIN-TURN-D2/v1"
SCOPE = {
    "protocol": PROTOCOL,
    "scenes": list(SCENES),
    "distances_mm": list(DISTANCES_MM),
    "train_distance_mm": TRAIN_DISTANCE_MM,
    "val_distance_mm": VAL_DISTANCE_MM,
    "train_seeds": list(TRAIN_SEEDS),
    "val_seeds": list(VAL_SEEDS),
    "repeats": REPEATS,
    "seconds": SECONDS,
    "tof_hz": TOF_HZ,
    "brain_hz": BRAIN_HZ,
    "steps_per_run": STEPS,
    "tof_frames_per_run": TOF_FRAMES,
    "runs": RUNS,
    "total_steps": TOTAL_STEPS,
    "trace_tau": TRACE_TAU,
    "ridge_lambda": RIDGE_LAMBDA,
    "random_seed": RANDOM_SEED,
    "device": "cpu",
    "batch": 1,
    "motor": False,
    "training": "external_linear_only",
    "synaptic_dW": 0.0,
    "wall_cap_s": WALL_CAP_S,
    "mem_cap_MiB": MEM_CAP_MIB,
}


class Refused(RuntimeError):
    pass


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def auth_payload(approval_text: str | None = None, approved_by: str | None = None) -> dict:
    return {
        "protocol": PROTOCOL,
        "status": "APPROVED" if approval_text else "NOT_AUTHORIZED",
        "approved_by": approved_by if approval_text else None,
        "approval_text": approval_text,
        "scope": SCOPE,
    }


def auth_with_checksum(approval_text: str | None = None, approved_by: str | None = None) -> dict:
    payload = auth_payload(approval_text, approved_by)
    payload["checksum"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def load_auth(path: str | Path = AUTH) -> dict:
    p = Path(path)
    if not p.is_file():
        raise Refused("AUTH_MISSING")
    data = json.loads(p.read_text(encoding="utf-8"))
    checksum = data.pop("checksum", None)
    if checksum != hashlib.sha256(canonical(data)).hexdigest():
        raise Refused("AUTH_CHECKSUM_MISMATCH")
    data["checksum"] = checksum
    if data.get("protocol") != PROTOCOL:
        raise Refused("AUTH_PROTOCOL_MISMATCH")
    if data.get("status") != "APPROVED" or data.get("approved_by") != "user":
        raise Refused("NOT_AUTHORIZED")
    if data.get("scope") != SCOPE:
        raise Refused("AUTH_SCOPE_MISMATCH")
    return data


def load_freeze(path: str | Path = FREEZE) -> dict:
    p = Path(path)
    if not p.is_file():
        raise Refused("FREEZE_MISSING")
    data = json.loads(p.read_text(encoding="utf-8"))
    for name, expected in (data.get("scripts") or {}).items():
        q = HERE / name
        if not q.is_file() or sha256(q) != expected:
            raise Refused(f"FREEZE_MISMATCH:{name}")
    return data


def lock_path(auth_checksum: str) -> str:
    return str(LOCK_DIR / f"{auth_checksum}.consumed.json")


def acquire_lock(auth_checksum: str, out_dir: str | Path) -> str:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = lock_path(auth_checksum)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise Refused("AUTHORIZATION_ALREADY_CONSUMED") from exc
    except OSError as exc:
        raise Refused(f"LOCK_CREATE_FAILED:{type(exc).__name__}:{exc}") from exc
    payload = {
        "protocol": PROTOCOL,
        "status": "CONSUMED_STARTED",
        "auth_checksum": auth_checksum,
        "out_dir": str(out_dir),
        "pid": os.getpid(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    return path


def run_name(seed: int, distance_mm: int, scene: str, repeat: int) -> str:
    return f"{seed}_{distance_mm}mm_{scene}_r{repeat}.npz"


def require_authorized() -> dict:
    auth = load_auth()
    load_freeze()
    return auth


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)
