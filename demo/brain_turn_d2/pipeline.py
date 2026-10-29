"""D2 one-shot parent: auth -> freeze -> lock -> static collect -> external fit -> held-out gates.

Default is fail-closed: without explicit user approval it prints REFUSED and
does not load a brain. No motors, no maze, no planner.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from demo.brain_turn_d2 import evaluate, protocol, readout, train


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D2 one-shot pipeline (not authorized by default)")
    ap.add_argument("--frames-root", default=str(protocol.HERE / "D2_FRAMES"))
    ap.add_argument("--out-dir", default=str(protocol.RUNS_DIR))
    ap.add_argument("--auth", default=str(protocol.AUTH))
    ap.add_argument("--run-meta", default=str(protocol.HERE / "D2_RUN_META.json"))
    args = ap.parse_args(argv)
    try:
        auth = protocol.load_auth(args.auth)
        protocol.load_freeze()
    except protocol.Refused as exc:
        print(f"REFUSED/NOT_AUTHORIZED: {type(exc).__name__}:{exc}")
        return 2
    if not Path(args.frames_root).is_dir():
        print(f"REFUSED/FRAMES_MISSING: {args.frames_root}")
        return 2
    try:
        lock = protocol.acquire_lock(auth["checksum"], args.out_dir)
    except protocol.Refused as exc:
        print(f"REFUSED: {type(exc).__name__}:{exc}")
        return 2

    meta = {
        "protocol": protocol.PROTOCOL,
        "status": "INCOMPLETE_NO_GO",
        "auth_checksum": auth["checksum"],
        "lock": lock,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": protocol.RUNS,
        "total_steps": protocol.TOTAL_STEPS,
        "motor": False,
        "planner": False,
        "training": "external_linear_only",
        "synaptic_dW": 0.0,
        "auto_retry": False,
    }
    try:
        from loop.p11_2_s1f_safe_job_launcher import run_under_job_safe
        child = run_under_job_safe(
            protocol.MEM_CAP_MIB * 1024 ** 2,
            module="demo.brain_turn_d2.collect",
            args=["--frames-root", args.frames_root, "--out-dir", args.out_dir,
                  "--auth", args.auth],
            wall_cap_s=protocol.WALL_CAP_S,
            mem_cap_mib=protocol.MEM_CAP_MIB,
            log_path=str(protocol.HERE / "D2_COLLECT.log"),
            cwd=str(protocol.ROOT),
            required_artifact=str(Path(args.out_dir) / "runs_manifest.json"),
        )
        meta["child"] = child
        if not child.get("experiment_succeeded"):
            meta["reason"] = "COLLECT_FAILED"
            protocol.write_json(args.run_meta, meta)
            return 3
        fit = train.fit_from_runs(args.out_dir)
        candidate = protocol.HERE / "D2_CANDIDATE.npz"
        candidate_meta = protocol.HERE / "D2_CANDIDATE.json"
        readout.save_model(candidate, fit["model"])
        protocol.write_json(candidate_meta, fit["meta"])
        report = evaluate.evaluate(args.out_dir, candidate)
        protocol.write_json(protocol.REPORT, report)
        meta["status"] = "EVALUATED"
        meta["verdict"] = report["verdict"]
        protocol.write_json(args.run_meta, meta)
        print(f"[d2-pipeline] {report['verdict']} -> {protocol.REPORT}")
        return 0 if report["verdict"] == "PASS" else 4
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "INCOMPLETE_NO_GO"
        meta["reason"] = f"{type(exc).__name__}:{exc}"
        protocol.write_json(args.run_meta, meta)
        print(f"[d2-pipeline] FAILED: {meta['reason']}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
