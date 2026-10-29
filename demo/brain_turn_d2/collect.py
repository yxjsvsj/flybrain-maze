"""D2 full-brain static collection: ToF frame packs -> MaleCNS -> 1314 DN Trace.

No motor, no training, no planner. Refuses unless the frozen D2 authorization
is APPROVED. Intended to run inside the existing named-Job launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from demo.brain_turn_d2 import frames, protocol


def _sha(path: Path) -> str:
    return protocol.sha256(path)


def collect(frames_root: str | Path, out_dir: str | Path, auth_path: str | Path) -> dict:
    auth = protocol.load_auth(auth_path)
    protocol.load_freeze()
    root = Path(frames_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    packs = {}
    for dist in protocol.DISTANCES_MM:
        for scene in protocol.SCENES:
            packs[(scene, dist)] = frames.load_pack(root, scene, dist)
            if len(packs[(scene, dist)]["t"]) < protocol.TOF_FRAMES:
                raise ValueError(f"{scene} {dist}: need {protocol.TOF_FRAMES} frames")

    from flybrain import Trace
    from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
    from loop.encode_tof import PhysicalToFEncoder
    from loop.run import make_setup

    cfg, _maze, _car, brain, groups, _enc, _dec = make_setup(
        brain_kind="real", map_name="gen", dt=None, device="cpu", seed=protocol.TRAIN_SEEDS[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]), dtype=np.int64)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=0.40)
    dt = float(brain.dt)
    steps = protocol.STEPS
    side_a, side_b = protocol.SIDE_WINDOW
    rec_a, rec_b = protocol.RECOVERY_WINDOW
    manifest = []
    t_start = time.perf_counter()
    for seed in protocol.TRAIN_SEEDS + protocol.VAL_SEEDS:
        for dist in protocol.DISTANCES_MM:
            for scene in protocol.SCENES:
                pack = packs[(scene, dist)]
                for repeat in range(protocol.REPEATS):
                    name = protocol.run_name(seed, dist, scene, repeat)
                    path = out / name
                    feats = np.empty((steps, len(dn_idx)), np.float32)
                    teacher = np.empty(steps, np.float32)
                    times = np.empty(steps, np.float64)
                    brain.reset(seed)
                    trace = Trace(brain, idx=dn_idx, tau=protocol.TRACE_TAU)
                    for k in range(steps):
                        t = k * dt
                        fi = min(int(round(t * protocol.TOF_HZ)), len(pack["t"]) - 1)
                        fr = frames.make_frame(pack, fi)
                        d = enc.sense(fr)
                        enc.update(fr, d)
                        inj, info = enc.inject()
                        fired = brain.step(inject=inj)
                        feats[k] = np.asarray(trace.observe(fired), dtype=np.float32)
                        teacher[k] = float(np.clip(3.0 * (info.get("prox_r", 0.0)
                                                         - info.get("prox_l", 0.0)), -1.0, 1.0))
                        times[k] = t
                    np.savez_compressed(
                        path, features=feats, teacher=teacher, t=times,
                        seed=np.asarray(seed), distance_mm=np.asarray(dist),
                        scene=np.asarray(scene), repeat=np.asarray(repeat),
                        side_window=np.asarray(protocol.SIDE_WINDOW),
                        recovery_window=np.asarray(protocol.RECOVERY_WINDOW),
                        brain_dt=np.asarray(dt), trace_tau=np.asarray(protocol.TRACE_TAU),
                    )
                    manifest.append({"name": name, "sha256": _sha(path),
                                     "bytes": path.stat().st_size,
                                     "seed": seed, "distance_mm": dist,
                                     "scene": scene, "repeat": repeat})
                    print(f"[d2-collect] {name} {feats.shape} {path.stat().st_size} bytes", flush=True)
    meta = {
        "protocol": protocol.PROTOCOL,
        "status": "COLLECTED_NOT_INTERPRETED",
        "auth_checksum": auth["checksum"],
        "frames_root": str(root),
        "out_dir": str(out),
        "runs": len(manifest),
        "steps_per_run": steps,
        "total_steps": len(manifest) * steps,
        "dn_features": int(len(dn_idx)),
        "device": "cpu", "batch": 1, "motor": False,
        "training": False, "synaptic_dW": 0.0,
        "wall_s": round(time.perf_counter() - t_start, 3),
        "manifest": manifest,
    }
    protocol.write_json(out / "runs_manifest.json", meta)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D2 static data collection (full brain; no motor)")
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out-dir", default=str(protocol.RUNS_DIR))
    ap.add_argument("--auth", default=str(protocol.AUTH))
    args = ap.parse_args(argv)
    try:
        meta = collect(args.frames_root, args.out_dir, args.auth)
    except (protocol.Refused, FileNotFoundError, ValueError) as exc:
        print(f"[d2-collect] REFUSED: {type(exc).__name__}:{exc}")
        return 2
    print(f"[d2-collect] collected {meta['runs']} runs -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
