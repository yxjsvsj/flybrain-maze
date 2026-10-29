"""Fit the frozen external zero-bias linear D2 readout. No brain, no motor."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from demo.brain_turn_d2 import protocol, readout


def load_train_runs(root: str | Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X, y, meta = [], [], []
    for seed in protocol.TRAIN_SEEDS:
        for repeat in range(protocol.REPEATS):
            for scene in protocol.SCENES:
                p = Path(root) / protocol.run_name(seed, protocol.TRAIN_DISTANCE_MM, scene, repeat)
                if not p.is_file():
                    raise FileNotFoundError(p)
                z = np.load(p, allow_pickle=False)
                if int(np.asarray(z["distance_mm"]).ravel()[0]) != protocol.TRAIN_DISTANCE_MM:
                    raise ValueError(f"{p}: wrong distance")
                X.append(np.asarray(z["features"], np.float32))
                y.append(np.asarray(z["teacher"], np.float32))
                meta.append({"path": str(p), "seed": seed, "scene": scene, "repeat": repeat})
    return np.vstack(X), np.concatenate(y), meta


def fit_from_runs(root: str | Path) -> dict:
    X, y, meta = load_train_runs(root)
    model = readout.fit_zero_bias(X, y, lam=protocol.RIDGE_LAMBDA)
    model_meta = {
        "protocol": protocol.PROTOCOL,
        "kind": "zero_bias_linear_ridge",
        "train_runs": len(meta),
        "train_samples": int(len(y)),
        "features": int(X.shape[1]),
        "lambda": protocol.RIDGE_LAMBDA,
        "fit_intercept": False,
        "teacher_used_at_inference": False,
        "source_runs": meta,
    }
    return {"model": model, "meta": model_meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D2 external readout fit (no brain)")
    ap.add_argument("--runs", default=str(protocol.RUNS_DIR))
    ap.add_argument("--out", default=str(protocol.HERE / "D2_CANDIDATE.npz"))
    ap.add_argument("--meta", default=str(protocol.HERE / "D2_CANDIDATE.json"))
    args = ap.parse_args(argv)
    try:
        protocol.require_authorized()
    except protocol.Refused as exc:
        print(f"[d2-train] REFUSED: {type(exc).__name__}:{exc}")
        return 2
    result = fit_from_runs(args.runs)
    readout.save_model(args.out, result["model"])
    protocol.write_json(args.meta, result["meta"])
    print(f"[d2-train] wrote {args.out} / {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
