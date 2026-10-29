"""Held-out D2 evaluation and frozen gates. No brain, no motor."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from demo.brain_turn_d2 import gates, protocol, readout


def load_val_runs(root: str | Path) -> list[dict]:
    runs = []
    for seed in protocol.VAL_SEEDS:
        for repeat in range(protocol.REPEATS):
            for scene in protocol.SCENES:
                p = Path(root) / protocol.run_name(seed, protocol.VAL_DISTANCE_MM, scene, repeat)
                if not p.is_file():
                    raise FileNotFoundError(p)
                z = np.load(p, allow_pickle=False)
                if int(np.asarray(z["distance_mm"]).ravel()[0]) != protocol.VAL_DISTANCE_MM:
                    raise ValueError(f"{p}: wrong distance")
                runs.append({
                    "path": str(p), "seed": seed, "scene": scene, "repeat": repeat,
                    "features": np.asarray(z["features"], np.float32),
                    "t": np.asarray(z["t"], np.float64),
                })
    return runs


def _mean_window(values: np.ndarray, t: np.ndarray, window: tuple[float, float]) -> float:
    m = (t >= window[0]) & (t < window[1])
    if not np.any(m):
        raise ValueError("empty window")
    return float(values[m].mean())


def evaluate(root: str | Path, model_path: str | Path) -> dict:
    model = readout.load_model(model_path)
    runs = load_val_runs(root)
    for r in runs:
        r["turn"] = readout.predict_turn(r["features"], model)
        r["side_mean"] = _mean_window(r["turn"], r["t"], protocol.SIDE_WINDOW)
        r["recovery_mean"] = _mean_window(r["turn"], r["t"], protocol.RECOVERY_WINDOW)
    by_seed_scene: dict[tuple[int, str], list[float]] = {}
    for r in runs:
        by_seed_scene.setdefault((r["seed"], r["scene"]), []).append(r["side_mean"])
    open_means = [float(np.mean(by_seed_scene[(s, "A_open")])) for s in protocol.VAL_SEEDS]
    fl_means = [float(np.mean(by_seed_scene[(s, "B_FL_only")])) for s in protocol.VAL_SEEDS]
    fr_means = [float(np.mean(by_seed_scene[(s, "C_FR_only")])) for s in protocol.VAL_SEEDS]
    recovery_means = []
    for s in protocol.VAL_SEEDS:
        vals = [r["recovery_mean"] for r in runs
                if r["seed"] == s and r["scene"] in ("B_FL_only", "C_FR_only", "D_FL_FR_symmetric")]
        recovery_means.append(float(np.mean(vals)))

    def direction_accuracy(rs: list[dict]) -> float:
        good = []
        for r in rs:
            if r["scene"] == "B_FL_only":
                good.append(r["side_mean"] < 0)
            elif r["scene"] == "C_FR_only":
                good.append(r["side_mean"] > 0)
        return float(np.mean(good)) if good else float("nan")

    original_accuracy = direction_accuracy(runs)
    all_x = np.vstack([r["features"] for r in runs])
    lengths = [len(r["features"]) for r in runs]
    rng = np.random.default_rng(protocol.RANDOM_SEED)
    shuffled = all_x[rng.permutation(len(all_x))]
    starts = np.cumsum([0] + lengths[:-1])
    shuffled_runs = []
    for r, start, n in zip(runs, starts, lengths):
        rr = dict(r)
        x = shuffled[start:start + n]
        rr["turn"] = readout.predict_turn(x, model)
        rr["side_mean"] = _mean_window(rr["turn"], rr["t"], protocol.SIDE_WINDOW)
        shuffled_runs.append(rr)
    shuffled_accuracy = direction_accuracy(shuffled_runs)
    result = gates.evaluate_validation(
        open_means=open_means, fl_means=fl_means, fr_means=fr_means,
        recovery_means=recovery_means, original_accuracy=original_accuracy,
        shuffled_accuracy=shuffled_accuracy, zero_directional=True,
    )
    return {
        "protocol": protocol.PROTOCOL,
        "status": "EVALUATED",
        "verdict": "PASS" if result["pass_all"] else "FAIL",
        "gates": result,
        "per_seed": {
            "seeds": list(protocol.VAL_SEEDS),
            "open_means": open_means, "fl_means": fl_means, "fr_means": fr_means,
            "recovery_means": recovery_means,
        },
        "shuffle": {"original_accuracy": original_accuracy,
                    "shuffled_accuracy": shuffled_accuracy},
        "runs": [{k: r[k] for k in ("seed", "scene", "repeat", "side_mean",
                                    "recovery_mean", "path")} for r in runs],
        "model": str(model_path),
        "motor": False, "planner": False, "teacher_at_inference": False,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D2 held-out evaluation (no brain)")
    ap.add_argument("--runs", default=str(protocol.RUNS_DIR))
    ap.add_argument("--model", default=str(protocol.HERE / "D2_CANDIDATE.npz"))
    ap.add_argument("--out", default=str(protocol.REPORT))
    args = ap.parse_args(argv)
    try:
        protocol.require_authorized()
    except protocol.Refused as exc:
        print(f"[d2-eval] REFUSED: {type(exc).__name__}:{exc}")
        return 2
    report = evaluate(args.runs, args.model)
    protocol.write_json(args.out, report)
    print(f"[d2-eval] {report['verdict']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
