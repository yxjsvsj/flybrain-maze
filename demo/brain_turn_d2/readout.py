"""Zero-bias linear DN readout: fitting, prediction, save/load.

Training is external to the connectome. There is no ToF, teacher, map, or
planner input at prediction time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def fit_zero_bias(features: np.ndarray, target: np.ndarray,
                  lam: float = 0.01) -> dict[str, np.ndarray]:
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("features must be (n, d) and target (n,)")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("non-finite fit data")
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + float(lam) * X.shape[0] * np.eye(X.shape[1])
    w = np.linalg.solve(A, Z.T @ y)
    return {"mean": mu.astype(np.float32), "scale": sd.astype(np.float32),
            "weights": w.astype(np.float32), "lambda": np.asarray(lam)}


def predict_turn(features: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    X = np.asarray(features, dtype=np.float64)
    mu = np.asarray(model["mean"], dtype=np.float64)
    sd = np.asarray(model["scale"], dtype=np.float64)
    w = np.asarray(model["weights"], dtype=np.float64)
    if X.shape[-1] != mu.shape[0] or sd.shape != mu.shape or w.shape != mu.shape:
        raise ValueError("feature dimension mismatch")
    return np.clip(((X - mu) / sd) @ w, -1.0, 1.0)


def save_model(path: str | Path, model: dict[str, np.ndarray]) -> None:
    np.savez(path, mean=model["mean"], scale=model["scale"],
             weights=model["weights"], lambda_=np.asarray(model["lambda"]),
             kind=np.asarray("zero_bias_linear_ridge"))


def load_model(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {"mean": np.asarray(z["mean"]), "scale": np.asarray(z["scale"]),
            "weights": np.asarray(z["weights"]), "lambda": float(z["lambda_"])}
