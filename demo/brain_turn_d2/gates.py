"""Frozen D2 gates and teacher contract.

These functions are offline design/synthetic-test helpers. They do not load a
brain, train a readout, touch a robot, or produce actuator commands.
"""
from __future__ import annotations

from typing import Any

import numpy as np

OPEN_MAX_ABS_MEAN = 0.05
SIDE_MIN_ABS_MEAN = 0.10
SIDE_MARGIN_MIN = 0.20
RECOVERY_MAX_ABS_MEAN = 0.05
MIN_SEED_FRACTION = 0.80
ORIGINAL_MIN_ACC = 0.75
SHUFFLE_MAX_ACC = 0.55
TEACHER_GAIN = 3.0


def teacher_turn(prox_l: float, prox_r: float, gain: float = TEACHER_GAIN) -> float:
    """Engineering teacher only: positive = turn left.

    prox = 1 - dmin/max_range. The teacher is used only while collecting
    training labels; it is never passed to inference.
    """
    return float(np.clip(gain * (float(prox_r) - float(prox_l)), -1.0, 1.0))


def turn_from_dn(features: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                 weights: np.ndarray) -> float:
    """Inference contract: DN features are the only input.

    No ToF, teacher, map, target, or motor state is accepted by this function.
    The zero-bias readout is deliberate: no temporary offset subtraction or
    sign flip is applied afterwards.
    """
    x = np.asarray(features, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(scale, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or mu.shape != x.shape or sd.shape != x.shape or w.shape != x.shape:
        raise ValueError("features/mean/scale/weights must be equal 1-D shapes")
    z = (x - mu) / sd
    return float(np.clip(float(z @ w), -1.0, 1.0))


def evaluate_validation(open_means, fl_means, fr_means, recovery_means,
                        original_accuracy: float, shuffled_accuracy: float,
                        zero_directional: bool) -> dict[str, Any]:
    """Evaluate the pre-registered gates on validation-only summaries.

    Inputs are per-seed means, not frame-level pseudo-replicates. This helper
    is used by synthetic tests; it never creates a success result by itself.
    """
    arrays = {
        "open": np.asarray(open_means, dtype=np.float64),
        "fl": np.asarray(fl_means, dtype=np.float64),
        "fr": np.asarray(fr_means, dtype=np.float64),
        "recovery": np.asarray(recovery_means, dtype=np.float64),
    }
    for name, arr in arrays.items():
        if arr.ndim != 1 or len(arr) == 0:
            raise ValueError(f"{name}_means must be a non-empty 1-D sequence")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}_means contains non-finite values")
    if not (len(arrays["open"]) == len(arrays["fl"]) == len(arrays["fr"]) == len(arrays["recovery"])):
        raise ValueError("all per-seed mean arrays must have the same length")

    open_max = float(np.max(np.abs(arrays["open"])))
    fl_mean = float(np.mean(arrays["fl"]))
    fr_mean = float(np.mean(arrays["fr"]))
    margin = fr_mean - fl_mean
    per_seed = (arrays["fl"] <= -SIDE_MIN_ABS_MEAN) & (arrays["fr"] >= SIDE_MIN_ABS_MEAN)
    seed_frac = float(np.mean(per_seed))
    recovery_max = float(np.max(np.abs(arrays["recovery"])))
    original_accuracy = float(original_accuracy)
    shuffled_accuracy = float(shuffled_accuracy)

    checks = {
        "open_zero_bias": open_max <= OPEN_MAX_ABS_MEAN,
        "unilateral_opposite_sign": (fl_mean <= -SIDE_MIN_ABS_MEAN
                                      and fr_mean >= SIDE_MIN_ABS_MEAN
                                      and margin >= SIDE_MARGIN_MIN),
        "seed_repeatability": seed_frac >= MIN_SEED_FRACTION,
        "removal_recovery": recovery_max <= RECOVERY_MAX_ABS_MEAN,
        "zero_ablation": bool(zero_directional),
        "shuffle_impairs": (original_accuracy >= ORIGINAL_MIN_ACC
                            and shuffled_accuracy <= SHUFFLE_MAX_ACC),
    }
    return {
        "checks": checks,
        "pass_all": bool(all(checks.values())),
        "details": {
            "open_max_abs_mean": open_max,
            "fl_mean": fl_mean,
            "fr_mean": fr_mean,
            "fr_minus_fl": margin,
            "per_seed_direction_fraction": seed_frac,
            "recovery_max_abs_mean": recovery_max,
            "original_accuracy": original_accuracy,
            "shuffled_accuracy": shuffled_accuracy,
            "zero_directional": bool(zero_directional),
        },
        "thresholds": {
            "OPEN_MAX_ABS_MEAN": OPEN_MAX_ABS_MEAN,
            "SIDE_MIN_ABS_MEAN": SIDE_MIN_ABS_MEAN,
            "SIDE_MARGIN_MIN": SIDE_MARGIN_MIN,
            "RECOVERY_MAX_ABS_MEAN": RECOVERY_MAX_ABS_MEAN,
            "MIN_SEED_FRACTION": MIN_SEED_FRACTION,
            "ORIGINAL_MIN_ACC": ORIGINAL_MIN_ACC,
            "SHUFFLE_MAX_ACC": SHUFFLE_MAX_ACC,
        },
    }
