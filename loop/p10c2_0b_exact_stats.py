"""P10-C2-0b：精确符号翻转修复 + 功效复验 + 冻结 CI（合成/历史复核）。

**不运行 A0、不训练、不更新真实 ΔW。**
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.stats import nct, norm, t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05
MIN_DZ = 0.65
VALID_N = 22
PLANNED_N = 25
SEEDS = list(range(101, 126))
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261


def _signed_sums(x):
    s = np.zeros(1)
    for v in x:
        s = np.concatenate([s + v, s - v])
    return s


def sign_flip_p(diffs):
    """精确双侧符号翻转（修复版）：用**补集区间**计数，避免 obs=0 双重计数。

    平局规则：|sum| >= |observed| 计为“至少同样极端”（计入 p）；补集为严格内部
    (-obs, obs)。容差 `tol = 1e-12*max(1,|obs|)` 仅用于浮点边界。
    """
    d = np.asarray(diffs, float)
    if d.size == 0:
        raise ValueError("empty")
    if not np.all(np.isfinite(d)):
        raise ValueError("non-finite")
    n = len(d)
    # 正比例归一化：符号翻转对正缩放不变；避免固定绝对容差在极小尺度上误判
    scale = float(np.max(np.abs(d)))
    if scale == 0.0:
        return 1.0
    dn = d / scale
    obs = abs(float(dn.sum()))
    if obs <= 1e-12:
        return 1.0
    tol = 1e-12 * max(1.0, obs)
    h = n // 2
    sa = _signed_sums(dn[:h]); sb = np.sort(_signed_sums(dn[h:]))
    inside = 0
    for x in sa:
        lo = np.searchsorted(sb, -obs - x + tol, side="right")   # first > lower+tol
        hi = np.searchsorted(sb, obs - x - tol, side="left")     # first >= upper-tol
        if hi > lo:
            inside += hi - lo
    p = 1.0 - inside / (2 ** n)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p out of range: {p}")
    return p


def sign_flip_p_enum(diffs):
    d = np.asarray(diffs, float); obs = abs(float(d.mean())); cnt = 0
    for m in range(1 << len(d)):
        s = np.array([1.0 if (m >> i) & 1 else -1.0 for i in range(len(d))])
        if abs(float((d * s).mean())) >= obs - 1e-12:
            cnt += 1
    return cnt / (1 << len(d))


def paired_power_normal(dz, n, alpha=ALPHA):
    z = norm.ppf(1 - alpha / 2); nc = dz * math.sqrt(n)
    return float(1 - norm.cdf(z - nc) + norm.cdf(-z - nc))


def paired_power_t(dz, n, alpha=ALPHA):
    df = n - 1; tc = tdist.ppf(1 - alpha / 2, df); delta = dz * math.sqrt(n)
    return float(nct.sf(tc, df, delta) + nct.cdf(-tc, df, delta))


def simulate_power(dz, n, sims=2000, seed=20260):
    rng = np.random.default_rng(seed); c = 0
    for _ in range(sims):
        d = rng.normal(dz, 1.0, n)
        if sign_flip_p(d) < ALPHA:
            c += 1
    p = c / sims
    return {"power": round(p, 4), "mc_se": round(math.sqrt(p * (1 - p) / sims), 4),
            "sims": sims, "seed": seed, "distribution": "N(dz,1) paired differences"}


def bootstrap_ci(diffs, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, alpha=ALPHA):
    d = np.asarray(diffs, float)
    if d.size == 0 or not np.all(np.isfinite(d)):
        raise ValueError("empty/non-finite")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(resamples)])
    return {"method": "percentile bootstrap", "resamples": resamples, "seed": seed,
            "ci_low": round(float(np.percentile(means, 100 * alpha / 2)), 8),
            "ci_high": round(float(np.percentile(means, 100 * (1 - alpha / 2))), 8)}


def paired_effect(diffs):
    d = np.asarray(diffs, float)
    mean = float(d.mean()); sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return {"mean": round(mean, 8), "sd": round(sd, 8),
            "dz": round(mean / sd, 4) if sd > 0 else None,
            "p_exact": round(sign_flip_p(d), 6), "ci": bootstrap_ci(d)}


def exact_stats_selftest():
    cases = {
        "all_zero_2": [0.0, 0.0],
        "cancel_2": [1.0, -1.0],
        "one_zero": [1.0, 2.0, 0.0],
        "multi_zero": [1.0, -2.0, 0.0, 0.0, 3.0],
        "tiny": [1e-15, -1e-15],
        "duplicates": [0.5, 0.5, 0.5, -0.5],
    }
    out = {}
    for k, d in cases.items():
        p = sign_flip_p(d)
        out[k] = {"p": round(p, 6), "in_range": 0.0 <= p <= 1.0}
    # enumeration agreement n=4..16 (随机，含零/重复)
    rng = np.random.default_rng(7)
    agree = True
    for n in range(4, 17):
        for _ in range(8):
            d = rng.integers(-2, 3, n).astype(float) / 2.0
            if abs(sign_flip_p(d) - sign_flip_p_enum(d)) > 1e-12:
                agree = False
    # 置换不变
    d = np.array([0.3, -0.1, 0.2, 0.0, 0.4, -0.2])
    perm = sign_flip_p(rng.permutation(d))
    perm_ok = abs(sign_flip_p(d) - perm) < 1e-12
    # 正比例不变（符号翻转对正缩放不变）
    scale_ok = abs(sign_flip_p(d) - sign_flip_p(3.7 * d)) < 1e-12
    # n=25 可行性
    d25 = rng.normal(0.2, 1.0, 25)
    p25 = sign_flip_p(d25)
    # 非有限/空拒绝
    def rejects(x):
        try:
            sign_flip_p(x); return False
        except ValueError:
            return True
    return {"cases": out, "enum_agree_4_16": agree, "permutation_invariant": perm_ok,
            "positive_scaling_invariant": scale_ok, "n25_p": round(p25, 6),
            "n25_in_range": 0.0 <= p25 <= 1.0,
            "rejects_empty": rejects([]), "rejects_nonfinite": rejects([1.0, float("nan")])}


def _json_default(o):
    if isinstance(o, (__import__("numpy").bool_,)): return bool(o)
    if isinstance(o, (__import__("numpy").integer,)): return int(o)
    if isinstance(o, (__import__("numpy").floating,)): return float(o)
    return str(o)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    s = exact_stats_selftest()
    sim20 = simulate_power(MIN_DZ, 20)
    sim22 = simulate_power(MIN_DZ, 22)
    print("[exact] enum agree:", s["enum_agree_4_16"], "| n25:", s["n25_p"],
          "| tiny:", s["cases"]["tiny"], "| cancel:", s["cases"]["cancel_2"])
    print("  power n20:", sim20, "| n22:", sim22)
    print("  normal n20:", round(paired_power_normal(MIN_DZ, 20), 4),
          "| t n20:", round(paired_power_t(MIN_DZ, 20), 4))
    json.dump({"selftest": s, "power": {"n20": sim20, "n22": sim22},
               "power_normal_n20": paired_power_normal(MIN_DZ, 20),
               "power_t_n20": paired_power_t(MIN_DZ, 20)},
              open(os.path.join(HERE, "p10c2_0b_exact_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=_json_default)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
