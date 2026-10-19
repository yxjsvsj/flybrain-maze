"""P11-1a 补充：**seed-level paired Cohen's dz** 静态勘误（只读归档；不运行任何动力学）。

对 25 个 seed：先对 320 KC 求 stim−none 平均差 -> 25 个独立模拟重复差值；
再算 mean、sample SD (ddof=1)、dz=mean/SD。
与 cross-cell dz / pooled seed-cell standardized ratio 明确区分。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from loop.p11_0_offline_diagnosis import load_archive

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_1a_seed_dz.json")
SEEDS = list(range(101, 126))
W = slice(100, 300)


def compute():
    a = load_archive()
    if not a.get("ok"):
        return {"status": a.get("status", "RAW_DATA_UNAVAILABLE")}
    data = a["data"]
    D = np.array([data[(s, "stim")]["kc"][W].mean(0) - data[(s, "none")]["kc"][W].mean(0)
                  for s in SEEDS])                       # 25 x 320
    seed_mean = D.mean(axis=1)                           # 25 independent replicate differences
    mean = float(seed_mean.mean())
    sd = float(seed_mean.std(ddof=1))
    dz = mean / sd if sd > 0 else None
    cross_cell = float(D.mean(axis=0).mean() / D.mean(axis=0).std(ddof=1))
    pooled = float(D.mean() / D.std(ddof=1))
    i114 = SEEDS.index(114)
    return {
        "status": "OK", "archive_status": a.get("status"),
        "n_seeds": len(SEEDS), "n_cells": int(D.shape[1]),
        "seed_level_paired_dz": {
            "definition": "per-seed mean over 320 KC of stim-none; mean/SD over 25 seeds (ddof=1)",
            "mean": mean, "sample_sd_ddof1": sd, "dz": dz,
            "n": len(SEEDS),
            "seed_mean_values": [round(float(x), 12) for x in seed_mean],
        },
        "comparisons": {
            "cross_cell_dz_P11_0": cross_cell,
            "cross_cell_definition": "mean(per-cell means)/sd(per-cell means) across 320 cells",
            "pooled_seed_cell_standardized_ratio": pooled,
            "pooled_definition": "mean(all 8000 seed-cell diffs)/sd(all seed-cell diffs)",
            "seed_level_paired_dz": dz,
            "all_three_distinct": bool(abs(dz - cross_cell) > 1e-6 and abs(dz - pooled) > 1e-6
                                       and abs(cross_cell - pooled) > 1e-6),
            "warning": ("cross-cell dz and pooled seed-cell ratio must NOT be presented as the "
                        "seed-level paired Cohen's dz"),
        },
        "seed_114": {"included": True, "index": i114, "value": round(float(seed_mean[i114]), 12)},
        "seed_selection": "none; all 25 seeds included; no deletion",
        "historical_unchanged": True, "a0_rerun": False,
        "full_brain_initialized": False, "training_authorized": False,
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    r = compute()
    json.dump(r, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else bool(o) if isinstance(o, np.bool_) else str(o))
    if r.get("status") != "OK":
        print("STATUS:", r.get("status")); return 1
    s = r["seed_level_paired_dz"]
    print("[seed dz] n:", s["n"], "| mean:", s["mean"], "| sd:", s["sample_sd_ddof1"], "| dz:", s["dz"])
    print("  cross-cell:", r["comparisons"]["cross_cell_dz_P11_0"],
          "| pooled:", r["comparisons"]["pooled_seed_cell_standardized_ratio"],
          "| all distinct:", r["comparisons"]["all_three_distinct"])
    print("  seed 114 included:", r["seed_114"])
    print("wrote p11_1a_seed_dz.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
