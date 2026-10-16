"""P5-1：基于果蝇真实视觉机制的**动态迫近（looming）**刺激验证（**唯一输入模型，冻结**）。

机制依据（step1）：T4(ON)/T5(OFF) 为方向选择运动检测器；LC4 迫近敏感；**LPLC2 大量接收 T4/T5**
（本连接组实测 LPLC2←T4a-d/T5a-d 平均入权 0.32、33,409 边）。**不**凭名称假定功能：另用 2 跳
（photo→ol→VPN）实测**感受野中心方位**核对拓扑（LC4/LPLC2 侧别一致性 1.00，L≈−0.5 / R≈+0.5）。

**唯一输入模型（冻结）**：视网膜方位带（6006 光感受器）上的**高斯带时空驱动**，经 `eye_drive` 加到
**光感受器**（**不**向 LC4/LPLC2/DN 注入任何"正确方向"）。条件：none / left_loom / right_loom /
left_contract / luminance；**各条件总线能量相等**（预算匹配）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

from loop.visual_motor_audit import BASE_STEPS, POST_STEPS, STIM_A, STIM_STEPS, group_idx, load_data

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [11, 12, 13, 14, 15]
C0_L, C0_R = -0.5, 0.5
SIG0, SIG1 = 0.10, 0.60
K_SIG = 3.0                 # 效应门槛 = K_SIG × 对照段每细胞发放 std（**预注册规则**）
CONDS = ("none", "left_loom", "right_loom", "left_contract", "luminance")
TOTAL = BASE_STEPS + STIM_STEPS + POST_STEPS
SL = slice(BASE_STEPS, BASE_STEPS + STIM_STEPS)
PO = slice(BASE_STEPS + STIM_STEPS, TOTAL)


def _groups(D):
    ct = D["cell_type"]; side = D["side"]
    t45 = [t for t in set(ct.tolist()) if t.startswith("T4") or t.startswith("T5")]
    g = {"photo": np.asarray(D["visual"], int),
         "T4T5": np.flatnonzero(np.isin(ct, t45))}
    for t in ("LC4", "LPLC2", "LPLC1"):
        ci = np.flatnonzero(ct == t)
        g[t + "_L"] = ci[side[ci] == "L"]; g[t + "_R"] = ci[side[ci] == "R"]
    return g


def _frames(az, cond):
    """逐帧 drive（仅 STIM 段非零），并对**总线能量**归一（预算匹配）。"""
    n = len(az)
    core = int((np.abs(az) <= 0.1).sum())
    budget = STIM_A * STIM_STEPS * max(1, core)
    F = np.zeros((TOTAL, n), np.float32)
    for k in range(BASE_STEPS, BASE_STEPS + STIM_STEPS):
        frac = (k - BASE_STEPS) / max(1, STIM_STEPS - 1)
        if cond == "none":
            pass
        elif cond == "luminance":
            F[k, :] = 1.0
        else:
            sig = SIG0 + (SIG1 - SIG0) * frac if cond.endswith("loom") else SIG1 - (SIG1 - SIG0) * frac
            c0 = C0_L if cond.startswith("left") else C0_R
            F[k] = np.exp(-0.5 * ((az - c0) / sig) ** 2)
    tot = float(F.sum())
    if tot > 0:
        F *= budget / tot
    return F


def _run(brain, groups, F):
    brain.reset()
    counts = {g: np.zeros(TOTAL, np.int64) for g in groups}
    for k in range(TOTAL):
        row = F[k]
        dv = brain.xp.asarray(row) if row.any() else None
        fired = np.asarray(brain.step(eye_drive=dv))
        for g, idx in groups.items():
            if len(idx):
                counts[g][k] = int(np.isin(fired, idx).sum())
    return counts


def run_frozen(D, *, device="cpu"):
    import numpy, scipy, flybrain
    from flybrain.brain import FlyBrain
    from flybrain.data import ensure_data
    data = ensure_data()
    Wcs = hashlib.sha256(open(str(data / "weights.npz"), "rb").read()).hexdigest()[:16]
    brain = FlyBrain(data=data, device=device, seed=SEEDS[0], batch=1, sensory_input=False)
    assert brain.device == device, f"device mismatch: got {brain.device}, expected {device}"   # step6
    groups = _groups(D)
    az = np.asarray(D["azimuth"], float)
    raw = {}
    per = []
    for sd in SEEDS:
        seg = {}
        for cond in CONDS:
            counts = _run(brain, groups, _frames(az, cond))
            raw[f"{sd}_{cond}"] = np.array([counts[g] for g in groups])
            seg[cond] = counts
        brain.reset(sd)   # 各 seed 用同一 seed 重置（_run 内已 reset()；此处仅为显式对齐）

        def cell(cond, g, s=SL):
            return float(seg[cond][g][s].mean()) / max(1, len(groups[g]))

        def ctrl_std(g):
            return float(np.std(seg["none"][g][SL])) / max(1, len(groups[g]))

        row = {"seed": sd}
        for t in ("LC4", "LPLC2"):
            base = cell("none", t + "_L")
            for side in ("L", "R"):
                row[f"{t}_{side}_none"] = round(cell("none", t + "_" + side), 5)
            row[f"{t}_left_loom_L"] = round(cell("left_loom", t + "_L") - cell("none", t + "_L"), 5)
            row[f"{t}_left_loom_R"] = round(cell("left_loom", t + "_R") - cell("none", t + "_R"), 5)
            row[f"{t}_right_loom_L"] = round(cell("right_loom", t + "_L") - cell("none", t + "_L"), 5)
            row[f"{t}_right_loom_R"] = round(cell("right_loom", t + "_R") - cell("none", t + "_R"), 5)
            row[f"{t}_left_lateral"] = round(row[f"{t}_left_loom_L"] - row[f"{t}_left_loom_R"], 5)
            row[f"{t}_right_lateral"] = round(row[f"{t}_right_loom_R"] - row[f"{t}_right_loom_L"], 5)
            row[f"{t}_sigma"] = round(max(ctrl_std(t + "_L"), ctrl_std(t + "_R")), 5)
            row[f"{t}_direction_ok"] = bool(row[f"{t}_left_lateral"] > 0 and row[f"{t}_right_lateral"] > 0
                                            and row[f"{t}_left_lateral"] > K_SIG * row[f"{t}_sigma"]
                                            and row[f"{t}_right_lateral"] > K_SIG * row[f"{t}_sigma"])
        per.append(row)
    n_ok = sum(r["LC4_direction_ok"] and r["LPLC2_direction_ok"] for r in per)
    ver = {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
           "flybrain": getattr(flybrain, "__version__", "?"), "weights_sha256_16": Wcs, "device": brain.device}
    go = n_ok >= 4
    np.savez_compressed(os.path.join(HERE, "p5_raw.npz"), **raw)
    return {"frozen": {"input_model": "retinal azimuth Gaussian-band, budget-matched; eye_drive on photoreceptors only",
                       "conds": CONDS, "seeds": SEEDS, "C0_L/C0_R": [C0_L, C0_R], "SIG0/SIG1": [SIG0, SIG1],
                       "window": [BASE_STEPS, STIM_STEPS, POST_STEPS], "K_SIG": K_SIG,
                       "metric": "control-subtracted (stim minus same-seed no-stim)"},
            "versions": ver, "per_seed": per, "LC4_and_LPLC2_direction_ok_seeds": n_ok,
            "verdict": "GO" if go else "NO-GO"}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    D = load_data()
    r = run_frozen(D, device=a.device)
    print("[P5-1] versions:", r["versions"])
    for row in r["per_seed"]:
        print("  seed%d LC4 lateral(L/R)=%.5f/%.5f σ=%.5f ok=%s | LPLC2 %.5f/%.5f ok=%s" % (
            row["seed"], row["LC4_left_lateral"], row["LC4_right_lateral"], row["LC4_sigma"], row["LC4_direction_ok"],
            row["LPLC2_left_lateral"], row["LPLC2_right_lateral"], row["LPLC2_direction_ok"]))
    print("  direction-ok seeds=%d/5 -> %s" % (r["LC4_and_LPLC2_direction_ok_seeds"], r["verdict"]))
    json.dump(r, open(os.path.join(HERE, "p51_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote p51_result.json + p5_raw.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
