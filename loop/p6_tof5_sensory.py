"""P6-1：五路 ToF → MaleCNS 神经**感觉**接口（唯一实验接口，冻结）。**不控车、不下发转向命令。**

审计（step1）：`loop/encode_tof.py` 中 `avoid_half=avoidance_fov_deg/2=60°` → `left={FL,F}`、`right={FR,F}`；
**L(+70°)/R(−70°) 被排除在神经避障通道外**（只参与建图）。本模块把**五路全部**接入神经注入。

注入靶（连接组/功能依据，`brain/neurons.py`）：`LPLC2`=looming、`LC4`=快速 looming/逃逸、`LC10a`=转向上游；
**本接口是人工 ToF→神经感觉转换，不冒充果蝇天然视网膜**。

**编码（冻结）**：每路 `size=clip(WALL/d,0,1.5)`、`growth=clip(Δsize/Δt_tof,0,·)`，
`loom=clip(loom_size_gain·size+loom_growth_gain·growth,0,cap)`、`threat=clip(threat_gain·growth,0,cap)`；
**方位权重** `wL=clip(0.5+0.5·θ/90°,0,1)`、`wR=1−wL` → `loom_L=Σ loom_i·wL_i`（五路均参与，保留方位/时间变化）。
**仅注入感觉组，不下发 fwd/转向/计时/回正命令。**
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from config import EncoderConfig
from loop.visual_motor_audit import dn_target_set_by_body, load_data

HERE = os.path.dirname(os.path.abspath(__file__))
ANGLES = {"L": 70.0, "FL": 30.0, "F": 0.0, "FR": -30.0, "R": -70.0}
ORDER = ("L", "FL", "F", "FR", "R")
SEEDS = [11, 12, 13, 14, 15]
PRE, STIM, POST = 25, 50, 25                      # 50Hz 步：0.5/1.0/0.5s
TOF_HZ = 10.0                                     # 与实际硬件一致
K_SIG = 3.0
NEAR_M, FAR_M = 0.30, 2.50
APPROACH = (1.50, 0.30)                           # 逼近：由远到近
SENSOR_COND = [(s, kind) for s in ORDER for kind in ("near", "away", "approach")] + [("none", "none")]
DN_TARGETS = None  # 延迟到 run 时按 bodyId 固定


def _cfg():
    return EncoderConfig()


def weights():
    th = np.array([ANGLES[s] for s in ORDER])
    wL = np.clip(0.5 + 0.5 * th / 90.0, 0.0, 1.0)
    return wL, 1.0 - wL


def per_sensor(dists, dt_tof, prev_size, cfg, max_speed):
    """五路 -> (size, growth, loom, threat)（冻结公式）。"""
    dists = np.maximum(np.asarray(dists, float), 0.05)
    size = np.clip(cfg.wall_size / dists, 0.0, 1.5)
    dsize_max = cfg.wall_size * max_speed / dists ** 2 * dt_tof
    growth = np.clip((size - np.asarray(prev_size, float)) / dt_tof, 0.0, dsize_max)
    loom = np.clip(cfg.loom_size_gain * size + cfg.loom_growth_gain * growth, 0.0, cfg.cap)
    threat = np.where(growth > cfg.threat_thresh, np.clip(cfg.threat_gain * growth, 0.0, cfg.cap), 0.0)
    return size, growth, loom, threat


def aggregate(dists, dt_tof, prev_size, cfg, max_speed):
    size, growth, loom, threat = per_sensor(dists, dt_tof, prev_size, cfg, max_speed)
    wL, wR = weights()
    return {"size": size, "growth": growth,
            "loom_L": float((loom * wL).sum()), "loom_R": float((loom * wR).sum()),
            "threat_L": float((threat * wL).sum()), "threat_R": float((threat * wR).sum())}


def _scenario_frames(cond, kind):
    """返回 STIM 段每步（10Hz 更新，帧间零阶保持）的 5 路距离。"""
    n = PRE + STIM + POST
    D = np.full((n, 5), FAR_M, np.float32)
    k = ORDER.index(cond) if cond in ORDER else None
    if kind == "none":
        return D
    lo, hi = (NEAR_M, NEAR_M) if kind == "near" else (FAR_M, FAR_M) if kind == "away" else APPROACH
    for b in range(PRE, PRE + STIM):
        frac = (b - PRE) / max(1, STIM - 1)
        D[b, k] = lo + (hi - lo) * frac
    return D


def run_frozen(D, *, device="cpu", brain_kind="real"):
    from flybrain.brain import FlyBrain
    from flybrain.data import ensure_data
    from brain.neurons import resolve
    from config import CarConfig
    data = ensure_data()
    brain = FlyBrain(data=data, device=device, seed=SEEDS[0], batch=1, sensory_input=False)
    assert brain.device == device, f"device mismatch {brain.device}!={device}"   # step6
    groups = resolve(brain, verbose=False)
    car = CarConfig()
    per = EncoderConfig()
    # 注入组
    inj_groups = {"loom_L": groups["loom_L"], "loom_R": groups["loom_R"],
                  "threat_L": groups["threat_L"], "threat_R": groups["threat_R"]}
    # 固定下行读出：LC4/LPLC2 的直接 DN 靶（bodyId 键，step4）
    dv = dn_target_set_by_body(D, vpn_types=("LC4", "LPLC2"))
    side = D["side"]; idx_of = {int(b): i for i, b in enumerate(D["ids"].tolist())}
    dnL = np.array([idx_of[int(b)] for b, v in dv.items() if v["side"] == "L"], int)
    dnR = np.array([idx_of[int(b)] for b, v in dv.items() if v["side"] == "R"], int)
    readout = {"LPLC2_L": groups["loom_L"], "LPLC2_R": groups["loom_R"],
               "LC4_L": groups["threat_L"], "LC4_R": groups["threat_R"],
               "DN_L": dnL, "DN_R": dnR}
    per_step = 1.0 - np.exp(-brain.dt / 0.100)
    raw = {}
    rows = []
    for sd in SEEDS:
        results = {}
        for cond, kind in SENSOR_COND:
            frames = _scenario_frames(cond, kind)
            brain.reset(sd)
            counts = {g: np.zeros(PRE + STIM + POST) for g in readout}
            contrib = []
            prev_size = np.zeros(5)
            tof_period = 1.0 / TOF_HZ
            last_t = -1e9; agg = aggregate(frames[0], tof_period, prev_size, per, car.max_speed)
            for b in range(PRE + STIM + POST):
                t = b * brain.dt
                if t - last_t >= tof_period - 1e-9:
                    dists = frames[b].astype(float)
                    agg = aggregate(dists, max(brain.dt, tof_period if last_t < 0 else t - last_t),
                                    prev_size, per, car.max_speed)
                    prev_size = agg["size"]
                    last_t = t
                    contrib.append(dict(agg))
                inj = []
                for name in ("loom_L", "threat_L", "loom_R", "threat_R"):
                    v = agg[name]
                    if v > 0:
                        inj.append((inj_groups[name], float(v * per_step)))
                fired = np.asarray(brain.step(inject=inj))
                for g, idx in readout.items():
                    if len(idx):
                        counts[g][b] = int(np.isin(fired, idx).sum())
            raw[f"{sd}_{cond}_{kind}"] = np.array([counts[g] for g in readout])
            results[(cond, kind)] = counts
        base = results[("none", "none")]
        sl = slice(PRE, PRE + STIM)
        def cell(cnt, g):
            return float(cnt[g][sl].mean()) / max(1, len(readout[g]))
        def ctrl_std(g):
            return float(np.std(base[g][sl])) / max(1, len(readout[g]))
        row = {"seed": sd, "sigma_L": round(max(ctrl_std("LPLC2_L"), ctrl_std("DN_L")), 5)}
        # 五路生效：每路的近障对其同侧 LPLC2 的对照差值
        for s, kind in SENSOR_COND[:-1]:
            if kind != "near":
                continue
            dL = cell(results[(s, kind)], "LPLC2_L") - cell(base, "LPLC2_L")
            dR = cell(results[(s, kind)], "LPLC2_R") - cell(base, "LPLC2_R")
            row[f"near_{s}_LPLC2L"] = round(dL, 5); row[f"near_{s}_LPLC2R"] = round(dR, 5)
            row[f"near_{s}_effective"] = bool(max(abs(dL), abs(dR)) > K_SIG * row["sigma_L"])
        # 左右镜像（L vs R）；下行读出左右差
        dlnear = cell(results[("L", "approach")], "DN_L") - cell(base, "DN_L")
        drnear = cell(results[("R", "approach")], "DN_R") - cell(base, "DN_R")
        row["DN_L_approachL"] = round(dlnear, 5); row["DN_R_approachR"] = round(drnear, 5)
        row["DN_lateral_ok"] = bool(dlnear > K_SIG * row["sigma_L"] and drnear > K_SIG * row["sigma_L"])
        rows.append(row)
    all5 = all(r[f"near_{s}_effective"] for r in rows for s in ORDER)
    n_eff5 = sum(all(r[f"near_{s}_effective"] for s in ORDER) for r in rows)
    n_dn = sum(r["DN_lateral_ok"] for r in rows)
    np.savez_compressed(os.path.join(HERE, "p6_raw.npz"), **raw)
    return {"frozen": {"interface": "5-way ToF -> LPLC2/LC4 sensory injection (per-sensor azimuth weighting)",
                       "angles": ANGLES, "ORDER": ORDER, "seeds": SEEDS,
                       "PRE/STIM/POST": [PRE, STIM, POST], "TOF_HZ": TOF_HZ,
                       "conds": [list(c) for c in SENSOR_COND], "K_SIG": K_SIG,
                       "dn_targets": {"L": len(dnL), "R": len(dnR)},
                       "note": "artificial ToF->neural sensory conversion; NOT the natural retina"},
            "per_seed": rows, "all5_effective_seeds": n_eff5, "DN_lateral_ok_seeds": n_dn,
            "verdict": "GO" if (all5 and n_dn >= 4) else "NO-GO"}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--brain", default="real")
    a = ap.parse_args(argv)
    D = load_data()
    r = run_frozen(D, device=a.device, brain_kind=a.brain)
    for row in r["per_seed"]:
        print("seed%d σ=%.5f | "%(row["seed"], row["sigma_L"])
              + " ".join("%s:%s" % (s, row[f"near_{s}_effective"]) for s in ORDER)
              + " | DN_L_apL=%.5f DN_R_apR=%.5f dnOK=%s" % (row["DN_L_approachL"], row["DN_R_approachR"], row["DN_lateral_ok"]))
    print("all5-effective seeds=%d/5  DN-lateral-ok=%d/5 -> %s" % (
        r["all5_effective_seeds"], r["DN_lateral_ok_seeds"], r["verdict"]))
    json.dump(r, open(os.path.join(HERE, "p61_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote p61_result.json + p6_raw.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
