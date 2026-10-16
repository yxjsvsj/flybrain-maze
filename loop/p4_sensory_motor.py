"""P4-1：真实 MaleCNS 感觉→运动**神经动力学模型**可行性实验（**唯一**一处机制修改，冻结）。

不覆盖 flybrain 参数/正式脑文件；不改 P1/正式权重/正式 readout/BASE/tags；不控车、不训练导航。
- 归档口径（P3-1a 更正）：一律用 **同 seed「刺激 − 无刺激对照」的同步时段差值**，不把 PRE→STIM 自然升高当感觉传递。
- **唯一修改**（冻结）：**逐神经元入权 L1 归一化**（synaptic normalization）——每条入边按 `1/Σ|w_in|` 缩放，
  使每个神经元的入权总量=1。**gain/tonic/noise/阈值/eye_gain 全部不变**。
  依据：连接组权重是**相对突触效能**而非绝对计数；当前未归一化（入权和中位 0.117，阈值 1.0）→ 级联过弱。
  这是**模型假设**（非连接组数据）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from loop.visual_motor_audit import (BASE_STEPS, GROUP_KEYS, STIM_A, STIM_STEPS, POST_STEPS,
                                     _groups, _run_segments, dn_target_set_by_body, load_data)

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [11, 12, 13, 14, 15]                 # 冻结
SOLO_CHANGE = "row-L1 synaptic normalization (incoming weights per neuron scaled to unit L1)"
# 冻结门槛
GO_STABLE = 4         # ≥4/5 seed: 左刺激 DN_L>DN_R 且 右刺激 DN_R>DN_L（对照差值）
GO_EFFECT = 0.5       # 对照差值的中位 |DN_L-DN_R| ≥ 0.5/步（群体）
GO_VPN = 0.5          # 对照差值 VPN 层显著 ≥ 0.5/步


def archive_p31a(raw_path=None):
    """(requirement 1) 用同 seed「刺激−对照」同步窗差值重算 P3-1a（不重跑）。"""
    d = np.load(raw_path or os.path.join(HERE, "p31_raw.npz"))
    sl = slice(BASE_STEPS, BASE_STEPS + STIM_STEPS)
    out = []
    for sd in SEEDS:
        c = d[f"{sd}_control"]; L = d[f"{sd}_left"]; R = d[f"{sd}_right"]
        def m(a, g):
            return float(a[GROUP_KEYS.index(g)][sl].mean())
        out.append({"seed": sd,
                    "photo_left_delta": round(m(L, "photo") - m(c, "photo"), 2),
                    "vpn_left_delta": round(m(L, "other_VPN") - m(c, "other_VPN"), 3),
                    "dn_L_left_delta": round(m(L, "DN_L") - m(c, "DN_L"), 3),
                    "dn_R_left_delta": round(m(L, "DN_R") - m(c, "DN_R"), 3),
                    "left_dnL_minus_dnR": round((m(L, "DN_L") - m(c, "DN_L")) - (m(L, "DN_R") - m(c, "DN_R")), 3)})
    med = float(np.median([abs(r["left_dnL_minus_dnR"]) for r in out]))
    return {"per_seed": out, "median_abs_left_dnL_minus_dnR": round(med, 3),
            "note": "control-subtracted = stimulus minus same-seed no-stimulus control",
            "conclusion": "NO lateral-specific sensory transmission under the unmodified LIF (>='%s')"
            % ("VPN delta sign-mixed/none, DN delta ~0")}


def row_l1_normalize(W):
    """**唯一修改**：逐神经元入权 L1 归一化（每行按 1/Σ|w| 缩放，行 L1=1）。**纯函数**。"""
    from scipy import sparse
    W = W.tocsr()
    rows = np.asarray(np.abs(W).sum(1)).ravel()
    rows[rows == 0] = 1.0
    return (sparse.diags(1.0 / rows) @ W)


def normalized_brain(D, *, device="cpu", seed=SEEDS[0]):
    """构造**唯一修改**下的 FlyBrain：逐神经元入权 L1 归一化（其余参数不变）。"""
    from scipy import sparse
    from flybrain.brain import FlyBrain
    from flybrain.data import ensure_data
    data = ensure_data()
    brain = FlyBrain(data=data, device=device, seed=seed, batch=1, sensory_input=False)
    sc = D["superclass"].astype(str)
    sensory = np.char.find(sc, "sensory") >= 0
    W = sparse.diags((~sensory).astype(np.float32)) @ D["W"].tocsr()   # 复刻 sensory_input=False
    Wn = row_l1_normalize(W).tocsc()
    brain.indptr, brain.indices, brain.weights = Wn.indptr, Wn.indices, Wn.data.astype(np.float32)
    return brain


def run_frozen(D, *, device="cpu"):
    brain = normalized_brain(D, device=device)
    groups = _groups(D, dn_target_set_by_body(D))
    vis = np.asarray(D["visual"]); az = D["azimuth"]
    Lm = az < -0.3; Rm = az > 0.3
    def drive(mask):
        d = np.zeros(len(vis), np.float32); d[mask] = STIM_A; return np.asarray(d)
    drv = {"control": None, "left": drive(Lm), "right": drive(Rm)}
    raw = {}
    rows = []
    for sd in SEEDS:
        seg = {}
        for cond in ("control", "left", "right"):
            brain.reset(sd)
            c = _run_segments(brain, groups, drv[cond])
            raw[f"{sd}_{cond}"] = np.array([c[g] for g in GROUP_KEYS])
            seg[cond] = c
        sl = slice(BASE_STEPS, BASE_STEPS + STIM_STEPS)
        po = slice(BASE_STEPS + STIM_STEPS, BASE_STEPS + STIM_STEPS + POST_STEPS)
        def mean(a, g, s=sl):
            return float(seg[a][g][s].mean())
        def cell(a, g, s=sl):
            return mean(a, g, s) / max(1, len(groups[g]))
        # 对照差值（per-cell）
        l_dnL = cell("left", "DN_L") - cell("control", "DN_L")
        l_dnR = cell("left", "DN_R") - cell("control", "DN_R")
        r_dnR = cell("right", "DN_R") - cell("control", "DN_R")
        r_dnL = cell("right", "DN_L") - cell("control", "DN_L")
        vpn_left = mean("left", "other_VPN") - mean("control", "other_VPN")
        rows.append({"seed": sd, "vpn_left_delta_pop": round(vpn_left, 3),
                     "left_dnL_minus_dnR_cell": round(l_dnL - l_dnR, 5),
                     "right_dnR_minus_dnL_cell": round(r_dnR - r_dnL, 5),
                     "dnL_pop_stim": round(mean("left", "DN_L"), 3),
                     "post_dnL_pop": round(mean("left", "DN_L", po), 3),
                     "ipsi_gt_contra": bool((l_dnL - l_dnR > 0) and (r_dnR - r_dnL > 0))})
    n_ok = sum(r["ipsi_gt_contra"] for r in rows)
    eff = float(np.median([abs(r["left_dnL_minus_dnR_cell"]) for r in rows]))
    vpn_med = float(np.median([abs(r["vpn_left_delta_pop"]) for r in rows]))
    go = (n_ok >= GO_STABLE) and (eff >= GO_EFFECT) and (vpn_med >= GO_VPN)
    np.savez_compressed(os.path.join(HERE, "p4_raw.npz"), **raw)
    return {"frozen": {"solo_change": SOLO_CHANGE, "seeds": SEEDS, "STIM_A": STIM_A,
                       "PRE/STIM/POST": [BASE_STEPS, STIM_STEPS, POST_STEPS],
                       "gain/tonic/noise/threshold/eye_gain": "UNCHANGED",
                       "stimulus": "left/right eye az<-0.3/>0.3, constant",
                       "metric": "control-subtracted (stim minus same-seed no-stim)",
                       "GO_STABLE": GO_STABLE, "GO_EFFECT": GO_EFFECT, "GO_VPN": GO_VPN},
            "per_seed": rows, "ipsi_gt_contra_seeds": n_ok,
            "median_abs_lateral_cell": round(eff, 5), "median_vpn_delta_pop": round(vpn_med, 3),
            "verdict": "GO" if go else "NO-GO"}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    arc = archive_p31a()
    print("[P4-1 archive] control-subtracted P3-1a: median |DN_L-DN_R| =", arc["median_abs_left_dnL_minus_dnR"])
    for r in arc["per_seed"]:
        print("  seed%d vpnΔ=%.3f dn_LΔ=%.3f dn_RΔ=%.3f dnL-dnR=%.3f" % (
            r["seed"], r["vpn_left_delta"], r["dn_L_left_delta"], r["dn_R_left_delta"], r["left_dnL_minus_dnR"]))
    D = load_data()
    res = run_frozen(D, device=a.device)
    print("[P4-1 frozen] solo change:", res["frozen"]["solo_change"])
    for r in res["per_seed"]:
        print("  seed%d vpnΔpop=%.3f lateral_cell L=%.5f R=%.5f ipsi>contra=%s dnL_pop_stim=%.3f post=%.3f" % (
            r["seed"], r["vpn_left_delta_pop"], r["left_dnL_minus_dnR_cell"], r["right_dnR_minus_dnL_cell"],
            r["ipsi_gt_contra"], r["dnL_pop_stim"], r["post_dnL_pop"]))
    print("  ipsi>contra %d/%d  median|lateral_cell|=%.5f  vpn=%.3f -> %s" % (
        res["ipsi_gt_contra_seeds"], len(SEEDS), res["median_abs_lateral_cell"],
        res["median_vpn_delta_pop"], res["verdict"]))
    json.dump({"archive": arc, "result": res}, open(os.path.join(HERE, "p41_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote p41_result.json + p4_raw.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
