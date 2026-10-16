"""P2d-27a：**从已保存时序重新分析**（不重跑 MaleCNS）。修正分析口径：
  - 区分 `phase_valid_duration_s`(R≥0.20) 与 `bump_sustain_duration_s`(R≥0.30 且发放≥MIN_SPK 连续)；
  - 加**真正的空间角标签打乱**对照；
  - 修正绘图时间切片（完整 5 s 观察段）；
  - 用原始时序重生成 逐seed表 / 裁决 / 图。**不改**预注册判据与原始结果。
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from loop.state_probe import (BUMP_R, CUE, MIN_SPK, PHASE_SEP_MIN, PRE, R_VALID,
                              SUSTAIN_MIN_S, TOTAL, WINDOW, epg_setup,
                              spatial_shuffle_control, trial_metrics, window_series)

COLS = ["phase", "seed", "condition", "pre_spikes", "pre_medianR", "cue_medianR", "obs_medianR",
        "valid_rate", "phase_valid_duration_s", "bump_sustain_duration_s", "final_phase_deg",
        "n_peaks", "fwhm_deg", "n_spikes_obs", "shuffle_obsR", "shuffle_p"]


def circ_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def recompute(series_path, ang):
    d = np.load(series_path)
    rows = []
    for k in d.files:
        ph, seed, cond = k.split("_", 2)
        ct = d[k]
        m = trial_metrics(ct, ang)
        sh = spatial_shuffle_control(ct, ang, n_perm=1000, seed=0)
        m.update({"phase": ph, "seed": int(seed), "condition": cond,
                  "shuffle_obsR": sh["obs_R"], "shuffle_p": sh["p"]})
        rows.append(m)
    rows.sort(key=lambda r: (r["phase"], r["seed"], r["condition"]))
    return rows


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    from flybrain.data import ensure_data
    _, _, ang = epg_setup(ensure_data())
    rows = recompute(os.path.join(HERE, "p2d27_timeseries.npz"), ang)
    with open(os.path.join(HERE, "p2d27_trials.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})

    def sub(ph, cond):
        return [r for r in rows if r["phase"] == ph and r["condition"] == cond]

    summary = {}
    for cond in ("nocue", "cue0", "cue120", "diffuse"):
        c = sub("confirm", cond)
        r0 = [x for x in sub("scan", cond)] if False else c
        summary[cond] = {
            "obs_medianR": round(float(np.median([x["obs_medianR"] for x in c])), 4),
            "cue_medianR": round(float(np.median([x["cue_medianR"] for x in c])), 4),
            "phase_valid_duration_s_med": round(float(np.median([x["phase_valid_duration_s"] for x in c])), 2),
            "bump_sustain_duration_s_med": round(float(np.median([x["bump_sustain_duration_s"] for x in c])), 2),
            "frac_bump_sustain_ge_1s": round(float(np.mean([x["bump_sustain_duration_s"] >= SUSTAIN_MIN_S for x in c])), 3),
            "frac_obsR_ge_BUMPr": round(float(np.mean([x["obs_medianR"] >= BUMP_R for x in c])), 3),
            "shuffle_p_med": round(float(np.median([x["shuffle_p"] for x in c])), 3)}
    # PRE 段活动
    pre_spk = [x["pre_spikes"] for x in rows]
    pre_R = [x["pre_medianR"] for x in rows if x["pre_medianR"] is not None]

    sep = []
    for s in sorted(set(x["seed"] for x in sub("confirm", "cue0"))):
        r0 = [x for x in sub("confirm", "cue0") if x["seed"] == s][0]
        r1 = [x for x in sub("confirm", "cue120") if x["seed"] == s][0]
        if r0["bump_sustain_duration_s"] >= SUSTAIN_MIN_S and r1["bump_sustain_duration_s"] >= SUSTAIN_MIN_S:
            sep.append(circ_diff(r0["final_phase_deg"], r1["final_phase_deg"]))
    any_bump_sustain = any(x["bump_sustain_duration_s"] >= SUSTAIN_MIN_S for x in rows)
    any_localized = any(x["obs_medianR"] >= BUMP_R or x["cue_medianR"] >= BUMP_R for x in rows)
    if any_bump_sustain and sep and np.mean([s >= PHASE_SEP_MIN for s in sep]) >= 0.5:
        verdict = "PASS"
    elif any_localized:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    out = {"verdict": verdict, "summary_confirm": summary,
           "pre_segment": {"median_pre_spikes": int(np.median(pre_spk)),
                           "median_pre_medianR": round(float(np.median(pre_R)), 4) if pre_R else None},
           "position_separation_deg": [round(s, 1) for s in sep],
           "any_bump_sustain_ge_1s": bool(any_bump_sustain),
           "any_localized_ge_0.30": bool(any_localized),
           "shuffle_control": {c: summary[c]["shuffle_p_med"] for c in summary},
           "criterion": {"R_VALID": R_VALID, "BUMP_R": BUMP_R, "MIN_SPK": MIN_SPK,
                         "SUSTAIN_MIN_S": SUSTAIN_MIN_S, "PHASE_SEP_MIN": PHASE_SEP_MIN},
           "note": "sensory_input=True (this round); analysis recomputed from saved timeseries only"}
    json.dump(out, open(os.path.join(HERE, "p2d27_verdict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))

    # ---- plot (fixed slicing: 0..6 s, cue shaded, full 5 s obs) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = np.load(os.path.join(HERE, "p2d27_timeseries.npz"))
        seed = 21
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for cond, col in (("nocue", "k"), ("cue0", "tab:blue"),
                          ("cue120", "tab:red"), ("diffuse", "tab:green")):
            ct = d[f"confirm_{seed}_{cond}"]
            Rs, _ = window_series(ct, ang)
            t = (np.arange(len(Rs)) + 0.5) * WINDOW * 0.02       # 窗中心，0..6 s
            axes[0].plot(t, Rs, color=col, label=cond)
        axes[0].axvspan(0, PRE * 0.02, color="0.95")
        axes[0].axvspan(PRE * 0.02, (PRE + CUE) * 0.02, color="0.80", label="cue")
        axes[0].axvspan((PRE + CUE) * 0.02, TOTAL * 0.02, color="0.93", label="OBS (5 s)")
        axes[0].axhline(BUMP_R, ls="--", c="tab:orange"); axes[0].axhline(R_VALID, ls=":", c="0.4")
        axes[0].set_xlim(0, TOTAL * 0.02); axes[0].set_ylim(0, 1)
        axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("population R")
        axes[0].set_title(f"EPG locality (confirm seed {seed}); orange=BUMP_R .30, dotted=R_VALID .20")
        axes[0].legend(fontsize=7, loc="upper right")
        ax = fig.add_subplot(122, projection="polar")
        for cond, col in (("cue0", "tab:blue"), ("cue120", "tab:red")):
            prof = d[f"confirm_{seed}_{cond}"][PRE + CUE:].sum(0)
            ax.bar(ang, prof / (prof.max() + 1e-9), width=0.12, color=col, alpha=0.5, label=cond)
        ax.set_title(f"OBS(5 s) angular profile seed {seed}"); ax.legend(fontsize=7, loc="lower right")
        fig.savefig(os.path.join(HERE, "p2d27_state.png"), dpi=110, bbox_inches="tight")
        print("plot -> loop/p2d27_state.png")
    except Exception as e:                               # noqa: BLE001
        print("plot skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
