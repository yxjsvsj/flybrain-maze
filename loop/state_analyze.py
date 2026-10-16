"""P2d-27 第二关：汇总 + 冻结判据裁决 + 图。读 p2d27_trials.csv / p2d27_timeseries.npz。"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from loop.state_probe import (BUMP_R, CUE, PHASE_SEP_MIN, PRE, R_VALID, SUSTAIN_MIN_S,
                              WINDOW, window_series)


def circ_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    rows = list(csv.DictReader(open(os.path.join(HERE, "p2d27_trials.csv"), encoding="utf-8")))
    for r in rows:
        for k in ("obs_medianR", "cue_medianR", "pre_medianR", "valid_rate"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else float("nan")
        r["sustain_s"] = float(r["sustain_s"]); r["seed"] = int(r["seed"])
        r["final_phase_deg"] = float(r["final_phase_deg"]) if r["final_phase_deg"] not in ("", "nan") else float("nan")

    def sub(ph, cond):
        return [r for r in rows if r["phase"] == ph and r["condition"] == cond]

    summary = {}
    for cond in ("nocue", "cue0", "cue120", "diffuse"):
        c = sub("confirm", cond)
        summary[cond] = {
            "obs_medianR_med": round(float(np.median([r["obs_medianR"] for r in c])), 4),
            "cue_medianR_med": round(float(np.nanmedian([r["cue_medianR"] for r in c])), 4),
            "sustain_med": round(float(np.median([r["sustain_s"] for r in c])), 3),
            "valid_rate_med": round(float(np.median([r["valid_rate"] for r in c])), 3),
            "frac_sustain_ge_1s": round(float(np.mean([r["sustain_s"] >= SUSTAIN_MIN_S for r in c])), 3),
            "frac_obsR_ge_BUMPr": round(float(np.mean([r["obs_medianR"] >= BUMP_R for r in c])), 3),
        }

    # 位置区分（confirm，按 seed 配对 cue0 vs cue120，仅当两者 obs_medianR 达 BUMP_R）
    sep = []
    for s in set(r["seed"] for r in sub("confirm", "cue0")):
        r0 = [r for r in sub("confirm", "cue0") if r["seed"] == s][0]
        r1 = [r for r in sub("confirm", "cue120") if r["seed"] == s][0]
        if r0["obs_medianR"] >= BUMP_R and r1["obs_medianR"] >= BUMP_R:
            sep.append(circ_diff(r0["final_phase_deg"], r1["final_phase_deg"]))
    any_localized = any(r["obs_medianR"] >= BUMP_R or r["cue_medianR"] >= BUMP_R for r in rows)
    any_sustain = any(r["sustain_s"] >= SUSTAIN_MIN_S and r["obs_medianR"] >= BUMP_R for r in rows)

    if any_sustain and sep and np.mean([s >= PHASE_SEP_MIN for s in sep]) >= 0.5:
        verdict = "PASS"
    elif any_localized:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    out = {"verdict": verdict, "summary_confirm": summary,
           "position_separation_deg": [round(s, 1) for s in sep],
           "max_obs_medianR": round(max(r["obs_medianR"] for r in rows), 4),
           "max_cue_medianR": round(max(r["cue_medianR"] for r in rows), 4),
           "any_localized_ge0.30": bool(any_localized),
           "criterion": {"BUMP_R": BUMP_R, "SUSTAIN_MIN_S": SUSTAIN_MIN_S,
                         "PHASE_SEP_MIN": PHASE_SEP_MIN}}
    json.dump(out, open(os.path.join(HERE, "p2d27_verdict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from loop.state_probe import epg_setup
        from flybrain.data import ensure_data
        _, _, ang = epg_setup(ensure_data())
        d = np.load(os.path.join(HERE, "p2d27_timeseries.npz"))
        seed = 21
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for cond, col in (("nocue", "k"), ("cue0", "tab:blue"),
                          ("cue120", "tab:red"), ("diffuse", "tab:green")):
            ct = d[f"confirm_{seed}_{cond}"]
            Rs, _ = window_series(ct, ang)
            axes[0].plot(np.arange(len(Rs)) * WINDOW * 0.02, Rs, color=col, label=cond)
        axes[0].axvspan(PRE * 0.02, (PRE + CUE) * 0.02, color="0.85", label="cue window")
        axes[0].axhline(BUMP_R, ls="--", c="0.5"); axes[0].set_xlabel("t (s)")
        axes[0].set_ylabel("population R"); axes[0].set_title(f"EPG bump locality (confirm seed {seed})")
        axes[0].legend(fontsize=7)
        ax = fig.add_subplot(122, projection="polar")
        for cond, col in (("cue0", "tab:blue"), ("cue120", "tab:red")):
            ct = d[f"confirm_{seed}_{cond}"][(PRE + CUE) * WINDOW:]
            prof = ct.sum(0)
            ax.bar(ang, prof / (prof.max() + 1e-9), width=0.12, color=col, alpha=0.5, label=cond)
        ax.set_title("obs angular profile (confirm seed 21)"); ax.legend(fontsize=7, loc="lower right")
        fig.savefig(os.path.join(HERE, "p2d27_state.png"), dpi=110, bbox_inches="tight")
        print("plot -> loop/p2d27_state.png")
    except Exception as e:                                # noqa: BLE001
        print("plot skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
