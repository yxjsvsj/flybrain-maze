"""P2d-7c 跨运行汇总：把多次 static_tof_diag CSV（A/B/C/D 多次重复）做成一张表。

复用 `loop/static_tof_report.py` 的解析/统计；**不**把连续帧当独立重复（只报实际效应与范围，
不做显著性声明）。每个运行独立进程采样，这里只做汇总。

用法：
  python -m loop.static_tof_crossrun --exclude-head-s 2.0 \
      --run A1 a1.csv --run B1 b1.csv --run C1 c1.csv --run D1 d1.csv \
      --run A2 a2.csv --run C2 c2.csv --run B2 b2.csv --run D2 d2.csv --run A3 a3.csv \
      [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys

from loop.static_tof_report import CH, _valid_ratio, exclude_head, load_rows, summarize, symmetry

SCENES = ("A", "B", "C", "D")


def _bt(rows, head_s):
    s = summarize(exclude_head(rows, head_s))["brain_turn_raw"]
    return (s["mean"], s["std"]) if s else (float("nan"), float("nan"))


def _dist_means(rows):
    out = {}
    for c in CH:
        d = summarize(rows)["channels"][c]["dist_mm"]
        out[c] = (d["mean"] if d else None)
    return out


def _valid(rows):
    return {c: _valid_ratio(rows, c) for c in CH}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                           # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="P2d-7c cross-run summary")
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "CSV"), required=True)
    ap.add_argument("--exclude-head-s", type=float, default=2.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    runs = []
    for label, path in a.run:
        rows = load_rows(path)
        m, s = _bt(rows, a.exclude_head_s)
        runs.append({"label": label, "scene": label[0].upper() if label else "?",
                     "bt_mean": m, "bt_std": s, "valid": _valid(rows),
                     "dist_mm": _dist_means(rows), "rows": len(rows),
                     "sym": (symmetry(rows) if label[:1].upper() == "D" else None)})

    A = [r["bt_mean"] for r in runs if r["scene"] == "A" and r["bt_mean"] == r["bt_mean"]]
    baseline = st.fmean(A) if A else float("nan")

    print(f"跨运行汇总（排除前 {a.exclude_head_s:g}s；正=偏左，负=偏右）")
    print(f"{'run':6s} {'bt_mean':>9s} {'bt_std':>8s} {'Δbase':>9s}  "
          f"{'FR_valid':>8s} {'FL_valid':>8s} {'FL_mm':>7s} {'FR_mm':>7s}  note")
    for r in runs:
        flv = r["valid"].get("FL", 0.0)
        frv = r["valid"].get("FR", 0.0)
        flm = r["dist_mm"].get("FL")
        frm = r["dist_mm"].get("FR")
        note = ""
        if r["scene"] == "D" and r["sym"]:
            note = f"sym rel_asym={r['sym']['rel_asym']:.3f}"
        print(f"{r['label']:6s} {r['bt_mean']:+9.4f} {r['bt_std']:8.4f} "
              f"{r['bt_mean']-baseline:+9.4f}  {frv:8.2f} {flv:8.2f} "
              f"{(flm if flm else float('nan')):7.0f} {(frm if frm else float('nan')):7.0f}  {note}")

    print(f"\nA 基线（池化均值）={baseline:+.4f}  n_A={len(A)}")
    Adrift = [r["bt_mean"] - A[0] for r in runs if r["scene"] == "A" and A] if A else []
    if Adrift:
        print(f"A 各次漂移(vs 首个 A): " +
              "  ".join(f"{r['label']}={d:+.4f}" for r, d in
                        zip([r for r in runs if r["scene"] == "A"], Adrift)))

    print("\n各场景相对基线的变化：")
    out_cmp = {}
    for sc in SCENES:
        vals = [r["bt_mean"] - baseline for r in runs if r["scene"] == sc]
        if vals:
            signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in vals]
            cons = max(signs.count(1), signs.count(-1), signs.count(0)) if signs else 0
            out_cmp[sc] = {"deltas": vals, "mean": st.fmean(vals), "range": (min(vals), max(vals)),
                           "sign_consistent": cons == len(signs)}
            print(f"  {sc}: Δ base mean={st.fmean(vals):+.4f} "
                  f"[{min(vals):+.4f},{max(vals):+.4f}] 同号={cons}/{len(signs)}")
    print("\n注：连续帧非独立重复；跨次样本少，只报效应与范围，不做显著性声明。")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"baseline": baseline, "runs": runs, "compare": out_cmp},
                      fh, indent=2, ensure_ascii=False, default=str)
        print(f"\n[json] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
