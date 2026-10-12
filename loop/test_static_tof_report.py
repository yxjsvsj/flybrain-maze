"""P2d-7 分析工具回归（合成 CSV，无硬件）。"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

from loop.static_tof_report import (CH, REQUIRED, ReportError, compare, exclude_head,
                                    load_rows, purity, summarize, symmetry)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def _row(t, status, mm, bt):
    d = {"t": t, "brain_turn_raw": bt}
    for c in CH:
        d[f"{c}_mm"] = ("" if status.get(c) != "VALID" else mm.get(c))
        d[f"{c}_status"] = status.get(c, "NO_TARGET")
    for s in ("L", "R"):
        for k in ("size", "growth", "chase"):
            d[f"{k}_{s}"] = "0.1"
    return d


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REQUIRED)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def t_errors(tmp):
    p = os.path.join(tmp, "missing.csv")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("t,foo\n0,1\n")
    raised = False
    try:
        load_rows(p)
    except ReportError:
        raised = True
    check("missing column -> ReportError", raised)

    pe = os.path.join(tmp, "empty.csv")
    open(pe, "w").close()
    raised = False
    try:
        load_rows(pe)
    except ReportError:
        raised = True
    check("empty CSV -> ReportError", raised)


def t_all_no_target(tmp):
    rows = [_row(i * 0.1, {}, {}, bt=-0.25 - 0.005 * i) for i in range(30)]
    p = os.path.join(tmp, "notarget.csv")
    _write(p, rows)
    s = summarize(load_rows(p))
    check("all NO_TARGET: no VALID distance", all(s["channels"][c]["dist_mm"] is None for c in CH))
    check("all NO_TARGET: brain_turn_raw present", s["brain_turn_raw"] is not None)
    check("all NO_TARGET: pos_frac 0", s["brain_turn_raw"]["pos_frac"] == 0.0)


def t_valid_distance_only(tmp):
    # VALID 仅 FR；距离用 VALID 计算，NO_TARGET 不计
    rows = [_row(i * 0.1, {"FR": "VALID"}, {"FR": 500 + i}, bt=0.0) for i in range(20)]
    p = os.path.join(tmp, "fr.csv")
    _write(p, rows)
    s = summarize(load_rows(p))
    check("FR VALID ratio 1.0", s["channels"]["FR"]["valid_ratio"] == 1.0)
    check("FR dist uses VALID only", s["channels"]["FR"]["dist_mm"]["n"] == 20)
    check("FL no VALID distance", s["channels"]["FL"]["dist_mm"] is None)


def t_purity(tmp):
    fr_only = [_row(i * 0.1, {"FR": "VALID"}, {"FR": 500}, 0.05) for i in range(20)]
    pf = os.path.join(tmp, "fr_only.csv")
    _write(pf, fr_only)
    check("FR-only -> pure", purity(load_rows(pf), "FR")["label"] == "pure")

    mixed = [_row(i * 0.1, {"FR": "VALID", "FL": "VALID"}, {"FR": 500, "FL": 500}, 0.0)
             for i in range(20)]
    pm = os.path.join(tmp, "mixed.csv")
    _write(pm, mixed)
    check("FR+FL both -> MIXED", purity(load_rows(pm), "FR")["label"] == "MIXED")
    check("FL+FR symmetric detected", symmetry(load_rows(pm))["symmetric"] is True)


def t_compare():
    c = compare(-0.268, 0.050, -0.320, 0.000, -0.265)
    check("dFR", abs(c["dFR"] - (0.050 + 0.268)) < 1e-9)
    check("dFL", abs(c["dFL"] - (-0.320 + 0.268)) < 1e-9)
    check("dSYM", abs(c["dSYM"] - (0.000 + 0.268)) < 1e-9)
    check("baseline_drift", abs(c["baseline_drift"] - (-0.265 + 0.268)) < 1e-9)


def t_exclude_head(tmp):
    rows = [_row(i * 0.5, {}, {}, 0.0) for i in range(10)]     # t=0..4.5
    tail = exclude_head(rows, 2.0)
    check("exclude_head drops first 2s", len(tail) < len(rows) and float(tail[0]["t"]) >= 2.0)


def main() -> int:
    print("=========== static_tof_report regression ===========")
    with tempfile.TemporaryDirectory() as tmp:
        t_errors(tmp)
        t_all_no_target(tmp)
        t_valid_distance_only(tmp)
        t_purity(tmp)
        t_exclude_head(tmp)
    t_compare()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
