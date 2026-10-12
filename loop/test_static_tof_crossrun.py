"""P2d-7c 跨运行汇总回归（合成 CSV）。"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

from loop.static_tof_crossrun import main as cross_main
from loop.static_tof_report import CH, REQUIRED

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


def _write(path, bt, stim):
    rows = [_row(i * 0.2, stim, {k: 500 for k in stim}, bt) for i in range(30)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REQUIRED)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    print("=========== crossrun regression (synthetic) ===========")
    with tempfile.TemporaryDirectory() as tmp:
        def P(name, bt, stim):
            p = os.path.join(tmp, name)
            _write(p, bt, stim)
            return p
        a1 = P("a1.csv", -0.27, {})
        b1 = P("b1.csv", +0.05, {"FR": "VALID"})
        c1 = P("c1.csv", -0.32, {"FL": "VALID"})
        d1 = P("d1.csv", 0.00, {"FL": "VALID", "FR": "VALID"})
        a2 = P("a2.csv", -0.265, {})
        argv = ["--run", "A1", a1, "--run", "B1", b1, "--run", "C1", c1,
                "--run", "D1", d1, "--run", "A2", a2]
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cross_main(argv)
        out = buf.getvalue()
        check("crossrun returns 0", rc == 0)
        check("baseline line present", "A 基线" in out)
        check("Δ for B positive-left", "B1" in out and "+0.3" in out)
        check("A drift reported", "漂移" in out)
        check("D symmetry reported", "rel_asym" in out)
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
