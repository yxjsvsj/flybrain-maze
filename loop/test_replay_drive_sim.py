"""P2d-8 replay_drive_sim 回归（合成 CSV，fake brain，无硬件）。"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import sys
import tempfile

from hardware.tof5 import ORDER
from loop.replay_drive_sim import load_frames
from loop.replay_drive_sim import main as replay_main

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def _write_csv(path, n=24, fr=None):
    cols = ["t"] + [f"{c}_mm" for c in ORDER] + [f"{c}_status" for c in ORDER]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(n):
            st = ["NO_TARGET"] * len(ORDER)
            mm = [""] * len(ORDER)
            if fr and "FR" in ORDER:
                st[ORDER.index("FR")] = "VALID"
                mm[ORDER.index("FR")] = "500"
            w.writerow([f"{i*0.02:.3f}"] + mm + st)


def main() -> int:
    print("=========== replay_drive_sim regression ===========")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "s.csv")
        _write_csv(p)
        check("load_frames parses", len(load_frames(p)) == 24)

        for mem in ("0", "1"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = replay_main(["--csv", p, "--brain", "fake", "--device", "cpu",
                                  "--mode", "dn", "--memory", mem])
            out = buf.getvalue()
            check(f"memory={mem} rc 0", rc == 0)
            check(f"memory={mem} prints net_heading", "net_heading" in out)
            check(f"memory={mem} prints brain_turn", "brain_turn mean" in out)

        pf = os.path.join(tmp, "fr.csv")
        _write_csv(pf, fr="FR")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = replay_main(["--csv", pf, "--brain", "fake", "--device", "cpu",
                              "--mode", "dn", "--memory", "0", "--out-csv",
                              os.path.join(tmp, "traj.csv")])
        check("FR scene runs + writes traj", rc == 0 and os.path.exists(os.path.join(tmp, "traj.csv")))
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
