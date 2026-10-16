"""P9-1b1 回归：参数协议/文献表/g_unit 阻碍/裁决（纯逻辑，**不**跑性能基准）。"""
from __future__ import annotations

import sys

from loop.p9_calibration import LIT, PROTOCOL, decision, g_unit_blocker

RESULTS = []


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p9_calibration ===========")
    check("LIT has >=8 entries", len(LIT) >= 8)
    check("every LIT entry has param/value/cells/paper/use",
          all({"param", "value", "cells", "paper", "use"} <= set(x) for x in LIT))
    check("LIT cites DOIs (10.x)", sum("10." in x["paper"] for x in LIT) >= 5)
    check("single-synapse conductance entry marked as NOT g_unit",
          any("非" in x["use"] and "归一化 W" in x["use"] for x in LIT))

    check("protocol declares g_unit MISSING",
          PROTOCOL["g_unit_nS_per_unit_weight"]["class"] == "MISSING")
    check("protocol forbids assigning nS to normalized W",
          "禁止" in PROTOCOL["g_unit_nS_per_unit_weight"]["rule"])
    check("protocol marks measured vs extrapolation vs synthetic",
          any(v["class"] == "cross-cell-type extrapolation" for v in PROTOCOL.values())
          and any(v["class"] == "measured(range)" for v in PROTOCOL.values())
          and any(v["class"] == "MISSING" for v in PROTOCOL.values()))

    b = g_unit_blocker()
    check("blocker: single synapse != g_unit", "≠" in b["issue"] or "!=" in b["issue"])
    check("blocker lists needed evidence", len(b["needed"]) >= 1)

    d = decision()
    check("engineering verdict conditional GO", "GO" in d["engineering"]["verdict"])
    check("science verdict NO-GO (g_unit missing)", "NO-GO" in d["science"]["verdict"])
    check("both-Go is False -> no next round", d["both_go_for_next_round"] is False)
    check("action says stop/report blocker (not claim biology)",
          "不得" in d["action"] and "停止" in d["action"])

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
