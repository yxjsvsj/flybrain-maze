"""P9-0 回归：p9_engine 的设计纯逻辑（数据清单/架构/参数表/验证/预算/裁决）。**无大规模运行**。"""
from __future__ import annotations

import sys

from loop.p9_engine import DATA, REFS, architecture, budget, param_table, validation, verdict

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== p9_engine ===========")
    check("data has real/lit/unknown tiers",
          {"real_available", "literature_estimate", "unknown"} <= set(DATA))
    check("unknown flags absolute conductance scale",
          any("conductance" in k for k in DATA["unknown"]))

    a = architecture()
    check("architecture is conductance-based", "电导型" in a["key_change"] and "E_exc" in a["neuron"])
    check("architecture keeps connectome identity/weights", "不变" in a["preserved"])
    check("architecture forbids P8 graded-release reuse", "P8" in a["forbidden"] and "渐变释放" in a["forbidden"])
    check("mechanism is NOT 'stronger gain'", "不**是" in a["not_a_gain"] or "非" in a["not_a_gain"])

    pt = param_table()
    check("param table has >=9 rows", len(pt) >= 9)
    check("every param row has unit/source/cells/basis/uncertainty",
          all({"unit", "source", "cells", "basis", "uncertainty"} <= set(r) for r in pt))
    check("absolute conductance scale marked unknown",
          any("g_syn" in r["symbol"] and "未知" in r["range"] for r in pt))
    check("weights/transmitter marked real data", any("真实数据" in r["source"] for r in pt))

    v = validation()
    check("validation respects P7 (0 direct edge)", "0 直接边" in v["layers"]["P7_respect"])
    check("validation has mirror control", any("镜像" in c for c in v["controls"]))
    check("gate fixed >=4/5 & 3sigma", "4/5" in v["gate"] and "3σ" in v["gate"])
    check("forbids ToF direct direction/timer", "不**" in v["forbidden_io"] or "无计时器" in v["forbidden_io"])

    b = budget()
    check("budget realistic (not 'a few minutes')", "非" in b["estimate"] or "数十分钟" in b["estimate"])
    check("budget notes CUDA unavailable", "CUDA" in b["current_lif_cost"] or "不可用" in b["current_lif_cost"])
    check("device assertion noted", "断言" in b["device_check"])

    vd = verdict()
    check("verdict GO conditional", "GO" in vd["decision"] and "有条件" in vd["decision"])
    check("verdict lists needed evidence", len(vd["need_extra_evidence"]) >= 2)
    check("refs have DOIs", all("doi" in x for x in REFS) and any("10.1038/s41586-022-04428-3" == x["doi"] for x in REFS))

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
