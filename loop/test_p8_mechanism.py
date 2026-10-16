"""P8-0 回归：p8_mechanism 的机制设计纯逻辑（缺陷表/瓶颈/提案/预注册）。**无模拟器运行**。"""
from __future__ import annotations

import sys

from loop.p8_mechanism import REFS, bottleneck, prereg_p8_1, proposal, simulator_defects

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== p8_mechanism ===========")
    d = simulator_defects()
    check("defect table has 7 rows", len(d) == 7)
    check("synaptic_current marked absent", "无" in d["synaptic_current"]["impl"])
    check("cell_type_dynamics marked absent", "无" in d["cell_type_dynamics"]["impl"])
    check("defect rows have 4 columns each",
          all({"impl", "data_known", "literature", "unknown"} <= set(v) for v in d.values()))

    b = bottleneck()
    check("bottleneck references P5/P6/P7", all(k in b for k in ("P5_visual", "P6_sensory_inject", "P7_static")))
    check("bottleneck localizes to postsynaptic threshold", "阈值" in b["localization"])

    p = proposal()
    check("proposal names graded release", "渐变" in p["name"])
    check("proposal has >=3 falsification criteria", len(p["falsification"]) >= 3)
    check("proposal forbids gain/search/heading", "不增大刺激" in p["not_allowed"] and "目标角" in p["not_allowed"])
    check("proposal respects EPG/PEN NO-GO", "NO-GO" in p["PFL3_PFL2"])
    check("proposal adds no free params", "不引入新自由参数" in p["params"])

    r = prereg_p8_1()
    check("prereg asserts device + versions", "device" in r["device"] and "sha256" in r["device"])
    check("prereg has mirror control", any("镜像" in c for c in r["conditions"]))
    check("prereg gate fixed >=4/5 & 3sigma", "4/5" in r["gate"] and "3σ" in r["gate"])
    check("references non-empty", len(REFS) >= 6)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
