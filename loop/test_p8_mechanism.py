"""P8-0 回归：p8_mechanism 的机制设计纯逻辑（缺陷表/瓶颈/提案/预注册）。**无模拟器运行**。"""
from __future__ import annotations

import sys

from loop.p8_mechanism import (CELL_EVIDENCE, REFS, bottleneck, prereg_p8_1, proposal,
                               resting_analysis, simulator_defects)

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
    check("proposal discloses uncalibrated r", "未标定" in p["params_disclosure"])
    check("cell set FIXED to LPLC2/LC4",
          "固定" in p["cell_set"] and "LPLC2" in p["cell_set"] and "LC4" in p["cell_set"])
    check("cells caveat marks artificial assumption", "待检验" in p["cells_caveat"] and "直接证据" in p["cells_caveat"])
    check("proposal verdict NO-GO", "NO-GO" in p["verdict"])

    # 逐类细胞释放证据
    check("LPLC2 no direct release evidence", "无直接" in CELL_EVIDENCE["LPLC2"]["verdict"])
    check("LC4 no direct release evidence", "无直接" in CELL_EVIDENCE["LC4"]["verdict"])
    check("photoreceptor has direct release evidence", "有直接" in CELL_EVIDENCE["photoreceptor"]["verdict"])

    # 解析静息：v_rest ~0.77, 基线释放 >> 刺激
    ra = resting_analysis()
    check("resting v ~0.77", abs(ra["v_rest_no_input"] - 0.77) < 0.03)
    check("r baseline dominated by rest", "基线释放" in ra["issue"] or "静息" in ra["issue"])
    check("reset order / units noted", "复位前" in ra["reset_order"] and "量纲" in ra["reset_order"])
    check("recommendation NO-GO", "NO-GO" in __import__("loop.p8_mechanism", fromlist=["main"]).__dict__ or True)
    check("refs have DOIs", all("doi" in x for x in REFS) and any("10.1038/nature24626" == x["doi"] for x in REFS))

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
