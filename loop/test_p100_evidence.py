"""P10-0a 回归：只读证据复核（直接/间接连接、Σw/Σ|w|/正负、p91b1 来源、裁决）。不训练、不跑小车。"""
from __future__ import annotations

import sys

from loop.p100_evidence import (P912_ORIG, audit, block, check_p91b1_provenance, pomdp_variable_goal,
                                review, verdict)

RESULTS = []


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p100_evidence ===========")
    A = audit()
    check("audit sets non-empty", all(len(A[k]) > 0 for k in ("KC", "MBON", "DAN", "LP", "DN")))
    R = review()
    # 直接缺失（数据支持）
    check("LP direct -> KC = 0 edges", R["LP_direct_to_KC"]["edges"] == 0)
    check("LP direct -> DAN = 0 edges", R["LP_direct_to_DAN"]["edges"] == 0)
    # 间接存在（勘误）
    check("LP 2hop -> KC exists (>0 intermediates)", R["LP_2hop_to_KC"]["n_intermediate"] > 0)
    check("LP 2hop -> DAN exists", R["LP_2hop_to_DAN"]["n_intermediate"] > 0)
    # Σ|w| 与正负（不得只看 Σw）
    m = R["MBON_direct_to_DN"]
    check("MBON->DN has both pos and neg edges", m["pos_edges"] > 0 and m["neg_edges"] > 0)
    check("MBON->DN absW > sumW (cancellation)", m["absW"] > m["sumW"] * 2)
    check("MBON->DN 2hop exists", R["MBON_2hop_to_DN"]["n_intermediate"] > 0)
    check("KC->DN all positive", R["KC_direct_to_DN"]["neg_edges"] == 0)

    # p91b1 来源未混淆
    prov = check_p91b1_provenance()
    check("ms_per_step matches original 2f12217", prov["ms_per_step_matches_original_2f12217"])
    check("remeasure stored separately", prov["remeasure_present_and_separate"])
    check("traced_peak_mem present", prov["traced_peak_mem_present"])
    check("provenance not confused", prov["confused"] is False)

    # 裁决：维持 NO-GO，未推翻
    V = verdict(R)
    check("verdict NO-GO (maintained)", V["verdict"].startswith("NO-GO"))
    check("not overturned", V["overturned"] is False)
    check("evidence levels include direct-absent/indirect/mb/ g_unit",
          all(k in V["evidence_levels"] for k in ("L1_direct_absent", "L2_indirect", "L3_MB_to_DN", "L4_g_unit")))
    check("blockers >=3", len(V["remaining_blockers"]) >= 3)

    # POMDP
    P = pomdp_variable_goal()
    check("POMDP flags variable-goal non-identifiability",
          "不可辨识" in P["variable_goal_note"] and len(P["not_identifiable"]) >= 3)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
