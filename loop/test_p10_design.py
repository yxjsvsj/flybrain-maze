"""P10-0 回归：学习回路审计/边界/提案/裁决（纯逻辑 + 只读实据断言）。不训练、不跑小车。"""
from __future__ import annotations

import sys

from loop.p10_design import (ablation_plan, audit_mb_circuit, decision, plasticity_proposal,
                             teacher_student_boundary)

RESULTS = []


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p10_design ===========")
    A = audit_mb_circuit()
    check("KC/MBON/DAN all present", A["counts"]["KC"] > 0 and A["counts"]["MBON"] > 0
          and A["counts"]["DAN(PPL/PAM)"] > 0)
    check("KC->MBON is the strong learning synapse", A["KC_to_MBON"]["edges"] > 10000)
    check("DAN->KC modulation present", A["DAN_to_KC"]["edges"] > 10000)
    check("ToF targets LPLC2/LC4 -> learning circuit ~0",
          (A["LPLC2_LC4_to_KC"]["sumW"] + A["LPLC2_LC4_to_MBON"]["sumW"]
           + A["LPLC2_LC4_to_DAN"]["sumW"]) < 0.1)
    check("MB->DN weak (<1)", A["MBON_to_DN"]["sumW"] < 1.0 and A["KC_to_DN"]["sumW"] < 1.0)
    check("direction doc = W[post,pre]", "W[post,pre]" in A["direction"])

    b = teacher_student_boundary()
    fo = b["student_forbidden_inputs"]
    check("student forbidden inputs include map/goal/Dijkstra/teacher",
          all(any(tok in f for f in fo) for tok in ("地图", "Dijkstra", "老师", "目标方向")))
    check("supervision forbids using correct turn as sensory input",
          "禁止" in b["supervision"])
    check("POMDP lists non-identifiable states", len(b["pomdp"]["not_identifiable_from_5tof"]) >= 3)

    p = plasticity_proposal()
    check("mechanism is DA-modulated three-factor KC->MBON", "三因素" in p["mechanism"] and "KC→MBON" in p["mechanism"])
    fs = p["freeze_spec"]
    check("plastic edges = real KC->MBON only, no new connections",
          "仅" in fs["plastic_edges"] and "不新建" in fs["plastic_edges"])
    check("official W read-only + separate dW overlay",
          "只读" in fs["W_readonly"] and "ΔW" in fs["W_readonly"])
    check("decoder frozen with neural basis (no external head)",
          "冻结" in fs["decoder_frozen"] and "不得" in fs["decoder_frozen"])
    check("unknown params marked unknown (eta/Delta_max)", "未知" in fs["learning_rate"]
          and "未知" in fs["bounds"])
    check("literature has DOIs", all("doi" in x for x in p["literature"]) and len(p["literature"]) >= 3)

    d = decision(A)
    check("verdict NO-GO", d["verdict"] == "NO-GO")
    check("has >=3 blockers (ToF->circuit, MB->DN weak, g_unit)",
          len(d["blockers"]) >= 3)
    check("distinguishes literature/anatomy/learns-navigation",
          all(k in d["distinction"] for k in ("literature_function", "anatomy_path", "learns_navigation")))
    check("does not auto-switch whole synapse set", "不" in d["alternative_note"] or "另立方案" in d["alternative_note"])

    a = ablation_plan()
    check("test process excludes teacher", "不加载" in a["independent_test_process"])
    check("ablation list complete (>=6)", len(a["ablations"]) >= 6)
    check("validity: external teacher in test => invalid", "无效" in a["validity"])

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
