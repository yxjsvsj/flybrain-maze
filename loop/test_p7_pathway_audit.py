"""P7-0 回归：p7_pathway_audit 的静态连接审计（实据断言）。无模拟器。"""
from __future__ import annotations

import sys

from loop.p7_pathway_audit import audit, conn, three_functions
from loop.visual_motor_audit import load_data

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== p7_pathway_audit ===========")
    D = load_data()
    A = audit(D)
    check("LC10a and DNa02 both exist", A["counts"]["LC10a"] > 0 and A["counts"]["DNa02"] > 0)
    # 核心实据：DNa02 <- LC10a 无直接边
    check("no direct LC10a->DNa02 edge", A["LC10a_to_DNa02"]["n_edges"] == 0)
    # DNa02 的视觉输入权重可忽略（净 ~0.01 << vnc 2.5）
    v = A["DNa02_receives_visual"]
    check("DNa02 visual input weight negligible", abs(v["w_sum"]) < 0.1)
    check("DNa02 visual pos/neg nearly cancel", abs(v["w_pos"] + v["w_neg"]) < 0.05)
    # DNa02 主要输入非视觉
    check("DNa02 top input not visual_projection", A["DNa02_top_inputs"][0][0] != "visual_projection")
    # visual->DN 以同侧为主
    check("visual->DN ipsi > contra",
          A["visual_to_DN_ipsi_contra"].get("ipsi", 0) > A["visual_to_DN_ipsi_contra"].get("contra", 0))
    # 三功能判定：障碍→转向 与 回中性 不支持
    F = three_functions(A)
    check("obstacle->turning NOT supported", F["obstacle_triggers_turning"]["supported"] is False)
    check("return-to-neutral NOT supported (static)", F["return_to_neutral"]["supported"] is False)
    # conn 方向自洽：DNa02 <- LC10a 与 LC10a <- DNa02 之和一致（都是 0）
    check("conn symmetric-zero", conn(D, ["DNa02"], ["LC10a"])["w_sum"] == conn(D, ["LC10a"], ["DNa02"])["w_sum"])

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
