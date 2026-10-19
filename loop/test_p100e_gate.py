"""P10-0e 回归：静态门控、预注册要素、无训练产物、维度独立裁决、负例。不训练/不控车。"""
from __future__ import annotations

import os
import sys
import tempfile

from loop.p100e_gate import dual_verdict, evidence_matrix_check, protocol_check, run

RESULTS = []


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p100e_gate ===========")
    R = run()
    c = R["checks"]
    check("w read-only hashes match", c["w_readonly"].get("ok") is True)
    check("P1 freeze empty", c["p1_freeze"].get("ok") is True)
    check("no training artifacts", c["no_training"].get("ok") is True)
    check("protocol required elements present", c["protocol"].get("ok") is True, str(c["protocol"].get("missing")))
    check("evidence matrix present", c["evidence_matrix"].get("ok") is True)
    check("historical verdicts still NO-GO", c["history_no_go"] is True)
    check("all static checks pass", c["all_static_ok"] is True)

    v = R["dual_verdict"]
    check("Biological = NO-GO", v["Biological"] == "NO-GO")
    check("go_biological False", v["go_biological"] is False)
    check("Computational = GO (conditional)", v["Computational_Hypothesis"].startswith("GO"))
    check("go_computational True", v["go_computational"] is True)
    check("training NOT authorized", v["training_authorized"] is False)

    # 负例：缺少关键要素的协议应判 False
    fd, tmp = tempfile.mkstemp(suffix=".md", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("# empty protocol\n")
        neg = protocol_check(tmp)
        check("negative protocol rejected", neg["ok"] is False and len(neg["missing"]) > 0)
    finally:
        os.remove(tmp)
    fd, tmp = tempfile.mkstemp(suffix=".md", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("# empty closeout\n")
        neg = evidence_matrix_check(tmp)
        check("negative evidence matrix rejected", neg["ok"] is False)
    finally:
        os.remove(tmp)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
