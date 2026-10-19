"""P10-0e/P10-C1a 回归：静态门控修正 + 负例（缺裁决/缺产物/训练产物/坏 hash）。不训练/不控车。"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

from loop.p100e_gate import (evidence_matrix_check, history_verdicts, no_training_check,
                             protocol_check, run, w_readonly_check)

RESULTS = []


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def main() -> int:
    print("=========== p100e_gate ===========")
    R = run()
    c = R["checks"]
    check("full SHA-256 match", c["w_readonly"].get("ok") is True)
    check("P1 freeze empty", c["p1_freeze"].get("ok") is True)
    check("no training artifacts", c["no_training"].get("ok") is True)
    check("history verdicts readable & empty problems", c["history"].get("problems") == [])
    check("p100 decision.verdict = NO-GO", c["history"]["verdicts"]["p100"]["verdict"] == "NO-GO")
    check("p100 go False", c["history"]["verdicts"]["p100"]["go"] is False)
    check("protocol elements present", c["protocol"].get("ok") is True)
    check("evidence matrix present", c["evidence_matrix"].get("ok") is True)
    check("historical NO-GO preserved", c["history_no_go"] is True)
    check("all static checks pass", c["all_static_ok"] is True)

    v = R["dual_verdict"]
    check("Biological NO-GO", v["Biological"] == "NO-GO")
    check("worth_doing_computational True", v["worth_doing_computational"] is True)
    check("execution_ready False (not hardcoded GO)", v["execution_ready"] is False)
    check("training NOT authorized", v["training_authorized"] is False)

    # ---- 负例 ----
    tmp = tempfile.mkdtemp(prefix="p100e_neg_")
    try:
        # 1) 缺失关键报告
        h = history_verdicts(base=tmp)
        check("negative: missing reports -> problems", len(h["problems"]) > 0)
        nt = no_training_check(base=tmp)
        check("negative: missing expected -> fail", nt["ok"] is False)
        # 2) 空裁决
        write(os.path.join(tmp, "p100.json"), {"decision": {"verdict": ""}})
        h = history_verdicts(base=tmp)
        check("negative: empty decision.verdict -> problems",
              any("decision.verdict" in p for p in h["problems"]))
        # 3) 禁止的训练产物
        open(os.path.join(tmp, "p10c1_delta_w.npz"), "wb").close()
        nt = no_training_check(base=tmp)
        check("negative: forbidden training artifact -> fail", nt["ok"] is False)
        # 4) 坏 hash（伪造 W 文件）
        wtmp = tempfile.mkdtemp(prefix="p100e_w_")
        try:
            open(os.path.join(wtmp, "brain.npz"), "wb").write(b"fake")
            open(os.path.join(wtmp, "weights.npz"), "wb").write(b"fake")
            wr = w_readonly_check(w_dir=wtmp)
            check("negative: bad hash -> fail", wr["ok"] is False)
        finally:
            shutil.rmtree(wtmp, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
