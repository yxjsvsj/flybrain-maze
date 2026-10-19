"""P10-C1a 回归：解码器审计、冻结规格、ΔW 规则、学生隔离（正/负例）、资源、执行准入。不训练/不控车。"""
from __future__ import annotations

import json
import os
import sys
import tempfile

from loop.p10c1a_gate import (StudentSandbox, audit_decoder, check_dw_rules, check_freeze_spec,
                              isolation_selftest, protocol_rules_check, resource_budget, run)

RESULTS = []


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p10c1a_gate ===========")
    dec = audit_decoder()
    check("decoder_ready False for current decode.py", dec["decoder_ready"] is False)
    check("risky direction sources detected", len(dec["risky_direction_sources"]) >= 5,
          str(dec["risky_direction_sources"]))
    check("decoder proposal present", "P10TestDecoder" in dec["proposed_interface"]["name"])

    fs = check_freeze_spec()
    check("freeze spec complete", fs["ok"] is True, str(fs.get("missing")))
    dw = check_dw_rules(fs)
    check("dW rules clear", dw["ok"] is True)
    check("sign/topology no new edges", "NO new edges" in dw["sign_topology"])
    check("no unrelated renormalization", "NO renormalization" in dw["l1_normalization"])

    pr = protocol_rules_check()
    check("A/B/C + controls + stats present", pr["ok"] is True, str(pr.get("missing")))

    iso = isolation_selftest()
    check("isolation positive allowed", iso["positive"] is True)
    check("all forbidden inputs caught", all(v == "caught" for v in iso["negative"].values()))
    check("forbidden modules caught", all(v == "caught" for v in iso["imports"].values()))
    check("forbidden files caught", all(v == "caught" for v in iso["files"].values()))

    res = resource_budget()
    check("subnet size 23218", res["subnetwork"]["n_neurons"] == 23218)
    check("nnz 3549643", res["subnetwork"]["nnz"] == 3549643)
    check("LIF benchmark MISSING -> not resource_ready", res["resource_ready"] is False)

    R = run()
    check("execution_ready False", R["execution_ready"] is False)
    check("training_authorized False", R["training_authorized"] is False)
    check("Biological NO-GO", R["verdicts"]["Biological"] == "NO-GO")
    check("Computational conditional GO", R["verdicts"]["Computational_Hypothesis"].startswith("GO"))
    check("Execution Readiness NO-GO", R["verdicts"]["Execution_Readiness"] == "NO-GO")
    check("blocking includes decoder", "3_decoder_ready" in R["blocking_reasons"])
    check("blocking includes resource", "9_resource_budget" in R["blocking_reasons"])

    # 负例：缺 eta 的冻结规格应失败
    spec = json.load(open(os.path.join(os.path.dirname(__file__), "p10c1a_freeze_spec.json"), encoding="utf-8"))
    fd, tmp = tempfile.mkstemp(suffix=".json", text=True)
    try:
        bad = dict(spec); bad["params"] = {k: v for k, v in spec["params"].items() if k != "eta"}
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(bad, f)
        neg = check_freeze_spec(tmp)
        check("negative freeze spec (no eta) rejected", neg["ok"] is False)
    finally:
        os.remove(tmp)

    # 负例：无 risky 的合成解码器应通过审计
    fd, tmp = tempfile.mkstemp(suffix=".py", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("def update(rates): return rates\n")
        ok = audit_decoder(tmp)
        check("clean synthetic decoder passes audit", ok["decoder_ready"] is True)
    finally:
        os.remove(tmp)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
