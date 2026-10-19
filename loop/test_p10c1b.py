"""P10-C1b 回归：专用解码器（合成）、ΔW 数学安全、真实子进程隔离、LIF 报告。不训练/不控车。"""
from __future__ import annotations

import json
import os
import sys

from loop.p10_test_decoder import (DecoderValidationError, P10TestDecoder, validate_groups)
from loop.p10c1b_lif_benchmark import dw_spec_check, isolation_suite

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def expect_raise(fn):
    try:
        fn()
        return False
    except (DecoderValidationError, ValueError, TypeError):
        return True


def test_decoder():
    groups = {"turn_L": {"bodyIds": [1]}, "turn_R": {"bodyIds": [2]},
              "fwd": {"bodyIds": [3]}, "back": {"bodyIds": [4]}}
    d = P10TestDecoder(groups)
    r0 = d.step({})
    check("no steering signal -> turn 0", r0["turn"] == 0.0 and r0["omega"] == 0.0)
    rL = d.step({"turn_L": 0.5, "turn_R": 0.0})
    rR = d.step({"turn_L": 0.0, "turn_R": 0.5})
    check("L>R -> positive turn", rL["turn"] > 0 and rL["omega"] > 0, str(rL["turn"]))
    check("R>L -> negative turn", rR["turn"] < 0 and rR["omega"] < 0, str(rR["turn"]))
    check("turn symmetric", abs(rL["turn"] + rR["turn"]) < 1e-9)
    rabl = d.step({"turn_L": 0.0, "turn_R": 0.0})
    check("neural ablation removes direction", rabl["turn"] == 0.0)
    check("fwd drives +v", d.step({"fwd": 1.0})["v"] > 0)
    check("back drives -v", d.step({"back": 1.0})["v"] < 0)
    for k in ("map", "pose", "dijkstra", "teacher_action", "teacher_label", "goal",
              "nav_memory", "planner", "turn_command"):
        check(f"forbidden input rejected: {k}", expect_raise(lambda k=k: d.step({k: 1})))
    check("NaN rejected", expect_raise(lambda: d.step({"turn_L": float("nan")})))
    check("Inf rejected", expect_raise(lambda: d.step({"fwd": float("inf")})))
    # 安全层：限速不改转向符号；急停同时停 v/omega
    s = d.step({"fwd": 1.0, "turn_L": 1.0}, {"speed_limit": 0.3})
    check("speed limit preserves turn sign", s["omega"] > 0 and abs(s["v"]) <= 0.3 + 1e-9, str(s))
    e = d.step({"fwd": 1.0, "turn_L": 1.0}, {"emergency_stop": True})
    check("emergency stop zeroes v and omega", e["v"] == 0.0 and e["omega"] == 0.0)
    # 空组/错误 bodyId/类型/侧别
    check("empty group rejected", expect_raise(lambda: P10TestDecoder({"turn_L": {"bodyIds": []},
          "turn_R": {"bodyIds": [2]}, "fwd": {"bodyIds": [3]}, "back": {"bodyIds": [4]}})))
    ids = [10, 11, 12, 13]
    ct = ["DNa02", "DNa02", "DNg100", "MDN"]
    side = ["L", "R", "L", "L"]
    good = {"turn_L": {"bodyIds": [10]}, "turn_R": {"bodyIds": [11]},
            "fwd": {"bodyIds": [12]}, "back": {"bodyIds": [13]}}
    check("valid groups accepted", validate_groups(good, ids, ct, side) is True)
    check("wrong bodyId rejected", expect_raise(lambda: validate_groups(
        {"turn_L": {"bodyIds": [99]}, "turn_R": {"bodyIds": [11]}, "fwd": {"bodyIds": [12]},
         "back": {"bodyIds": [13]}}, ids, ct, side)))
    check("wrong type rejected", expect_raise(lambda: validate_groups(
        {"turn_L": {"bodyIds": [12]}, "turn_R": {"bodyIds": [11]}, "fwd": {"bodyIds": [12]},
         "back": {"bodyIds": [13]}}, ids, ct, side)))
    check("wrong side rejected", expect_raise(lambda: validate_groups(
        {"turn_L": {"bodyIds": [11]}, "turn_R": {"bodyIds": [11]}, "fwd": {"bodyIds": [12]},
         "back": {"bodyIds": [13]}}, ids, ct, side)))


def test_dw_safety():
    d = dw_spec_check()
    s = d["synthetic"]
    check("current dW spec NOT sufficient", d["current_spec_sufficient"] is False)
    check("negative dW can make W_eff<0", s["negative_dW_makes_Weff_negative"] is True)
    check("W_eff floor fixes negativity", s["proposed_Weff_floor_fixes"] is True)
    check("L1 bound can break", s["l1_break"] is True, f"{s['l1_before']}->{s['l1_after']}")
    check("unmasked dW creates new edges", s["new_edges_if_unmasked"] >= 1)
    check("masked dW leaves unapproved unchanged", s["unapproved_unchanged_with_mask"] is True)
    check("erratum pending approval (not implemented)", "PENDING_APPROVAL" in d["proposed_erratum"]["status"])


def test_isolation():
    iso = isolation_suite()
    check("mock process isolation passes", iso["mock_process_isolation"] is True)
    check("real student isolation NOT yet verified", iso["real_student_isolation"] is False)
    check("positive child OK", iso["positive"]["returncode"] == 0 and iso["positive"]["stdout"] == "OK")
    check("all forbidden children rejected", all(v["returncode"] == 2 for v in iso["negative"].values()))


def test_report():
    p = os.path.join(HERE, "p10c1b_report.json")
    if not os.path.exists(p):
        check("report exists", False); return
    d = json.load(open(p, encoding="utf-8"))
    b = d["benchmark"]; v = b["verification"]
    check("subnet n=23218", v["n_neurons"] == 23218)
    check("subnet nnz=3549643", v["nnz"] == 3549643)
    check("bodyId unique", v["bodyId_unique"] is True)
    check("W[post,pre] direction correct", v["direction_check"]["correct"] is True)
    check("relevant DN preserved (others reported as truncated)",
          v["n_DN_preserved"] > 0 and v["n_DN_preserved"] + v["n_DN_outside_subnet"] == v["n_DN_total"])
    check("LIF benchmark ok", b["ok"] is True)
    check("repeatable measurement", b["repeatable"] is True)
    check("peak memory measured (>0)", b["peak_mem_MB"] > 0, str(b.get("peak_mem_MB")))
    check("resource ready (1890 trials <= budget)", d["resource_budget"]["resource_ready"] is True)
    check("1890 trials reported", d["resource_budget"]["trials_1890"] == 1890)
    check("decoder_ready False", d["decoder_freeze"]["decoder_ready"] is False)
    check("overall execution NO-GO", d["verdicts"]["Overall_Execution_Readiness"] == "NO-GO")
    check("training not authorized", d["training_authorized"] is False)


def main() -> int:
    print("=========== p10c1b ===========")
    test_decoder()
    test_dw_safety()
    test_isolation()
    test_report()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
