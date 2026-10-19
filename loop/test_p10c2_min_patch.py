"""P10-C2 回归：极小尺度精确 p、正式审批配置门槛、合成隔离接口、资源与边界。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

import numpy as np

from loop.p10c2_0b_adjudication import (APPROVAL_CONFIG_PATH, adjudicate_raw,
                                        _adjudicate_raw_with_threshold, make_payload,
                                        resource_budget, selftest, verdicts, load_approval_config,
                                        canonical_checksum)
from loop.p10c2_0b_exact_stats import sign_flip_p, sign_flip_p_enum

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def test_tiny_exact_p():
    check("[1e-14]*6 -> 0.03125", abs(sign_flip_p([1e-14] * 6) - 0.03125) < 1e-12,
          str(sign_flip_p([1e-14] * 6)))
    check("[1.0]*6 -> 0.03125", abs(sign_flip_p([1.0] * 6) - 0.03125) < 1e-12,
          str(sign_flip_p([1.0] * 6)))
    check("scale invariant 1e-14 vs 1.0", sign_flip_p([1e-14] * 6) == sign_flip_p([1.0] * 6))
    check("all zero -> 1.0", sign_flip_p([0.0, 0.0]) == 1.0)
    check("full cancel -> 1.0", sign_flip_p([1.0, -1.0]) == 1.0)
    rng = np.random.default_rng(11)
    agree = True
    for n in range(4, 17):
        for _ in range(5):
            x = rng.integers(-2, 3, n).astype(float) / 2.0
            if abs(sign_flip_p(x) - sign_flip_p_enum(x)) > 1e-12:
                agree = False
    check("enum agreement n=4..16", agree)
    for bad in ([], [float("nan")], [float("inf")]):
        try:
            sign_flip_p(bad); ok = False
        except ValueError:
            ok = True
        check(f"invalid input rejected {bad[:1]}", ok)


def test_approval_default_pending():
    ap = load_approval_config(APPROVAL_CONFIG_PATH)
    check("formal config valid", ap["valid"] is True)
    check("status PENDING_APPROVAL", ap["status"] == "PENDING_APPROVAL")
    check("no production threshold", ap["threshold"] is None)
    r = adjudicate_raw(make_payload("pass"))
    check("production default -> INCONCLUSIVE", r["state"] == "INCONCLUSIVE")
    check("threshold status PENDING_APPROVAL", r["eligibility"]["threshold_status"] == "PENDING_APPROVAL")
    check("no A0_PASS without approval", r["state"] != "A0_PASS")


def _write_cfg(cfg):
    body = json.dumps({k: v for k, v in cfg.items() if k != "checksum"},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    cfg = dict(cfg); cfg["checksum"] = hashlib.sha256(body.encode()).hexdigest()
    fd, p = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return p


def test_approval_paths():
    base = {"protocol_version": "P10-C2-0a/v1", "status": "APPROVED", "rule": "min_evoked_mean",
            "value": 0.01, "approved_by": "unit-test", "approved_at": "2026-01-01T00:00:00Z"}
    p = _write_cfg(base)
    try:
        r = adjudicate_raw(make_payload("pass"), approval_config_path=p)
        check("approved config -> A0_PASS (synthetic)", r["state"] == "A0_PASS")
        check("approved threshold used", r["eligibility"]["threshold"] == 0.01)
    finally:
        os.remove(p)
    # checksum mismatch
    p = _write_cfg(base)
    try:
        bad = json.load(open(p, encoding="utf-8")); bad["value"] = 0.02
        json.dump(bad, open(p, "w", encoding="utf-8"))
        r = adjudicate_raw(make_payload("pass"), approval_config_path=p)
        check("tampered checksum -> INCONCLUSIVE", r["state"] == "INCONCLUSIVE")
    finally:
        os.remove(p)
    # wrong version
    p = _write_cfg({**base, "protocol_version": "old/v0"})
    try:
        r = adjudicate_raw(make_payload("pass"), approval_config_path=p)
        check("wrong version -> INCONCLUSIVE", r["state"] == "INCONCLUSIVE")
    finally:
        os.remove(p)
    # incomplete approval metadata
    p = _write_cfg({**base, "approved_by": None})
    try:
        r = adjudicate_raw(make_payload("pass"), approval_config_path=p)
        check("incomplete metadata -> INCONCLUSIVE", r["state"] == "INCONCLUSIVE")
    finally:
        os.remove(p)


def test_synthetic_isolated():
    s = selftest()
    check("synthetic five states correct", s["all_correct"] is True)
    check("A0_PASS only via internal test interface",
          s["cases"]["A0_PASS"] == "A0_PASS" and s["cases"]["PENDING_APPROVAL"] == "INCONCLUSIVE")
    r = _adjudicate_raw_with_threshold(make_payload("pass"), None)
    check("internal without threshold -> INCONCLUSIVE", r["state"] == "INCONCLUSIVE")


def test_resource_and_verdicts():
    r = resource_budget()
    check("30,000 steps / 39 s", r["lif_steps"] == 30000 and abs(r["core_lif_s"] - 39.0) < 1e-6)
    check("worst < 15 min", r["worst_case_2x_min"] < 15.0)
    v = verdicts(True, True, True, True)
    check("eligibility PENDING_APPROVAL", v["Eligibility_Threshold"] == "PENDING_APPROVAL")
    check("A0 not executed", v["A0_Executed"] is False)
    check("training False", v["Training_Authorization"] is False)
    check("B/C/Biological NO-GO", v["Level_B"] == "NO-GO" and v["Level_C"] == "NO-GO" and v["Biological"] == "NO-GO")


def main() -> int:
    print("=========== p10c2 minimal patch ===========")
    test_tiny_exact_p()
    test_approval_default_pending()
    test_approval_paths()
    test_synthetic_isolated()
    test_resource_and_verdicts()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
