"""P10-C2-0b：**原始逐 seed 记录 -> 五状态裁决**流水线（纯数据；不运行神经动力学）。

输入必须是原始观测（entrance/KC/MBON rates + eligibility 时间序列），不是手填 p 值/效应量。
资格迹最小有效效应**未获独立依据** -> 默认 `PENDING_APPROVAL`，不得暗自选阈值。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

from loop.p10c2_0b_exact_stats import (ALPHA, VALID_N, PLANNED_N, SEEDS, bootstrap_ci,
                                       sign_flip_p)

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_VERSION = "P10-C2-0a/v1"
APPROVAL_CONFIG_PATH = os.path.join(HERE, "p10c2_eligibility_threshold_approval.json")
DT = 0.02
TAU_PRE = 0.05
TAU_POST = 0.05
TAU_E = 0.5
DV = 0.2
AMOUNT = 0.036254
STEPS = 400
STIM_WINDOW = [100, 300]
MAX_EXCLUSIONS = 3
CONDITIONS = ("none", "stim", "recover")


def _finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:                            # noqa: BLE001
        return False


def validate_params(params):
    if not isinstance(params, dict):
        return False
    exp = {"dt": DT, "tau_pre": TAU_PRE, "tau_post": TAU_POST, "tau_e": TAU_E,
           "dv": DV, "steps": STEPS, "stim_window": STIM_WINDOW}
    for k, v in exp.items():
        if params.get(k) != v:
            return False
    return True


def canonical_checksum(cfg):
    body = {k: v for k, v in cfg.items() if k != "checksum"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_approval_config(path=None):
    """正式审批配置：版本 + 校验和 + 批准状态/数值。默认 PENDING_APPROVAL。"""
    p = path or APPROVAL_CONFIG_PATH
    if not os.path.exists(p):
        return {"valid": False, "reason": "approval config missing", "status": None}
    cfg = json.load(open(p, encoding="utf-8"))
    if cfg.get("checksum") != canonical_checksum(cfg):
        return {"valid": False, "reason": "approval config checksum mismatch", "status": None}
    if cfg.get("protocol_version") != PROTOCOL_VERSION:
        return {"valid": False, "reason": "approval protocol version mismatch", "status": None}
    status = cfg.get("status")
    if status == "APPROVED":
        v = cfg.get("value")
        if not (isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) >= 0):
            return {"valid": False, "reason": "approved threshold invalid", "status": status}
        if not cfg.get("rule") or not cfg.get("approved_by") or not cfg.get("approved_at"):
            return {"valid": False, "reason": "approval metadata incomplete", "status": status}
        return {"valid": True, "status": status, "threshold": float(v), "config": cfg}
    if status in ("PENDING_APPROVAL", "REJECTED"):
        return {"valid": True, "status": status, "threshold": None, "config": cfg}
    return {"valid": False, "reason": "unknown approval status", "status": status}


def adjudicate_raw(payload, approval_config_path=None):
    """**正式入口**：只从带校验和的审批配置取阈值；无批准 -> INCONCLUSIVE。"""
    ap = load_approval_config(approval_config_path)
    if not ap.get("valid"):
        return {"state": "INCONCLUSIVE", "reasons": [ap.get("reason", "invalid approval")],
                "approval": ap}
    return _adjudicate_raw_with_threshold(payload, ap.get("threshold"), approval=ap)


def _adjudicate_raw_with_threshold(payload, eligibility_threshold, approval=None):
    reasons = []
    if not isinstance(payload, dict):
        return {"state": "INCONCLUSIVE", "reasons": ["payload not a dict"]}
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        return {"state": "INCONCLUSIVE", "reasons": ["protocol version mismatch"]}
    if not validate_params(payload.get("params")):
        return {"state": "INCONCLUSIVE", "reasons": ["unfrozen/mismatched params"]}
    recs = payload.get("records")
    if not isinstance(recs, list) or len(recs) == 0:
        return {"state": "INCONCLUSIVE", "reasons": ["missing raw records"]}
    by = {}
    hard_invalid = []
    for r in recs:
        if not isinstance(r, dict) or "seed" not in r or "condition" not in r:
            hard_invalid.append("record missing seed/condition")
            continue
        s, c = r["seed"], r["condition"]
        if s not in SEEDS:
            hard_invalid.append(f"unregistered seed {s}")
            continue
        if c not in CONDITIONS:
            hard_invalid.append(f"bad condition {c}")
            continue
        if (s, c) in by:
            hard_invalid.append(f"duplicate seed/condition {s}/{c}")
            continue
        by[(s, c)] = r
    if hard_invalid:
        return {"state": "INCONCLUSIVE", "reasons": hard_invalid[:10]}
    # 每 seed 三条件完整；非有限/缺失 -> 排除该 seed
    valid, exclusions = [], []
    for s in SEEDS:
        trio = [by.get((s, c)) for c in CONDITIONS]
        if any(t is None for t in trio):
            hard_invalid.append(f"missing condition for seed {s}")
            continue
        ok = all(_finite(t.get("entrance_rate")) and _finite(t.get("kc_rate"))
                 and _finite(t.get("mbon_rate")) and isinstance(t.get("eligibility_series"), (list, tuple))
                 and len(t["eligibility_series"]) == STEPS
                 and all(_finite(x) for x in t["eligibility_series"]) for t in trio)
        if not ok:
            exclusions.append({"seed": s, "reason": "non-finite/malformed raw data"})
            continue
        valid.append(s)
    if hard_invalid:
        return {"state": "INCONCLUSIVE", "reasons": hard_invalid[:10], "exclusions": exclusions}
    if not valid:
        return {"state": "INCONCLUSIVE", "reasons": ["no valid seeds"], "exclusions": exclusions}
    # 逐 seed 配对
    ent = np.array([by[(s, "stim")]["entrance_rate"] - by[(s, "none")]["entrance_rate"] for s in valid])
    kc = np.array([by[(s, "stim")]["kc_rate"] - by[(s, "none")]["kc_rate"] for s in valid])
    mb = np.array([by[(s, "stim")]["mbon_rate"] - by[(s, "none")]["mbon_rate"] for s in valid])
    # 资格迹：刺激诱发的时间分辨差异（stim - none），窗口均值
    a, b = STIM_WINDOW
    el = np.array([np.mean(np.asarray(by[(s, "stim")]["eligibility_series"])[a:b]
                          - np.asarray(by[(s, "none")]["eligibility_series"])[a:b]) for s in valid])
    def rel(d, base):
        m = float(base)
        return float(d.mean()) / m if m else float("nan")
    ent_base = float(np.mean([by[(s, "none")]["entrance_rate"] for s in valid]))
    kc_base = float(np.mean([by[(s, "none")]["kc_rate"] for s in valid]))
    mb_base = float(np.mean([by[(s, "none")]["mbon_rate"] for s in valid]))
    ent_p = float(sign_flip_p(ent)) if len(ent) > 1 else 1.0
    kc_p = float(sign_flip_p(kc)) if len(kc) > 1 else 1.0
    ent_ci = bootstrap_ci(ent); kc_ci = bootstrap_ci(kc)
    result = {
        "valid_seeds": len(valid), "n_exclusions": len(exclusions), "exclusions": exclusions[:10],
        "entrance": {"diff_mean": round(float(ent.mean()), 8), "relative": round(rel(ent, ent_base), 6),
                     "p": round(ent_p, 6), "ci": ent_ci, "baseline": round(ent_base, 6)},
        "kc": {"diff_mean": round(float(kc.mean()), 8), "relative": round(rel(kc, kc_base), 6),
               "p": round(kc_p, 6), "ci": kc_ci, "baseline": round(kc_base, 6)},
        "mbon": {"diff_mean": round(float(mb.mean()), 8), "relative": round(rel(mb, mb_base), 6),
                 "baseline": round(mb_base, 6)},
        "approval": approval if approval is not None else {"status": "SYNTHETIC_TEST_ONLY"},
        "eligibility": {"evoked_mean": round(float(el.mean()), 8),
                        "threshold": eligibility_threshold,
                        "threshold_status": ("OK" if eligibility_threshold is not None else "PENDING_APPROVAL"),
                        "stim_window": STIM_WINDOW},
        "ceiling": {"kc_baseline": round(kc_base, 6), "mbon_baseline": round(mb_base, 6),
                    "limit": 0.85},
    }
    # 五状态（优先级）
    if len(valid) < VALID_N or len(exclusions) > MAX_EXCLUSIONS:
        result.update(state="INCONCLUSIVE", reasons=["insufficient valid seeds / too many exclusions"])
        return result
    if not (result["entrance"]["diff_mean"] > 0 and ent_p < ALPHA
            and result["entrance"]["relative"] >= 0.10):
        result.update(state="INPUT_INVALID", reasons=["entrance positive control failed"])
        return result
    if not (np.all(np.isfinite(el)) and len(el) > 0):
        result.update(state="METRIC_INVALID", reasons=["eligibility series invalid"])
        return result
    if eligibility_threshold is None:
        result.update(state="INCONCLUSIVE",
                      reasons=["eligibility minimum effect threshold PENDING_APPROVAL"])
        return result
    if not (result["eligibility"]["evoked_mean"] > eligibility_threshold):
        result.update(state="METRIC_INVALID", reasons=["stimulus-evoked eligibility below frozen threshold"])
        return result
    if kc_base >= 0.85 or mb_base >= 0.85:
        result.update(state="INCONCLUSIVE", reasons=["ceiling (baseline rate >=0.85)"])
        return result
    if not (result["kc"]["diff_mean"] > 0 and kc_p < ALPHA
            and result["kc"]["relative"] >= 0.05 and kc_ci["ci_low"] > 0):
        result.update(state="SENSORY_NO_EFFECT", reasons=["KC primary endpoint failed"])
        return result
    result.update(state="A0_PASS", reasons=["all pre-registered gates met"])
    return result


def _trio(seed, entrance, kc, mbon, el_none, el_stim):
    return [
        {"seed": seed, "condition": "none", "entrance_rate": entrance[0], "kc_rate": kc[0],
         "mbon_rate": mbon[0], "eligibility_series": el_none},
        {"seed": seed, "condition": "stim", "entrance_rate": entrance[1], "kc_rate": kc[1],
         "mbon_rate": mbon[1], "eligibility_series": el_stim},
        {"seed": seed, "condition": "recover", "entrance_rate": entrance[0], "kc_rate": kc[0],
         "mbon_rate": mbon[0], "eligibility_series": el_none},
    ]


def make_payload(kind="pass", seeds=SEEDS):
    rng = np.random.default_rng(4242)
    recs = []
    for s in seeds:
        ent_n, ent_s = 0.2, 0.2 + (0.2 if kind in ("pass", "invalid_metric") else 0.0)
        kc_n, kc_s = 0.4, 0.4 + (0.1 if kind in ("pass", "invalid_metric") else 0.0)
        mb_n, mb_s = 0.4, 0.4 + (0.05 if kind in ("pass",) else 0.0)
        if kind == "invalid_entrance":
            ent_s = ent_n
        if kind == "invalid_metric":
            el_n = [1.0] * STEPS; el_s = [1.0] * STEPS
        else:
            el_n = list(0.1 + 0.01 * rng.normal(size=STEPS))
            el_s = list(np.asarray(el_n) + (0.05 if kind == "pass" else 0.0))
        if kind == "ceiling":
            kc_n = kc_s = 0.9
        recs.extend(_trio(s, (ent_n, ent_s), (kc_n, kc_s), (mb_n, mb_s), el_n, el_s))
    return {"protocol_version": PROTOCOL_VERSION,
            "params": {"dt": DT, "tau_pre": TAU_PRE, "tau_post": TAU_POST, "tau_e": TAU_E,
                       "dv": DV, "steps": STEPS, "stim_window": STIM_WINDOW},
            "records": recs}


def selftest():
    cases = {}
    cases["A0_PASS"] = _adjudicate_raw_with_threshold(make_payload("pass"), 0.01)["state"]
    cases["INPUT_INVALID"] = _adjudicate_raw_with_threshold(make_payload("invalid_entrance"), 0.01)["state"]
    cases["METRIC_INVALID"] = _adjudicate_raw_with_threshold(make_payload("invalid_metric"), 0.01)["state"]
    p = make_payload("pass")
    for r in p["records"]:
        if r["condition"] == "stim":
            r["kc_rate"] = r["entrance_rate"]  # kc diff ~0
    cases["SENSORY_NO_EFFECT"] = _adjudicate_raw_with_threshold(p, 0.01)["state"]
    p = make_payload("pass", seeds=SEEDS[:10])
    cases["INCONCLUSIVE"] = _adjudicate_raw_with_threshold(p, 0.01)["state"]
    cases["PENDING_APPROVAL"] = _adjudicate_raw_with_threshold(make_payload("pass"), None)["state"]
    # 恶意输入
    cases["summary_only"] = _adjudicate_raw_with_threshold({"protocol_version": PROTOCOL_VERSION,
                                            "params": make_payload()["params"],
                                            "kc_p": 0.001, "effect": 0.5}, None)["state"]
    p = make_payload("pass"); p["records"].append(dict(p["records"][0]))
    cases["duplicate"] = _adjudicate_raw_with_threshold(p, 0.01)["state"]
    p = make_payload("pass"); p["records"] = [r for r in p["records"] if not (r["seed"] == 101 and r["condition"] == "stim")]
    cases["missing_condition"] = _adjudicate_raw_with_threshold(p, 0.01)["state"]
    p = make_payload("pass"); p["protocol_version"] = "old/v0"
    cases["version_mismatch"] = _adjudicate_raw_with_threshold(p, 0.01)["state"]
    allowed = {"A0_PASS", "INPUT_INVALID", "METRIC_INVALID", "SENSORY_NO_EFFECT", "INCONCLUSIVE"}
    ok = (cases["A0_PASS"] == "A0_PASS" and cases["INPUT_INVALID"] == "INPUT_INVALID"
          and cases["METRIC_INVALID"] == "METRIC_INVALID" and cases["SENSORY_NO_EFFECT"] == "SENSORY_NO_EFFECT"
          and cases["INCONCLUSIVE"] == "INCONCLUSIVE" and cases["PENDING_APPROVAL"] == "INCONCLUSIVE"
          and all(cases[k] in allowed and cases[k] != "A0_PASS" for k in
                  ("summary_only", "duplicate", "missing_condition", "version_mismatch")))
    return {"cases": cases, "all_correct": bool(ok)}


def resource_budget():
    lif_steps = PLANNED_N * len(CONDITIONS) * STEPS
    core_s = lif_steps * 1.3 / 1000.0
    init_s = PLANNED_N * len(CONDITIONS) * 1.0
    elig_s = 20.0; save_s = 5.0
    total_s = core_s + init_s + elig_s + save_s
    return {"lif_steps": lif_steps, "core_lif_s": round(core_s, 2), "init_rebuild_s": round(init_s, 2),
            "eligibility_s": elig_s, "save_stats_s": save_s,
            "total_min": round(total_s / 60.0, 3), "worst_case_2x_min": round(2 * total_s / 60.0, 3),
            "cap_min": 15.0, "cap_MB": 2048, "class3_learning": "UNKNOWN",
            "old_doc_error": "39,000 steps / 50.7 s was WRONG; correct 30,000 steps / 39.0 s",
            "ready": bool(2 * total_s / 60.0 <= 15.0)}


def verdicts(exact_ok, power_ok, adj_ok, res_ok):
    ready = bool(exact_ok and power_ok and adj_ok and res_ok)
    return {"Exact_P_Algorithm_Fixed": bool(exact_ok),
            "Primary_Power_At_n22_ge_0.80": bool(power_ok),
            "Raw_Record_Adjudication": bool(adj_ok),
            "Eligibility_Threshold": "PENDING_APPROVAL",
            "Resource_Consistent_And_Ready": bool(res_ok),
            "Ready_To_Request_One_A0": ready,
            "A0_Executed": False,
            "Level_A_Execution": "NO-GO (approval/run pending)",
            "Level_B": "NO-GO", "Level_C": "NO-GO", "Biological": "NO-GO",
            "Training_Authorization": False}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    from loop.p10c2_0b_exact_stats import exact_stats_selftest
    from loop.p10c2_0b_exact_stats import (paired_power_normal, paired_power_t)
    ex = exact_stats_selftest()
    st = json.load(open(os.path.join(HERE, "p10c2_0b_exact_stats.json"), encoding="utf-8"))
    p20, p22 = st["power"]["n20"]["power"], st["power"]["n22"]["power"]
    power_ok = (p22 >= 0.80 and st["power"]["n22"]["mc_se"] < 0.02)
    adj = selftest(); res = resource_budget()
    exact_ok = (ex["enum_agree_4_16"] and ex["permutation_invariant"]
                and ex["positive_scaling_invariant"] and ex["n25_in_range"]
                and ex["rejects_empty"] and ex["rejects_nonfinite"]
                and all(v["in_range"] for v in ex["cases"].values()))
    v = verdicts(exact_ok, power_ok, adj["all_correct"], res["ready"])
    report = {"protocol_version": PROTOCOL_VERSION, "exact_stats": ex,
              "power_recheck": {"n20": st["power"]["n20"], "n22": st["power"]["n22"],
                                "power_normal_n20": round(paired_power_normal(0.65, 20), 4),
                                "power_t_n20": round(paired_power_t(0.65, 20), 4),
                                "note": "0.831 is KC single-primary hypothetical power, NOT joint A0 power"},
              "adjudication": adj, "resource": res, "verdicts": v,
              "a0_executed": False, "training_authorized": False}
    json.dump(report, open(os.path.join(HERE, "p10c2_0b_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print("[P10-C2-0b] exact ok:", exact_ok, "| power n22:", p22, "(SE", st["power"]["n22"]["mc_se"], ")",
          "| n20:", p20)
    print("  adjudication:", adj["cases"])
    print("  resource:", res["lif_steps"], "steps,", res["core_lif_s"], "s core, worst",
          res["worst_case_2x_min"], "min")
    print("  verdicts:", v)
    print("wrote p10c2_0b_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
