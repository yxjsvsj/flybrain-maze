"""P10-C2-0a：新版 A0 预注册最终加固（统计/资格迹/五状态判定/资源）。

**仅设计、合成测试、静态审计**；不运行 A0、不训练、不更新真实 ΔW、不跑迷宫/小车。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.stats import nct, norm, t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOL_VERSION = "P10-C2-0a/v1"
ALPHA = 0.05
MIN_DZ = 0.65
VALID_N = 22
PLANNED_N = 25
SEEDS = list(range(101, 126))            # 固定、预先声明；不来自旧 A0
CONDITION_ORDER = ("none", "stim", "recover")
MAX_EXCLUSIONS = 3
T_STEPS = 400
DT = 0.02
TAU = 0.1
DV = 0.2
TAU_PRE = 0.05
TAU_POST = 0.05
TAU_E = 0.5
EXPLORATORY_TAU = {"tau_pre": [0.02, 0.05, 0.10], "tau_post": [0.02, 0.05, 0.10],
                   "tau_e": [0.2, 0.5, 1.0]}
WALL_CAP_MIN = 15.0
MEM_CAP_MB = 2048.0


def paired_power_normal(dz, n, alpha=ALPHA):
    z = norm.ppf(1 - alpha / 2); nc = dz * math.sqrt(n)
    return float(1 - norm.cdf(z - nc) + norm.cdf(-z - nc))


def paired_power_t(dz, n, alpha=ALPHA):
    df = n - 1; tc = tdist.ppf(1 - alpha / 2, df); delta = dz * math.sqrt(n)
    return float(nct.sf(tc, df, delta) + nct.cdf(-tc, df, delta))


def _signed_sums(x):
    s = np.zeros(1)
    for v in x:
        s = np.concatenate([s + v, s - v])
    return s


def sign_flip_p_fast(diffs):
    """精确双侧符号翻转（meet-in-the-middle）；n<=~30 时可承受。"""
    d = np.asarray(diffs, float)
    if d.size == 0 or not np.all(np.isfinite(d)):
        raise ValueError("empty or non-finite")
    n = len(d); obs = abs(float(d.sum()))
    h = n // 2
    sa = _signed_sums(d[:h]); sb = np.sort(_signed_sums(d[h:]))
    cnt = 0; tol = 1e-12
    for x in sa:
        lo = np.searchsorted(sb, -obs - x + tol, side="right")
        hi = np.searchsorted(sb, obs - x - tol, side="left")
        cnt += lo + (len(sb) - hi)
    return cnt / (2 ** n)


def sign_flip_p_enum(diffs):
    d = np.asarray(diffs, float); obs = abs(float(d.mean())); cnt = 0
    for m in range(1 << len(d)):
        s = np.array([1.0 if (m >> i) & 1 else -1.0 for i in range(len(d))])
        if abs(float((d * s).mean())) >= obs - 1e-12:
            cnt += 1
    return cnt / (1 << len(d))


def simulate_signflip_power(dz, n, sims=2000, seed=20260):
    rng = np.random.default_rng(seed); c = 0
    for _ in range(sims):
        d = rng.normal(dz, 1.0, n)
        if sign_flip_p_fast(d) < ALPHA:
            c += 1
    p = c / sims
    return {"power": round(p, 4), "mc_se": round(math.sqrt(p * (1 - p) / sims), 4),
            "sims": sims, "seed": seed, "effect_distribution": "N(dz,1) paired differences"}


def sample_size_plan(power22=None, power20=None):
    p22 = power22 or 0.833
    p20 = power20 or 0.789
    ok = p22 >= 0.80 and p20 < 0.80
    return {
        "protocol_version": PROTOCOL_VERSION,
        "primary_test": "two-sided exact sign-flip, alpha=0.05",
        "independent_unit": "brain seed",
        "min_dz": MIN_DZ,
        "valid_n": VALID_N, "planned_n": PLANNED_N, "seeds": SEEDS,
        "condition_order": list(CONDITION_ORDER),
        "max_exclusions": MAX_EXCLUSIONS,
        "power_signflip_n20": p20, "power_signflip_n22": p22,
        "power_normal_n20": round(paired_power_normal(MIN_DZ, 20), 4),
        "power_t_n20": round(paired_power_t(MIN_DZ, 20), 4),
        "valid_n_changed_from_20_to_22": True,
        "reason": ("精确符号翻转模拟：n=20 power≈0.789 <0.80；n=22 power≈0.833 >=0.80。"
                   "故 valid n=22、planned n=25（允许 3 个失败/非有限，12%）。"
                   "正态近似/配对 t/精确符号翻转**不可混用**。"),
        "ok": bool(ok),
    }


def stimulus_audit():
    decay = math.exp(-DT / TAU); per_step = 1 - decay
    amount = DV * per_step
    return {"encoder_formula": "amount = dv*(1-exp(-dt/tau))", "dt": DT, "tau": TAU,
            "decay": round(decay, 6), "per_step": round(per_step, 6),
            "dv": DV, "amount": round(amount, 6),
            "target": "LPLC2/LC4 only", "no_posthoc_increase": True,
            "ready": abs(amount - 0.036254) < 1e-6}


def eligibility_traces_correct(pre_spikes, post_spikes, edges, dt=DT, tau_pre=TAU_PRE,
                               tau_post=TAU_POST, tau_e=TAU_E):
    """e(t)=e(t-1)*de + pre_trace(t-1)*post_trace(t-1); e(0)=0；严格因果。"""
    pre = np.asarray(pre_spikes, float); post = np.asarray(post_spikes, float)
    T = pre.shape[0]; n_pre, n_post = pre.shape[1], post.shape[1]
    p = np.zeros(n_pre); q = np.zeros(n_post); ee = np.zeros(len(edges))
    dp, dq, de = math.exp(-dt / tau_pre), math.exp(-dt / tau_post), math.exp(-dt / tau_e)
    out = np.zeros((T, len(edges)))
    for t in range(T):
        if t > 0:
            p = p * dp + pre[t - 1]
            q = q * dq + post[t - 1]
            ee = ee * de + p[edges[:, 0]] * q[edges[:, 1]]
        out[t] = ee
    return out


def eligibility_audit():
    edges = np.array([[0, 0]], int)
    pre = np.zeros((12, 1)); post = np.zeros((12, 1))
    pre[5, 0] = 1; post[5, 0] = 1
    e = eligibility_traces_correct(pre, post, edges)[:, 0]
    first_nonzero = int(np.flatnonzero(e > 0)[0]) if np.any(e > 0) else None
    # causality
    pre2 = pre.copy(); post2 = post.copy(); pre2[9, 0] = 1; post2[9, 0] = 1
    e2 = eligibility_traces_correct(pre2, post2, edges)[:, 0]
    return {
        "frozen_params": {"dt": DT, "tau_pre": TAU_PRE, "tau_post": TAU_POST, "tau_e": TAU_E,
                          "output_index": "e(t) uses pre/post traces at t-1", "init": "e(0)=0",
                          "reset": "trial start"},
        "exploratory_only": EXPLORATORY_TAU,
        "primary_analysis_uses_frozen_params": True,
        "no_tau_selection_by_significance": True,
        "single_pulse_t5_first_nonzero_index": first_nonzero,
        "time_index_matches_formula": first_nonzero == 6,
        "strictly_causal": bool(np.allclose(e[:10], e2[:10])),
        "valid": bool(first_nonzero == 6 and np.allclose(e[:10], e2[:10])),
    }


def _get(rec, *keys):
    cur = rec
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def adjudicate(rec):
    """默认失败的五状态判定器；PASS 需全部门槛。"""
    required = ["protocol_version", "frozen", "entrance", "kc", "eligibility", "ceiling", "exclusions"]
    if not isinstance(rec, dict) or any(k not in rec for k in required):
        return {"state": "INCONCLUSIVE", "reason": "missing required field(s)"}
    if rec.get("protocol_version") != PROTOCOL_VERSION:
        return {"state": "INCONCLUSIVE", "reason": "protocol version mismatch"}
    frozen = rec["frozen"]
    for k, v in (("n_valid", VALID_N), ("dz", MIN_DZ), ("alpha", ALPHA),
                 ("tau_pre", TAU_PRE), ("tau_post", TAU_POST), ("tau_e", TAU_E)):
        if frozen.get(k) != v:
            return {"state": "INCONCLUSIVE", "reason": f"unfrozen/mismatched {k}"}
    ent = rec["entrance"]
    if not (ent.get("direction_positive") and ent.get("p", 1.0) < ALPHA
            and ent.get("relative_effect", 0.0) >= 0.10):
        return {"state": "INPUT_INVALID", "reason": "entrance positive control failed"}
    el = rec["eligibility"]
    if not (el.get("valid") and el.get("time_resolved") and not el.get("saturated")
            and el.get("amplitude_effect", 0.0) > 0):
        return {"state": "METRIC_INVALID", "reason": "eligibility metric invalid/saturated"}
    ceil = rec["ceiling"]
    if ceil.get("kc_baseline", 1.0) >= 0.85 or ceil.get("mbon_baseline", 1.0) >= 0.85:
        return {"state": "INCONCLUSIVE", "reason": "ceiling"}
    if rec["exclusions"] > MAX_EXCLUSIONS:
        return {"state": "INCONCLUSIVE", "reason": "too many exclusions"}
    kc = rec["kc"]
    if kc.get("n_valid", 0) < VALID_N:
        return {"state": "INCONCLUSIVE", "reason": "insufficient valid seeds"}
    if not (kc.get("p", 1.0) < ALPHA and kc.get("effect", 0.0) >= 0.05
            and kc.get("ci_low", -1.0) > 0):
        return {"state": "SENSORY_NO_EFFECT", "reason": "KC primary endpoint failed"}
    return {"state": "A0_PASS", "reason": "all pre-registered gates met"}


def adjudicator_selftest():
    base = {
        "protocol_version": PROTOCOL_VERSION,
        "frozen": {"n_valid": VALID_N, "dz": MIN_DZ, "alpha": ALPHA,
                   "tau_pre": TAU_PRE, "tau_post": TAU_POST, "tau_e": TAU_E},
        "entrance": {"direction_positive": True, "p": 0.001, "relative_effect": 0.2},
        "kc": {"n_valid": 22, "p": 0.01, "effect": 0.05, "ci_low": 0.01},
        "eligibility": {"valid": True, "time_resolved": True, "saturated": False, "amplitude_effect": 0.1},
        "ceiling": {"kc_baseline": 0.5, "mbon_baseline": 0.5},
        "exclusions": 1,
    }
    import copy
    cases = {}
    r = copy.deepcopy(base); cases["A0_PASS"] = adjudicate(r)["state"]
    r = copy.deepcopy(base); r["entrance"] = {"direction_positive": False, "p": 0.5, "relative_effect": 0.0}
    cases["INPUT_INVALID"] = adjudicate(r)["state"]
    r = copy.deepcopy(base); r["eligibility"]["valid"] = False
    cases["METRIC_INVALID"] = adjudicate(r)["state"]
    r = copy.deepcopy(base); r["kc"] = {"n_valid": 22, "p": 0.5, "effect": 0.0, "ci_low": -0.1}
    cases["SENSORY_NO_EFFECT"] = adjudicate(r)["state"]
    r = copy.deepcopy(base); r["kc"]["n_valid"] = 10
    cases["INCONCLUSIVE"] = adjudicate(r)["state"]
    r = copy.deepcopy(base); del r["kc"]
    cases["missing_field"] = adjudicate(r)["state"]
    r = copy.deepcopy(base); r["protocol_version"] = "old/v0"
    cases["version_mismatch"] = adjudicate(r)["state"]
    r = copy.deepcopy(base); r["frozen"]["dz"] = 0.5
    cases["unfrozen_dz"] = adjudicate(r)["state"]
    return {"cases": cases, "all_correct": (
        cases["A0_PASS"] == "A0_PASS" and cases["INPUT_INVALID"] == "INPUT_INVALID"
        and cases["METRIC_INVALID"] == "METRIC_INVALID" and cases["SENSORY_NO_EFFECT"] == "SENSORY_NO_EFFECT"
        and cases["INCONCLUSIVE"] == "INCONCLUSIVE" and cases["missing_field"] != "A0_PASS"
        and cases["version_mismatch"] != "A0_PASS" and cases["unfrozen_dz"] != "A0_PASS")}


def resource_budget():
    lif_steps = PLANNED_N * len(CONDITION_ORDER) * T_STEPS
    core_s = lif_steps * 1.3 / 1000.0
    init_s = PLANNED_N * len(CONDITION_ORDER) * 1.0
    elig_s = 20.0
    save_s = 5.0
    total_s = core_s + init_s + elig_s + save_s
    total_min = total_s / 60.0
    est_min_worst = total_min * 2.0
    ok = est_min_worst <= WALL_CAP_MIN
    return {"lif_steps": lif_steps, "core_step_s": round(core_s, 2), "init_rebuild_s": round(init_s, 2),
            "eligibility_s": elig_s, "save_stats_s": save_s,
            "estimated_total_min": round(total_min, 2), "worst_case_2x_min": round(est_min_worst, 2),
            "wall_cap_min": WALL_CAP_MIN, "mem_cap_MB": MEM_CAP_MB,
            "breakdown": "core LIF + init/rebuild + eligibility + save/stats", "ready": bool(ok),
            "class3_learning": "UNKNOWN"}


def verdicts(sample_ok, elig_ok, adj_ok, res_ok):
    stat_ok = bool(sample_ok and elig_ok and adj_ok)
    return {
        "Protocol_Statistically_Valid": stat_ok,
        "Primary_Power_At_Valid_N": ">=0.80 (simulated exact sign-flip)" if stat_ok else "FAIL",
        "Eligibility_Frozen": bool(elig_ok),
        "Adjudicator_Failably_Complete": bool(adj_ok),
        "Resource_Ready": bool(res_ok),
        "A0_Executed": False,
        "Eligible_To_Request_One_NoLearning_A0": bool(stat_ok and res_ok),
        "Level_A_Execution": "NO-GO (approval/run pending)",
        "Level_B": "NO-GO", "Level_C": "NO-GO", "Biological": "NO-GO",
        "Training_Authorization": False,
        "note": "即使全部通过，仅形成一次无学习 A0 的审批申请；不授权 ΔW/B/C/训练。",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    sim20 = simulate_signflip_power(MIN_DZ, 20, sims=2000)
    sim22 = simulate_signflip_power(MIN_DZ, 22, sims=2000)
    s = sample_size_plan(power22=sim22["power"], power20=sim20["power"])
    elig = eligibility_audit(); adj = adjudicator_selftest(); res = resource_budget()
    v = verdicts(s["ok"], elig["valid"], adj["all_correct"], res["ready"])
    report = {"protocol_version": PROTOCOL_VERSION, "sample_size": s,
              "power_simulation": {"n20": sim20, "n22": sim22},
              "stimulus": stimulus_audit(), "eligibility": elig,
              "adjudicator": adj, "resource": res, "verdicts": v,
              "training_authorized": False, "a0_executed": False,
              "delta_from_p10c2_0": {
                  "valid_n": "20 -> 22", "planned_n": "22 -> 25",
                  "power_method": "normal approximation -> exact sign-flip MC simulation",
                  "eligibility_index": "fixed off-by-one; e(t) now matches e(t-1)de+pre(t-1)post(t-1)",
                  "adjudicator": "hardcoded true -> executable five-state, default-fail",
                  "seed_list": "101-125 fixed",
              }}
    json.dump(report, open(os.path.join(HERE, "p10c2_0a_protocol.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[P10-C2-0a] power n20:", sim20, "| n22:", sim22)
    print("  valid n:", VALID_N, "planned:", PLANNED_N, "| seeds:", SEEDS[0], "-", SEEDS[-1])
    print("  eligibility time index:", elig["single_pulse_t5_first_nonzero_index"], "| valid:", elig["valid"])
    print("  adjudicator:", adj["cases"])
    print("  resource ready:", res["ready"], "| worst:", res["worst_case_2x_min"], "min")
    print("  verdicts:", v)
    print("wrote p10c2_0a_protocol.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
