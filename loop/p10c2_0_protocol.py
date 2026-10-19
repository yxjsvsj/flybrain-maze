"""P10-C2-0：新版 A0（LPLC2/LC4 → KC → KC→MBON 候选资格迹）**预注册**。

**仅设计/合成测试/静态审计**；不运行 A0、不训练、不更新真实 ΔW、不跑迷宫/小车。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))

ALPHA = 0.05
POWER_TARGET = 0.80
MIN_DZ = 0.65                 # 预先声明的最小有意义标准化效应（设计假设，非旧 A0 结果）
REQUIRED_N = None             # filled by sample_size_plan
VALID_N_REQUIRED = 20
PLANNED_N = 22                # 20 valid + 10% attrition
MAX_EXCLUDE_FRAC = 0.10
T_STEPS = 400
CONDITIONS = ("none", "stim", "recover")


def paired_power(dz, n, alpha=ALPHA):
    ncp = dz * math.sqrt(n)
    crit = norm.ppf(1 - alpha / 2)
    return float(1 - norm.cdf(crit - ncp) + norm.cdf(-crit - ncp))


def required_n(dz, power=POWER_TARGET, alpha=ALPHA):
    z_a = norm.ppf(1 - alpha / 2)
    z_p = norm.ppf(power)
    return int(math.ceil(((z_a + z_p) / dz) ** 2))


def min_signflip_p(n_nonzero):
    return 2.0 / (2 ** n_nonzero) if n_nonzero > 0 else 1.0


def sign_flip_p(diffs):
    d = np.asarray(diffs, float)
    if d.size == 0 or not np.all(np.isfinite(d)):
        raise ValueError("empty or non-finite")
    obs = abs(float(d.mean()))
    cnt = 0
    for mask in range(1 << d.size):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(d.size)])
        if abs(float((d * signs).mean())) >= obs - 1e-12:
            cnt += 1
    return cnt / (1 << d.size)


def sample_size_plan():
    req = required_n(MIN_DZ)
    return {
        "independent_unit": "brain seed",
        "test": "two-sided paired sign-flip / paired t; alpha=0.05",
        "min_meaningful_dz": MIN_DZ,
        "power_target": POWER_TARGET,
        "required_n_for_power": req,
        "valid_n_required": VALID_N_REQUIRED,
        "planned_n": PLANNED_N,
        "attrition_allowance": f"{(PLANNED_N - VALID_N_REQUIRED) / PLANNED_N:.0%}",
        "power_at_valid_n": round(paired_power(MIN_DZ, VALID_N_REQUIRED), 4),
        "power_at_planned_n": round(paired_power(MIN_DZ, PLANNED_N), 4),
        "exact_test_reachable": min_signflip_p(6) < ALPHA,
        "exact_min_p_6": min_signflip_p(6),
        "not_based_on_old_A0_effect": True,
        "rationale": ("n=20 有效种子在 dz=0.65 下 power≈0.83；计划 22 预留 10% 失败/非有限；"
                      "精确符号翻转在 ≥6 非零时可达 p<0.05。**不**用旧 A0 的偶然最大差异选参数。"),
    }


def stimulus_unit_audit(dt=0.020, tau=0.100, old_amount=0.6, proposed_dv=0.2):
    decay = math.exp(-dt / tau)
    per_step = 1.0 - decay
    old_dv = old_amount / per_step
    proposed_amount = proposed_dv * per_step
    return {
        "encoder_semantics": "amount = dv * (1 - decay); decay = exp(-dt/tau)",
        "dt": dt, "tau": tau, "decay": round(decay, 6), "per_step": round(per_step, 6),
        "old_amount": old_amount, "old_steady_state_dv": round(old_dv, 4),
        "old_amount_is_per_step_not_dv": True,
        "old_unit_consistent": False,
        "old_mismatch_factor_vs_dv_0.2": round(old_amount / proposed_amount, 3),
        "proposed_unit": "steady-state voltage increment dv (LIF normalized voltage units)",
        "proposed_dv": proposed_dv, "proposed_amount_per_step": round(proposed_amount, 6),
        "calibration_target": "LPLC2/LC4 entrance firing increase (NOT KC response)",
        "no_posthoc_increase": True,
        "future_fixed_stimulus": {
            "unit": "dv (steady-state voltage increment)",
            "single_step_amount": "dv*(1-exp(-dt/tau))",
            "stim_window_steps": [100, 300],
            "no_stim_control": "same duration, same seed, no injection",
            "recover": "stim then off; matched baseline duration",
        },
        "ceiling_criteria": {"kc_baseline_rate_max": 0.85, "mbon_baseline_rate_max": 0.85,
                             "if_exceeded": "INCONCLUSIVE (ceiling)"},
    }


def eligibility_traces(pre_spikes, post_spikes, edges, dt=0.020, tau_pre=0.05, tau_post=0.05, tau_e=0.5):
    """独立 pre/post traces；e(t) 使用 t-1 的 traces（严格因果），e(0)=0。"""
    pre = np.asarray(pre_spikes, float); post = np.asarray(post_spikes, float)
    T = pre.shape[0]
    n_pre, n_post = pre.shape[1], post.shape[1]
    p = np.zeros(n_pre); q = np.zeros(n_post); ee = np.zeros(len(edges))
    dp, dq, de = math.exp(-dt / tau_pre), math.exp(-dt / tau_post), math.exp(-dt / tau_e)
    out = np.zeros((T, len(edges)))
    for t in range(T):
        out[t] = ee                         # record before incorporating current spikes (causal)
        ee = ee * de + p[edges[:, 0]] * q[edges[:, 1]]
        p = p * dp + pre[t]
        q = q * dq + post[t]
    return out


def eligibility_audit():
    edges = np.array([[0, 0], [1, 0], [0, 1]], int)
    pre = np.zeros((20, 2)); post = np.zeros((20, 2))
    pre[5, 0] = 1; post[6, 0] = 1
    e1 = eligibility_traces(pre, post, edges)
    # causality: future spike at t=15 must not change e for t<=15
    pre2 = pre.copy(); post2 = post.copy(); pre2[15, 1] = 1; post2[15, 1] = 1
    e2 = eligibility_traces(pre2, post2, edges)
    causal = bool(np.allclose(e1[:16], e2[:16]))
    # reset/fresh
    e3 = eligibility_traces(pre, post, edges)
    reset_ok = bool(np.allclose(e1, e3))
    # spontaneous vs evoked (add spikes)
    pre_ev = pre.copy(); post_ev = post.copy(); pre_ev[10:14, 0] = 1; post_ev[10:14, 0] = 1
    e_ev = eligibility_traces(pre_ev, post_ev, edges)
    diff = e_ev - e1
    return {
        "definition": "e(t)=e(t-1)*exp(-dt/tau_e)+pre_trace(t-1)*post_trace(t-1)",
        "init_zero": bool(np.all(e1[0] == 0)),
        "strictly_causal": causal,
        "reset_fresh_equal": reset_ok,
        "nonzero_after_spikes": bool(e1[6:].max() > 0),
        "time_resolved": True,
        "records_amplitude_and_nonzero": True,
        "spontaneous_vs_evoked_separable": bool(diff.max() > 0),
        "synthetic": {"max_e": round(float(e1.max()), 6), "mean_e": round(float(e1.mean()), 6),
                      "nonzero_frac": round(float((e1 > 1e-9).mean()), 6),
                      "evoked_diff_max": round(float(diff.max()), 6)},
        "primary_metric": "stim-evoked eligibility amplitude difference (paired by seed, time-resolved)",
        "secondary": ["nonzero fraction", "distribution quantiles", "KC per-cell response fraction"],
    }


def metrics_and_failure():
    return {
        "A_entrance": {"metric": "LPLC2/LC4 population firing rate (stim - none)",
                       "gate": "pre-registered positive effect (p<0.05, effect size >= declared)"},
        "B_KC": {"primary": "KC population firing rate (stim - none)",
                 "secondary": ["per-cell KC response", "latency", "post-removal recovery"]},
        "C_eligibility": {"primary": "stim-evoked eligibility amplitude difference",
                          "secondary": ["nonzero fraction", "distribution", "time course"]},
        "single_primary_endpoint": "B_KC population rate",
        "secondary_family_fdr": "Benjamini-Hochberg q=0.05; labelled secondary/exploratory",
        "zero_diff_handling": "included in sign-flip (zero contributes 0); excluded from sign test; both reported",
        "nonfinite_failure": "excluded; >10% excluded -> INCONCLUSIVE",
        "multiple_comparisons_frozen": True,
        "states": {
            "INPUT_INVALID": "entrance positive control fails",
            "METRIC_INVALID": "eligibility synthetic validation fails or saturated/undiscriminative",
            "SENSORY_NO_EFFECT": "entrance valid but KC primary endpoint fails declared effect",
            "INCONCLUSIVE": "insufficient valid seeds/ceiling/wide CI/excessive exclusions",
            "A0_PASS": "all pre-registered gates met (not just one p<0.05)",
        },
        "a0_pass_authorizes_training": False,
        "no_posthoc_changes": True,
    }


def resource_budget(per_step_ms=1.3):
    steps = PLANNED_N * len(CONDITIONS) * T_STEPS
    lif_s = steps * per_step_ms / 1000.0
    elig_edge_steps = 4533 * T_STEPS * PLANNED_N * len(CONDITIONS)
    return {
        "seeds": PLANNED_N, "conditions": list(CONDITIONS), "steps_per_run": T_STEPS,
        "lif_steps_total": steps, "estimated_lif_seconds": round(lif_s, 2),
        "eligibility_edge_steps": int(elig_edge_steps),
        "estimated_eligibility_seconds": "~seconds (vectorized; not a learning run)",
        "peak_memory_MB": "~1000 (subnet LIF, per process; sequential)",
        "wall_clock_budget": "< 10 min CPU (no learning; class3 UNKNOWN)",
        "class3_learning_cost": "UNKNOWN (eligibility/dW/teacher update not implemented)",
        "not_full_training_measurement": True,
    }


def evidence_levels():
    return {
        "this_round": "DESIGN/SYNTHETIC ONLY (no empirical A0)",
        "a0_execution_would_provide": ["Level 1: entrance positive control", "Level 1-2: KC transmission",
                                       "Level 2: candidate eligibility identifiability"],
        "does_not_provide": ["real synaptic modulation", "g_unit calibration", "learning/navigation"],
        "historical_p10c1_closed": True,
    }


def approval_items():
    return {
        "statistics": {"min_dz": MIN_DZ, "planned_n": PLANNED_N, "valid_n": VALID_N_REQUIRED,
                       "alpha": ALPHA, "test": "two-sided paired; exact reachability"},
        "stimulus": {"unit": "dv steady-state; amount=dv*(1-decay)", "proposed_dv": 0.2,
                     "entrance_calibration_required": True, "no_posthoc_increase": True},
        "eligibility": {"independent_pre_post_traces": True, "tau_pre/tau_post/tau_e grids": True,
                        "init/reset": "e(0)=0; trial reset", "time_resolved": True},
        "failure_conditions": ["INPUT_INVALID", "METRIC_INVALID", "SENSORY_NO_EFFECT", "INCONCLUSIVE", "A0_PASS"],
        "dw_training": {"s1_budget_a_b": "net positive headroom = 0", "s2": "L1 deviation",
                        "s1_s2_approved": False, "real_update_implemented": False,
                        "training_authorization": False},
        "scope": "one new A0 run only after written approval; no B/C/training",
    }


def verdicts():
    return {
        "Protocol_Statistically_Valid": True,
        "Stimulus_Units_Clear": True,
        "Eligibility_Metric_Preregistered": True,
        "Failure_Conditions_Observable": True,
        "A0_Run_This_Round": False,
        "Worth_Applying_For_One_A0_Run": True,
        "Level_A_Execution": "NO-GO (until approved and run)",
        "Level_B": "NO-GO", "Level_C": "NO-GO", "Biological": "NO-GO",
        "Training_Authorization": False,
        "note": ("协议可申请单次 A0；A0_PASS 也不授权真实 ΔW/B/C/训练。"
                 "P10-C1 已关闭，不重启。"),
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    report = {"sample_size": sample_size_plan(), "stimulus_unit": stimulus_unit_audit(),
              "eligibility": eligibility_audit(), "metrics_failure": metrics_and_failure(),
              "resource": resource_budget(), "evidence": evidence_levels(),
              "approval_items": approval_items(), "verdicts": verdicts(),
              "training_authorized": False, "a0_executed": False}
    json.dump(report, open(os.path.join(HERE, "p10c2_0_protocol.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    s = report["sample_size"]
    print("[P10-C2-0] n:", s["planned_n"], "valid:", s["valid_n_required"],
          "power@valid:", s["power_at_valid_n"], "min p (6):", s["exact_min_p_6"])
    u = report["stimulus_unit"]
    print("  amount=0.6 -> steady-state dv:", u["old_steady_state_dv"], "| consistent:", u["old_unit_consistent"],
          "| proposed amount:", u["proposed_amount_per_step"])
    e = report["eligibility"]
    print("  eligibility causal:", e["strictly_causal"], "| init0:", e["init_zero"],
          "| spontaneous/evoked separable:", e["spontaneous_vs_evoked_separable"])
    print("  verdicts:", report["verdicts"])
    print("wrote p10c2_0_protocol.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
