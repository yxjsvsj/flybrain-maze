"""P10-C1e：A0 方法学复核 + S1 Budget B 勘误 + 合成内核审计（**只读/合成**）。

不重跑 A0、不更新真实 ΔW、不训练、不进入 B/C。仅使用已保存数据、源码静态检查与合成数学测试。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

from loop.p10c1d_delta_sandbox import (apply_dw_sandbox, frozen_eval, persistence_roundtrip,
                                       synthetic_case, zero_weight_semantics)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_a0():
    return json.load(open(os.path.join(HERE, "p10c1d_a0_probe.json"), encoding="utf-8"))


def sign_flip_p(diffs, include_zeros=True):
    """精确双侧符号翻转 p。`include_zeros=True`：零差值保留（符号翻转后仍为 0）。"""
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        raise ValueError("empty diffs")
    if not np.all(np.isfinite(d)):
        raise ValueError("non-finite diffs")
    d = d if include_zeros else d[d != 0]
    if d.size == 0:
        return 1.0
    obs = abs(float(d.mean()))
    cnt = 0
    for mask in range(1 << d.size):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(d.size)])
        if abs(float((d * signs).mean())) >= obs - 1e-12:
            cnt += 1
    return cnt / (1 << d.size)


def min_achievable_p(n_nonzero):
    return 2.0 / (2 ** n_nonzero) if n_nonzero > 0 else 1.0


def can_reach_alpha(n_nonzero, alpha=0.05):
    return min_achievable_p(n_nonzero) <= alpha


def _signs(mask, n):
    return np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])


def a0_stats_audit():
    d = load_a0()
    rows = d["rows"]
    seeds = d["protocol"]["seeds"]
    diffs = {"kc": [], "mb": [], "e_nz": []}
    for s in seeds:
        n = [r for r in rows if r["seed"] == s and r["condition"] == "none"][0]
        a = [r for r in rows if r["seed"] == s and r["condition"] == "stim"][0]
        diffs["kc"].append(a["kc_stim"] - n["kc_stim"])
        diffs["mb"].append(a["mb_stim"] - n["mb_stim"])
        diffs["e_nz"].append(a["e_nonzero_frac"] - n["e_nonzero_frac"])
    out = {"diffs": {k: [round(float(x), 8) for x in v] for k, v in diffs.items()}}
    for k, v in diffs.items():
        nz = [x for x in v if x != 0]
        out[k] = {"n_nonzero": len(nz), "n_zero": len(v) - len(nz),
                  "p_include_zeros": round(sign_flip_p(v, include_zeros=True), 6),
                  "p_exclude_zeros": round(sign_flip_p(v, include_zeros=False), 6),
                  "min_p_possible_with_5(nonzero)": 0.0625,
                  "min_p_possible_current_nonzero": round(min_achievable_p(len(nz)), 6),
                  "effect_mean": round(float(np.mean(v)), 8),
                  "sign_consistency": int(max(sum(1 for x in nz if x > 0), sum(1 for x in nz if x < 0)))}
    # 合成案例
    cases = {
        "all_same_sign_5": [1, 2, 3, 4, 5],
        "mixed_signs_5": [1, -2, 3, -4, 5],
        "one_zero_4_nonzero": [1, 2, 3, 4, 0],
        "all_zero_5": [0, 0, 0, 0, 0],
    }
    out["synthetic"] = {k: {"p": round(sign_flip_p(v), 6), "min_possible": round(min_achievable_p(
        sum(1 for x in v if x != 0)), 6)} for k, v in cases.items()}
    try:
        sign_flip_p([1.0, float("nan")])
        out["nonfinite_rejected"] = False
    except ValueError:
        out["nonfinite_rejected"] = True
    out["stat_gate_valid"] = False
    out["reason"] = ("5 个非零配对的双侧精确符号翻转最小 p=2/32=0.0625；实际 1 个零 + 4 非零 "
                     "-> 最小 p=2/16=0.125。均不能达到 p<0.05；**不得**改单侧/改 α/加 seed 事后挽救。")
    return out


def eligibility_audit():
    src = open(os.path.join(HERE, "p10c1d_a0_probe.py"), encoding="utf-8").read()
    same_step = "kc[t][E.col] * mb[t][E.row]" in src
    final_only = "e > 1e-6).mean()" in src and "e_step" not in src
    entrance_recorded = "lp_pos] = 1" in src or "lp_fired" in src
    return {
        "definition_matches_prereg": not same_step,   # 预注册要求独立 pre/post traces
        "uses_same_step_product": bool(same_step),
        "time_resolved": not final_only,
        "final_only_binary_fraction": bool(final_only),
        "entrance_positive_control_recorded": bool(entrance_recorded),
        "amplitude_timecourse_saved": False,
        "stimulus_units_verified_vs_encoder": False,
        "high_spontaneous_ceiling": True,
        "recover_baseline_matched": False,
        "recover_vs_stim_comparable": False,
        "validity": "INVALID/LIMITED",
        "missing": ["per-step eligibility amplitude/timecourse", "LPLC2/LC4 entrance firing (positive control)",
                    "Encoder-consistent stimulus calibration", "matched recover baseline"],
        "interpretations": {
            "stimulus_truly_no_transmission": "NOT ESTABLISHED",
            "stimulus_drowned_by_background": "LIKELY (high spontaneous; eligibility nonzero=1.0 in 4/5 controls)",
            "metric_lacks_discriminative_power": "YES (same-step product; final-only binarized fraction)",
            "current_data_cannot_discriminate": "YES",
        },
        "note": "以上四类必须分开；不得把'指标无效'说成'感觉通路机制失败'。",
    }


def s1_budget_erratum():
    from loop.visual_motor_audit import load_data
    from loop.p100_evidence import audit
    from loop.p100b_path_audit import chain_layers
    D = load_data(); A = audit(D); L = chain_layers(D, A)
    W = D["W"].tocsr().copy(); W.eliminate_zeros()
    L2, L3 = L[2], L[3]
    l1_total, l1_plastic = [], []
    for mb_i in L3:
        l1_total.append(float(np.abs(W.getrow(mb_i).data).sum()))
        l1_plastic.append(float(np.abs(W[mb_i][:, L2].data).sum()))
    lt = np.array(l1_total); lp = np.array(l1_plastic); ln = lt - lp
    head_A = np.maximum(0.0, lp - lp)                    # budget = plastic sum
    head_B = np.maximum(0.0, 1.0 - ln - lp)              # budget = L1=1
    # 合成负例：非可塑 0.9585 + 可塑 0.0415 + 0.01 正向 -> 1.01 > 1
    synth_total = 0.9585 + 0.0415 + 0.01
    return {
        "n_MBON": int(len(lt)),
        "l1_total": {"mean": round(float(lt.mean()), 6), "min": round(float(lt.min()), 6),
                     "max": round(float(lt.max()), 6), "all_one": bool(np.allclose(lt, 1.0))},
        "l1_plastic": {"mean": round(float(lp.mean()), 6)},
        "l1_nonplastic": {"mean": round(float(ln.mean()), 6)},
        "budgetA_headroom": {"mean": round(float(head_A.mean()), 6),
                             "n_zero": int((head_A <= 1e-12).sum())},
        "budgetB_headroom": {"mean": round(float(head_B.mean()), 9),
                             "max": round(float(head_B.max()), 9),
                             "n_zero_1e-6": int((head_B <= 1e-6).sum()),
                             "note": "浮点残差 ~1e-7；净余量为 0"},
        "previous_wrong_budgetB_headroom": 0.9585,
        "corrected_budgetB_headroom": 0.0,
        "erratum": ("P10-C1d 把 L1_nonplastic≈0.9585 误当作**未使用**学习容量；"
                    "实际 L1_total = L1_nonplastic + L1_plastic = 1.0，故 headroom=0。"),
        "synthetic_negative": {"nonplastic": 0.9585, "plastic": 0.0415, "added_positive": 0.01,
                               "resulting_total_L1": round(synth_total, 6),
                               "exceeds_one": bool(synth_total > 1.0)},
        "conclusion": ("Budget A 与 Budget B 的**净正向余量均为 0**；保持 L1≤1 只能通过"
                       "可塑突触间增强/减弱抵消实现重新分配。正向净增长需 S2（放宽 L1）"
                       "或减少非可塑权重（不允许，未批准边不变）。"),
        "s1_a_feasible": True, "s1_b_feasible": True, "s2_feasible": True,
    }


def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def save_checked(path, W, eff, approved, params, l1):
    """候选（审计用，非历史 sandbox）：带元数据的持久化。"""
    sp.save_npz(path, eff.tocsr())
    meta = {"w_sha256": _sha(W.toarray()), "approved_sha256": _sha(approved),
            "params": params, "l1": l1}
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_checked(path, W, approved, params, l1):
    meta = json.load(open(path + ".meta.json", encoding="utf-8"))
    if meta["w_sha256"] != _sha(W.toarray()):
        raise ValueError("official W mismatch")
    if meta["approved_sha256"] != _sha(approved):
        raise ValueError("approved mask mismatch")
    if meta["params"] != params:
        raise ValueError("params mismatch")
    if meta["l1"] != l1:
        raise ValueError("L1 strategy mismatch")
    eff = sp.load_npz(path).tocsr()
    return eff, (W.tocsr() + eff).tocsr()


def kernel_audit():
    c = synthetic_case(l1="S1")
    checks = c["checks"]
    fe = frozen_eval()
    # 历史 save_state 无元数据；FROZEN_EVAL 不校验 W/掩码/参数/L1
    return {
        "requested_effective_consistent": bool(checks["effective_equals_Weff_minus_W"]),
        "frozen_eval_retains": bool(fe["retains_learned_dW"] and fe["frozen_eval_stable"]),
        "zero_weight_keeps_anatomical_identity": int(c["zero_semantics"]["anatomical_edge_deleted"]) == 0,
        "original_save_state_has_metadata": False,
        "original_frozen_eval_validates_source": False,
        "candidate_checked_loader_validates": ["W sha256", "approved mask sha256", "params", "L1 strategy"],
        "limitations": ["历史 save_state 仅存 delta，无 W/掩码/参数/L1 元数据",
                        "历史 FROZEN_EVAL 加载不校验来源；仅合成沙箱内使用"],
        "status": "SYNTHETIC_ONLY; real update NOT implemented",
    }


def candidate_loader_tests():
    c = synthetic_case(l1="S1")
    W, approved, out = c["W"], c["approved"], c["out"]
    params = {"delta_max": 0.6}
    path = os.path.join(HERE, "_tmp_p10c1e_state.npz")
    res = {}
    try:
        save_checked(path, W, out["effective_delta"], approved, params, "S1")
        eff, weff = load_checked(path, W, approved, params, "S1")
        res["matching_ok"] = bool(np.allclose(weff.toarray(), out["W_eff"].toarray(), atol=1e-7))
        for label, kwargs in (("wrong_W", {"W": W + sp.eye(W.shape[0])}),
                              ("wrong_approved", {"approved": ~approved}),
                              ("wrong_params", {"params": {"delta_max": 0.1}}),
                              ("wrong_l1", {"l1": "S2"})):
            args = {"W": W, "approved": approved, "params": params, "l1": "S1"}
            args.update(kwargs)
            try:
                load_checked(path, **args)
                res[label] = False
            except ValueError:
                res[label] = True
    finally:
        for p in (path, path + ".meta.json"):
            if os.path.exists(p):
                os.remove(p)
    return res


def future_a0_preregistration():
    return {
        "status": "PENDING_APPROVAL; NOT run",
        "statistics": {"min_nonzero_pairs": 6,
                       "reason": "5 非零 -> min p 0.0625；6 非零 -> 2/64=0.03125 <0.05",
                       "rule": "pre-register n and one-sided/two-sided; no post-hoc change",
                       "power_analysis_required": True},
        "eligibility": {"definition": "independent pre/post traces (exponential filters), not same-step product",
                        "time_resolved": "per-step/per-window amplitude + nonzero fraction",
                        "metrics": ["mean", "quantiles", "nonzero fraction", "decay time"],
                        "no_final_only_binary": True},
        "stimulus": {"entrance_positive_control": "record LPLC2/LC4 firing; require significant increase",
                     "calibration": "match Encoder steady-state voltage semantics; pre-register amount",
                     "no_kc_injection": True},
        "controls": {"matched_baseline": "same duration no-stim; recover baseline matched",
                     "background": "quantify spontaneous rate; pre-register ceiling criteria"},
        "inference": {"separate": ["no transmission", "drowned by background", "metric invalid", "indeterminate"]},
        "approval_required": True,
    }


def verdicts():
    return {
        "A0_Historical_Protocol_Result": "FAIL (retained)",
        "A0_Statistical_Gate_Validity": "INVALID",
        "A0_Sensory_Transmission_Evidence": "INCONCLUSIVE",
        "Eligibility_Measurement_Validity": "INVALID/LIMITED",
        "S1_Budget_A": "feasible; zero net positive headroom (only redistribution)",
        "S1_Budget_B": "erratum: headroom=0 (previous 0.9585 wrong)",
        "S2": "feasible; positive net growth allowed with explicit L1 deviation",
        "Synthetic_Kernel": "synthetic-only; invariants pass; persistence metadata LIMITED",
        "Level_A_Execution": "NO-GO",
        "Level_B": "NO-GO",
        "Level_C": "NO-GO",
        "Biological": "NO-GO",
        "Training_Authorization": False,
        "route": "FORMALLY CLOSED (A0 protocol invalid + A-level not ready)",
        "restart_requires": ["new preregistered A0 protocol (statistics + metric + entrance control)",
                             "real DW erratum written approval", "real update implementation + acceptance",
                             "real-student isolation (for C) and B decoder"],
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    report = {"a0_stats": a0_stats_audit(), "eligibility": eligibility_audit(),
              "s1_budget_erratum": s1_budget_erratum(), "kernel": kernel_audit(),
              "candidate_loader_tests": candidate_loader_tests(),
              "future_a0_preregistration": future_a0_preregistration(),
              "verdicts": verdicts(), "training_authorized": False}
    json.dump(report, open(os.path.join(HERE, "p10c1e_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    a = report["a0_stats"]
    print("[P10-C1e] A0 KC diffs:", a["diffs"]["kc"])
    print("  KC n_nonzero:", a["kc"]["n_nonzero"], "p(include zeros):", a["kc"]["p_include_zeros"],
          "p(exclude):", a["kc"]["p_exclude_zeros"], "min possible:", a["kc"]["min_p_possible_current_nonzero"])
    print("  stat_gate_valid:", a["stat_gate_valid"])
    e = report["eligibility"]
    print("  eligibility:", e["validity"], "| same_step:", e["uses_same_step_product"],
          "| final_only:", e["final_only_binary_fraction"], "| entrance:", e["entrance_positive_control_recorded"])
    s = report["s1_budget_erratum"]
    print("  S1 BudgetA headroom:", s["budgetA_headroom"], "| BudgetB headroom:", s["budgetB_headroom"],
          "| erratum:", s["corrected_budgetB_headroom"])
    print("  kernel loader tests:", report["candidate_loader_tests"])
    print("  verdicts:", report["verdicts"])
    print("wrote p10c1e_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
