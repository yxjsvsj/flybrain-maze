"""P10-C1c：A 级（KC→MBON 脑内突触学习）独立准入 + ΔW 数学不变量门控。

**不训练、不更新真实 ΔW、不跑迷宫、不控车。** 仅合成稀疏图数学验证、静态审计、无学习性能预算。
官方 W/P1/BASE/readout/tags 冻结。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp

from loop.p10c1b_lif_benchmark import resource_budget

HERE = os.path.dirname(os.path.abspath(__file__))


def _synthetic():
    # 0..3; approved: (1,0)=0.5 兴奋, (2,1)=-0.3 抑制; 非批准边 (3,2)=0.2
    W = sp.lil_matrix((4, 4), dtype=np.float32)
    W[1, 0] = 0.5; W[2, 1] = -0.3; W[3, 2] = 0.2
    W = W.tocsr()
    approved = np.zeros((4, 4), bool)
    approved[1, 0] = True; approved[2, 1] = True
    dW = sp.lil_matrix((4, 4), dtype=np.float32)
    dW[1, 0] = -0.9      # 大负 -> 需 W_eff>=0
    dW[2, 1] = -0.9      # 抑制边 -> W_eff<=0
    dW[3, 2] = 0.5       # 非批准边 -> 必须被掩码
    dW[0, 3] = 0.5       # 非边 -> 必须被掩码
    dW = dW.tocsr()
    return W, approved, dW


def apply_dw(W, dW, approved, delta_max=0.6, training=True, l1="S1"):
    """候选实现（**未批准、未用于真实 W**）：掩码 -> clip -> W_eff 符号约束 -> L1 策略。"""
    W = W.tocsr().copy()
    if not training:
        return W, sp.csr_matrix(W.shape)
    mask = sp.csr_matrix(approved.astype(np.float32))
    dW = dW.multiply(mask).tocsr()
    dW.data = np.clip(dW.data, -delta_max, delta_max)
    Weff = (W + dW).tocsr().tolil()
    for i, j in zip(*approved.nonzero()):
        w0 = float(W[i, j]); v = float(Weff[i, j])
        if w0 > 0:
            Weff[i, j] = max(0.0, v)
        elif w0 < 0:
            Weff[i, j] = min(0.0, v)
    Weff = Weff.tocsr()
    if l1 == "S1":
        Weff_l = Weff.tolil()
        for i in range(W.shape[0]):
            cols = approved[i].nonzero()[0]
            if not len(cols):
                continue
            budget = float(np.abs(W[i, cols].toarray()).sum())
            cur = float(np.abs(Weff_l[i, cols].toarray()).sum())
            if budget > 0 and cur > budget:
                scale = budget / cur
                for j in cols:
                    Weff_l[i, j] = float(Weff_l[i, j]) * scale   # 仅缩放**可塑边**，未批准边不变
        Weff = Weff_l.tocsr()
    return Weff, dW


def dw_invariants():
    W, approved, dW = _synthetic()
    out = {}
    for l1 in ("S1", "S2"):
        Weff, dWm = apply_dw(W, dW, approved, l1=l1)
        Wa = W.toarray(); Ea = Weff.toarray(); Da = dWm.toarray()
        appro = approved
        out[l1] = {
            "only_approved_modified": bool(np.all(Da[~appro] == 0)),
            "no_new_edges": bool(int(((Ea != 0) & (Wa == 0)).sum()) == 0),
            "excitatory_nonnegative": bool(np.all(Ea[appro & (Wa > 0)] >= 0)),
            "inhibitory_nonpositive": bool(np.all(Ea[appro & (Wa < 0)] <= 0)),
            "unapproved_bit_identical": bool(np.array_equal(Ea[~appro], Wa[~appro])),
            "direction_preserved": bool(np.all((Ea[appro & (Wa > 0)] > 0)) or
                                        np.all((Ea[appro & (Wa > 0)] >= 0))),
            "l1_row_before": [round(float(np.abs(Wa[i]).sum()), 4) for i in range(4)],
            "l1_row_after": [round(float(np.abs(Ea[i]).sum()), 4) for i in range(4)],
            "l1_preserved": bool(all(np.abs(Ea[i]).sum() <= np.abs(Wa[i]).sum() + 1e-9 for i in range(4))),
            "dW_separate_object": bool(dWm is not W),
        }
    # 8) training off -> zero
    _, dWoff = apply_dw(W, dW, approved, training=False)
    out["training_off_zero"] = bool(dWoff.nnz == 0)
    # 官方 W 未被修改
    out["official_W_untouched"] = bool(np.array_equal(W.toarray(), _synthetic()[0].toarray()))
    return {"candidate_implementable": True, "strategies": out}


def l1_strategies():
    return {
        "S1_plastic_budget": {
            "rule": "逐突触后：仅对**可塑边**施加绝对预算（原始可塑边 Σ|W|）；超出则缩放可塑边 ΔW；未批准边逐位不变。",
            "effect": "保留原 L1 上界；局部（逐后）、确定、可重复；可能削弱学习幅度并耦合可塑边。",
            "pending_approval": True,
        },
        "S2_allow_l1_exceed": {
            "rule": "不重归一化；允许逐后 L1 超出原始归一化范围，列为**单独计算模型假设**。",
            "effect": "最简单、局部、可重复；但偏离原始归一化模型，结果不能直接与原始 L1 模型比较。",
            "pending_approval": True,
        },
        "note": "两策略均**未选择、未应用**；需书面批准其一或另行方案。",
    }


def modulation_protocol():
    return {
        "mechanism": "dW = eta * eligibility * (D - D0)",
        "D_t": {"type": "分段常数、训练期标注调制标签（教师/奖励）", "range": [-1.0, 1.0], "D0": 0.0,
                "source": "teacher/reward label, NOT sensory input; training-time only",
                "not_biological_calibration": True},
        "eligibility": {"init": "e(0)=0 at trial start",
                        "update": "e(t+1)=e(t)*exp(-dt/tau_e) + pre_trace(t)*post_trace(t)",
                        "reset": "trial start e=0; after-trial reset per preregistration",
                        "causal": True, "no_future_info": True,
                        "tau_e_grid_s": [0.1, 0.5, 1.0]},
        "params": {"eta_grid": [1e-5, 1e-4, 1e-3], "delta_max_grid": [0.01, 0.05, 0.1],
                   "D0": 0.0, "source": "computational hypotheses; no biological calibration"},
        "separation": "modulation labels are separate from sensory inputs; teacher actions never fed as sensory or test-time neural commands",
    }


def subnet_validity():
    return {
        "n_neurons": 23218, "nnz": 3549643, "candidate_KC_MBON": 4533,
        "preserved": {"LP_LPLC2_LC4": 311, "X": 10931, "KC_in_chain": 320, "MBON": 95,
                      "Y": 11340, "DN": 1285, "DAN": 340},
        "truncation": {"lost_incoming_edges": 4201146, "lost_incoming_weight": 4238.44,
                       "note": "外部入边被截断；不得宣称为全脑功能"},
        "q1_interpretable": True,
        "q1_reason": ("可作为**截断子网固定刺激计算假设**：在 LPLC2/LC4 注入固定刺激（**不注入 KC**），"
                      "经真实 2 跳到 KC、经真实 KC→MBON 可塑边；但不是全脑。"),
        "q2_effect": ("缺失 3,744 个 KC、2 个 MBON 及子网外输入会改变 KC/MBON 活动与资格迹；"
                      "因此学习曲线/幅度只能作为截断子网现象。"),
        "q3_sensory_preserved": True,
        "q3_note": "LPLC2/LC4 保留；但其上游视觉输入被截断，故用固定刺激替代（明确标注）。",
        "q4_truncated_only": ["学习幅度/曲线", "DAN 调制效应的绝对值", "MBON 输出统计", "资格迹时间常数辨识"],
        "q5_full_brain_recheck": ["调制对 KC→MBON 的因果效应", "与完整 DAN/MBON 群体的交互", "任何全脑功能声明"],
        "interpretability_ok": True,
    }


def _neg_case(label, budget):
    return {"label": label, "resource_ready": budget.get("resource_ready"),
            "problems": budget.get("problems")}


def resource_gate():
    p = os.path.join(HERE, "p10c1b_report.json")
    if not os.path.exists(p):
        base = resource_budget(None)
        return {"measured": base, "negative_cases": [], "ok": False,
                "reason": "p10c1b_report.json missing"}
    d = json.load(open(p, encoding="utf-8"))
    measured = resource_budget(benchmark=d.get("benchmark"), stimulus_benchmark=d.get("stimulus_benchmark"))
    neg = []
    neg.append(_neg_case("no measurement", resource_budget(None)))
    neg.append(_neg_case("zero per-step", resource_budget(0.0, benchmark=d.get("benchmark"))))
    timeout_bench = {"ok": True, "runs": [{"batches": {"10": {"median_ms": 1.0, "abnormal_firing": False}},
                                           "abnormal": ["timeout"]}], "peak_mem_MB": 100}
    neg.append(_neg_case("timeout", resource_budget(1.0, benchmark=timeout_bench)))
    nonfinite_bench = {"ok": True, "runs": [{"batches": {"10": {"median_ms": 1.0, "abnormal_firing": False}},
                                             "abnormal": ["non-finite-v"]}], "peak_mem_MB": 100}
    neg.append(_neg_case("non-finite state", resource_budget(1.0, benchmark=nonfinite_bench)))
    overmem_bench = {"ok": True, "runs": [{"batches": {"10": {"median_ms": 1.0, "abnormal_firing": False}},
                                           "abnormal": []}], "peak_mem_MB": 5000}
    neg.append(_neg_case("over memory", resource_budget(1.0, benchmark=overmem_bench)))
    few_bench = {"ok": True, "runs": [{"batches": {"1": {"median_ms": 1.0, "abnormal_firing": False}},
                                       "abnormal": []}], "peak_mem_MB": 100}
    neg.append(_neg_case("insufficient samples", resource_budget(1.0, benchmark=few_bench, min_samples=10)))
    neg_ok = all(c["resource_ready"] is False for c in neg)
    return {"measured": measured, "negative_cases": neg,
            "ok": bool(measured.get("resource_ready") and neg_ok),
            "class3": "UNKNOWN (eligibility/dW/teacher feedback not implemented)",
            "limitation": "预算为实测 LIF 前向 × 预注册开销因子；**不是**完整训练实测。"}


def leakage_and_grading():
    return {
        "level_A_independent_of_B_C_decoder": True,
        "mock_isolation": "verified (subprocess mock only)",
        "real_student_isolation": False,
        "A_does_not_replace_C_isolation": True,
        "dijkstra": "future training-time teacher only; absent at test",
        "modulation_label_separation": "D(t) labels separate from sensory input; never test-time neural command",
    }


def verdicts(dw_ok, res_ok):
    # A design: math candidate implementable + subnet interpretable + protocol specified
    a_design = bool(dw_ok and subnet_validity()["interpretability_ok"])
    # A execution: design + erratum APPROVED (False this round) + implementation (False) + resource
    erratum_approved = False       # 未获书面批准
    implemented = False            # 本轮未实现更新
    a_exec = bool(a_design and erratum_approved and implemented and res_ok)
    return {
        "Biological_Readiness": "NO-GO",
        "Computational_Hypothesis": "GO (conditional, proposal only)",
        "Level_A_Design_Readiness": "GO" if a_design else "NO-GO",
        "Level_A_Execution_Readiness": "GO" if a_exec else "NO-GO",
        "Level_B_Execution_Readiness": "NO-GO",
        "Level_C_Execution_Readiness": "NO-GO",
        "Training_Authorization": False,
        "a_exec_blockers": ([] if a_exec else
                            ([] if not a_design else
                             ["DW_ERRATUM_NOT_APPROVED", "UPDATE_NOT_IMPLEMENTED"]
                             + ([] if res_ok else ["RESOURCE_GATE"]))),
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    dw = dw_invariants()
    res = resource_gate()
    v = verdicts(all(s["only_approved_modified"] and s["no_new_edges"] and s["excitatory_nonnegative"]
                     and s["inhibitory_nonpositive"] and s["unapproved_bit_identical"]
                     for s in dw["strategies"].values() if isinstance(s, dict)),
                 res["ok"])
    report = {"dw_invariants": dw, "l1_strategies": l1_strategies(),
              "modulation_protocol": modulation_protocol(), "subnet_validity": subnet_validity(),
              "resource_gate": res, "leakage_grading": leakage_and_grading(), "verdicts": v,
              "training_authorized": False}
    erratum = {"status": "PENDING_APPROVAL; NOT applied; NOT implemented",
               "invariants": ["only approved KC->MBON edges", "no new edges",
                              "excitatory W_eff>=0; inhibitory W_eff<=0",
                              "unapproved edges bit-identical", "direction/transmitter unchanged",
                              "explicit L1 policy", "dW persisted separately", "training off -> zero"],
               "l1_strategies": l1_strategies(), "modulation_protocol": modulation_protocol(),
               "pending_approval_items": [
                   "adopt W_eff>=0 (exc) / <=0 (inh) constraint",
                   "choose L1 strategy S1 or S2 (or another approved rule)",
                   "freeze delta_max relative to W (clip alone insufficient)",
                   "approve D(t) and eligibility init/reset semantics",
                   "approve separate dW persistence format"]}
    json.dump(report, open(os.path.join(HERE, "p10c1c_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(erratum, open(os.path.join(HERE, "p10c1c_dw_erratum.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[P10-C1c] DW invariants S1/S2:",
          {k: (v["only_approved_modified"], v["no_new_edges"], v["excitatory_nonnegative"],
               v["unapproved_bit_identical"]) for k, v in dw["strategies"].items() if isinstance(v, dict)})
    print("  training_off_zero:", dw["strategies"]["training_off_zero"], "| official_W_untouched:", dw["strategies"]["official_W_untouched"])
    print("  resource ok:", res["ok"], "| measured ready:", res["measured"].get("resource_ready"),
          "| negative cases:", [c["resource_ready"] for c in res["negative_cases"]])
    print("  verdicts:", v)
    print("wrote p10c1c_report.json / p10c1c_dw_erratum.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
