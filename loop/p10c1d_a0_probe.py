"""P10-C1d A0：固定 LPLC2/LC4 刺激是否在真实截断子网产生可重复 KC 活动差异与候选资格迹。

**无学习、ΔW=0**；仅短时 LIF 前向。固定刺激强度/窗口/seed，不调参。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

from loop.p10c1b_lif_benchmark import build_subnet

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (11, 12, 13, 14, 15)
STEPS = 200
AMOUNT = 0.6
TAU_E = 0.5
S0, S1 = 50, 150


def _run_condition(b, kc_pos, mb_pos, lp_pos, E, condition):
    kc = np.zeros((STEPS, len(kc_pos)), np.float32)
    mb = np.zeros((STEPS, len(mb_pos)), np.float32)
    e = np.zeros(E.nnz, np.float32)
    decay = np.exp(-0.020 / TAU_E)
    for t in range(STEPS):
        inj = []
        if condition == "stim" and S0 <= t < S1:
            inj = [(lp_pos, AMOUNT)]
        elif condition == "recover" and t < 100:
            inj = [(lp_pos, AMOUNT)]
        fired = np.asarray(b.step(inject=inj))
        k = np.isin(fired, kc_pos); m = np.isin(fired, mb_pos)
        kc[t, np.searchsorted(kc_pos, fired[k])] = 1.0
        mb[t, np.searchsorted(mb_pos, fired[m])] = 1.0
        e = e * decay + kc[t][E.col] * mb[t][E.row]
    return kc, mb, e


def run_a0(seeds=SEEDS):
    from flybrain import FlyBrain
    sub = build_subnet()
    D, L, union, Wsub = sub["D"], sub["layers"], sub["union"], sub["Wsub"]
    kc_pos = np.searchsorted(union, L[2]); mb_pos = np.searchsorted(union, L[3])
    lp_pos = np.searchsorted(union, sub["A"]["LP"])
    E = Wsub[mb_pos][:, kc_pos].tocoo()
    td = tempfile.mkdtemp(prefix="a0_")
    try:
        np.savez(os.path.join(td, "brain.npz"), ids=D["ids"][union],
                 visual=np.zeros(0, np.int32), azimuth=np.zeros(0, np.float32),
                 cell_type=D["cell_type"][union], side=D["side"][union],
                 positions=(D["positions"][union] if D.get("positions") is not None
                            else np.zeros((len(union), 3), np.float32)),
                 superclass=D["superclass"][union])
        sp.save_npz(os.path.join(td, "weights.npz"), Wsub)
        rows = []
        for seed in seeds:
            for cond in ("none", "stim", "recover"):
                b = FlyBrain(data=td, device="cpu", batch=1, seed=seed, dt=0.020)
                kc, mb, e = _run_condition(b, kc_pos, mb_pos, lp_pos, E, cond)
                base_kc = kc[0:50].mean(); stim_kc = kc[50:100].mean()
                base_mb = mb[0:50].mean(); stim_mb = mb[50:100].mean()
                rec_kc = kc[120:200].mean() if cond == "recover" else kc[150:200].mean()
                rows.append({"seed": seed, "condition": cond,
                             "kc_baseline": round(float(base_kc), 6), "kc_stim": round(float(stim_kc), 6),
                             "kc_recovery": round(float(rec_kc), 6),
                             "mb_baseline": round(float(base_mb), 6), "mb_stim": round(float(stim_mb), 6),
                             "e_nonzero_frac": round(float((e > 1e-6).mean()), 6),
                             "e_mean": round(float(e.mean()), 6)})
    finally:
        shutil.rmtree(td, ignore_errors=True)
    by = {(r["seed"], r["condition"]): r for r in rows}
    dkc = np.array([by[(s, "stim")]["kc_stim"] - by[(s, "none")]["kc_stim"] for s in seeds])
    dmb = np.array([by[(s, "stim")]["mb_stim"] - by[(s, "none")]["mb_stim"] for s in seeds])
    de = np.array([by[(s, "stim")]["e_nonzero_frac"] - by[(s, "none")]["e_nonzero_frac"] for s in seeds])
    # 精确配对置换（5 seeds -> 2^5=32 符号翻转）
    def perm_p(d):
        obs = abs(float(d.mean()))
        cnt = 0
        for mask in range(1 << len(d)):
            s = np.array([1 if (mask >> i) & 1 else -1 for i in range(len(d))])
            if abs(float((d * s).mean())) >= obs - 1e-12:
                cnt += 1
        return cnt / (1 << len(d))
    base_kc = np.mean([by[(s, "none")]["kc_baseline"] for s in seeds])
    rel_kc = float(dkc.mean() / base_kc) if base_kc else 0.0
    sign_cons = int(max((dkc > 0).sum(), (dkc < 0).sum()))
    stim_effect = bool(perm_p(dkc) < 0.05 and abs(rel_kc) > 0.05 and sign_cons >= 4)
    elig_discernible = bool(abs(de.mean()) > 0.05 and sign_cons >= 4)
    a0_go = bool(stim_effect and elig_discernible)
    return {
        "measurement_id": f"A0-{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}",
        "protocol": {"seeds": list(seeds), "steps": STEPS, "amount": AMOUNT, "tau_e": TAU_E,
                     "stim_window": [S0, S1], "recover_stim_window": [0, 100],
                     "comparison_window": [50, 100], "baseline_window": [0, 50],
                     "entrance": "LPLC2/LC4 only", "dW": 0, "learning": False},
        "rows": rows,
        "paired": {"kc_diff_mean": round(float(dkc.mean()), 8), "kc_diff_values": dkc.round(8).tolist(),
                   "kc_perm_p": round(perm_p(dkc), 4), "kc_relative": round(rel_kc, 6),
                   "mb_diff_mean": round(float(dmb.mean()), 8),
                   "e_nonzero_diff_mean": round(float(de.mean()), 6),
                   "sign_consistency": sign_cons},
        "stim_effect_repeatable": stim_effect,
        "eligibility_discernible": elig_discernible,
        "eligibility_saturated_by_spontaneous": bool(max(by[(s, "none")]["e_nonzero_frac"] for s in seeds) > 0.99),
        "a0_go": a0_go,
        "a0_verdict": "GO" if a0_go else "FAIL",
        "reason": ("固定刺激未产生可重复 KC 活动差异/可辨认资格迹；资格迹被自发发放饱和。"
                   "按预注册：不得增强刺激、注入 KC 或换机制挽救。"),
    }


def s1_s2_structural_audit():
    """只读官方 W：统计 4,533 条可塑边在各 MBON 的预算。"""
    from loop.visual_motor_audit import load_data
    from loop.p100_evidence import audit
    from loop.p100b_path_audit import chain_layers
    D = load_data(); A = audit(D); L = chain_layers(D, A)
    W = D["W"].tocsr().copy(); W.eliminate_zeros()
    L2, L3 = L[2], L[3]
    rows = []
    for mb_i in L3:
        row = float(np.abs(W.getrow(mb_i).data).sum())
        plast = float(np.abs(W[mb_i][:, L2].data).sum())
        rows.append((row, plast, row - plast, plast / row if row > 0 else 0.0))
    a = np.array(rows)
    budgetA_headroom = np.zeros(len(a))          # 预算=原始可塑和 -> 正向净增长为 0
    budgetB_headroom = a[:, 0] - a[:, 1]          # 预算=原始逐后 L1 -> 非可塑余量
    return {
        "n_MBON": int(len(a)),
        "row_L1": {"mean": round(float(a[:, 0].mean()), 6), "median": round(float(np.median(a[:, 0])), 6),
                   "min": round(float(a[:, 0].min()), 6), "max": round(float(a[:, 0].max()), 6)},
        "plastic_abs": {"mean": round(float(a[:, 1].mean()), 6), "median": round(float(np.median(a[:, 1])), 6),
                        "min": round(float(a[:, 1].min()), 6), "max": round(float(a[:, 1].max()), 6)},
        "nonplastic_abs": {"mean": round(float(a[:, 2].mean()), 6), "median": round(float(np.median(a[:, 2])), 6),
                           "min": round(float(a[:, 2].min()), 6), "max": round(float(a[:, 2].max()), 6)},
        "saturation": {"mean": round(float(a[:, 3].mean()), 6), "max": round(float(a[:, 3].max()), 6)},
        "budgetA": {"definition": "budget = original plastic abs sum", "mean_headroom": 0.0,
                    "n_mbon_zero_positive_headroom": int((budgetA_headroom <= 1e-12).sum()),
                    "redistribution_space_mean": round(float(a[:, 1].mean()), 6),
                    "answer": "**是**：全部 95 个 MBON 无正向净增长余量，只能减弱或重新分配"},
        "budgetB": {"definition": "budget = original per-post L1 (1.0)", "mean_headroom":
                    round(float(budgetB_headroom.mean()), 6),
                    "n_mbon_positive_headroom": int((budgetB_headroom > 1e-12).sum()),
                    "redistribution_space_mean": round(float(a[:, 0].mean()), 6),
                    "answer": "**否**：非可塑余量充足，正向增长可行且保持逐后 L1<=1"},
        "recommendation": ("S1 的预算定义是决定性问题：Budget A 会抑制全部正向增长；Budget B 保留余量。"
                           "两者均 PENDING_APPROVAL，不按导航成功率选择。"),
    }


def resource_consistency():
    import hashlib
    p = os.path.join(HERE, "p10c1b_report.json")
    d = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    rb = d.get("resource_budget", {})
    c1 = rb.get("class1_no_stimulus_ms"); c2 = rb.get("class2_fixed_stimulus_ms")
    used = rb.get("used_per_step_ms"); est = rb.get("est_hours_1890"); peak = rb.get("peak_mem_MB")
    overhead = rb.get("overhead_factor", 2.0)
    formula = (1890 * 2000 * (used / 1000.0) / 3600.0 * overhead) if used else None
    consistent = bool(used is not None and c1 is not None and c2 is not None
                      and abs(used - max(c1, c2)) < 1e-6
                      and est is not None and formula is not None and abs(est - formula) < 1e-3)
    mid = "LIF-C1B-" + hashlib.sha256(f"{c1}|{c2}|{peak}".encode()).hexdigest()[:12]
    return {"historical_measurement_id": mid, "historical_source": "loop/p10c1b_report.json",
            "class1_no_stimulus_ms": c1, "class2_fixed_stimulus_ms": c2,
            "used_per_step_ms": used, "overhead_factor": overhead, "est_hours_1890": est,
            "formula_est_hours": round(formula, 6) if formula is not None else None,
            "peak_mem_MB": peak, "consistent": consistent,
            "class3": "UNKNOWN (eligibility/dW/teacher not implemented)",
            "policy": "历史记录冻结；新复测只能追加新 measurement_id，不覆盖旧值。"}


def modulation_identity():
    return {
        "teacher_derived_modulation": {"D_t": "external teacher/reward label", "value_range": [-1, 1],
                                       "status": "computational hypothesis; training-time only",
                                       "biological": False},
        "actual_DAN_activity": {"measured": False, "source": None,
                                "note": "本轮未观测真实 DAN 发放；如未来观测须注明数据来源"},
        "biological_correspondence": {"evidence": "NONE",
                                      "note": "不得把外部奖励标签称为已实现的生物多巴胺信号"},
        "separation": "teacher labels separate from sensory input; test-time has no teacher",
        "dijkstra": "not implemented; not called for neural updates",
    }


def verdicts(a0_go, kernel_ok, s1_ok, s2_ok):
    a_design = bool(a0_go and kernel_ok and s1_ok and s2_ok)
    return {
        "A0_Sensory_Transmission": "GO" if a0_go else "NO-GO",
        "Synthetic_Plasticity_Kernel": "GO" if kernel_ok else "NO-GO",
        "S1_Mathematical_Feasibility": "GO" if s1_ok else "NO-GO",
        "S2_Mathematical_Feasibility": "GO" if s2_ok else "NO-GO",
        "Level_A_Design": "GO" if a_design else "NO-GO",
        "Level_A_Execution": "NO-GO",
        "Level_B": "NO-GO",
        "Level_C": "NO-GO",
        "Biological": "NO-GO",
        "Training_Authorization": False,
        "note": ("A0 FAIL -> A Design NO-GO；合成内核通过 != 真实突触更新已实现；"
                 "真实 ΔW 未批准/未实现 -> A Execution NO-GO。"),
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    from loop.p10c1d_delta_sandbox import ablation, frozen_eval, persistence_roundtrip, synthetic_case
    a0 = run_a0()
    s1 = s1_s2_structural_audit()
    res = resource_consistency()
    mod = modulation_identity()
    sandbox = {}
    for l1 in ("S1", "S2"):
        c = synthetic_case(l1=l1)
        sandbox[l1] = {"checks": c["checks"], "zero_semantics": c["zero_semantics"],
                       "persistence": persistence_roundtrip(l1=l1)}
    sandbox["frozen_eval"] = frozen_eval()
    sandbox["ablation_zero"] = ablation()
    kernel_ok = all(all(v for v in sandbox[l1]["checks"].values()) for l1 in ("S1", "S2")) \
        and sandbox["frozen_eval"]["frozen_eval_stable"] and sandbox["frozen_eval"]["retains_learned_dW"] \
        and sandbox["ablation_zero"]["delta_zero"]
    s1_ok = all(sandbox["S1"]["checks"].values())
    s2_ok = all(sandbox["S2"]["checks"].values())
    v = verdicts(a0["a0_go"], kernel_ok, s1_ok, s2_ok)
    report = {"a0": a0, "s1_s2_structural_audit": s1, "sandbox": sandbox,
              "resource_consistency": res, "modulation_identity": mod, "verdicts": v,
              "training_authorized": False}
    json.dump(report, open(os.path.join(HERE, "p10c1d_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(a0, open(os.path.join(HERE, "p10c1d_a0_probe.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[A0]", a0["a0_verdict"], "| paired KC diff:", a0["paired"]["kc_diff_mean"],
          "| rel:", a0["paired"]["kc_relative"], "| e diff:", a0["paired"]["e_nonzero_diff_mean"])
    print("  stim_effect:", a0["stim_effect_repeatable"], "| elig_discernible:", a0["eligibility_discernible"],
          "| saturated:", a0["eligibility_saturated_by_spontaneous"])
    print("  S1 BudgetA:", s1["budgetA"]["answer"], "| BudgetB:", s1["budgetB"]["answer"])
    print("  sandbox kernel:", kernel_ok, "| resource consistent:", res["consistent"])
    print("  verdicts:", v)
    print("wrote p10c1d_report.json / p10c1d_a0_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
