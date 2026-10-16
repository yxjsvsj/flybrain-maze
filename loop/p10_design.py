"""P10-0：Dijkstra 教师指导、**真实连接组内突触可塑性**的可行性设计（**只设计/只读审计**）。

不训练、不运行 MaleCNS 新动力学、不运行虚拟/实体小车；不改 P1/官方 W/正式 readout/BASE/tags。
Dijkstra 仅在**训练**阶段当老师；**测试**阶段必须彻底移除。
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

from loop.visual_motor_audit import load_data

HERE = os.path.dirname(os.path.abspath(__file__))


def audit_mb_circuit(D=None):
    """只读审计真实连接组的学习回路（bodyId/类型/侧别/`W[post,pre]`）。"""
    D = D or load_data()
    ct = D["cell_type"]; sc = D["superclass"]; side = D["side"]; W = D["W"].tocsr()
    KC = [t for t in set(ct.tolist()) if t.startswith("KC")]
    MBON = [t for t in set(ct.tolist()) if t.startswith("MBON")]
    DAN = [t for t in set(ct.tolist()) if t.startswith("PPL") or t.startswith("PAM")]
    KI = np.flatnonzero(np.isin(ct, KC)); MI = np.flatnonzero(np.isin(ct, MBON)); DI = np.flatnonzero(np.isin(ct, DAN))
    vp = np.flatnonzero(sc == "visual_projection"); dn = np.flatnonzero(sc == "descending_neuron")
    lp = np.flatnonzero(np.isin(ct, ["LPLC2", "LC4"]))          # P6 的 ToF 感觉靶

    def blk(rows, cols):
        if not len(rows) or not len(cols):
            return {"edges": 0, "sumW": 0.0}
        M = W[rows][:, cols]
        return {"edges": int(M.nnz), "sumW": round(float(M.data.sum()), 3)}

    # 视觉->KC 的主要来源类型
    Mv = W[KI][:, vp].tocoo(); agg = collections.Counter()
    for j, w in zip(Mv.col, Mv.data):
        agg[str(ct[vp[j]])] += abs(w)
    vp2kc = [(t, round(float(w), 2)) for t, w in agg.most_common(10)]
    return {
        "counts": {"KC": len(KI), "MBON": len(MI), "DAN(PPL/PAM)": len(DI),
                   "visual_projection": len(vp), "descending_neuron": len(dn)},
        "KC_to_MBON": blk(MI, KI),
        "DAN_to_KC": blk(KI, DI),
        "DAN_to_MBON": blk(MI, DI),
        "visual_projection_to_KC": blk(KI, vp),
        "visual_projection_to_MBON": blk(MI, vp),
        "MBON_to_DN": blk(dn, MI),
        "KC_to_DN": blk(dn, KI),
        "LPLC2_LC4_to_KC": blk(KI, lp),
        "LPLC2_LC4_to_MBON": blk(MI, lp),
        "LPLC2_LC4_to_DAN": blk(DI, lp),
        "visual_to_KC_top_types": vp2kc,
        "direction": "W[post,pre]（row=post, col=pre）",
    }


def teacher_student_boundary():
    return {
        "teacher_privileged": ["地图/栅格", "目标坐标", "真实位姿(全局)", "Dijkstra 最短路径与动作"],
        "student_inputs_only": ["五路 ToF 距离 + 其时间变化", "自身神经活动产生的内部状态", "自身运动指令反馈"],
        "student_forbidden_inputs": ["地图", "全局真值位姿", "目标方向", "Dijkstra 路径", "老师动作(作为推理输入)"],
        "supervision": "训练阶段可提供**明确标注的监督/奖励**给学习系统；"
                       "**禁止**把'正确左/右转'直接当**感觉输入**或以之作为**测试**期的神经命令。",
        "pomdp": {
            "not_identifiable_from_5tof": ["已探索/未探索状态(纯 ToF 帧序列可能混淆)", "全局朝向/漂移",
                                           "目标相对可达性(门后/走廊尽头)", "死胡同 vs 环路(局部相似)"],
            "implications": "学生**无法复现**拥有全地图的教师；需要内部记忆/目标线索。**但任何显式目标/航向输入被禁止** → "
                            "可辨识性缺口必须在**脑内状态**中解决，而非外部注入。",
        },
    }


def plasticity_proposal():
    """唯一机制提案：多巴胺调制的三因素 KC->MBON 局部学习（含资格迹）。"""
    return {
        "mechanism": "**DA 调制的三因素 KC→MBON 突触可塑性**：ΔW = η · e_ij · (D(t) − D0)，"
                     "`e_ij`=KC→MBON 资格迹(局部、因果)，`D(t)`=DAN 多巴胺信号(奖励/惩罚)，`D0`=基线。",
        "literature": [
            {"cite": "Heisenberg 2003, Nat Rev Neurosci 4:266", "doi": "10.1038/nrn1054",
             "phenomenon": "MB 为联想学习中心（果蝇）", "evidence": "behavioral/genetic"},
            {"cite": "Aso et al. 2014, eLife 3:e04577", "doi": "10.7554/eLife.04577",
             "phenomenon": "KC→MBON 与 DAN 的解剖与功能分区", "evidence": "anatomy+function"},
            {"cite": "Hige et al. 2015, Neuron 85:1341", "doi": "10.1016/j.neuron.2014.12.045",
             "phenomenon": "DA 门控 KC→MBON 可塑性", "evidence": "electrophysiology"},
            {"cite": "Aso & Rubin 2016, Curr Biol 26:R225", "doi": "10.1016/j.cub.2016.01.046",
             "phenomenon": "DAN 的价值/强化编码", "evidence": "review+behavior"},
        ],
        "freeze_spec": {
            "plastic_edges": "**仅**真实存在的 KC→MBON 边（`W[post,pre]`, bodyId 逐一记录）；**不新建连接**。",
            "sign": "KC→MBON 为兴奋(ACh)；更新方向由 DA 符号决定(KC 可正可负门控)。",
            "bounds": "ΔW ∈ [−Δmax, +Δmax]（Δmax 与 g_unit 同类未定标，**标为未知**）。",
            "timescale": "资格迹 τ_e（文献量级，**待定标**）；DA 信号时间尺度由训练产生。",
            "learning_rate": "η **未知**（无本数据集定标）→ **禁止**按训练成功率挑 η。",
            "modulator": "DAN 群体（PPL1/PAM，bodyId 记录）；**训练**阶段由教师奖励生成 DA。",
            "W_readonly": "官方 `W` **永久只读**；学习只写**独立 ΔW 覆盖层**。",
            "decoder_frozen": "运动解码器必须**训练前冻结且有神经依据的固定解码器**；"
                              "**不得**用外部读出头学教师标签再称'脑内学习'。",
        },
    }


def decision(A=None):
    A = A or audit_mb_circuit()
    lc_edges = A["LPLC2_LC4_to_KC"]["edges"] + A["LPLC2_LC4_to_MBON"]["edges"] + A["LPLC2_LC4_to_DAN"]["edges"]
    lc_w = A["LPLC2_LC4_to_KC"]["sumW"] + A["LPLC2_LC4_to_MBON"]["sumW"] + A["LPLC2_LC4_to_DAN"]["sumW"]
    blockers = []
    if lc_w < 0.1:                                              # 1 条 0 权边不算通路
        blockers.append(f"**P6 的 ToF 感觉靶(LPLC2/LC4) → 学习回路(KC/MBON/DAN) 权重≈0**"
                        f"（边 {lc_edges}、Σw {lc_w:.3f}）：五路 ToF 当前**接不到**学习回路。")
    if A["MBON_to_DN"]["sumW"] < 1.0 and A["KC_to_DN"]["sumW"] < 1.0:
        blockers.append(f"学习回路 → 下行运动 很弱：MBON→DN {A['MBON_to_DN']}，KC→DN {A['KC_to_DN']}。")
    g_unit = "COBA 绝对电导 g_unit 仍未知（P9-1b1 科学 NO-GO）→ **不得自动解锁真实神经训练**。"
    blockers.append(g_unit)
    # 是否有替代感觉->KC 通路（供报告，不代表本轮 GO）
    alt = A["visual_projection_to_KC"]["edges"] > 0
    return {
        "verdict": "NO-GO",
        "reason": "按 §3.1：五路 ToF（当前感觉靶）→ 学习回路**缺必要连接**（0 边）；且学习回路→下行运动很弱；"
                  "加之 g_unit 未定标。**不得随意改选全部突触**。",
        "blockers": blockers,
        "alternative_note": (f"存在**其它**视觉投射(motion VPN: {A['visual_to_KC_top_types'][:5]}) → KC 的通路"
                             f"(visual_projection→KC {A['visual_projection_to_KC']})；但将其作为 ToF 感觉靶属**另立方案**，"
                             f"本轮**不**据此 GO，也不复活已关闭路线。") if alt else "无可信替代感觉→KC 通路。",
        "distinction": {
            "literature_function": "KC→MBON/DAN 有**学习功能**的实验证据（果蝇）。",
            "anatomy_path": "连接组**存在** KC→MBON/DAN 路径（真实边）。",
            "learns_navigation": "**不能**由此推断该路径能学会**导航**——尤其 ToF→学习回路**无连接**。",
        },
    }


def ablation_plan():
    return {
        "train_test_separation": "训练数据与迷宫**测试集严格分离**；教师示范/反馈/塑性更新**仅**在训练阶段。",
        "independent_test_process": "测试从**独立学生进程**启动：**不加载** Dijkstra/地图/教师标签/特权输入；"
                                    "以**日志、接口检查与断言**证明老师不存在。",
        "freeze_dW": "训练后 ΔW **冻结并保存**；与**未训练**的同一连接组对比。",
        "ablations": ["关闭神经输出", "恢复 ΔW=0(原始 W)", "打乱训练反馈", "冻结可塑突触",
                      "纯教师上限(Dijkstra)", "相同五路 ToF 的简单工程基线"],
        "metrics": ["独立迷宫完成率", "碰撞", "开始转弯", "左右方向", "结束转弯", "恢复直行", "**是否只是记住训练地图**"],
        "validity": "成功必须依赖**真实神经活动 + 内部学习 ΔW**；一旦测试期出现**外部 Dijkstra/导航记忆/老师动作/隐藏目标**→**判定无效**。",
        "goal": "不以'完全模仿最短路径'为唯一目标；区分**可达出口的局部避障学习** 与 **真正全局最短路径规划**。",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    A = audit_mb_circuit()
    d = decision(A)
    print("[P10-0] counts:", A["counts"])
    for k in ("KC_to_MBON", "DAN_to_KC", "DAN_to_MBON", "visual_projection_to_KC",
              "MBON_to_DN", "KC_to_DN", "LPLC2_LC4_to_KC", "LPLC2_LC4_to_MBON", "LPLC2_LC4_to_DAN"):
        print("  %-26s %s" % (k, A[k]))
    print("[P10-0] visual->KC top:", A["visual_to_KC_top_types"][:6])
    print("[P10-0] verdict:", d["verdict"])
    for b in d["blockers"]:
        print("   blocker:", b)
    out = {"audit": A, "teacher_student": teacher_student_boundary(),
           "proposal": plasticity_proposal(), "decision": d, "ablation": ablation_plan()}
    json.dump(out, open(os.path.join(HERE, "p100.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote p100.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
