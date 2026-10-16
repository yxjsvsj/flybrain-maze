"""P9-0：真实 MaleCNS 连接组的新**神经动力学引擎**设计（**只设计；不运行大规模模拟**）。

目标（不变）：五路 ToF 提供感觉，真实 MaleCNS 网络活动决定**转弯开始/左右/结束/恢复直行**；
任何人工层只允许限速或急停。**不改真实连接组身份与连接**；不按迷宫成绩改权重；
**不**重包装已关闭的 P8 未定标渐变释放。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 原始文献（完整题名 + DOI；用于**新动力学**的参数来源） ----
REFS = [
    {"cite": "Groschner, Malis, Zuidinga & Borst 2022, Nature 603(7899):119-123 [P9-1a 勘误：原写 602:92]",
     "title": "A biophysical account of multiplication by a single neuron", "doi": "10.1038/s41586-022-04428-3",
     "use": "T4 **电导型**(passive conductance-based) 模型：E_exc(胆碱能)、E_inh(GluCl 分流抑制)、输入电阻测量。"
            "**勘误/边界：该定标仅针对 T4，不可外推为全脑统一参数**。"},
    {"cite": "Seelig & Jayaraman 2015, Nat Neurosci 18:718",
     "title": "Neural dynamics for landmark orientation and angular path integration", "doi": "10.1038/nn.3996",
     "use": "CX 神经元膜动力学/时间常数量级"},
    {"cite": "Turner-Evans et al. 2017, eLife 6:e23496",
     "title": "Angular velocity integration in a fly heading circuit", "doi": "10.7554/eLife.23496",
     "use": "P-EN 渐变电导整合的动力学量级"},
    {"cite": "Nagel et al. 2019 / Fisher et al. 2022", "title": "T4/T5 运动检测（电生理/电导）",
     "doi": "10.1016/j.cub.2021.10.063", "use": "T4/T5 输入为 graded-potential；τ 量级"},
    {"cite": "Borst et al. 2010, Annu Rev Neurosci 33:49", "title": "Fly motion vision",
     "doi": "10.1146/annurev-neuro-060909-153155", "use": "LPTC 渐变传导/时间常数"},
    {"cite": "flybrain 构建 (build.py)", "title": "per-neuron/per-tbar 递质预测 -> 权重符号",
     "doi": "—", "use": "连接组真实：突触计数权重 + 递质符号（兴奋/抑制）"},
]

# ---- 数据清单（真实可得 vs 未知） ----
DATA = {
    "real_available": {
        "connectome": "weights.npz：166,700 神经元、25,582,938 条有符号边（62% 正/38% 负）",
        "transmitter_per_neuron": "body-neurotransmitters-male-cns-v1.0.feather (43 MB, GCS)",
        "transmitter_per_tbar": "tbar-neurotransmitters-male-cns-v1.0.feather (2.65 GB, GCS) —— **逐突触(t-bar)**递质",
        "identity": "cell_type / side / superclass / positions / visual-azimuth（brain.npz）",
        "annotation": "body-annotations-...feather：class/subclass/supertype/entryNerve/...（P2d-24c 已核验）",
    },
    "literature_estimate": {
        "E_rev": "E_exc≈0 mV、E_inh≈-70 mV、E_l≈-60 mV、V_th≈-45 mV（果蝇 patch-clamp 量级；逐细胞有别）",
        "tau_syn": "兴奋(nAChR)~1-5 ms；抑制(GABA/GluCl)~5-20 ms（果蝇电生理量级）",
        "tau_m": "~10-100 ms，逐类型不同（CX/T4 有测量；多数类型缺）",
    },
    "unknown": {
        "absolute_synaptic_conductance": "**逐连接电导尺度**（连接组只给突触**计数**，未给电导）——**最关键未知**",
        "per_type_membrane_params": "本连接组多数细胞类型的 C/g_l/V_rest/τ_m 无直接测量",
        "receptor_composition": "逐连接受体类型/动力学（需 t-bar 递质 + 假设）",
    },
}


def architecture():
    """新引擎：**电导型 (COBA) LIF**——保留连接组身份/连接，替换动力学核心。"""
    return {
        "neuron": "C_i dV_i/dt = -g_l,i (V_i - E_l,i) - g_exc,i (V_i - E_exc) - g_inh,i (V_i - E_inh) + I_noise,i",
        "synapse": "d g_x,i/dt = -g_x,i / tau_syn,x + sum_j w_ij * s_j(t)；w 符号来自**递质预测**（兴奋->g_exc，抑制->g_inh）",
        "spike": "V_i>=V_th -> spike、V_i<-V_reset、不应期（同 flybrain 接口）；**发放为脉冲事件**——"
                  "COBA 以 **spike 事件**驱动突触电导（`g += g_unit·(W@S)`），**不使用复位前 V 的连续释放**；"
                  "P8 未定标渐变释放**已关闭**，不重用。",
        "key_change": "由'电流型、瞬时、全同参数'改为'**电导型 + 反转电位 + 突触时间常数**'；"
                      "抑制为**分流(shunting)**（不是简单负电流）→ 非线性积分、正确驱动力 (E-V)。",
        "not_a_gain": "机制解释**不**是'增益更强故 DN 发放'，而是：分流抑制与驱动力 (E-V) 改变**积分/门控**；"
                      "τ_syn 让突触事件在时间上叠加。",
        "preserved": "神经元身份、左右侧别、真实连接、突触计数权重、递质符号**均不变**；不重构连接、不按成绩改权重。",
        "forbidden": "不重包装 P8 的 r=clip(V,0,1) 未定标渐变释放；不注入航向/目标角；不手写转向/计时/回正。",
    }


def param_table():
    """参数来源表（单位/来源/细胞/依据/不确定性）。"""
    def row(sym, unit, rng, source, cells, basis, unc):
        return {"symbol": sym, "unit": unit, "range": rng, "source": source, "cells": cells,
                "basis": basis, "uncertainty": unc}
    lit = "文献"
    return [
        row("E_exc", "mV", "≈0 (相对静息 +55~+70)", lit+" Groschner2022 等", "胆碱能靶(多数)",
            "反转电位", "中(逐细胞有别)"),
        row("E_inh", "mV", "≈-70 (GluCl/GABA)", lit+" Groschner2022", "GluCl/GABA 靶", "分流抑制反转", "中"),
        row("E_l", "mV", "≈-60 (静息)", lit, "全部", "漏电反转", "中"),
        row("V_th", "mV", "≈-45", lit, "全部", "发放阈值", "中"),
        row("tau_syn_e", "ms", "1-5 (nAChR)", lit, "兴奋突触", "EPSC 衰减", "中"),
        row("tau_syn_i", "ms", "5-20 (GABA/GluCl)", lit, "抑制突触", "IPSC 衰减", "中"),
        row("tau_m=C/g_l", "ms", "10-100 (逐类型)", lit, "全部(**多类型缺测量**)", "膜时间常数", "高(多数未知)"),
        row("w_ij(相对)", "无单位", "连接组突触计数", "**真实数据** brain.npz", "全部", "相对突触效能", "低(计数已知)"),
        row("sign(w_ij)", "±", "兴奋/抑制", "**真实数据** 递质预测", "全部", "兴奋->g_exc/抑制->g_inh", "低"),
        row("g_syn 绝对尺度", "nS/突触", "**未知**", "**需要额外证据**", "全部", "计数->电导换算", "**高(最关键)**"),
    ]


def validation():
    """验证协议（分层；用 P6 冻结 ToF 输入作对照；公平比较原 LIF）。"""
    return {
        "fairness": "同一连接组、同一冻结刺激、可比执行环境；**原 LIF vs 新引擎**并排。",
        "layers": {
            "P5_layer": "视叶/光感受器 -> 视觉投射(LPLC2/LC4)：层内是否出现**方向/左右特异**（对照无刺激）。",
            "P6_layer": "视觉投射 -> 下行靶：冻结 ToF 输入下，DN 靶是否出现**>3σ** 的对照差值。",
            "P7_respect": "**不**默认 LC10a->DNa02 直接存在（P7：0 直接边、弱两跳）；转向读出须用**解剖固定**的 DN 靶。",
            "recovery": "撤除刺激后是否回落至对照（**必要条件**，不据此宣称会结束转弯）。",
        },
        "controls": ["无刺激(同 seed)", "左刺激", "右刺激", "左右镜像", "（可选）收缩/亮度"],
        "metrics": "同 seed **刺激-对照**同步窗；per-cell + 群体；K_SIG=3×对照每细胞 std；跨 seed 逐报。",
        "gate": "GO 门槛**运行前冻结**：左右特异 ≥4/5 seed 且 >3σ；否则 NO-GO。",
        "no_nav_tuning": "**不**以导航成功率反向调模型参数。",
        "forbidden_io": "ToF **不**直接输出左/右/固定转角/回正；无计时器、无手写方向。",
    }


def budget():
    """资源/性能预算（**真实估计**，不假设全脑几分钟跑完）。"""
    from flybrain.brain import FlyBrain
    return {
        "neurons": 166700, "edges": 25582938,
        "current_lif_cost": "P2d-28 实测：CPU ~5 s / 40 s-sim(2000 步)；CUDA 不可用（本机 'CUDA path could not be detected'）。",
        "new_engine_overhead": "增加 2 个突触电导状态(g_exc,g_inh)+每突触非线性(分流/驱动力) → 估 ~2-4× 于原 LIF。",
        "estimate": "**单次 40 s 全脑 trial ≈ 10-20 s(CPU)**；一次 5 seed × 5 条件 × 5 s 短程筛选 ≈ 数分钟；"
                    "**全脑多 seed × 40 s 实验 ≈ 数十分钟~小时**，非'几分钟'。",
        "memory": "g_exc/g_inh 2×166,700 float32 ≈ 1.3 MB；稀疏 W 不变(~25.6M nnz)。",
        "plan": "**先**小规模(子网络/少数类型)机制测试 -> 通过后再全脑；先 CPU，CUDA 需先验证设备生效(checklist)。",
        "device_check": "运行前**断言并记录** device；CUDA 路径需核对 `_W` 生效（P4-1a 教训：只改 CPU 数组不等于 CUDA 已运行）。",
        "override_module": "**独立模块**覆盖 step（不改 flybrain 安装包/正式引擎）。",
    }


def verdict():
    return {
        "decision": "GO（有条件，小规模先行）",
        "because": "①架构有**定量实验依据**（Groschner 2022 电导型 T4 模型等）；②连接组/递质数据**真实可得**（含逐 t-bar 递质）；"
                   "③主要不确定（**绝对电导尺度**）以**声明的单位约定 + 有界范围**处理，**非**隐藏增益。",
        "conditions": [
            "params 依 §param_table（单位/来源/细胞/依据/不确定性齐备）；**绝对电导尺度须声明约定并标注不确定**，不得为出方向而调。",
            "**先小规模机制测试**（子网络）→ 过**预注册门槛**后才允许全脑。",
            "公平比较原 LIF；尊重 P7（LC10a->DNa02 0 直接边）；用 P6 冻结 ToF 输入作对照。",
            "若小规模测试即**证伪**（无层内方向特异或无 DN 对照差>3σ）→ **关闭**，不加第二候选。",
        ],
        "need_extra_evidence": [
            "**逐类型电导/膜参数**（本连接组的 C/g_l/V_rest/τ_m）——当前多数类型缺直接测量。",
            "**逐连接电导尺度**（计数->nS 的换算），最好来自本数据集或同源电生理。",
        ],
        "if_unavailable": "若上述证据无法取得，则**只能**在**声明约定+不确定性**下做小规模测试；**不得**为转弯效果编造参数。",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    out = {"data": DATA, "references": REFS, "architecture": architecture(),
           "param_table": param_table(), "validation": validation(), "budget": budget(),
           "verdict": verdict()}
    json.dump(out, open(os.path.join(HERE, "p90_engine.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[P9-0] architecture:", out["architecture"]["key_change"][:50], "...")
    print("[P9-0] params rows:", len(out["param_table"]), "| unknown:", list(DATA["unknown"]))
    print("[P9-0] budget:", out["budget"]["estimate"][:60], "...")
    print("[P9-0] verdict:", out["verdict"]["decision"])
    print("wrote p90_engine.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
