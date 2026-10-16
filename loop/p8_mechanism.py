"""P8-0：MaleCNS 神经动力学**机制设计**（只写提案/审查；**不运行 MaleCNS**、不写车控）。

组装：当前 flybrain 0.1.0 LIF 的**实际实现**缺陷对照表 + 用 P5/P6/P7 归档数据定位瓶颈 +
**唯一**机制提案（细胞类型特异**渐变/非脉冲**释放）+ 可证伪判据 + P8-1 预注册协议。
不改 P1/官方权重/正式 readout/BASE/tags；不开发导航器；不接/上电任何车。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 文献依据（完整题名 + DOI + 支持的**具体生理现象** + 证据类型） ----
# evidence 标记：direct_electro = 直接电生理(膜电位/电流)；release = 直接**释放动力学**；calcium = 钙成像；anatomy = 解剖
REFS = [
    {"cite": "Seelig & Jayaraman 2015, Nature Neuroscience 18:718", "title": "Neural dynamics for landmark orientation and angular path integration",
     "doi": "10.1038/nn.3996", "phenomenon": "EPG/PEN 头朝向（CX）以**非脉冲/渐变**膜电位编码", "evidence": "direct_electro"},
    {"cite": "Turner-Evans et al. 2017, eLife 6:e23496", "title": "Angular velocity integration in a fly heading circuit",
     "doi": "10.7554/eLife.23496", "phenomenon": "P-EN 角速度积分（渐变膜电位）", "evidence": "direct_electro"},
    {"cite": "Rayshubskiy et al. 2020, Cell 182:1372", "title": "Neural circuit mechanisms for steering control in walking Drosophila",
     "doi": "10.1016/j.cell.2020.07.016", "phenomenon": "PFL3 为转向输出（接 EPG 罗盘 + PFN 航向）", "evidence": "calcium"},
    {"cite": "Hulse et al. 2021, Nature 591:91", "title": "A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and context-dependent action selection",
     "doi": "10.1038/s41586-021-03307-7", "phenomenon": "PFL2/PFL3 左右转向与步态", "evidence": "anatomy"},
    {"cite": "Klapoetke et al. 2017, Nature 551:230", "title": "Ultra-selective looming detection from radial motion opponency",
     "doi": "10.1038/nature24626", "phenomenon": "**LPLC2** 径向运动对立；**钙成像**证明 looming 选择性", "evidence": "calcium"},
    {"cite": "Wu et al. 2016, eLife 5:e12244", "title": "Visual projection neurons in the Drosophila lobula link feature detection to distinct behavioral programs",
     "doi": "10.7554/eLife.12244", "phenomenon": "**LC4** looming 反应（钙成像）+ 光遗传行为；部分 LC 单侧可致转向", "evidence": "calcium"},
    {"cite": "Maisak et al. 2013, Nature 500:212", "title": "A directional tuning map of Drosophila elementary motion detectors",
     "doi": "10.1038/nature12320", "phenomenon": "**T4/T5** 方向选择性（**钙成像**；T4 ON / T5 OFF）", "evidence": "calcium"},
    {"cite": "Groschner et al. 2022, Nature 602:92", "title": "A biophysical account of multiplication by a single neuron",
     "doi": "10.1038/s41586-022-04428-3", "phenomenon": "**T4 全细胞膜电位**(渐变)记录 + 电导模型；输入神经元为 graded-potential", "evidence": "direct_electro"},
    {"cite": "Gruntman, Reimers, Romani & Reiser 2022, Current Biology 32:P94", "title": "Non-preferred contrast responses in the Drosophila motion pathways reveal a receptive field structure that explains a common visual illusion",
     "doi": "10.1016/j.cub.2021.10.063", "phenomenon": "T4/T5 **全细胞**膜电位（渐变），非线性/整流", "evidence": "direct_electro"},
    {"cite": "Borst, Haag & reviewers (Lobula Plate 综述)", "title": "Fly motion vision / graded transmission of lobula plate tangential cells",
     "doi": "(综述；见 Borst et al. 2010, Annu Rev Neurosci 33:49, 10.1146/annurev-neuro-060909-153155)",
     "phenomenon": "LPTC/视觉中间神经元以**渐变电位**传递；果蝇 lobula 输出轴突极细、多为渐变传导", "evidence": "direct_electro/anatomy"},
    {"cite": "fly64 / flybrain 0.1.0", "title": "参考 LIF 标定（tonic/gain 手工标定）", "doi": "—",
     "phenomenon": "LIF 全同参数、瞬时突触", "evidence": "impl"},
]

# 逐类细胞对"非脉冲/渐变**释放**"的**直接**证据核查（临床级要求：不得以钙成像或解剖代替释放动力学）
CELL_EVIDENCE = {
    "photoreceptor": {"graded_potential": "direct (光转导)", "graded_release": "direct (组胺紧张性释放)",
                      "verdict": "有直接释放证据"},
    "T4/T5": {"graded_potential": "direct (whole-cell: Groschner 2022/ Gruntman 2022)",
              "graded_release": "inferred (非脉冲->假定渐变释放，未直接测释放)", "verdict": "膜电位直接、释放仅推断"},
    "LPLC2": {"graded_potential": "calcium only (Klapoetke 2017)", "graded_release": "none",
              "verdict": "**无直接释放动力学证据**（钙成像/解剖不足）"},
    "LC4": {"graded_potential": "calcium (Wu 2016)", "graded_release": "none",
            "verdict": "**无直接释放动力学证据**"},
}


def resting_analysis():
    """解析核对（不运行网络）：无刺激静息电位与 r=clip(v,0,1) 的基线释放。"""
    from flybrain.brain import FlyBrain
    import math
    decay = math.exp(-FlyBrain.dt / FlyBrain.tau)
    tonic = FlyBrain.tonic  # 已按 dt 重标定（dt=0.02 时=0.14）
    v_rest = tonic / (1.0 - decay)
    return {
        "dt": FlyBrain.dt, "tau": FlyBrain.tau, "decay": round(decay, 4), "tonic": tonic,
        "v_rest_no_input": round(v_rest, 4),
        "threshold": 1.0,
        "issue": f"无输入静息 v_rest≈{v_rest:.3f}（=tonic/(1-decay)）→ r=clip(v,0,1) 的**基线释放≈{v_rest:.2f}**；"
                 f"而刺激诱发的 Δv 量级 ~0.1 → 基线释放**远大于**刺激分量，r 被静息电位主导。",
        "reset_order": "[**该渐变释放方案已关闭，被 P9 COBA 脉冲事件传递取代**；以下为历史记录] `brain.step()` 先 `fired=v>=1`、再 `v[fired]=0`；故渐变释放若用 v 必须取**复位前**的 v，"
                       "且与 0/1 脉冲**量纲不同**（脉冲=事件增量；渐变=速率/电导），二者混用会改变 W@out 的物理含义。",
        "conclusion": "朴素 r=clip(v,0,1) **不可用**（基线释放过大且未标定）；要可用需减静息并标定斜率 -> **引入新的未标定自由参数**。",
    }


def simulator_defects():
    """当前 flybrain 0.1.0 LIF 实际实现 vs 数据/文献/未知。"""
    from flybrain.brain import FlyBrain
    return {
        "membrane": {"impl": f"LIF v<-exp(-dt/tau)v + gain*W@spikes + tonic + noise; dt={FlyBrain.dt} tau={FlyBrain.tau}",
                     "data_known": "—", "literature": "LIF 为一阶近似", "unknown": "所有细胞共用一个 tau"},
        "threshold_reset": {"impl": "v>=1.0 -> spike, reset 0, refractory=0", "data_known": "—",
                            "literature": "all-or-none 发放", "unknown": "各类型阈值/复位差异"},
        "synaptic_current": {"impl": "**无**：W@spikes 瞬时加到 v（无突触电流状态）",
                             "data_known": "—", "literature": "存在 EPSC/渐变释放（ms~持续）",
                             "unknown": "tau_syn 未建模；dt=20ms 已粗于常见 EPSC"},
        "cell_type_dynamics": {"impl": f"**无**：gain={FlyBrain.gain} 对所有 166,700 神经元相同",
                               "data_known": "细胞类型/侧别/递质符号已知", "literature": "果蝇视觉/CX 神经元**大量为非脉冲(渐变)**",
                               "unknown": "各类型脉冲/渐变比例与 IO 增益"},
        "receptor_effects": {"impl": "仅**符号化**权重（递质->±）", "data_known": "递质身份（符号）",
                             "literature": "nAChR/GABA/NMDA 动力学历时", "unknown": "受体动力学未建模"},
        "network_state": {"impl": f"tonic={FlyBrain.tonic} noise={FlyBrain.noise_hz}Hz/{FlyBrain.noise_amp}（手工标定）",
                          "data_known": "—", "literature": "—", "unknown": "标定正确性"},
        "weight_norm": {"impl": "逐突触后 L1<=1（build.py:137）", "data_known": "连接组相对权重",
                        "literature": "相对突触效能", "unknown": "绝对电导尺度"},
    }


def bottleneck():
    """用 P5/P6/P7 归档数据定位瓶颈（感觉层有反应、下行不足）。"""
    def load(f):
        p = os.path.join(HERE, f)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    p5 = load("p51_result.json"); p6 = load("p61_result.json"); p7 = load("p70a_audit.json")
    return {
        "P5_visual": {"verdict": (p5 or {}).get("verdict"),
                      "evidence": "光感受器/T4T5 有反应(扩张+/收缩-)，但 LC4/LPLC2 无左右特异"},
        "P6_sensory_inject": {"verdict": (p6 or {}).get("verdict"),
                              "evidence": "直接注入 LPLC2/LC4 使感觉层响应，但 DN 靶对照差 0.013-0.018 << 3σ≈0.15"},
        "P7_static": {"verdict": (p7 or {}).get("verdict_scope"),
                      "evidence": "DNa02 输入以 cb_intrinsic 主导(abs 0.80)，视觉输入 abs 0.025-0.030（净≈0）"},
        "localization": "瓶颈在**突触后整合/发放**（感觉层→VPN→DN 的**逐级 all-or-none 阈值**衰减）："
                        "感觉层可被驱动，但模拟的下行输入（~0.1×gain）远低于阈值 1.0，**弱而持续的**感觉信号无法累积到 DN。",
    }


def proposal():
    """唯一首选提案（**P8-0a 修订：如实披露缺陷，判 NO-GO**）。"""
    return {
        "name": "细胞类型特异**渐变（非脉冲）释放**（graded release）——**已因证据/标定不足关闭**",
        "cell_set": "**固定为 LPLC2 / LC4**（对应五路 ToF 方案，见 `CELL_EVIDENCE`）；**不**'可含 CX'、**不**事后扩大。",
        "cells_caveat": "**LPLC2/LC4 的渐变释放缺直接证据**（对其仅有钙成像/解剖，非释放动力学）→ 本提案属"
                        "**待检验的人工假设**，**不冒充**已知果蝇机制。",
        "change": "对 LPLC2/LC4，把 all-or-none 输出替换为渐变释放率 r(v)，下游 W@spikes -> W@r。",
        "why": "IF LPLC2/LC4 为渐变释放，则弱而持续的感觉信号可无阈值地传到下行靶——对应 P6 瓶颈。",
        "params_disclosure": "**r=clip(v,0,1) 是未经生理定标的**人工传递函数**，不是'无新假设/无自由尺度'**。"
                             "且解析（见 resting_analysis）显示**无输入静息 v_rest≈0.77** → r 的**基线释放≈0.77**，"
                             "远大于刺激诱发 Δv(~0.1) → r 被静息主导；要可用须减静息+标定斜率 = **新的未标定自由参数**。",
        "release_units": "[已关闭方案的历史记录] 释放必须取自**复位前**的 v，且**脉冲(0/1 事件)**与**渐变(速率/电导)**量纲不同，混用会改变 W@out 的物理含义。",
        "implementability": "新模块对指定 cell_type 覆盖输出（不改 flybrain/官方权重）。",
        "not_allowed": "不增大刺激、不调电机增益、不搜索最佳神经群、不注入航向/目标角（尊重 EPG/PEN NO-GO）。",
        "falsification": [
            "LC4/LPLC2 -> DN 左右特异 < 预注册阈值（<4/5 seed 或 <3σ）→ 证伪。",
            "渐变输出未使 DN 靶'刺激-对照'增量 > 3σ → 证伪。",
            "撤除后 DN 不回落或跨 seed 符号混乱 → 证伪。",
        ],
        "turn_stopping_hypothesis": "若成立，转向输出连续跟随驱动（非计时器）→ 停止转弯的**必要**机制；充分性待验。",
        "PFL3_PFL2": "PFL3/PFL2 转向依赖 EPG/PEN 罗盘（已 NO-GO）；本提案**不**注入航向/目标角。",
        "verdict": "NO-GO（见 recommendation）",
    }


def prereg_p8_1():
    """未来**唯一**离线验证实验的预注册协议（不运行）。"""
    return {
        "device": "运行前**显式打印并断言** brain.device（含 CUDA 时的 _W 路径核对）；记录 python/numpy/scipy/flybrain 版本与 weights sha256。",
        "conditions": ["无刺激(none)", "左视觉刺激", "右视觉刺激", "左右镜像对照", "（可选）收缩/亮度对照"],
        "layers": ["光感受器", "T4T5", "LPLC2_L/R", "LC4_L/R", "解剖固定的 DN 靶"],
        "metric": "同 seed **刺激 − 无刺激对照**（同步窗，逐帧）；per-cell + 群体；K_SIG=3× 对照每细胞 std（预注册）。",
        "recovery": "撤除后是否回落至对照水平（仅作**必要条件**，不据此宣称会结束转弯）。",
        "stability": "≥5 seed，报逐 seed；不因导航成功率反向调模型。",
        "gate": "GO 门槛在运行前固定：左右特异 ≥4/5 seed 且 >3σ；否则 NO-GO。",
        "forbidden": "不接车/不上电；不改 P1/官方权重/正式 readout/BASE/tags；不训练导航器。",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    out = {"simulator_defects": simulator_defects(), "bottleneck": bottleneck(),
           "references": REFS, "cell_evidence": CELL_EVIDENCE, "resting_analysis": resting_analysis(),
           "proposal": proposal(), "prereg_p8_1": prereg_p8_1(),
           "recommendation": "NO-GO：①**LPLC2/LC4 的渐变释放缺直接释放动力学证据**（仅钙成像/解剖，不得替代）→ 属待检验人工假设；"
                             "②`r=clip(v,0,1)` **未经生理定标**，且解析显示**无输入静息 v≈0.77 → 基线释放≈0.77 ≫ 刺激 Δv(~0.1)**，"
                             "r 被静息主导；要可用需**新的未标定自由参数**（减静息+斜率）→ 违反'不新增自由参数'。"
                             "→ **关闭该机制**（不得为得到转向信号硬调释放强度）。",
           "workload": "本轮为设计修订，无新增运行；若将来有新机制提案另议。"}
    json.dump(out, open(os.path.join(HERE, "p80_mechanism.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[P8-0] defects rows:", len(out["simulator_defects"]))
    print("[P8-0] bottleneck:", out["bottleneck"]["localization"])
    print("[P8-0] proposal:", out["proposal"]["name"])
    print("[P8-0] recommendation:", out["recommendation"][:60], "...")
    print("wrote p80_mechanism.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
