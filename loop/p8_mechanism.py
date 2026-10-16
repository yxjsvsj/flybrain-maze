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

# ---- 文献依据（按作者/年/期刊；仅引用确凿的存在性） ----
REFS = [
    "Seelig & Jayaraman 2015, Nature Neuroscience — EPG/PEN 头朝向环、非脉冲式动力学",
    "Turner-Evans et al. 2017, eLife — 中央复合体角速度积分（P-EN）动力学",
    "Green et al. 2017 / 2019 — 中央复合体 EPG/PFN 群体",
    "Rayshubskiy et al. 2020, Cell — PFL3 转向输出（接收 EPG 罗盘 + PFN 航向）",
    "Hulse et al. 2021, Nature — PFL2/PFL3 左右转向与步态",
    "Borst, Haag & colleagues (Lobula Plate) — 视觉中间神经元以**渐变（非脉冲）**信号传递",
    "Juusola & French (phototransduction) — 光感受器**渐变电位**编码",
    "Nagel et al. 2019 / Fisher et al. 2022 — T4/T5 为**渐变**运动检测器",
    "fly64 / flybrain 0.1.0 — 参考 LIF 标定（tonic/gain 手工标定）",
]


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
    """唯一首选提案。"""
    return {
        "name": "细胞类型特异**渐变（非脉冲）释放**（graded release）",
        "change": "对指定细胞类型（视觉投射 LPLC2/LC4 + 上游视叶/光感受器，可含 CX），把 all-or-none 输出"
                  "替换为**渐变释放率** r = clip(v, 0, 1)（模拟膜电位的模拟量），下游输入由 W@spikes 改为 W@r。",
        "why": "果蝇视觉中间神经元与 CX 神经元**以渐变（非脉冲）信号传递**（文献）；all-or-none 阈值在**每一级**衰减模拟信号，"
               "而渐变释放让**弱而持续**的感觉信号无阈值地传到下行靶——正好对应 P6 的'感觉层有、下行无'瓶颈。",
        "params": "**不引入新自由参数**：沿用现有 tau/gain/tonic/noise/阈值 与**未改的连接权重**；唯一结构改动是"
                  "'all-or-none -> 渐变 rate'。若需要，r 的斜率可直接沿用现有噪声尺度（noise_amp），非搜索得到。",
        "implementability": "在**新模块**中仅对指定 cell_type 覆盖 step 的输出（不改 flybrain 文件、不改官方权重）。",
        "not_allowed": "不增大刺激、不调电机增益、不搜索最佳神经群、不注入航向/目标角（尊重 EPG/PEN NO-GO）。",
        "falsification": [
            "若在渐变释放下，LC4/LPLC2 -> DN 的**左右特异**仍 < 预注册阈值（<4/5 seed 或 <3σ），则**证伪**。",
            "若渐变输出**未能**使 DN 靶对'刺激-对照'的增量 > 3σ（即仍无下行响应），则**证伪**。",
            "若撤除刺激后 DN 不回落（持续同向）或跨 seed 符号混乱，则**证伪**。",
        ],
        "turn_stopping_hypothesis": "渐变释放下转向输出**连续跟随感觉驱动**（非计时器/固定转角）：驱动消失则输出随之衰减"
                                    "-> 具备'神经自主停止转弯'的**必要**机制（是否充分需 P8-1 验证）。",
        "PFL3_PFL2": "文献中 PFL3/PFL2 为转向输出（接 EPG 罗盘+PEN 航向）。**尊重 EPG/PEN NO-GO**："
                     "本提案**不**注入航向、不注入目标角，仅改**传递动力学**，不补足被关闭的罗盘。",
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
           "references": REFS, "proposal": proposal(), "prereg_p8_1": prereg_p8_1(),
           "recommendation": "GO (conditional): 机制有文献支持、可实施、**不引入新自由参数**（复用现有标定）；"
                             "唯一结构改动 all-or-none->渐变释放。条件：P8-1 仅用现有标定参数、预注册证伪、不得按导航成功率反调。",
           "workload": "P8-1 预估：新模块 ~150 行 + 单次离线实验（5 seed × 5 条件 × 100 步，CPU 数分钟）+ 测试/报告。"}
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
