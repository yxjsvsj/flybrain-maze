# P10-0c 真实 MaleCNS 多巴胺区室调制证据审计（**只读**）

**基线** `d444262`。只读解剖/元数据审计；**不**训练、**不**更新 ΔW、**不**运行 MaleCNS 动力学、**不**控车；
官方 W/P1/BASE/readout/tags **冻结**；**未**改 ToF 靶、**未**建连接、**未**找第二机制。

**数据来源/版本**：本地 `flybrain` MaleCNS（bodyId/类型/侧别/`W[post,pre]`）+ **官方 neuPrint `male-cns:v1.0`**
（targeted `fetch_neurons` 逐神经元 ROI 计数；`fetch_synapses` 可返回逐突触 `roi`/`compartment`）。
权重为**有符号、逐突触后 L1 归一化相对权重**（无单位；符号来自递质预测）。仅用**版本核对 bodyId**，未跨数据集强制匹配。

## 1. 区室信息：**已找到**
官方 ROIs 含蘑菇体区室：`a1,a2,a3,b1,b2,g1,g2,g3,g4,g5` 及 `a'1,a'2,a'3,b'1,b'2`（均有 L/R）；
逐神经元 `roi` 计数含 `pre/post`；逐突触查询可返回 `roi` 与 `compartment`。

## 2. 学习链对接（320 KC / 95 MBON / 4,533 条 KC→MBON）
| 指标 | 值 |
|---|---|
| KC→MBON 真实边 | **4,533** |
| 有**神经元级**区室重叠的边 | **4,533（全部）** |
| 有匹配 DAN 的边 | **4,533（全部）** |
| 有 MB 区室的 KC / MBON / DAN | **320 / 95 / 334**（340 DAN 中 6 个无 MB 区室） |
| 共享区室数分布 | 1–12（**266 种不同共享集合**） |
**关键限定**：由于 KC/MBON 的区室集合**宽泛且多重**，重叠对**全部 4,533 条边都成立** → **不具区分力**，
**不能**据此定位任何特定 KC→MBON 突触，也**不能**证明其受 DA 调制。

## 3. 具体突触调制证据：**MISSING**
- 同源 MaleCNS 的**逐突触/区室**注释存在（`fetch_synapses` 的 `roi`/`compartment`），但**本轮未**对 4,533 条边逐条重建
  伙伴特异性突触（避免大规模下载）；**神经元级 ROI 重叠 ≠ 逐突触证据**。
- **受体类型/位置**：**MISSING**。
- **生理/钙成像/光遗传**对特定 MaleCNS KC→MBON 突触的调制：**MISSING**。
- **DAN→KC 普通突触或共享 ROI ≠ 已证明该 KC→MBON 突触受 DA 功能调制。**

## 4. 独立文献核验（跨数据集，注明不确定性）
| 文献 | 层次 | 同源 MaleCNS bodyId |
|---|---|---|
| Aso et al. 2014 eLife 3:e04577 | 一般 + 区室化解剖 | 否 |
| Hige et al. 2015 Neuron 85:1341 | **特定区室 α3** 的 DA 依赖 KC→MBON 可塑性 | 否 |
| Aso & Rubin 2016 Curr Biol 26:R225 | 环路/区室 | 否 |
| Perisse 2013 / Cohn 2015 / Berry 2018 | 区室化 DAN 调制 | 否 |
**分级**：一般 KC→MBON 可塑性=文献支持；特定区室 DAN 调制=文献支持（其它数据集）；同源 MaleCNS bodyId 解剖=**部分**（neuron-level）；
**某条 MaleCNS KC→MBON 突触受调制的直接证据=MISSING**。**不因一般性 MB 学习文献自动宣称当前五路 ToF 导航机制有效。**

## 5. 补充：一条完整简单路径（无重复 bodyId）
`11498(LPLC2,L) → 84698(PVLP148,L) → 137302(KCg-d,L) → 519128(MBON27,L) → 69730(CB2245,L) → 555726(DNa13,L)`
逐段权重（`W[post,pre]`）：`+0.000814 → +0.002058 → +0.004634(KC→MBON) → +0.021912 → −0.000254(Y→DN)`。
**全部非零、6 个不同 bodyId**；负号仅表示预测符号，不代表已标定生理效应。

## 6. 证据分级
| 维度 | 裁决 |
|---|---|
| **Structural Path** | **存在**（P10-0b，数据支持） |
| **Compartment Match** | **部分/非特异**（neuron-level ROI 重叠；对全部边都成立，无区分力） |
| **DAN Modulation** | **不足**（文献一般性支持；无同源 MaleCNS 逐突触证据） |
| **Synapse Specificity** | **MISSING**（未逐边重建；无受体/生理证据） |
| **Biological Calibration** | **`g_unit` 未解决**（P9-1b1） |
| **Learning Readiness** | **否（NO-GO 维持）** |

## 7. 剩余阻碍
1. **特定 KC→MBON 突触的 DA 调制证据**（逐突触/受体/生理）。
2. **`g_unit` 独立定标**。
3. **可变目标信息不可辨识**（P10-0a）。

## 8. 不可推断事项
- 不以 neuron-level ROI 重叠代替逐突触调制；不以 `DAN→KC` 权重推断调制；不以一般 MB 学习文献推断本导航机制有效；
- 不用 walk 代替 simple path；路径权重为**结构代理**，非已标定生理增益；未做动力学/功能验证。

**已提交后立即停止；不自动进入 P10-1，不运行训练。**
