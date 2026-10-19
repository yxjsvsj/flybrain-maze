# P10-0d MaleCNS 真实突触定位、DAN 区室对应与文献勘误（**只读**）

**基线** `4613e8c`。只读审计；**不**训练、**不**更新 ΔW、**不**运行 MaleCNS 动力学、**不**控车；
官方 W/P1/BASE/readout/tags **冻结**；**未**改五路 ToF 靶、**未**新建连接、**未**找第二机制。

**数据/API/版本**：本地 `flybrain` MaleCNS + 官方 neuPrint `male-cns:v1.0`；
`fetch_synapse_connections(source, target, min_total_weight=1)`（**严格限定 source+target**）。
权重 `W[post,pre]`（有符号、逐突触后 L1 归一化相对、无单位）；**neuPrint 突触计数 ≠ flybrain 归一化权重**。

## 1. 左右侧 ROI 修复（`_base`/`mb_comps` 现支持 `keep_side`）
- `keep_side=True` **保留完整 ROI**（`g1(L)` vs `g1(R)`，不合并）；旧默认 `False` 保留以兼容 P10-0c。
- **修正前后**：4,533 条边中，合并左右时 **4,533** 条有重叠；保留完整 ROI 后 **4,486** 条；
  **47 条为跨侧候选**（合并后有、完整 ROI 后无）——**未删除**，但需同源突触数据证明。
- 该结果仍仅属**神经元级筛选**。

## 2. 伙伴特异性突触查询（10 对：9 阳性 + 1 阴性对照）
固定可复算选取：simple-path 对 + 不同 lobe/side 的首个候选 + 一个 **W=0 阴性对照**。

| 类型 | KC→MBON | 突触数 | 主要 ROI（lobe 级） |
|---|---|---|---|
| simple_path | 137302→519128 | **25** | gL(L) 24 + unspecified 1 |
| positive | 19788→10495 | 24 | gL(L) |
| positive | 18540→10013 | 113 | gL(R) 55, CRE(R) 49, unspecified 8, b'L(R) 1 |
| positive | 44069→10704 | 31 | gL(L) 30 + unspecified 1 |
| positive | 52267→11805 | 4 | b'L(R) 2, aL(R) 1, gL(R) 1 |
| positive | 21778→10704 | 57 | gL(L) 55 + unspecified 2 |
| positive | 18540→10704 | 1 | bL(R) |
| positive | 21778→28622 | 5 | a'L(L) 1 + unspecified 4 |
| positive | 21778→10495 | 78 | gL(L) 72 + unspecified 3 + CRE(L) 3 |
| **negative_control** | 18540→10079 | **0** | — |
- **查询完整性**：`fetch_synapse_connections` 返回逐突触记录（坐标/置信度/`compartment_pre=axon`、`compartment_post=dendrite`）；
  最大 113 条，远低于 batch_size，未见截断。阴性对照为 0，验证了查询的方向与限定。
- **与本地 W 的对应**：9 对阳性在本地均存在非零 `W[MBON,KC]`；阴性对照 W=0 且查询为 0。**未发现方向不一致**。

## 3. 突触定位与区室：**lobe 级，非 fine 区室**
逐突触 `roi_pre` 为 **`gL/aL/bL/a'L/b'L`（叶级）**，**不是** `g1–g5` 等精细区室。因此：
- **Synapse Localization：部分**（9 对阳性已定位到叶级；非全部 4,533 边）。
- **Compartment Assignment：非特异/部分**（叶级，非精细区室）。

## 4. DAN 释放共定位：**B 有证据但非特异**
对每个阳性突触的叶+侧，从同源注释找释放位点（pre）同叶+侧的 DAN：
- 匹配 DAN 数 **11–317**（如 gL 187、a'L 11、bL 104）；**全部 `B_specific=False`**（叶级覆盖过宽，**无区分力**）。
- **C（更具体局部空间关系）**：未查询（需 DAN 释放点坐标 + 逐突触最近邻；本轮仅叶级）。
- **D（功能调制）**：**MISSING**（无受体/生理/钙成像/光遗传证据）。
- **A（突触位置）已确认 ≠ B/D 成立**；**A+B 也不得推断 D**。

## 5. 证据分级
| 维度 | 裁决 |
|---|---|
| **Synapse Localization** | **部分**（9 对阳性；叶级） |
| **Compartment Assignment** | **非特异/部分**（叶级 gL/aL/bL/a'L/b'L） |
| **DAN Release Co-localization** | **有证据但非特异**（B：同叶+侧，11–317 DAN；C 未做） |
| **Functional DA Modulation** | **未验证** |
| **g_unit Calibration** | **未解决**（P9-1b1） |
| **Variable-goal Identifiability** | **仍受限**（P10-0a） |
| **Learning Readiness** | **NO-GO** |

## 6. 停止条件（已触发）
有限查询**仍无法**建立**足够具体**的突触—DAN 对应（叶级匹配 11–317 个 DAN，无区分力；精细区室与 D 级证据缺失）。
→ **记录证据缺口并停止该审计分支**；**不扩大到全脑搜索**，**不启动 P10-1**。

## 7. 局限/不可推断
- 叶级 ROI ≠ 精细区室；突触计数 ≠ 归一化权重/电导；共定位 ≠ 功能调制；
- 未做受体/扩散范围/生理实验；未用导航表现筛选；未改官方资产。
