# P10-C1e A0 方法学复核 + S1 Budget B 勘误（**只读/合成**）

**基线** `61cdb30`。**未重跑 A0、未更新真实 ΔW、未训练、未进入 B/C。**
仅使用已保存数据、源码静态检查、合成数学测试。历史 `p10c1d_*` 原始数据**保留**，本文件新增勘误。

## 1. A0 统计检验复核：**门槛不可通过**
原始配对差（stim − none，5 seeds）：
- KC：`[-6.3e-05, +5.63e-04, -3.13e-04, -6.87e-04, 0.0]` → **4 非零 + 1 零**。
- 双侧精确符号翻转：`p(含零) = p(排除零) = 0.625`；**当前非零数 4 的最小可达 p = 2/16 = 0.125**。
- 即便 **5 个全非零**，最小 p = **2/32 = 0.0625**，仍 **> 0.05**。
**结论：A0 Statistical Gate = INVALID**。该设计**不可能**通过 `p<0.05`；**不得**改单侧检验、改 α、加 seed 或改阈值事后挽救。
**合成检验**：全同号 5 → 0.0625；1 零+4 同号 → 0.125；全零 → 1.0；混合符号 → ≥0.0625；非有限/缺失 → 拒绝。
**保留原始 A0 FAIL**，但明确：**该统计门槛不能单独证明感觉通路机制失败**。

## 2. 资格迹与感觉传递指标：**INVALID/LIMITED**
| 核查项 | 结果 |
|---|---|
| 资格迹用独立 pre/post traces | **否**：`e = e·decay + kc[t]·mb[t]`（**同一时刻乘积**），与预注册不符 |
| `e_nonzero_frac` 时间分辨 | **否**：仅在最后一步统计 |
| 1.0 是否只是二值化上限 | **是**：`(e>1e-6).mean()` 是阈值比例，非幅度 |
| 是否有幅度饱和证据 | **MISSING**：无逐时间步/逐边幅度记录 |
| LPLC2/LC4 入口阳性对照 | **MISSING**（只记录 KC/MBON） |
| 刺激量 0.6 的单位/语义 | **未验证**与 Encoder 稳态电压定义一致 |
| KC/MBON 自发高活动 | 是（KC ~0.94/步；对照资格迹非零=1.0 于 4/5 seeds）→ **天花板/淹没** |
| `recover` 与 `stim` 窗口可直接比较 | **否**（recover 基线期已被刺激） |
**四类解释必须分开**：
- 刺激**确实无功能传递**：**NOT ESTABLISHED**。
- 刺激响应**被背景淹没**：**LIKELY**。
- 指标**缺乏区分力**：**YES**（同刻乘积 + 最终二值比例）。
- 当前数据**无法判别**：**YES**。

## 3. S1 Budget B 勘误：**headroom = 0**
`L1_total = L1_plastic + L1_nonplastic`；`headroom = max(0, 1 − L1_nonplastic − L1_plastic)`。
- 原始 **95/95 MBON 的 L1_total = 1.0**。
- `L1_plastic` mean **0.0415**；`L1_nonplastic` mean **0.9585**。
- **Budget A**（预算=原始可塑和）headroom = **0**（95/95）。
- **Budget B**（预算=原始逐后 L1=1）headroom = **0**（95/95，浮点残差 ~1e-7）。
**勘误**：P10-C1d 把 `L1_nonplastic≈0.9585` 当作**未使用**学习容量是**错误**——它已被非可塑连接占用，且计入 L1_total。
**正确结论**：S1（A/B）**净正向余量为 0**；保持 L1≤1 只能通过可塑突触间增强/减弱**抵消**实现**重新分配**。正向净增长需 **S2**（放宽 L1，显式模型偏离）或减少非可塑权重（不允许）。
**合成负例**：非可塑 0.9585 + 可塑 0.0415 + 新增 +0.01 → 总 L1 **1.01 > 1**，不得声称仍 ≤1。

## 4. 合成内核复核
- `requested_delta` vs `effective_delta = W_eff − W_official`：一致 ✅。
- `FROZEN_EVAL` 保留学成 ΔW、多次运行稳定 ✅；但**不校验**官方 W 版本/允许边掩码/参数/L1。
- 历史 `save_state()` **无元数据**（仅 delta）→ 加载来源不可验证（LIMITED）。
- 训练关闭**保留**已学权重（`FROZEN_EVAL`）；`ABLATION_ZERO` 才归零 ✅。
- 生效权重为 0 时**保留解剖边身份**（deleted=0）✅。
- **候选**带元数据加载器（审计用）：校验 W sha256 / 掩码 sha256 / params / L1；错误输入全部拒绝 ✅。

## 5. 未来 A0 预注册建议（**PENDING_APPROVAL；未运行**）
- **统计**：≥ **6 个非零配对**（最小 p=2/64=0.03125）或预注册功效分析；禁止事后改检验/α/seed。
- **资格迹**：独立 pre/post traces（指数滤波）、逐时间步幅度+非零比例（非最终二值）。
- **入口阳性对照**：记录 LPLC2/LC4 发放并要求显著升高。
- **刺激标定**：与 Encoder 稳态电压语义对齐；预注册幅度。
- **基线匹配**：各条件同时长无刺激基线；recover 基线匹配。
- **推断分离**：预注册区分「无传递/被淹没/指标无效/不确定」。

## 6. 裁决
- A0 Historical Protocol Result：**FAIL（保留）**
- A0 Statistical Gate Validity：**INVALID**
- A0 Sensory Transmission Evidence：**INCONCLUSIVE**
- Eligibility Measurement Validity：**INVALID/LIMITED**
- S1 Budget A：可行，**净正向余量 0**（仅重新分配）
- S1 Budget B：**勘误 headroom=0**（原 0.9585 错误）
- S2：可行，允许正向净增长但**显式 L1 偏离**
- Synthetic Kernel：仅合成；不变量通过；持久化元数据 LIMITED
- Level A Execution：**NO-GO**；B/C：**NO-GO**；Biological：**NO-GO**
- Training Authorization：**False**

**不因原始统计设计有误而自动升级 GO。**
