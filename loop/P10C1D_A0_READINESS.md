# P10-C1d A0 功能预检 + 合成可塑性内核（**无真实 ΔW/无训练**）

**基线** `e0cca1c`。**未对真实 MaleCNS W 执行任何 ΔW 更新**；无训练、无迷宫、无控车。
真实截断子网短时**无学习** LIF 前向 + 合成稀疏矩阵内核。官方 W/P1/BASE/readout/tags 冻结。

## 1. A0：固定 LPLC2/LC4 刺激能否产生学习资格迹 —— **FAIL**
**协议（固定、未调参）**：种子 11–15；200 步；刺激量 0.6；仅 LPLC2/LC4 入口；刺激窗 50–150；恢复条件 0–100 后撤除；比较窗 50–100；`τ_e=0.5`；ΔW=0。
| 指标 | 结果 |
|---|---|
| 配对 KC 发放差（stim−none） | **−1.0×10⁻⁴**（相对 −2.1×10⁻⁴） |
| 配对 MBON 发放差 | ≈ +3.4×10⁻⁴（符号不一致） |
| 资格迹非零比例差 | **+3.5×10⁻⁴** |
| 资格迹非零比例（无刺激对照） | 种子 11: 0.34；**种子 12–15: 1.00（自发饱和）** |
| 刺激效应可重复 | **False** |
| 资格迹可辨认 | **False** |
**结论：A0 FAIL**。固定刺激**未**产生可重复 KC 活动差异，资格迹被自发发放饱和；**不得**增强刺激、注入 KC 或换机制挽救。功能扩展**停止**。

## 2. 合成 ΔW 内核（`p10c1d_delta_sandbox.py`，仅合成）
- 区分 **`requested_delta`** 与 **`effective_delta = W_eff − W_official`**。
- **S1/S2 全部不变量通过**：仅批准边、无新边、兴奋性 `W_eff≥0`、抑制性 `W_eff≤0`、未批准边逐位不变、因果、ΔW 独立。
- **持久化**：保存/加载后 `W_official + loaded_effective_delta == W_eff`（max abs err **0.0**）。
- **三模式**：`LEARN`（仅合成）/`FROZEN_EVAL`（保留学成 ΔW，多次运行权重不变 ✅）/`ABLATION_ZERO`（ΔW=0 ✅）。
- **零权语义**：区分「解剖边存在」「生效权重=0」「解剖边删除」；本内核**从不删除**解剖边；功能沉默**不**等于连接消失。

## 3. S1/S2 真实结构预算（只读官方 W）
| 指标（95 MBON） | 值 |
|---|---|
| 原始逐后 L1 总量 | mean **1.0000**（min=max=1） |
| 可塑边绝对权重和 | mean **0.0415**（median 0.0277，max 0.3959） |
| 非可塑边绝对权重和 | mean **0.9585**（min 0.6041） |
| 可塑饱和比 | mean 0.0415，max 0.3959 |
- **Budget A（预算=原始可塑和）**：**95/95 MBON 无正向净增长余量** → 只能**减弱或重新分配**。
- **Budget B（预算=原始逐后 L1=1）**：**95/95 有非可塑余量**（mean 0.9585）→ 正向增长可行且保持 L1≤1。
**回答**：S1 是否严重限制取决于**预算定义**；Budget A 是严重限制，Budget B 不是。两者均 **PENDING_APPROVAL**，不按导航成功率选择。

## 4. 调制信号的科学身份
- `D(t)` = **教师/奖励标签** → **外部计算调制假设**；**不是**已验证的真实 DAN 发放或 DA 浓度。
- **实际 DAN 神经活动**：本轮**未测量**（`measured=False`）。
- **生物对应证据**：**NONE**。
- 教师标签与感觉输入**分离**；测试期**无**教师；Dijkstra **未实现、未参与**神经更新。

## 5. 资源记录一致性
- 历史 measurement ID：`LIF-C1B-<hash>`（来源 `p10c1b_report.json`，**冻结**）。
- 新 A0 measurement ID：`A0-<timestamp>`（仅追加，不覆盖历史）。
- 一致性校验：`used_per_step = max(class1, class2)`；`est_hours_1890` 公式匹配 = **True**。
- **class3（资格迹/ΔW/教师）仍为 UNKNOWN**；**不**宣称完整学习训练资源已实测。

## 6. 独立门控
| 门控 | 结果 |
|---|---|
| **A0 Sensory Transmission** | **NO-GO**（FAIL） |
| **Synthetic Plasticity Kernel** | **GO**（仅合成；≠真实突触更新已实现） |
| **S1 Mathematical Feasibility** | **GO**（Budget A 有严重容量限制） |
| **S2 Mathematical Feasibility** | **GO**（L1 偏离显式声明） |
| **Level A Design** | **NO-GO**（A0 FAIL） |
| **Level A Execution** | **NO-GO** |
| **Level B** | **NO-GO** |
| **Level C** | **NO-GO** |
| **Biological** | **NO-GO** |
| **Training Authorization** | **False** |

## 7. 结论
- **A0 FAIL**：固定感觉刺激在本截断子网**不能**形成可辨认的感觉驱动资格迹；功能扩展**到此停止**。
- 合成内核与 S1/S2 数学可行，但**不等于**真实突触学习已实现。
- **Level A Design/Execution 均 NO-GO**；B/C/Biological NO-GO；**不授权训练**。
- 若未来重启，必须是**新的预注册 A0 协议**（不得用增强刺激/注入 KC/换机制做**事后**挽救）。
