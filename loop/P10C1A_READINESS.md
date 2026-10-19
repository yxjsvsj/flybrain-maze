# P10-C1a 执行前门控：真实连接组计算学习实验（**工程/静态/隔离，不训练**）

**基线** `95a27c2`。**不训练、不更新 ΔW、不跑真实神经动力学、不跑虚拟/实体小车。**
官方 W/P1/BASE/readout/tags 冻结；未改五路 ToF 靶、未建连接、未找第二机制。
**`training_authorized` 恒为 False。**

## 1. 静态门控修正（`p100e_gate.py`）
- 正确读取 `p100.json` 的 **`decision.verdict`**（= `"NO-GO"`）；缺失/空裁决 → 失败。
- **全量 SHA-256**：`brain.npz`=`cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca`、
  `weights.npz`=`c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c`（**均匹配**）。
- 训练产物：**预期清单**（5 个历史 JSON）+ **明确禁止项**（`p10c1_delta/training/learned/delta_w/...`）。
- **分开**：`worth_doing_computational=True`（值得做） vs `execution_ready=False`（未满足启动） vs `training_authorized=False`。
- 负例测试：缺报告/空裁决/训练产物/坏 hash → 必失败。

## 2. DN 解码器审计（`loop/decode.py`，P1 冻结，**未修改**）
| 检查项 | 结果 |
|---|---|
| `memory_gain` | **存在**（外部 pursuit 偏置） |
| `pursuit_turn` | **存在**（外部规划方向） |
| `scripted` | **存在**（手写沿墙） |
| `readout` | **存在**（训练线性读出头） |
| `bias` | **存在**（escape/stall 方向） |
| `escape` | **存在**（bias 方向 + 后退） |
| `stall_turn_mode` / `stall_dir` | **存在**（人工旋转脱困） |
| `turn_sign` | 存在（DN 组映射符号） |
**结论：`decoder_ready=False`。** 测试期方向必须只来自**冻结、有神经依据的 DN 输出映射**；上述外部/人工方向来源必须由**独立 P10 测试解码接口**禁用（**不改** `loop/decode.py`）。安全层仅限速/急停，**不得**决定左右/旋转脱困。所需：独立接口、固定增益的神经依据与冻结审计、证明无外部方向、安全层只限速/急停。

## 3. 计算参数与 ΔW 规则（`p10c1a_freeze_spec.json`）
`ΔW = η × eligibility × (D − D0)`

| 参数 | 计算假设 | 预注册网格 |
|---|---|---|
| η | 无定标 | `[1e-5, 1e-4, 1e-3]` |
| τ_e | 文献量级 | `[0.1, 0.5, 1.0]` s |
| Δmax | 相对权重上界 | `[0.01, 0.05, 0.1]` |
| D0 | 基线 DA | `0.0` |
- **eligibility**：pre-trace × post-trace，**因果**（不含未来/教师动作输入）。
- **合成**：`W_eff = W_official + ΔW`（**覆盖层**；官方 W 只读）。
- **边界**：逐突触 `clip(ΔW, ±Δmax)`。
- **符号/拓扑**：**仅**真实 KC→MBON 边；递质符号；**不新建**连接；**不改**感觉靶。
- **L1 归一化**：**不得**重归一化无关连接；默认**不**重归一化；若做须预注册、隔离，且不得改变未批准突触。
- **敏感性**：27 点**有限预固定网格**；**全部报告**；**禁止**按导航成绩事后挑选；**不**解释为真实电导标定。
- **本轮仅冻结规格，未实现更新算法。**

## 4. 独立学生进程隔离（mock，不跑 MaleCNS）
- **允许**：`tof`、`tof_delta`、`neural_state`、`motor_feedback`。
- **禁止**：地图/占用栅格、真值位姿、Dijkstra、教师动作/标签、目标坐标/方向、外部导航记忆、路径规划器、人工转向命令。
- **负例已捕获**：9 类禁止输入（map/pose/dijkstra/teacher_action/teacher_label/goal/nav_memory/planner/turn_command）；
  3 类危险模块（dijkstra/planner/networkx）；3 类危险文件（map/teacher/planner）。
- **说明**：仅 mock；真实学生进程仍需**接口白名单 + 依赖检查 + 文件访问限制 + 运行时断言 + 日志**，**不**以单一 `sys.modules` 检查宣称无泄漏。

## 5. 子网络与资源（静态，无动力学）
- 子网：**N=23,218**；**nnz=3,549,643**；CSR(float32) **≈28.49 MB**（data+indices+indptr 下界）。
- 层规模：`[311, 10931, 320, 95, 11340, 1285]`；DAN 340；子网内真实 KC→MBON 可塑边 **4,533**（bodyId 可追踪）。
- 试验：**10 trial × 7 条件 = 70**（预注册上限，**未执行**）。
- **LIF 子网逐步耗时：MISSING**（本轮不跑动力学；**P9 COBA 数据仅参考，不能替代 LIF 实测**）→ `resource_ready=False`。
- **不**将截断子网宣称为完整全脑功能。

## 6. 九项执行准入条件
| # | 条件 | 结果 |
|---|---|---|
| 1 | 官方 W/冻结资产完整（全量 SHA-256） | ✅ |
| 2 | 历史裁决读取正确（`decision.verdict=NO-GO`） | ✅ |
| 3 | DN 解码器满足无人工转向并冻结 | ❌ `decoder_ready=False` |
| 4 | 计算参数与灵敏度网格预注册 | ✅ |
| 5 | ΔW 拓扑/符号/幅度/归一化规则明确 | ✅ |
| 6 | A/B/C 独立失败条件明确 | ✅ |
| 7 | 七组对照与统计预注册 | ✅ |
| 8 | 独立学生隔离测试通过 | ✅ |
| 9 | 子网络资源预算有依据 | ❌ 缺 LIF 实测 |
**`execution_ready=False`；阻碍：`3_decoder_ready`、`9_resource_budget`。**

## 7. 三项独立裁决（+ 训练授权）
| 裁决 | 结果 |
|---|---|
| **Biological** | **NO-GO**（缺同源逐突触调制/受体/生理/`g_unit`） |
| **Computational Hypothesis** | **GO（有条件，仅提案）**——值得另行申请 |
| **Execution Readiness** | **NO-GO**（解码器未冻结合格；缺 LIF 资源实测） |
| **Training Authorization** | **False** |

## 8. 仍缺的证据（才能申请有限计算训练）
1. **独立 P10 测试解码接口 + 固定增益神经依据 + 冻结审计**（`decoder_ready=True`）。
2. **LIF 子网逐步耗时实测**与完整资源预算（`resource_ready=True`）。
3. 参数敏感性网格的**书面签署**与 ΔW 实现前的独立审计。
4. 真实学生进程的隔离**运行时证明**（非仅 mock）。
5. 之后仍需**另行书面批准**；**不自动启动 P10-C1/P10-1**。

**本轮停止。**
