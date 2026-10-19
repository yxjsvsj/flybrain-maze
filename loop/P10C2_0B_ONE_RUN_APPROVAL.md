# P10-C2 一次性无学习 A0：审批请求（**未批准；未运行**）

**授权状态：NOT GRANTED。**
本文件仅为**审批请求**，不构成用户批准。正式审批配置
`loop/p10c2_eligibility_threshold_approval.json` 仍为 **PENDING_APPROVAL**、`value=null`、`approved_by=null`。
未获批准前 `training_authorized=False`，**不得运行 A0**。
**自行填写 `approved_by` 或重算 checksum 不构成授权。**

## 0. 请用户逐项书面选择/确认
- [ ] **(1) 运行范围**：25 seeds（101–125）× 3 条件（none/stim/recover）× 400 步。
- [ ] **(2) 无学习**：ΔW=0；无训练；无 B/C；无实体控车。
- [ ] **(3) 入口**：仅 LPLC2/LC4；`dv=0.2`，`amount = dv·(1−e^{−dt/τ}) ≈ 0.036254`。
- [ ] **(4) 资源上限**：15 min / 2 GB；超限 → INCONCLUSIVE。
- [ ] **(5) 资格迹审批模式**：选择 **Mode A** 或 **Mode B**（见 §2）。
- [ ] **(6) 禁止项**：Dijkstra、地图、目标、真实 ΔW、训练、B/C、实体控车。

## 1. 冻结协议
- 协议版本：**`P10-C2-0a/v1`**；主检验：**双侧精确符号翻转**，α=0.05。
- 有效 **n=22**、计划 **n=25**（排除 ≤3）；主终点：**KC 群体发放差**。
- `dt=0.02`、`tau_pre=tau_post=0.05`、`tau_e=0.5`；`e(0)=0`；trial 重置；单脉冲 t=5 → e[6]。
- 95% CI：percentile bootstrap，10,000 次，seed=20261。
- 所有统计由**原始逐 seed 记录**重算；不得手填 p/效应量。

## 2. 资格迹审批模式（用户必须二选一）
### Mode A：探索性统计检测
- 仅报告**刺激诱发**资格迹幅度、配对差、CI、背景与时间过程。
- **不声明**独立生物学意义的最小效应；正式阈值保持 **PENDING**。
- **canonical A0 判定不得升级为 A0_PASS**（即使统计显著）；可另报 descriptive detection。

### Mode B：预先批准的最小有意义效应
- 用户须提供**事前确定、具独立依据**的数值阈值或可复算规则。
- 该阈值将写入正式审批配置并绑定 `protocol_version`、`approved_by`、`approved_at`、`checksum`。
- **不得**从历史或本次 A0 结果中选择阈值；测试值 `0.01` **不是**已批准阈值。

## 3. 授权绑定规则
实际用户书面授权必须记录为：
- `protocol_version = P10-C2-0a/v1`
- `status = APPROVED`
- `rule`（Mode B 时必须）/ `value`（Mode B 时必须）
- `approved_by`、`approved_at`
- 重新计算的 `checksum`
并仅以此正式配置作为生产裁决输入；**不接受**任意传入的阈值。

## 4. 明确不在本次请求内
- 真实 KC→MBON ΔW 更新/训练；B/C 阶段；Dijkstra 教师；导航；P10-C1 重启。
- 即使 A0_PASS，也不自动授权上述；需另行书面申请。

## 5. 当前静态状态（未运行 A0）
| 项 | 状态 |
|---|---|
| 正式审批配置 | **PENDING_APPROVAL** |
| 生产裁决（无批准） | **INCONCLUSIVE** |
| A0 已运行 | **否** |
| `training_authorized` | **False** |
| Biological / B / C | **NO-GO** |

**未获批准前：不运行 A0、不训练、不更新真实 ΔW、不进入 B/C。**
