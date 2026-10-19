# P10-C2 一次性无学习 A0 结果（Mode A：仅探索性统计检测）

**授权**：用户书面批准 **Mode A**（仅一次、无学习）。正式审批配置
`loop/p10c2_eligibility_threshold_approval.json`：`status=APPROVED`、`mode=A`、
`threshold=null`、`run_authorized=true`、`canonical_pass_allowed=false`、
`approved_by="user (chat approval: P10-C2 Mode A)"`、`approved_at=2026-10-19T05:14:03Z`、
checksum `ac822321...7d8874`。
**运行 commit**：`4835fd8`；**协议版本**：`P10-C2-0a/v1`；**脚本 SHA-256** 见报告。
**本任务不构成任何真实突触训练授权。**

## 1. 运行完整性
| 项 | 结果 |
|---|---|
| seeds | **25/25**（101–125） |
| 条件 | none / stim / recover |
| 每条件步数 | 400 |
| 完成 run | **75/75** |
| 墙钟 | **82.78 s**（<15 min） |
| 峰值内存 | **817.54 MB**（<2 GB） |
| 官方 brain/weights SHA-256 | 校验通过（见下） |
| 失败日志 | 无 |
**未减小 seed、未提高刺激、未选择有利参数、未补跑。**

## 2. LPLC2/LC4 入口（阳性对照）
- 刺激−无刺激：**+17.77** 群体发放/步；相对 **+520%**；精确 p **<1e-4**（报告 0.0）；
  bootstrap 95% CI **[+17.63, +17.91]**；基线 3.415。
- **结论：入口被有效驱动（PASS）。**

## 3. KC 传递（主终点）
- 刺激−无刺激：**−2.0×10⁻⁵**（相对 **−2.2×10⁻⁵**）；精确 p **0.834**；
  bootstrap 95% CI **[−1.86×10⁻⁴, +1.63×10⁻⁴]**（含 0）；基线 **0.9118**。
- **结论：未检测到可重复的 KC 群体刺激诱发差异。**
- **天花板**：KC 基线 **0.9118**、MBON 基线 **0.8815**，均 **>0.85** 预注册阈值 → **ceiling**。

## 4. 资格迹（Mode A：仅探索性）
- 刺激诱发资格迹均值：**−0.01142**（负向、幅度小）；阈值 **PENDING_APPROVAL**。
- **无正向可检测差异**；**canonical A0_PASS 被禁用**（Mode A）。
- 仅作 descriptive detection 报告；**不声明**独立生物学意义或学习成功。

## 5. 五状态与未满足项
| 门控 | 结果 |
|---|---|
| 入口阳性对照 | **PASS** |
| KC 主终点 | **FAIL**（p=0.834，CI 含 0，基线近天花板） |
| 资格迹 | **PENDING_APPROVAL**（Mode A 无阈值；描述性为负向小量） |
| 天花板 | **FAIL**（KC 0.9118 / MBON 0.8815 >0.85） |
| **canonical 五状态** | **INCONCLUSIVE** |
| A0_PASS | **否**（Mode A 禁用 + 未满足项） |
**未满足项**：KC 主终点、资格迹阈值（PENDING）、天花板。

## 6. 原始数据 / 日志 / 资源
- 原始目录（仓库外）：`C:\Users\yang\AppData\Local\Temp\opencode\p10c2_a0\20261019T131616`
- 文件：**76**（75 run npz + records.json）；**114.2 MB**（压缩）。
- **manifest SHA-256**：`a6940d3ba99f2b9d2b737ebc3c2e485e10977898a17b678ca9d9ac4eb989314d`
- 仓库仅提交报告（含逐文件 manifest 哈希）；**不提交原始大文件**。
- 官方哈希：
  - `brain.npz` = `cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca`
  - `weights.npz` = `c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c`

## 7. 科学解释与限制
- 入口**被强驱动**，但**未**观察到 KC 群体变化；KC/MBON 基线接近饱和 → 可能**淹没**小效应。
- 资格迹**无正向检测**；Mode A 仅探索性，**不声称**“感觉驱动学习成立/失败”。
- 截断子网（320 KC / 95 MBON / 4,533 可塑边）；**不外推全脑**。
- 未使用 Dijkstra/地图/目标；**ΔW=0**；无训练/迷宫/小车。

## 8. 权限
- **Training Authorization = False**
- 未更新真实 KC→MBON ΔW；未进入 B/C；未恢复 Dijkstra 导航。
- 本结果**不**自动授权任何后续训练；需另行书面申请。
