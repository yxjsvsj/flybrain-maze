# P11-2-C1 一次性审批（**已批准；已运行一次；25/25**）

**状态：APPROVED；RUN COMPLETED；CONSUMED。**
`P11-2-C1 Execution Authorization = APPROVED_ONE_RUN; CONSUMED`。

**用户授权原文**：> 批准一次 P11-2-C1：A/B，25 seeds，每 seed 400 步，共 20,000 步，15 分钟 / 2 GiB，一次性，不自动重跑。

本文件是授权与执行记录；只覆盖一次 A/B；不覆盖训练/ΔW/Dijkstra/迷宫/B/C/实体车。

## 1. 请求

申请**一次** P11-2-C1 网络级因果实验：

- 条件 **A intact** 与 **B KC→KC zero**；
- 完整 MaleCNS 166,700 神经元；
- seeds **101–125（25 个）**；每 seed **400 步**；**无刺激**；
- 总步数 **20,000**；主要窗口 **[100,300)**；
- 整体 **15 min / 2 GiB**；一次性；**不自动重跑**；
- 两条件在同一次冻结审批内完成。

## 2. 干预（已冻结）

- **A**：官方 `weights.npz` 只读，原始 W。
- **B**：仅在内存派生 `W_B`，删除现有 KC→KC 的 **642,933** 条边；
  - `W_B` 非零 **24,940,005**；无权重重归一化；无新增边；
  - 其他所有边的连接与权重值逐位不变；
  - 官方 W 文件前后 SHA-256 相等；不写任何派生 weights 文件；
  - B 中 KC 输入总量下降（删除权和 **2,275.627441**）是干预的一部分。
- 不改 gain、tonic、noise、dt、tau、阈值、reset、readout、其他 W。
- 不训练、不更新 ΔW、不 Dijkstra、不迷宫、不 B/C、不实体车。

## 3. 噪声配对（已核验）

- 源码：`step()` 每步**一次** `self.rng.random((self.n, B))`，在 current/tonic 后、firing 前；
  `reset(seed)` 重建 `default_rng(seed)`；无其他 RNG。
- 条件：同 seed、同 n/batch、同步数、每条件 `reset(seed)`、CPU only、无 stim/inject。
- 合成测试：同 seed 原始 RNG 逐位相等。
- **逐神经元噪声严格配对**；A/B 的唯一差异是 W。

## 4. 统计（已冻结）

- 主要终点：**匹配 320 KC 在 [100,300) 的每步发放占比**，per seed。
- 效应：**B − A** 的 25 个 paired seed 差值；seed 为独立单位。
- 检验：**exact two-sided sign-flip**（2^25；最小 p = **5.960464477539063e-08**）。
- 95% CI：paired bootstrap 10,000，seed 20261019。
- 最低有意义效应：**绝对下降 0.10**（`mean(B−A) ≤ −0.10`）。
- 支持：`mean ≤ −0.10` **且** p < 0.05；不确定：`mean ≤ −0.10` 但 p ≥ 0.05；不支持：`mean > −0.10` 或 B 增加。
- 所有 25 seeds 保留；不删除 seed 114；不挑选；不把 cell-step 当独立样本。
- 超时 >900 s 或 Job 峰值 >2048 MiB → **NO_GO**；缺失 → **INCONCLUSIVE**。

## 5. 资源（估算，未验证）

- P11-1c 152.54 s / 10,000 步 → 15.254 ms/步；
- 20,000 步 ≈ **305.08 s** < 900 s；
- P11-1b 峰值 1497.87 MB + W_B 190.91 MiB ≈ **1688.78 MiB** < 2048 MiB；
- 本轮**未用完整脑验证**；执行时 Job 硬限制 2 GiB、父进程 15 min 截止；
- 子进程结果与 Job 峰值在后处理前持久化；不自动重试。

## 6. 安全

- Windows `CREATE_SUSPENDED → AssignProcessToJobObject → ResumeThread`；
- 进程级 + Job 级 2 GiB；`KILL_ON_JOB_CLOSE`；
- 官方 brain/weights 完整 SHA-256 前后相等；脚本 SHA-256 固定；
- 原始 W 不变；3 个既有无关改动不清理；
- 合成测试覆盖内存限制、超时、启动/分配失败、崩溃、禁止重试。

## 7. 解释边界

- 若 B 下降 ≥0.10 且显著：最多支持 **KC→KC 边对当前模型饱和表型有因果贡献**；
  不证明特定反馈环路、生物机制或学习能力。
- 若 B 不下降：删除 KC→KC 在所测条件下不能解除饱和；继续调查非 KC 输入。
- 模型因果 ≠ 生物因果；未保存的非 KC 活动未重建。

## 8. 批准边界

批准只覆盖本文件冻结的：

- A/B 两条件；25 seeds（101–125）；400 步；无刺激；20,000 步；[100,300)；
- 15 min / 2 GiB；一次性；不重跑；
- 上述统计与解释边界。

**不覆盖**：训练、ΔW、Dijkstra、迷宫、B/C、实体车、A1、A2、A3、B。
`Training Authorization=False`；`REAL_SYNAPTIC_LEARNING_READY=NO`；
Level A/B/C/Biological = **NO-GO**。

## 9. 执行结果（本次一次性运行）

- **25/25 A/B pairs**；全部 seed 有限；A/B 逐 seed RNG 状态配对 **25/25 true**。
- 匹配 320 KC [100,300)：**A 0.956807，B 0.194136，B−A −0.762671**；
  exact two-sided sign-flip **p = 5.960464477539063e-08**；
  95% CI **[-0.766145, -0.759004]**。
- **实测 Job 峰值 1942.7 MiB / wall 290.38 s**；exit 0；无超时/超内存；< 2 GiB / 15 min。
- W_B：删除 **642,933** 条 KC→KC；`W_B` 非零 **24,940,005**；无新增/无重归一化；官方 W 前后哈希相同。
- 判定：**CAUSAL_CONTRIBUTION_SUPPORTED**（模型内）；B 后仅 2 个 KC per-cell ≥0.85
  （2460 0.9545、3971 0.9516；2650 0.6016），匹配 320 降至 0.194136。
- 后续感觉传递：**仅可另提独立提案**；本轮不授权训练/ΔW/Dijkstra/迷宫/B/C/实体车。
- `Training Authorization=False`；`REAL_SYNAPTIC_LEARNING_READY=NO`；Level A/B/C/Biological **NO-GO**。

**本次一次性授权已消耗；不自动重跑。**
