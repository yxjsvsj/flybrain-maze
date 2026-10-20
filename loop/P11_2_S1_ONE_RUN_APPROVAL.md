# P11-2-S1 一次性审批（**已批准；执行一次；RUN FAILED/INCOMPLETE**）

**状态：APPROVED_ONE_RUN；RUN_FAILED/INCOMPLETE。**
`P11-2-S1 Execution Authorization = APPROVED_ONE_RUN; CONSUMED。`

**用户批准原文**：

> 我批准 P11-2-S1 仅执行一次：完整 MaleCNS 的合成 B（KC→KC-zero）网络，none/stim 两条件，seeds 101–125，每 seed 每条件 400 步，共 20,000 步。仅精确 LPLC2/LC4 共 311 个靶点，dv=0.2、amount=0.036254、窗口 [100,300)，CPU，15 分钟及 2 GiB 硬上限，不自动重跑。不批准训练、ΔW、Dijkstra、迷宫、B/C 或实体控车。

`P11-2-C1` 的一次性授权已消费，不适用于本阶段；本文件记录一次 S1 审批与执行状态。

## 0. S1 静态预检状态（已批准；RUN PENDING）

- 静态预检：`p11_2_s1_preflight_report.json`；未启动完整脑、未生成 S1 原始数据/run_meta/统计报告。
- 两跳勘误：**7,883 = 聚合 source–KC 端点配对**；**独立 2-hop 路径 = 8,162**；
  7,783/100 是**聚合端点符号**；signed sum 0.003759783 不是动态刺激效应。
- MBON：2-hop 端点 2,528 / 独立路径 3,265；KC→MBON 61,210 全正。
- 资源：1948.43 MiB 静态估计 / 2048 MiB；`RESOURCE_PLAUSIBLE_TIGHT`；未验证。
- **没有自我批准配置**；需用户明确批准下方一次 S1。

---

## 1. 请求

申请**一次** P11-2-S1 感觉传递实验：

- 完整 MaleCNS，**仅 CPU**；**仅 B 合成网络**（仅删除原 642,933 条 KC→KC 边）。
- seeds **101–125（25 个）**；每 seed **400 步**；条件 **none / stim** 同实验配对。
- 总步数 **20,000**；主窗口 **[100,300)**。
- 感觉：**精确 LPLC2/LC4（311 个；LPLC2 185 + LC4 126）**；
  `dv=0.2`；`amount=0.036254`；[100,300)；**不是 0.6**。
- 原 LIF 参数固定；`W_B` 仅内存派生；官方 `weights.npz` 只读；`ΔW=0`。
- 不直接刺激 KC/MBON；不新增连接；不改变五 ToF 编码/分组；无方向/Dijkstra/地图/教师。
- 整体 **15 min / 2 GiB**；一次性；**不自动重跑**；不训练/不更新 ΔW。

---

## 2. 统计（已冻结）

- **入口阳性对照**：精确 LPLC2/LC4（311）[100,300) 的 stim−none；exact sign-flip；
  要求 mean > 0 且 p < 0.05；失败 → `SENSORY_ENTRY_FAILED`。
- **主终点**：匹配 320 KC [100,300) 的 stim−none；seed 为独立单位（25 对）；
  exact two-sided sign-flip；95% paired bootstrap（10,000，seed 20261019）；
  **最低有意义效应 +0.05 绝对**；要求 mean ≥ +0.05、p < 0.05、CI 排除 0。
- **完整 4,064 KC** 探索性；**匹配 95 / 全部 97 MBON** 次级；匹配 95 最低 +0.05。
- **背景/天花板**；**三个低反馈 KC 单独报告**（2460/109321、2650/117446、3971/545375），
  不作为总体代表。
- **无刺激漂移**：none-only matched-320 [100,200) vs [200,300) 的 exact 检验；
  若 p<0.05 标记漂移；随机噪声由 within-seed paired、exact、25 seeds 控制。
- **失败条件**：
  - 入口无响应 → `SENSORY_ENTRY_FAILED`；
  - 入口有响应但 KC 主终点失败 → **`SENSORY_TRANSMISSION_NOT_DETECTED`**；
  - KC 有响应但 MBON 无 → `KC_RESPONSE_WITHOUT_MBON_TRANSMISSION`；
  - KC+MBON 均通过 → `SENSORY_TRANSMISSION_DETECTED_IN_SYNTHETIC_B`（仅计算现象）。
- 所有 seeds 保留，含 114；不挑选、不删除；不把 cell-step 当独立样本；不事后换终点。

---

## 3. 路径与资源

- 静态审计：LPLC2/LC4 → KC **无直接边**；前馈到 KC 是 **2-hop via non-KC（7,883 条）**；
  B 保留全部 7,883（0 条经 KC 中间被删除）；KC→MBON **61,210 条，全正**，B 保留。
- 前馈 signed sum **0.003759783**，很小；**解剖可达 ≠ 动力学可检测**。
- C1 实测 Job 峰值 **1942.7 MiB**；余量 **105.3 MiB**。
- S1 额外记录约 **3.728 MiB/seed（两条件）**；静态估计峰值 **1948.43 MiB**，余量 **99.57 MiB**。
- **`RESOURCE_PLAUSIBLE_TIGHT`；`RESOURCE_VERIFIED=NO`**。
  2 GiB 不能称为已验证；S1 必须在 Job 下实测；>2048 MiB 或 >900 s → `RESOURCE_NO_GO`；
  **不自动提高上限**。
- 低内存逐 seed/条件保存 NPZ 后释放；父进程在后处理前持久化 Job 峰值；
  沿用 `CREATE_SUSPENDED → Job Object → ResumeThread`。
- 脚本/源码/官方 W 全量 SHA-256 固定；官方 W 只读。

---

## 4. 科学边界

- 即使 S1 通过，也只支持**合成消融模型中的计算现象**；
  不代表原始完整 MaleCNS 已恢复生理功能。
- 真实 KC→MBON 学习未授权；DN 自主转向未验证；实车自动导航 **NO-GO**。
- `Training Authorization=False`；`REAL_SYNAPTIC_LEARNING_READY=NO`；
  Level A/B/C/Biological = **NO-GO**。
- **不生成自我批准配置**；需用户明确书面批准一次。

---

## 5. 请用户明确批准

如批准，请回复类似：

> **批准一次 P11-2-S1：B synthetic KC→KC-zero；none/stim；25 seeds；每 seed 400 步；共 20,000 步；精确 LPLC2/LC4（311）；dv=0.2 amount=0.036254；[100,300)；CPU；15 分钟 / 2 GiB；一次性；不自动重跑；不训练/ΔW/Dijkstra/迷宫/B/C/实体车。**

未批准前不运行。

## 9. 运行结果（失败；一次性授权已消耗）

- child exit 1；wall **0.82 s**；Job 峰值 **521.05 MiB**；xperiment.log **0 字节**；无 child_result.json；**0/25**；无逐 seed 原始数据；无统计报告。
- 未触发 2 GiB / 900 s 上限（521 < 2048；0.82 < 900）；失败发生在子进程/管线阶段。日志为空，**无法从已保存记录确定具体异常**；不臆造原因。
- 一次性授权已消耗；**不自动重跑**；不补跑、不调参、不提高上限。无模型因果/生物/学习结论。

**Training Authorization=False；REAL_SYNAPTIC_LEARNING_READY=NO；Level A/B/C/Biological=NO-GO。**
