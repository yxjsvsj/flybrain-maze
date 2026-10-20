# P11-2-S1 静态勘误与安全检查（**NOT_AUTHORIZED；NO_RUN**）

**状态：STATIC_PREFLIGHT_COMPLETE；NOT_AUTHORIZED；NO_RUN。**
基线：`6fe7db9`。本文件只做静态勘误、源码检查、资源预检、统计预注册和合成测试；
**未启动完整脑、未运行 FlyBrain、未重跑 C1、未执行 S1、未训练、未更新 ΔW、未运行 Dijkstra/迷宫/B/C/实体车。**

---

## 1. 授权核验

- 当前任务**没有**给出“批准一次 P11-2-S1”的明确执行授权。
- `P11_2_S1_ONE_RUN_APPROVAL.md` 是 **NOT_AUTHORIZED** 的审批申请，不是授权。
- **P11-2-C1 的一次性授权已消费**，不得复用。
- 因此：
  - `P11-2-S1 Execution Authorization = False`
  - `FULL_BRAIN_RUN = False`
  - `S1_RUN_GENERATED = False`
  - 未生成 S1 原始数据、run_meta、统计报告或运行脚本。
- **没有伪造执行结果、没有自我批准配置。**

---

## 2. 两跳统计勘误

S0 报告把 **7,883** 写成“前馈路径条数”是不准确的。正确口径：

| 统计量 | 值 | 含义 |
|---|---|---|
| `nnz(W_KC_source @ W_source_bool)` | **7,883** | **聚合后的 source–KC 端点配对数量**（布尔 2-hop 乘积的非零元素数） |
| 独立 2-hop 路径数 | **8,162** | 每条 source→intermediate→KC 的独立路径，按 multiplicity 计数 |
| 每个端点配对的路径数 | max **3**；mean **1.035392642** | 同一 source–KC 端点可能由多个中间神经元到达 |
| 聚合端点正 / 负 | **7,783 / 100** | 这 7,883 个**聚合值的符号**，不是路径符号 |
| 聚合 signed sum | **0.003759782761** | 静态权重统计量；**不是**神经动态刺激效应 |
| MBON 2-hop 端点配对 | **2,528** | 聚合端点配对 |
| MBON 独立 2-hop 路径 | **3,265** | max 6；mean 1.291535 |
| B 后 KC 端点配对 / 独立路径 | **7,883 / 8,162** | 前馈保留；经 KC 中间被删除 = **0** |
| KC→MBON | **61,210，全部为正** | B 保留 |
| 直接 LPLC2/LC4→KC | **0** | 无直接边 |
| 直接 LPLC2/LC4→MBON | **1** | 单条直接边 |

**不得**：
- 把 7,883 直接称为独立两跳路径数；
- 把 7,783/100 当作路径正负号；
- 把 signed sum 0.003759783 当作神经动态刺激效应；
- 为“让刺激有效”而修改预注册刺激或神经网络。

本勘误不改变预注册刺激、靶点、阈值或网络；S0 报告保留，未被覆盖。

---

## 3. 负向 KC/MBON 响应规则

若未来 S1 观察到显著的**负向** KC/MBON 响应：

- 它是**真实的探索性观察**，必须单独报告；
- **不能**冒充已冻结的**正向**主终点通过（主终点要求 `mean ≥ +0.05`）；
- **不能**说成“完全没有感觉变化”；
- **不能**改变主终点方向或事后换终点；
- 入口阳性对照与正向主终点仍按 S0 冻结规则裁决。

---

## 4. S1 正式范围（仅在单独批准后）

- 完整 MaleCNS 166,700；**仅 B synthetic KC→KC-zero**；仅删除已冻结的 **642,933** 条 KC→KC 边。
- `W_B` 内存派生；无新增边、不重归一化；其他原始连接、LIF 参数不变；官方 W 只读。
- **none / stim 同一次实验配对**；seeds **101–125**；每 seed 每条件 **400 步**；总计 **20,000 步**；主窗口 **[100,300)**；CPU only。
- 每条件同 seed、reset、同步数、相同随机流；stim 仅按 FlyBrain inject 语义加电压。
- **精确 `LPLC2`/`LC4` 311 个**；`dv=0.2`；`amount=0.036254`；仅 [100,300) 注入。
- 不刺激 KC/MBON；不换靶点/不加强度/不加连接；无方向/地图/Dijkstra/教师；不训练/不更新 ΔW。

---

## 5. 统计门槛（沿用 S0，未改）

1. **入口**：311 个 LPLC2/LC4 [100,300) 的 stim−none；mean > 0 且 exact p < 0.05；失败 → `SENSORY_ENTRY_FAILED`。
2. **主终点**：匹配 320 KC [100,300) 的 stim−none；25 个 seed 差值；exact two-sided sign-flip；95% paired bootstrap（10,000，seed 20261019）；**最低正向 +0.05**；要求 mean ≥ +0.05、p < 0.05、CI 排除 0。
3. **完整 4,064 KC** 探索性；**匹配 95 / 完整 97 MBON** 次级；匹配 95 最低 +0.05；背景/天花板；三个低反馈 KC 单独报告；none-only 漂移。
4. **失败分级**：`SENSORY_ENTRY_FAILED` / `SENSORY_TRANSMISSION_NOT_DETECTED`（仅指预注册正向标准未达到） / `KC_RESPONSE_WITHOUT_MBON_TRANSMISSION` / `SENSORY_TRANSMISSION_DETECTED_IN_SYNTHETIC_B` / `RESOURCE_NO_GO` / `INCONCLUSIVE`。
5. 不挑 seed、不改门槛、不伪重复、不事后换终点；保留 seed 114。

---

## 6. 资源与安全

- C1 实测 Job 峰值 **1942.7 MiB**；余量 **105.3 MiB**。
- S1 额外记录约 **3.728 MiB/seed（两条件）**；静态估计峰值 **1948.43 MiB**，余量 **99.57 MiB**。
- 时间估计 **290.38 s** < 900 s；时间不是瓶颈。
- **`RESOURCE_PLAUSIBLE_TIGHT`；`RESOURCE_VERIFIED=NO`。**
- 2 GiB 不能称为已验证；S1 必须在 Job 下实测；峰值 >2048 MiB 或 wall >900 s → `RESOURCE_NO_GO`；**不自动提高上限**。
- 冻结安全框架：`CREATE_SUSPENDED → Job Object → ResumeThread`；W_B 派生和全部仿真在 Job 内；低内存逐 seed/条件保存；父进程先持久化 Job 峰值、退出码、时间，再做统计后处理；不重试、不补跑、不调参。

---

## 7. 结论

| 项目 | 值 |
|---|---|
| AUTHORIZATION | **NOT_AUTHORIZED** |
| S1_EXECUTION | **False** |
| FULL_BRAIN_RUN | **False** |
| S0_ENDPOINT_PAIRS_7883 | **True（不是独立路径）** |
| S0_INDEPENDENT_PATHS | **8,162** |
| S0_SIGNED_SUM_IS_DYNAMIC | **False** |
| NEGATIVE_RESPONSES_EXPLORATORY | **True** |
| PRIMARY_FROZEN | **True** |
| RESOURCE | **RESOURCE_PLAUSIBLE_TIGHT** |
| RESOURCE_VERIFIED | **False** |
| S1_READY_TO_APPLY | **True** |
| Training Authorization | **False** |
| REAL_SYNAPTIC_LEARNING_READY | **NO** |
| Level A/B/C/Biological | **NO-GO** |
| A1/A2/A3/B | **NO-GO** |

**未授权则只提交静态勘误及安全检查；未伪造执行结果；未自动启动 S1。**
如用户明确批准，S1 仍需一次独立运行、独立 commit 和完整原始数据/run_meta/统计交付。
