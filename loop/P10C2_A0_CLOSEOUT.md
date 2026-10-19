# P10-C2-A0 离线归档、独立复核与正式关闭

**基线** `b11c74c`。**本任务不是新实验授权**：未运行任何 MaleCNS 动力学、未重跑 A0、未追加 seed/条件、
未改变刺激/资格迹参数/判定规则、未训练、未更新真实 ΔW、未运行 Dijkstra/迷宫/B/C/小车。
未覆盖 `p10c2_a0_report.json` 及原始结果。

## 1. 原始数据归档
| 项 | 结果 |
|---|---|
| 源 raw_dir | `C:\Users\yang\AppData\Local\Temp\opencode\p10c2_a0\20261019T131616` |
| 文件 | **76**（75 条件 NPZ + `records.json`） |
| 大小 | **119,728,940 B**（114.2 MB） |
| 缺失 / 损坏 / 多余 | **0 / 0 / 0** |
| 逐文件 SHA-256 对比 `raw_manifest` | **全部一致** |
| manifest 总哈希 | **`a6940d3b…989314d`**（与报告一致） |
| 归档目录 | `C:\AA_project\flybrain-maze-archive\p10c2_a0\20261019T131616` |
| 独立备份 | `C:\Users\yang\p10c2_a0_archive_backup\20261019T131616` |
| 状态 | **`ARCHIVED_VERIFIED`** |
**原始文件内容未修改；原始大文件未加入 Git。**

## 2. 离线独立复算（仅读 NPZ/records；未调用 `FlyBrain.step()`）
- seeds **101–125**、条件齐全唯一、每条件 **400 步**；KC 320、MBON 95、资格迹边 **4,533**。
- NPZ 数组与 metadata 对应；从 NPZ 重算的 rates 与 `records.json` 一致。
- **Entrance**：配对差 **+17.767**；精确双侧符号翻转 **p_exact = 5.960464477539063×10⁻⁸**（=2/2²⁵；
  运行报告展示 0.0 是 `round(.,6)`）；95% CI 与报告一致。
- **KC**：配对差 **−1.9996×10⁻⁵**（报告展示 −2×10⁻⁵）；**p_exact = 0.8341633**；CI 含 0；
  基线 **0.9118**。
- **MBON** 基线 **0.8815**。
- **资格迹**：刺激诱发均值 **−0.0114243**；阈值 **PENDING_APPROVAL**。
- `matches_report`：entrance/KC/eligibility/CI 全部一致；仅**展示舍入**差异（已记录）。
- 成功运行 **75/75**；资源（报告记录）**82.78 s / 817.54 MB**。

## 3. 运行来源审计
- **`run_commit = 4835fd8`**（运行时基线）；**运行脚本在后续结果提交 `b11c74c` 中新增**。
  → 二者**必须区分**：脚本来源由 `script_sha256` 标识，**不是**由旧 commit 标识；
  当前脚本 SHA-256 **与报告一致**。
- **`brain.reset(seed)` 审计**（读源码，不重跑）：重置 `rng = default_rng(seed)`、`v=0`、
  `fired=empty`、`steps=0`；每个 seed/条件前调用 → **无残留状态**；三条件同 seed → **RNG 配对**。
- **资格迹**：`if t > 0:` 后才用 `t−1` 的 traces → **严格因果**；边来自
  `Wsub[mb_pos][:, kc_pos]`（**仅真实 KC→MBON 边，4,533 条**），无新边。
- **授权**：配置 `status=APPROVED`、`mode=A`、`threshold=null`、
  `approved_by="user (chat approval: P10-C2 Mode A)"`、`approved_at=2026-10-19T05:14:03Z`、
  checksum `ac822321…7d8874`。**原始授权文本来自聊天，已在配置中记录；未独立签名核验**（不补造）。

## 4. 科学表述与限制
- **seed = 固定连接组模型下的随机模拟重复**，**不是**独立真实果蝇个体。
- **入口刺激有效**（LPLC2/LC4 被强驱动）。
- **KC 群体未检出效应**（p=0.834，CI 含 0）。
- KC/MBON 高自发背景（0.912/0.881）→ **天花板**，下游结论受限。
- **不能**推广到完整 MaleCNS 或真实生物脑。
- **资格迹结果仅为 Mode A 探索性描述**，无独立意义阈值。

## 5. 最终裁决
| 门控 | 结果 |
|---|---|
| A0 运行完整性 | **VERIFIED 75/75** |
| Entrance | **PASS** |
| KC Main Endpoint | **NOT_DETECTED** |
| Ceiling | **FAIL** |
| Eligibility Meaningful Threshold | **PENDING_APPROVAL** |
| Canonical A0 | **INCONCLUSIVE** |
| Level A Training | **NO-GO** |
| Level B / C | **NO-GO / NO-GO** |
| Biological | **NO-GO** |
| **Training Authorization** | **False** |

**无论离线复核结果如何，不从本次 Mode A 升级为训练授权。**
**已停止：不重跑 A0、不改参数、不更新真实 ΔW、不启动导航实验。**
