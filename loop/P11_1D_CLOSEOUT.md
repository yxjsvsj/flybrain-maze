# P11-1d 关闭：P11-1c 原始数据归档、KC 全饱和离线核验与模型诊断

**基线** `f4176df`。**只读归档/静态审计**；未运行 FlyBrain、未重跑 P11-1c、未改参数/W/ΔW、未训练、未进入 B/C。
**Training Authorization=False。**

## 1. 归档
- 29 文件（25 seed NPZ + ids.npz + matched_ids.json + child_result.json + log）；**逐文件 SHA-256 全部匹配**。
- 归档：`C:\AA_project\flybrain-maze-archive\p11_1c_full\20261019T153322`
- 备份：`C:\Users\yang\p11_1c_full_archive_backup\20261019T153322`
- 状态 **ARCHIVED_VERIFIED**；原始文件未修改；**未加入 Git**。

## 2. KC 全饱和核验：**NOT_EXACTLY_FULL（重要修正）**
- 25 seeds × 400 步；`kc` **400×4064**、`mb` **400×97**；值仅 **0/1**；与 `child_result` 发放总量一致（容差 0.01，child 记录到 2 位小数）。
- **匹配 320 KC：在 [100,300) 的每个时间步全部为 1**（25/25 seeds；0 个零）。
- **完整 4,064 KC：并非每个时间步全部为 1**——全窗口共 **3 个零**：
  - seed **104**、**114**、**115** 各 **1 个零**（各 1 个细胞 × 1 个时间步）；
  - 每个细胞的窗口占比最小 **0.995**（仍 ≥0.95）；群体均值 ~1.0（四舍五入）。
- **完整全 1 状态的最早时间**：seed 间 **17–24 步**（seed 101 = 20）；**并非对所有 seed 持续到第 400 步**
  （5 个 seed 在后续存在非全 1 步；其中 102/118 在 [300,400)）。
- 启动箱（跨 seed 均值）：**0.1463（0–20）→ 0.9991（20–40）→ 1.0（40–60 起）**。
- 完整 MBON 0.9226；匹配 95 MBON 0.9327。
**修正**：P11-1c 的“完整脑 4,064 KC 全部 1.0”应表述为
**“匹配 320 KC 每步全 1；完整 4,064 KC 均值≈1.0，但有 3 个零（3 个 seed × 1 细胞 × 1 步）”**。
**DATA_EXTRACTION = VERIFIED**（形状/二值/总量/映射一致）；`KC_FULL_SATURATION = NOT_EXACTLY_FULL`。

## 3. bodyId / 提取映射：**VERIFIED**
- 官方 166,700 bodyId 唯一；KC **4,064**、MBON **97**。
- `kc_idx/mb_idx` 与官方 `cell_type`（`brain.npz`）完全一致；索引有序。
- 匹配 320/95 均属于对应完整群体；`np.searchsorted` 结果逐值精确相等。
- `matched_ids.json` 的 bodyId 与 `ids.npz` 的匹配位置对应。
- `brain.step()` 返回**完整 N 空间的 flat indices**；`v` 形状 `(n,batch)`；`kc_map`/`mb_map` 映射到局部列。
- 全部检查通过。

## 4. 独立复算完整/截断比较：**REPRODUCED**
- 匹配 320 KC：完整 **1.000** vs 截断 **0.911799**；差值 **+0.088201**；
  95% CI **[+0.05989751, +0.14159824]**（冻结 bootstrap 10,000/seed 20261）。
- **seed 114 保留**：差值 **+0.711781**（截断 0.288 vs 完整 1.0）；其余 seed 约 +0.059～+0.062。
- 完整 95 MBON 0.932691；截断 95 MBON 0.881472。
- 启动瞬态：完整 0.146→1.0（20–40 步）；截断 0.030→0.818→0.912（~80–100 步）。
- **同 seed 标签 ≠ 逐神经元噪声配对**；完整/子网 N 不同；差异只能解释为**整体网络条件**。
- **科学表述（已核验）**：完整网络在当前模型参数下**同样出现高基线**，因此**截断不是高基线的必要条件**；
  完整与截断条件存在差异，但**具体根因仍未建立**。**不得**断言 gain/tonic/noise 或某类连接是原因。

## 5. 来源审计：**PARTIAL**
- 运行时 HEAD **`7978938`**；正式结果提交 **`f4176df`**。
- 当前脚本 SHA-256 与报告中的 `script_sha256`：**不能独立证明运行时字节完全一致**——实验运行后对
  父进程侧做了补丁（numpy 导入、compare-only、匹配索引映射）；**子进程代码路径未变，但文件字节改变**。
  → **`SOURCE_PROVENANCE = PARTIAL`**。
- 启动失败记录 `p11_1c_experiment_report_launch_failure.json` 保留；父进程后处理异常来源为
  `_stats` 中 `NameError: np`（子进程已完成，**未重跑**）。
- **RESOURCE = INCONCLUSIVE**（Job 峰值内存在写报告前丢失；**未**从 P11-1b 推断、**未**补造）。

## 6. 模型诊断（静态）
- **源码可确认**：`v ← exp(−dt/τ)v + gain·(W@spikes) + tonic + noise`；`dt=0.02`、`τ=0.1`、`gain=3.0`、
  `tonic=0.14`、`noise_hz=1.2`、`noise_amp=0.22`；阈值 `v≥1.0`、复位 0、不应期 0；递归 W；
  逐突触后 `w /= max(incoming,1)`；20 ms 步长封顶 1 脉冲/步。
- **已观察**：完整 KC ≈1.0、MBON ≈0.92；截断 KC 0.91、MBON 0.88；启动 17–24 步达到全 1。
- **候选机制（待因果验证）**：高 gain+tonic、递归兴奋、噪声、L1 归一化、步长上限。
- **MODEL_ROOT_CAUSE = NOT_ESTABLISHED。**

## 7. 裁决
| 项 | 结果 |
|---|---|
| ARCHIVE | **VERIFIED** |
| KC_FULL_SATURATION | **NOT_EXACTLY_FULL**（匹配 320 全 1；完整 4,064 有 3 个零） |
| DATA_EXTRACTION | **VERIFIED** |
| BODYID_MAPPING | **VERIFIED** |
| FULL_VS_SUBNET_DIFFERENCE | **REPRODUCED** |
| SOURCE_PROVENANCE | **PARTIAL** |
| RESOURCE | **INCONCLUSIVE** |
| MODEL_ROOT_CAUSE | **NOT_ESTABLISHED** |
| REAL_SYNAPTIC_LEARNING_READY | **NO** |
| Level A / B / C / Biological | **NO-GO** |
| **Training Authorization** | **False** |

## 8. 当前模型是否具备真实突触学习条件
**否**：完整脑匹配 KC 已全部饱和（每步 1.0），动态范围为 0；完整 4,064 KC 均值≈1.0 且有 3 个零；
在饱和读出上无法可靠检验 KC→MBON 学习。`REAL_SYNAPTIC_LEARNING_READY=NO`。

## 9. 原始数据与限制
- 归档/备份路径见 §1；逐文件 SHA-256 见 `p11_1d_archive_manifest.json`。
- 未覆盖 P11-1c/P10-C2-A0 历史；3 个既有无关工作区改动保留未动。
- 同 seed 标签不等于噪声配对；不得把整体差异归因于某类连接。
