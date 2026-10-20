# P11-2-C1a：授权来源与运行记录审计

**状态：AUDIT COMPLETE；NO RERUN；NO DYNAMICS。**
基线：`5e68988`。本文件只读核对授权文本、脚本冻结、run_meta、child_result、
官方数据哈希、W_B/字段切换和 Job 记录；不重新执行实验。

---

## 1. 授权文件与 checksum

`loop/p11_2_c1_authorization.json`：

- `protocol`：`P11-2-C1/v1`
- `status`：`APPROVED`
- `approved_by`：`user`
- approval text：**“批准一次 P11-2-C1：A/B，25 seeds，每 seed 400 步，共 20,000 步，15 分钟 / 2 GiB，一次性，不自动重跑。”**
- `checksum`：`57ac353a449edb8a6bdaa8f4da7039db34df24b8356bb559e4333503662b57b4`
- 重新按 payload 规范序列化计算：**一致**。
- `run_meta.authorization_sha256`：`57ac353a...`
- `child_result.auth_sha256`：`57ac353a...`

**重要区别**：

- checksum 只能证明 **配置 payload 未被改动**；它**不是**独立用户签名、证书或第三方时间戳。
- 授权文本与本会话中用户的原始批准消息**完全对应**。
- 因此：
  - `AUTHORIZATION_PROVENANCE = VERIFIED`（可信聊天记录）
  - `AUTHORIZATION_SIGNATURE = NOT_AVAILABLE`
  - 没有伪造聊天记录、签名或精确时间；没有声称有独立签名。
- 若审计标准要求“独立签名/第三方证明”，则签名层面应记为 `UNVERIFIED`；
  这不改变已发生实验的记录、统计重算或资源事实。

---

## 2. 脚本冻结与运行元数据

| 项目 | 值 |
|---|---|
| 实验脚本 | `loop/p11_2_c1_experiment.py` |
| 脚本 SHA-256 | `136bb62890165af551e1a65ae3db9d24b8085755106de3d4b15b19c11eeb1543` |
| freeze SHA-256 | `136bb62890165af551e1a65ae3db9d24b8085755106de3d4b15b19c11eeb1543` |
| `run_meta.script_sha256` | 同上 |
| `child_result.script_sha256` | 同上 |
| 授权 checksum | `57ac353a449edb8a6bdaa8f4da7039db34df24b8356bb559e4333503662b57b4` |
| 官方 brain SHA-256 | `cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca` |
| 官方 weights SHA-256 | `c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c` |

`run_meta.json` 在统计后处理之前写出；`child_result.json` 与 50 个 A/B NPZ 已持久化。
脚本 SHA、授权 SHA、freeze SHA、官方哈希在 `run_meta` 与 `child_result` 中相互一致。

---

## 3. 运行记录与资源

- **25/25 A/B pairs**；25 seeds × 2 conditions × 400 steps = **20,000 步**。
- 全部 seed 有限；`noise_pairing_all=true`，逐 seed A/B RNG 状态配对 **25/25**。
- **Job 峰值 1942.7 MiB**（进程 1941.0 MiB）；**wall 290.38 s**；exit 0；
  `timeout=false`、`mem_exceeded=false`；< 2048 MiB / 900 s。
- 官方 brain/weights 完整 SHA-256 相等；原始 W 未被修改。
- `W_B`：删除 **642,933** 条 KC→KC；`W_B` 非零 **24,940,005**；
  `KC_KC_nnz=0`；field switch：A 恢复逐位一致、非 KC posts 不变、1,478 个 KC posts 改变。
- 原始数据 **57** 文件；源、持久归档、独立备份逐文件 SHA-256 全部匹配；
  `p11_2_c1a_archive_manifest.json` 记录三处副本；未把 NPZ 加入 Git。

---

## 4. 统计勘误与因果裁决

- 历史报告使用 full-400 数组；预注册主窗口 **[100,300)**。
- 从原始 NPZ 重算：A=**1.000000000**、B=**0.201393125**、B−A=**−0.798606875**；
  exact sign-flip **p=5.960464477539063e-08**；95% CI **[-0.802570, -0.794391]**；
  最低有意义效应 **−0.10**；门槛通过。
- `MODEL_CAUSAL_CONTRIBUTION = SUPPORTED`（模型内）。
- `BIOLOGICAL_CAUSATION = NOT_ESTABLISHED`。
- 三个低反馈 KC 对完整 B 均值的贡献仅 **0.000646**；其余 4,061 个 KC 为 **0.350706**。
  不能把三个 KC 称为剩余 0.35 群体活动的主要来源。

---

## 5. 最终门控

| 项目 | 值 |
|---|---|
| EXPERIMENT_RUN | **COMPLETE** |
| RESOURCE | **MEASURED_UNDER_LIMIT** |
| PRIMARY_ERRATA | **CORRECTED** |
| CAUSAL_STATISTICAL_GATE | **CAUSAL_CONTRIBUTION_SUPPORTED** |
| AUTHORIZATION_PROVENANCE | **VERIFIED**（chat transcript） |
| AUTHORIZATION_SIGNATURE | **NOT_AVAILABLE** |
| MODEL_CAUSAL_CONTRIBUTION | **SUPPORTED**（模型内） |
| BIOLOGICAL_CAUSATION | **NOT_ESTABLISHED** |
| REAL_SYNAPTIC_LEARNING_READY | **NO** |
| Training Authorization | **False** |
| Level A/B/C/Biological | **NO-GO** |
| A1/A2/A3/B | **NO-GO** |

**已停止。未重跑 C1；未训练；未更新真实 ΔW；未运行 Dijkstra/迷宫/B/C/实体车。**
