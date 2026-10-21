# DEMO-BRAIN-TURN-D2 — 双向神经读出可行性准备

**基线** `0d1d845`
**D2 结论** `INSUFFICIENT_EVIDENCE`
**D1 结论** `NO_GO`（永久保留，未改写）

本轮只做离线审查、预注册和合成门控测试；未运行完整脑、未训练、未接电机。

## 1. 旧路线为什么失败

- `readout_real.npz` SHA `632cddb1c9ae43de7bde63a5bd8851036009c4625d12c113bf0e96119ace083e`；linear ridge imitates a hand-written teacher; not a natural fly motor program; not a natural motor readout。
- D1 A 无遮挡：`mean=-0.2773`，`96.5%` 负向，持续同号 `8.48 s`；不是零偏置。
- D1 C 右侧：外层 `R` 有效但 `dmin_R` 仍为 `5.78`，没有进入神经侧向输入；不能把 R-only 数据当 FR 神经刺激。
- D1 B 左侧：无归档；没有可用的左向真实 ToF/DN 反应。
- P2d-29 L50/R50：`FAIL_OFFLINE`，已关闭；不能重开或调参。

## 2. 候选双向架构

```text
真实 FL/F/FR ToF
  → PhysicalToFEncoder
  → MaleCNS（W 固定，ΔW=0）
  → 1314 DN Trace
  → 外部零偏置线性读出
  → omega
  → 左右轮（未来仅预览/受控授权）
```

- DN 特征维数：**1314**。
- 左向专用归档：**0**；有效神经右向归档：**0**；双侧归档：**4**。
- 现有 `static_corr_*` 是双侧，不是 FL-only/FR-only；不能替代左右独立刺激。
- `p2_readout_recoverability.py` 期望 `static_A/B/C/D`；当前无 B，且 `static_corr*` 会被其 `_label` 误标成 C，因此不能直接运行它当成双向证据。
- 候选 `.npz` 是否存在：**0**；未训练、未产生结果。

**候选结论：`INSUFFICIENT_EVIDENCE`**。没有左侧数据，无法判断 1314 DN 是否足够支持双向识别；这不是 CANDIDATE_GO，也不是对现有 readout 的翻案。

## 3. 预注册的训练/验证边界

- 协议：`D2-STATIC-EXTERNAL-READOUT/v1`；状态：`PREREGISTERED; NOT_AUTHORIZED; NOT_RUN`。
- 连接组 W 只读；生物突触 ΔW=0；只训练外部线性读出。
- 老师：`clip(3.0*(prox_R-prox_L), -1, +1)`，正=左；仅训练时使用，推理绝不读取。
- 推理输入只有 1314 DN Trace；禁止原始 ToF、老师、地图、Dijkstra、目标方位、临时偏置、反转符号、手动左右切换。
- 训练：seeds [3001, 3002, 3003]，400mm，3 次；验证：seeds [3051, 3052]，700mm，独立距离。
- 数据量：120 次 × 10s，约 60,000 完整脑步；只做静态 ToF，不接电机。
- 读出：1314 特征 z-score、零截距 ridge、λ=0.01；不搜索、不选特征。

## 4. 冻结成功门槛

| 门槛 | 固定值 |
|---|---|
| 无障碍 | 所有验证 seed `|mean| ≤ 0.05` |
| 左右相反 | FL `≤ -0.1`，FR `≥ 0.1`，FR−FL `≥ 0.2` |
| 独立重复 | 至少 80% 验证 seed 同号 |
| 撤除恢复 | 撤除后 1s 内 `|mean| ≤ 0.05` |
| ZERO 消融 | neural turn=0，omega=0 |
| 打乱特征 | 原始方向准确率 `≥ 0.75`，打乱后 `≤ 0.55` |

全部门槛必须在前瞻性冻结的验证集上通过；任何失败均报告，不调门槛、不挑 seed、不换路线。

## 5. 一次性下一步授权范围

- `D2-AUTH-1-STATIC-TRAIN-AND-HELDOUT`：`authorized_now=False`。
- 授权范围：`D2-STATIC-v1` 数据采集 + 外部线性读出训练 + 留出验证；
  A/B/C/D，400mm 训练、700mm 验证，3 次重复，5 seeds。
- 完整脑只在静态 ToF 采样中运行；不接电机；不训练连接组；
  资源/Job 必须实测，未验证前不得声称安全。
- 任何门失败即停止并保留失败素材；不自动换读出/神经群/增益/滤波/老师。

## 6. 演示因果链与视频

- 复用 D1 REPLAY 页面：`python -m demo.brain_turn_7d.replay --open`；页面只读，无 `/api/drive`，无电机接口。
- 分镜与诚实失败备选见 `STORYBOARD.md`。
- 当前页面只能展示历史 readout/失败证据，不能展示实体车运动。

## 7. 限制

- No new full-brain dynamics were run.
- No readout was trained; no candidate weights exist.
- No left-only real ToF/DN data exists; this is the primary evidence gap.
- The right outer R-only archives are not valid neural right-side stimuli.
- P2d-29 remains FAIL and closed; D1 remains NO_GO.

`FULL_BRAIN=NO`, `TRAINING=NO`, `ROBOT_MOTION=NO`, `BIOLOGICAL_VALIDITY=UNVALIDATED`。
