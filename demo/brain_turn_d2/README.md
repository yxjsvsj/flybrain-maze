# DEMO-BRAIN-TURN-D2 — 双向神经读出可行性准备

**D2 结论：`INSUFFICIENT_EVIDENCE`**
**D1 结论：`NO_GO`，永久保留，未改写**

本轮只做离线审查、预注册、合成门控测试；没有执行新的完整脑动力学，没有训练 readout，
没有修改连接组/突触权重，没有驱动电机。

## 文件

| 文件 | 作用 |
|---|---|
| `D2_REVIEW.md` / `D2_REVIEW.json` | 旧路线失败结论、现有证据审查、D2 判定 |
| `SHA256SUMS.txt` | D2 审查引用的 readout/归档/源码 SHA 清单 |
| `D2_PREREG.md` | 未来数据采集、外部线性训练、独立验证和冻结门槛 |
| `review.py` | 只读生成审查 JSON/Markdown；不加载脑、不训练 |
| `dashboard.py` | 独立 D2 启动器，复用 D1 REPLAY 页面；无电机路径 |
| `gates.py` | 预注册老师/推理契约/门控函数 |
| `test_d2.py` | 合成门控与证据回归 |
| `protocol.py` | 120-run 固定协议、授权、freeze、一次性锁 |
| `D2_AUTHORIZATION.json` / `D2_SCRIPT_FREEZE.json` | 当前 `NOT_AUTHORIZED`；无生产锁 |
| `frames.py` / `capture.py` | 真实 ToF 帧包校验 / 采集（不跑脑、不动电机） |
| `collect.py` | 静态完整脑采集；默认无授权拒绝 |
| `train.py` | 外部零偏置线性拟合；默认无授权拒绝 |
| `evaluate.py` | 独立 700mm 留出验证与 G1–G7 |
| `pipeline.py` | 一次性 fail-closed 总入口；默认拒绝 |
| `test_pipeline.py` | 合成数据采集/拟合/验证/授权回归 |
| `STORYBOARD.md` | 7 天视频分镜、真实失败素材备选 |

## 运行

```powershell
# 离线审查；不会加载完整脑
.\.venv\Scripts\python.exe -m demo.brain_turn_d2.review

# 合成门控测试
.\.venv\Scripts\python.exe -m unittest demo.brain_turn_d2.test_d2 -v
.\.venv\Scripts\python.exe -m unittest demo.brain_turn_d2.test_pipeline -v

# 完整流水线：默认会 REFUSED/NOT_AUTHORIZED，不会加载脑
.\.venv\Scripts\python.exe -m demo.brain_turn_d2.pipeline

# 复用 D1 只读 REPLAY 仪表盘（D2 启动器）
.\.venv\Scripts\python.exe -m demo.brain_turn_d2.dashboard --open
```

当前 `D2_AUTHORIZATION.json` 是 `NOT_AUTHORIZED`，没有授权锁；
`D2_RUNS/` 不存在，`D2_CANDIDATE.npz` 不存在，`D2_EVAL.json` 不存在。
`pipeline.py` 只有显式批准后才可能运行 120 个静态完整脑 run。

D1 页面（`http://127.0.0.1:8765/`）继续只读：没有 `/api/drive`、没有
`/api/manual/drive`、没有 Arduino/串口/电机输出。D2 不另造一套可驱动页面。

## 旧路线失败结论

1. `readout_real.npz`（SHA `632cddb1...083e`）是模仿手写贴右墙老师的线性解码器，
   不是天然果蝇运动回路，也没有独立左右输出。
2. D1 A 无遮挡已持续负向：均值 `-0.2773`、`96.5%` 负、最长同号 `8.48s`。
3. D1 C 右侧的最近外层 `R` 在 380–450mm 有效，但 `dmin_R` 仍为 `5.78`；
   外层 `R(-70°)` 不在 ±60° 神经避障 FOV 内，因此不是有效神经右障碍刺激。
4. D1 B 左侧缺失；没有左向真实 ToF/DN 归档。
5. P2d-29 L50/R50 保持 `FAIL_OFFLINE`/`FAIL`，正式关闭；未重启、未调参。

## 候选是否有足够神经信息？

**`INSUFFICIENT_EVIDENCE`**。

- 现有 1314 DN 特征来自 A/C/D；没有 FL-only 左向数据。
- C 的右向数据不进入神经侧向输入；D 是双侧，不能建立左右独立反号。
- `p2_readout_recoverability.py` 要求 `static_A/B/C/D`；当前没有 B，
  且 `static_corr*` 会被其 `_label` 误标为 C，不能直接拿它当双向证据。
- 没有 `readout_real_p2_5tof_candidate*.npz`；未训练、未产生新结果。
- 不把“没有数据”说成“没有信息”，也不把任何单次场景说成成功。

## 候选设计（未训练）

```text
FL/F/FR ToF → PhysicalToFEncoder → MaleCNS（W 固定，ΔW=0）
→ 1314 DN Trace → 外部零偏置线性读出 → omega → 左/右轮
```

- 老师：`clip(3.0*(prox_R-prox_L), -1, +1)`，正=左；仅训练时使用。
- 推理只接收 DN 特征；禁止原始 ToF、老师、地图、Dijkstra、临时偏置、符号反转、手动左右。
- 外部线性读出：1314 特征、零截距 ridge、λ=0.01、不搜参。
- 训练 seeds 3001–3003 / 400mm；验证 seeds 3051–3052 / 700mm。

## 冻结成功门槛

- A 无遮挡：所有验证 seed `|mean| ≤ 0.05`。
- FL `≤ -0.10`，FR `≥ +0.10`，`FR-FL ≥ 0.20`。
- ≥80% 验证 seed 同号。
- 撤除后 1s 内 `|mean| ≤ 0.05`。
- ZERO 消融：omega=0。
- 打乱 DN 特征：原始方向准确率 ≥0.75，打乱后 ≤0.55。

全部门槛必须通过；所有失败原样报告，不调门槛、不挑 seed、不换路线。

## 下一步授权

`D2-AUTH-1-STATIC-TRAIN-AND-HELDOUT` 当前 **NOT_AUTHORIZED**。

一次明确授权只覆盖：`D2-STATIC-v1` 静态数据采集 + 外部线性读出训练 + 独立 700mm
seeds 验证。预计 120 runs / 约 60,000 完整脑步；不接电机、不跑迷宫、不训练连接组，
资源/Job 必须实测。详见 `D2_PREREG.md`。

## 边界

```text
NEW_FULL_BRAIN_AUTHORIZATION=False
READOUT_TRAINING_AUTHORIZATION=False
ROBOT_MOTION_AUTHORIZATION=False
REAL_SYNAPTIC_LEARNING_READY=NO
BIOLOGICAL_VALIDITY=UNVALIDATED
```

D1 历史、P2d-29 原始失败结果、S1R/S3/S4/S5 归档、授权锁和无关 dirty 文件均保留。
