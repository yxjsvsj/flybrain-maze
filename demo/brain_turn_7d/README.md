# DEMO-BRAIN-TURN-D1 — 离线脑源转向回放

**判定：`NO_GO`**（不是 CANDIDATE_GO）

本轮只读归档与冻结 `readout_real.npz`；没有构建完整脑、没有运行 MaleCNS 动力学、没有训练、
没有启动电机，也没有新增授权。

## 快速启动

```powershell
.\.venv\Scripts\python.exe -m demo.brain_turn_7d.audit
.\.venv\Scripts\python.exe -m demo.brain_turn_7d.replay --open
.\.venv\Scripts\python.exe -m unittest demo.brain_turn_7d.test_d1 -v
```

默认地址 `http://127.0.0.1:8765/`。回放器只允许 loopback，只读 GET；没有
`/api/drive`、`/api/manual/drive`、Arduino、串口或电机接口。

## 第一关结论

| 项目 | 结果 |
|---|---|
| A 无遮挡 | **FAIL**：冻结 readout 持续负向，`mean=-0.2773`，`96.5%` 负，最长同号 `8.48 s` |
| B 左侧障碍 | **MISSING**：没有左向真实 ToF/DN 归档；未用合成或镜像伪造 |
| C 右侧障碍 | **INVALID TEST**：`R` 380–450 mm 有效，但 `dmin_R` 仍为 5.78；外层 R 不在当前神经侧向掩码 |
| D 左右同时 | **AVAILABLE**：走廊 350/400/500/1000 mm；不是左右独立刺激，无法回答方向对置 |
| 左右方向相反 | `INSUFFICIENT_EVIDENCE` |
| 撤除后恢复 | `INSUFFICIENT_EVIDENCE`：没有同一运行的接近/撤除序列 |
| 运行间稳定 | `INSUFFICIENT_EVIDENCE`：每个场景只有一次记录 |
| 真实五路 ToF 域 | `DOMAIN_MISMATCH_EVIDENCE` |

最终 `NO_GO` 的依据不是“数据不够”本身，而是已有 A 场景持续偏转，以及 C 最近右传感器
没有进入神经侧向输入；这两点已足以否定当前 5-ToF → DN → readout 链的直接可演示性。

完整报告：`D1_ACCEPTANCE.md`、`D1_ACCEPTANCE.json`。
逐项原始路径与 SHA 在 JSON 的 `archive_manifest`。

## readout 是什么

- SHA-256：`632cddb1c9ae43de7bde63a5bd8851036009c4625d12c113bf0e96119ace083e`
- `readout_real.npz`：1314 个下行神经元 Trace → 单个标量 turn。
- 训练代码：`loop/train_readout.py`；目标是模仿手写 `teacher_turn =
  clip(0.6*(0.9-dmin_R), -1, 1)`，即贴右墙控制器。
- `ridge`，120 PCs，λ=0.001，CV score=`-0.029744`。
- 它是**模仿手写控制器的线性解码器**，不是天然果蝇运动神经回路，也没有独立左右输出。
- 不能把它的输出解释为“果蝇想左转/右转”。

历史 `loop/P2D29_REPORT.md` 的 L50/R50 方案已是
`FAIL_OFFLINE` / 闭环 `FAIL`，并正式关闭。本目录没有重启、调参、换群或换 readout。

## 回放数据

从仓库根目录原有的未跟踪归档复制为字节完全一致的快照，放在 `data/`：

- `A_OPEN`：`static_open.csv` + `.npz`
- `C_RIGHT_370`：`static_right_370.csv` + `.npz`
- `C_RIGHT_PERP_400`：`static_right_perp_400.csv` + `.npz`
- `D_BOTH_350/400/500/1000`：`static_corr_*.csv` + `.npz`
- 额外：`static_front_1000.*`、`static_fr_front_400.*`
- 冻结 readout：`data/readout_real.npz`、`data/readout_real_sensor.npz`

每个 CSV 的 `brain_turn_raw` 与用冻结 readout 对该行 1314 维 feats 的重新预测一致
（最大误差 < `2e-4`）。这确认回放用的确实是历史神经输出，而不是事后生成的转向。

## 模式

| 模式 | 方向来源 | 用途 |
|---|---|---|
| `BRAIN` | 冻结 readout 的历史原始输出 | 唯一正式回放模式 |
| `ZERO` | 神经转向置零，omega=0 | 对照；左/右轮仍显示固定前进预览 |
| `FLIP` | 仅反转神经输出 | 消融；不得作为正式演示方向 |

- `memory_gain=0`；`pursuit_turn=0`；无地图、无 Dijkstra、无目标方位。
- 无手写 ToF 左右规则产生 omega；ToF 只显示与提供来源。
- 固定前进速度 `0.35 m/s` 是独立工程预览设定。
- 左/右轮速是 `v ± omega*track/2` 的公式预览，`sent=false`，不写任何执行器。
- 安全状态固定为 `REPLAY_ONLY` / `ACTUATORS=DISCONNECTED` / `DRIVE_API=NOT_PRESENT`。

## 页面

顶部有醒目 **REPLAY** 标识。页面显示：

```text
ToF L / FL / F / FR / R
→ Neural activity (1314 DN Trace bins + raw turn)
→ Brain turn
→ Left / Right motor command (PREVIEW ONLY)
→ Safety state
```

页面只做 GET 数据读取；不提供发送运动命令的按钮或 API。B 左障碍在场景列表中明确标为
`NO ARCHIVE` 且不可选。

## 下一项需要用户授权的具体实验范围

本轮 `NO_GO`，不自动换神经群、读出头、增益、滤波器或训练新模型。

若用户决定继续挑战当前冻结路线，唯一可考虑的下一步是一次**新的静态真实五路 ToF 脑响应测试**
（不是正式演示、不是实体车）：

- 显式授权：`NEW_FULL_BRAIN_AUTHORIZATION=True`，且单独批准一次静态实验；仍
  `ROBOT_MOTION_AUTHORIZATION=False`、`TRAINING_AUTHORIZATION=False`。
- 固定现有 MaleCNS、BASE 编码器、冻结 `readout_real.npz`、`memory_gain=0`；不改任何参数。
- A/B/C/D 四场景；左右各自独立；每场景至少 3 次独立运行；每次包含“无遮挡→单侧→撤除→无遮挡”。
- 10 s 静态、只记录原始 5 路 ToF、DN Trace、readout 原始标量；不接电机。
- 预注册门槛：左右符号相反；无遮挡持续偏转 `|mean|<0.05`；撤除后 1 s 内回到基线；
  3 次重复同号率 ≥ 2/3。任一不满足即保持 NO_GO。
- 该测试只能补 B/恢复/稳定性证据，**不能**修复 A 的持续偏置或 C 的外层传感器覆盖问题；
  若仍失败，应单独报告替代路线，等待用户决定。

在上述授权之前不启动完整脑或实体车。

## 边界

```text
NEW_FULL_BRAIN_AUTHORIZATION=False
ROBOT_MOTION_AUTHORIZATION=False
TRAINING_AUTHORIZATION=False
REAL_SYNAPTIC_LEARNING_READY=NO
BIOLOGICAL_VALIDITY=UNVALIDATED
```
