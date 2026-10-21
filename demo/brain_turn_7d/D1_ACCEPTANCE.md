# DEMO-BRAIN-TURN-D1 — 第一关验收

**基线** `1855e91`
**判定** `NO_GO`
**readout** `632cddb1c9ae43de7bde63a5bd8851036009c4625d12c113bf0e96119ace083e`

本轮只读归档与冻结 readout；未创建完整脑、未跑 MaleCNS 动力学、未训练、未启动电机。

## 结论

现有真实五路 ToF 归档**不足以证明**左右相反方向；且已有数据已经显示无遮挡持续偏转、右侧最近传感器未进入神经侧向输入。按第一关规则判 **NO_GO**，不是 CANDIDATE_GO。

## A/B/C/D

| 场景 | 归档 | 神经转向均值 | 正/负比例 | 持续同号 | 关键限制 |
|---|---|---:|---:|---:|---|
| A 无遮挡 | `static_open.csv` | -0.2773 | 0.035/0.966 | 8.48s | archived static |
| B 左侧障碍 | MISSING | - | - | - | 无左向归档；未合成 |
| C 右侧墙 370mm | `static_right_370.csv` | -0.2720 | 0.045/0.955 | 8.48s | outer R valid but not in neural side |
| C 右侧墙垂直 400mm | `static_right_perp_400.csv` | -0.2683 | 0.045/0.955 | 8.48s | outer R valid but not in neural side |
| D 双侧走廊 350mm | `static_corr_0350.csv` | +0.0710 | 0.739/0.261 | 1.01s | archived static |
| D 双侧走廊 400mm | `static_corr_0400.csv` | +0.0480 | 0.693/0.307 | 0.71s | archived static |

B（左侧障碍）没有归档。TOF 链的 120° 神经避障 FOV 只覆盖 `FL/F/FR`；外层 `L(+70°)`、`R(-70°)` 不进入 DN 侧向输入。C 的 `R` 380–450mm 有效，但 `dmin_R` 仍为 5.78，所以 C 不是有效的“右障碍神经响应”测试。

## 五个问题

1. 左右刺激是否相反？ **INSUFFICIENT_EVIDENCE** —— no left-only archive; right-only archives do not enter the neural side input。
2. 无障碍是否持续偏转？ **FAIL** —— 均值 `-0.2773`，负向 `96.5%`，最长同号 `8.48s`。
3. 撤除后能否恢复？ **INSUFFICIENT_EVIDENCE** —— all archived scenes are constant; no readout_real approach/removal sequence exists。
4. 不同运行是否稳定？ **INSUFFICIENT_EVIDENCE** —— one recording per scene/class; no repeated independent runs of the same condition。
5. 真实五路 ToF 域偏移？ **DOMAIN_MISMATCH_EVIDENCE** —— readout is a simulated hand-teacher imitation; real outer R is valid but excluded from the neural right-side input; real open-loop output is strongly biased。

## readout 身份

- `C:\AA_project\guoying\demo\brain_turn_7d\data\readout_real.npz` / `632cddb1c9ae43de7bde63a5bd8851036009c4625d12c113bf0e96119ace083e`。
- `ridge`，120 PCs，λ=0.001，CV score=-0.029744。
- 目标：`clip(0.6 * (0.9 - dmin_R), -1, 1) for scripted follow_side=R`。
- 它是模仿手写贴墙控制器的线性解码器，**不是**天然果蝇运动神经回路，也没有独立左右输出。
- P2d-29 的 L50/R50 方案仍为 FAIL_OFFLINE/FAIL，已关闭；本报告没有重启或调参。

## 回放

`python -m demo.brain_turn_7d.replay` 启动本地只读回放。BRAIN 使用冻结 readout 的原始输出；ZERO 置零；FLIP 仅为消融反转。左/右轮速是公式预览，不发送，界面没有驱动/手动驱动/Arduino 接口。

## 下一项需要授权的范围

仅当用户明确批准时，才可做一次 `D1-NEXT-STATIC-TOF`：固定现有 MaleCNS、BASE 编码器、`readout_real.npz`、`memory_gain=0`；A/B/C/D 各 3 次独立、每次 10 s、“无遮挡→单侧→撤除→无遮挡”；只记录 ToF/DN/readout，不接电机、不训练。预注册门槛为左右反号、无遮挡 `|mean|<0.05`、撤除 1 s 内回基线、重复同号 ≥2/3。该测试只能补 B/恢复/稳定性，不能修复 A 的持续偏置或 C 的外层传感器覆盖；未获授权前不运行。

## 限制

- No new MaleCNS/FlyBrain run was performed.
- No natural motor program is claimed; the readout is a linear imitation of a hand controller.
- B-left and recovery/stability are missing or insufficient; no result was fabricated.
- P2d-29 L50/R50 route remains FAIL and closed; it is not reused as a success.

`FULL_BRAIN=NO`, `MOTOR=NO`, `TRAINING=NO`, `BIOLOGICAL_VALIDITY=UNVALIDATED`。
