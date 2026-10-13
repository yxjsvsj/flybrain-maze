# P2 演示：真实 MaleCNS L50/R50 群体最小闭环（虚拟仿真）

> **这是离线虚拟仿真，不是实体车拍摄画面。** 图中标明 `[VIRTUAL SIMULATION]`。

## 一条可复现命令
```bash
python -m loop.p2_demo --out-dir C:\tmp\p2demo --device cuda
```
输出（写到 `--out-dir`，不提交二进制）：
- `demo_success_1001_must_right.png` —— brain seed 1001 / `must_right` / 群体控制，并排对比 **群体 / 置零 / 反号**（同地图、同 seed、同运动包络）：轨迹、群体差分 `D(t)`、`turn_cmd`、`v/omega`、停车状态。
- `demo_failure_808_must_right.png` —— 同参数的**未通过**案例，标注其持续旋转/失去进展处。

## 参考复现值
实测（`loop.p2_demo`）：**seed 1001 / must_right / 群体控制：到达，t_goal = 24.2 s，独立碰撞事件 = 0**。
同条件下置零、反号**均不到达**。

## 正式统计（同时给出，不隐去失败）
| 组 | POP | DN_ema(单DNa02) | POP_zero | POP_swap |
|---|---|---|---|---|
| screen 64/101/202/303 | **1/8** | **2/8** | 0/8 | 0/8 |
| confirm 606/808/1001 | **1/6** | **0/6** | 0/6 | 0/6 |

在线受控刺激（A→左/右障碍→A）：**左右可分**（左→负 turn、右→正 turn，全 screen seed）且**撤除后减弱**；
但**动态转弯仍不稳定**（首转向常对、累计转角大→打圈；confirm 仅 1/6）。置零/反号消融使到达率归零，
说明成功**依赖时变神经活动**。

## 冻结不变
L50/R50 神经元（`loop/population_turn.py`，位置索引 + 全脑神经元 ID + 各自校验和）、Trace τ=0.2、
`gain=8`、`sign=-1`、每 seed 无遮挡 b0 校准、BASE 编码器、`P2Control`、`world.car` 动力学。
**不重选神经元、不训练、不调参、不为适配成功而改迷宫、无 Dijkstra/手写老师。**

索引审计：`L_pos_ck=15ba62fd7381 R_pos_ck=ecd24b8c4759 L_brain_ck=8729a058f8e1 R_brain_ck=165acfaeae37`（不合并）。

## 归因边界（必须区分）
- **MaleCNS 真实连接结构 + 模拟神经活动**：`flybrain` 的连接组与 spiking 脑；本演示只用其**当前/过去**活动。
- **人工选出的 L50/R50 群体**：由实验规则在 screen seeds 上筛出、**固化为常数**，非天然运动回路。
- **人工的神经元→电机映射**：`D(t)=mean(L)−mean(R)`、`gain/sign`、`turn = sign*gain*(D−b0)` 均为**人工设计**，不冒充果蝇天然运动解码。
- **独立的前方安全停车**：`front_safety` 是手写层，与神经转向解耦。
- **虚拟碰撞 ≠ 实体安全**：`world.car` 的圆盘碰撞模型不代表实体防撞外廓。

## 明确不宣称
- ❌ "果蝇脑已学会随机迷宫" / "已具备稳定自主导航"。
- ❌ 把虚拟画面当实体车。
- 本演示只是把**已经完成的神经闭环成果**整理为一个清楚、可复现、可审计的离线样例。

## 相关模块
`loop/population_turn.py`（固化群体 + 在线读出）、`loop/pop_closedloop.py`（在线门控 + 动态对照）、
`loop/p2_demo.py`（本演示）、`loop/test_p2_demo.py`、`loop/test_population_turn.py`。
