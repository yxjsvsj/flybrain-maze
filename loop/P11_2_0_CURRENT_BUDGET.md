# P11-2-0 KC 高发放的理论电流预算与因果实验准入（**静态/数学**）

**基线** `7d5a37e`。**未运行 FlyBrain.step、未创建真实神经动力学、未做单神经元时间演化。**
仅读安装源码、官方 W（只读）、归档报告。**W/ΔW 未修改；Training Authorization=False。**

## 1. 源码更新式（安装的 `flybrain/brain.py`）
```
current = self.synaptic_input(self.fired) * self.gain   # 上一时刻 fired
self.v *= self.decay                                    # decay = exp(-dt/tau)
self.v += current + self.tonic
self.v += (self.rng.random((N,B)) < self.noise_hz*self.dt) * noise_amp
fired = flatnonzero(self.v >= 1.0); v[fired] = 0.0
```
- `decay = exp(-0.020/0.100)`；`gain` 作用于**突触电流**；`tonic` 与 current 同加；噪声为**伯努利**（p=`noise_hz*dt`=0.024），幅度 0.22；阈值 1.0；复位 0；不应期默认 0。
- `tonic` 在 `__init__` 按 dt 重标定；**dt=0.02 时因子为 1，tonic=0.14**。
- `build.py`：`w /= max(incoming[post],1)`（逐突触后 L1）；实际 KC 行 L1 **≈1.0**（mean 1.000000，median 1.0）→ 与运行 W 一致。

## 2. 数学必要条件（无递归/无噪声，或均值噪声）
| 量 | 值 |
|---|---|
| decay | **0.818730753** |
| tonic 单独固定点 | **0.772332**（**<1**） |
| **tonic 单独能否发放** | **否（数学排除）** |
| 从 reset=0 单步发放所需（post-gain） | **0.86** |
| 同一条件的原始 `W@spikes` | **0.286667** |
| 持续亚阈（非发放）固定点所需（post-gain） | 0.041269（raw 0.013756） |
| **每步发放所需突触电流**（post-gain / raw） | **0.86 / 0.286667** |
| 噪声 p / 均值电压 | 0.024 / 0.00528 |
| tonic+均值噪声固定点 | **0.8008（<1，期望上不足）** |
> 以上是**数学必要条件**，不是因果结果；噪声的精确发放概率需时间演化（A1）。

## 3. 完整脑 KC 输入权重预算（官方 W，只读）
- **4,064** 个 KC；输入边总数（KC 行）见报告；**0 个零输入行**。
- **正权重和**：mean **0.883**、median 0.882、**min 0.802**、max 0.957。
- 负权重绝对值：mean 0.1169；**逐后 L1** mean 1.000、median 1.0。
- 边数：mean 204、median 190、min 16、max 670。
- **所有 4,064 个 KC 的正权重和 ≥0.802 ≫ 0.286667** → 静态权重**在原则上**足以让每个 KC 每步发放
  （若足够多的正输入同时发放）；post-gain 最大电流 **≥2.405**（min），远高于 0.86。
- **主要输入类型**（对 KC 行求和）：**KCab +974.7**、**KCg-m +711.3**（**递归 KC→KC 兴奋**）、
  **APL −378.4**（抑制）、KCapbp-ap2 +164.3、KCg-d +124.8、PAM08/DPM 等。
- **实际 `W@spikes` 未知**：P11-1c 未保存全部 166,700 个神经元的逐步 spikes，不能从 KC/MBON 记录重建完整电流。
- **理论最大可能输入 ≠ 真实运行时输入**。

## 4. 噪声与递归
- **tonic 单独**：数学排除（固定点 0.772 <1）。
- **tonic+均值噪声**：期望固定点 0.8008 <1 → **期望上不足**；精确概率/长连续发放需 A1。
- **递归**：KC→KC 正权重和很大（KCab/KCg-m）；静态上**允许**强反馈输入；实际贡献依赖逐步 spikes（UNKNOWN）。
- **候选**：递归 KC→KC 兴奋（gain 3.0）+ 噪声触发 + APL 抑制平衡 + 20 ms 步长上限。
- **`MODEL_ROOT_CAUSE = NOT_ESTABLISHED`**；不得把权重统计/理论界限当作已证实的因果根因。

## 5. 来源勘误（P11-1c 脚本哈希）
- 当前 GitHub 文件 **LF** SHA-256：`e523d68a0b476f7ed5ee19e203c77f27ffe74a6001ec6e0c63dc30487b8a9290`
- 转 **CRLF** 后 SHA-256：`27b121f11f5b5413bfd094a481c0834cf8f9252364e42caf7c4ef8d8caa6fd6d`
- 历史报告 `script_sha256`：`27b121f1...` → **与当前文件的 CRLF 形式一致**。
- **但**该哈希是在 **compare-only 后处理**（实验之后、父进程补丁之后）计算的；运行时的脚本是**补丁前**版本。
  → **哈希一致不能证明运行时源码字节**；`SOURCE_PROVENANCE = PARTIAL` 维持。
- 后处理才计算的哈希不能自动证明实验运行时源码版本；不修改历史报告，仅新增本勘误。

## 6. A1 因果测试准入（仅提案）
- **P11-2-A1**：无递归单神经元对照；原参数；**噪声关/开**；预固定 100 realizations × 10,000 步；
  判据：tonic-only 期望 0 发放；tonic+noise 期望稀疏、非连续；报告每步发放概率与最长连续发放。
- 目的：**只验计算机制**（tonic/noise vs 递归），**不**寻找让 KC 好看的参数，**不**声称生理标定。
- **A1 若需任何新的时间演化仿真，必须先获得用户单独批准**；A2/A3（零递归/重排 W）与阶段 B 敏感性 **NO-GO**。
- 详见 `P11_2_A1_ONE_TEST_APPROVAL.md`。

## 7. 裁决
| 项 | 结果 |
|---|---|
| TONIC_ALONE_REACHES_THRESHOLD | **NO**（0.772 <1） |
| EVERY_STEP_RAW_Wx_THRESHOLD | **0.286667** |
| STATIC_WEIGHTS_SUFFICIENT_IN_PRINCIPLE | **YES**（4,064/4,064 KC pos_sum ≥0.2867） |
| ACTUAL_Wx_CONTRIBUTION | **UNKNOWN**（无完整 spike 历史） |
| MECHANISMS_MATHEMATICALLY_EXCLUDED | **tonic 单独**；**tonic+均值噪声**（期望固定点 0.8008 <1） |
| CANDIDATE_MECHANISMS | 递归 KC→KC 兴奋、噪声触发、APL 抑制平衡、步长上限 |
| MODEL_ROOT_CAUSE | **NOT_ESTABLISHED** |
| SOURCE_PROVENANCE | **PARTIAL** |
| A1_WORTH_APPLYING | **YES**（单独一次授权；低成本、可区分 tonic/noise vs 递归） |
| A2/A3/B | **NO-GO** |
| REAL_SYNAPTIC_LEARNING_READY | **NO** |
| Level A/B/C/Biological | **NO-GO** |
| **Training Authorization** | **False** |

**已停止：未启动 A1/A2/A3/B；未更新真实 ΔW；未训练、Dijkstra、迷宫、B/C 或实体控车。**
