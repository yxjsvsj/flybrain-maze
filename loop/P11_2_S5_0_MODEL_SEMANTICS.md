# P11-2-S5-0 — FlyBrain LIF 模型与神经驱动语义审查

基线：`68dca94`
状态：`S5_0_SOURCE_SEMANTICS_COMPLETE; NO_NEW_DYNAMICS`
没有调用 `FlyBrain.step`、没有完整脑采样、没有新剂量试验。

源码文件：`C:\AA_project\guoying\.venv\Lib\site-packages\flybrain\brain.py`
SHA-256：`142ecfffd5d32e30655829e34b38b04024e49beae662bd078279e9610c789b0c`

## 1. 实际 LIF 执行语义

`FlyBrain.step()` 实际顺序（源码行号）：

```text
current = synaptic_input(previous fired) * gain      # 179
v *= decay                                           # 180
v += current + tonic                                 # 181
v += (rng.random((n,batch)) < noise_hz*dt) * noise_amp  # 182
[optional eye_drive]                                 # 184-185
for idx, amount in inject: v[idx] += amount           # 186-187
[refractory zeroing if enabled]                      # 188-189
fired = flatnonzero(v >= 1.0)                        # 190
v.ravel()[fired] = 0.0                               # 191
last_spike[fired] = steps                             # 192-193
steps += 1
return fired indices only                            # 198
```

`reset(seed)` 行 131/134：`v=0`、`rng=default_rng(seed)`、`fired=empty`、`steps=0`；S3 的 `run_condition` 对每个条件都调用 `reset(seed)`，因此 sham/low/primary 的噪声序列从同一起点出发。

### v 的数值定义和单位

- `v` 是**无量纲的模型膜电位**，阈值固定为 `1.0`，发放后重置为 `0.0`。
- 源码没有 mV 转换、没有电容/电导单位，也没有把 `v` 映射到真实膜电位。
- 因此 `amount=0.0362538494` 是 **v 的增量**，不是 nA、pA 或真实电压。

### leak / dt / tau / tonic / gain / noise

- `dt=0.020 s`，`tau=0.100 s`，`decay=exp(-dt/tau)=0.8187307530779818`。
- `tonic=0.14`；`__init__` 会按 dt 重标定，dt=0.020 时仍为 0.14。
- `gain=3.0`：先对上一 step 的发放求稀疏输入，再乘 gain。
- 噪声不是高斯：每 step 每个神经元抽一个 Bernoulli 掩码，`p=noise_hz*dt=1.2*0.020=0.024`，命中时加 `noise_amp=0.22`；期望每步约 0.00528。
- 以上是模型工程参数，不是从本数据集测得的生物膜参数。

### W 的方向、符号、归一化

- CPU 路径把权重存为 CSC；`_propagate` 按“spiked 神经元”的列累加到 post-synaptic 行，语义是 `W[post, pre]`。
- 权重是 signed float32；step 中**没有 abs、裁剪或归一化**。
- 输入总电流 = `gain * sum(W[post, pre] for fired pre)`；单位仍是模型单位，不是 nA。
- 连接组提供的是拓扑/权重结构，不等于已校准的突触电导。

## 2. inject 的索引、时间与单位

- 索引是完整脑 `v` 的全局 flat row indices（S3 用 S1R matched-320 KC 的全局索引）。
- 注入发生在 leak、tonic、noise（及可选 eye）之后，在 refractory、阈值、reset 之前。
- `amount` 是电压增量（v 单位），每步注入；不是突触电流。
- S3 冻结：primary `0.03625384938440364`，low `0.01812692469220182`，sham 无注入；窗口 `[100,300)`。

## 3. 静态无发放数学核查

忽略突触输入、噪声、发放和 reset，常数注入 `a`：

```text
v_inf = (tonic + a) / (1 - exp(-dt/tau))
```

代入 `dt=0.020, tau=0.100, tonic=0.14`：

| 条件 | a | v_inf |
|---|---:|---:|
| sham | 0 | 0.772332 |
| low | 0.01812692469220182 | 0.872332 |
| primary | 0.03625384938440364 | 0.972332 |

与预期 `0.7723 / 0.8723 / 0.9723` 一致（差异来自精确常数）。
但这三条都 **低于阈值 1.0**；这是“无发放近似”的固定点，不能直接预测发放率增量。噪声、网络输入和 reset 会改变实际结果。

S3 实测 KC rate 增量：

- primary `+0.022858125`；
- low `+0.011631875`；
- low/primary 观测比 `0.508872666`，接近 ΔV 比 0.5，但只有两档，不能外推。

理论电压变化与观测发放率变化必须分开；二者不是同一量。

## 4. 间接网络反馈

即使 B 网络直接 KC→KC 边为 0，静态结构上仍存在大量 **KC→non-KC→KC** 两跳通路：

- 直接 KC→KC（B）：`0`；
- 经 non-KC 的 2-hop：**16,216,622 个聚合端点对 / 125,657,209 条独立路径**（静态可及性，不代表动态影响；更长路径未穷举）。

因此“删除 KC→KC 直接边”不等于“消除所有网络反馈”。实际是否存在状态驱动反馈必须由电压/电流轨迹验证，不能仅由静态路径断言。

## 5. 当前 API 的可观测性

- `brain.v`：step 后可读；对发放细胞是 **reset 后**的 0，不是发放前电压。
- `brain.fired`：只保存最近一 step 的发放索引。
- `brain.synaptic_input(brain.fired)`：可重算最近一步的**聚合突触电流**，但无历史、无分量拆分。
- step 不保存 leak、tonic、noise 掩码、inject 的逐项贡献；发放前的 pre-reset v 丢失。
- `brain.rng` 可读状态，但噪声抽样值没有记录。

**结论：现有 API 本身不足以提供可靠的发放前电压/各输入贡献轨迹；需要 instrumentation。**

## 6. 生物真实性证据表

| 组件 | 源码实际假设 | 生物依据 | 证据等级 | 可支持 | 不可支持 |
|---|---|---|---|---|---|
| dt=0.020, tau=0.100 | LIF 离散化、fly64 风格 | 源码只作模型比较，未用本数据校准 | UNVALIDATED_FOR_THIS_MODEL | 模型内时间常数 | 真实膜时间常数 |
| threshold=1.0, reset=0.0 | 归一化 LIF | 无实验校准 | ENGINEERING_NORMALIZATION | 模型内 spike 定义 | mV 阈值/真实 spike threshold |
| tonic=0.14, gain=3.0 | 手工调参 | 源码明确写 hand-calibrated, not measured | ENGINEERING_HAND_CALIBRATED | 模型内基线/增益 | 生物 tonic/gain |
| noise_hz=1.2, noise_amp=0.22 | Bernoulli 噪声 | 手工调参，非测量 | ENGINEERING_HAND_CALIBRATED | 模型内噪声统计 | 生物噪声率/幅度 |
| 稀疏 W | 官方连接组 signed weights | 只有解剖结构；step 无归一化 | CONNECTOME_STRUCTURE_ONLY | 模型内拓扑/权重 | 真实突触电导/符号/幅度 |
| inject | v 直接增量 | 工程刺激，无 current-clamp 校准 | ENGINEERING_STIMULUS | 模型内电压扰动 | 真实电流/光遗传剂量等价 |

“有真实连接组”不等于“LIF/噪声/阈值参数具有真实生物校准”。

## 7. S3 数据能证明/不能证明

**能证明（模型内）：**

- 对 matched-320 KC 每步注入 0.0362538494（ΔV∞=0.20）使 `[100,300)` 群体发放率增加 `+0.022858`；
- 低剂量产生较小但显著的正向效应；
- 该效应在 25/25 seeds 同向，广泛分布于 311/320 matched KC，非饱和，非少数细胞主导。

**不能证明：**

- 自然感觉通路能产生同样驱动；
- 该电压增量等价于真实生物电流/光遗传剂量；
- 单条 KC→MBON 突触的专一因果；
- MBON 下游功能；
- 生物学习或功能结论。

## 8. 未来电压轨迹观测方案（只设计）

建议未来新增独立的 `InstrumentedFlyBrain` 包装/子类（不改已安装包），在**完全复现 step 顺序**的前提下，只记录预注册小集合：

- matched-320 KC；
- matched-95 MBON；
- 预先由静态结构选定的中继细胞（排除事后挑选；记录选择规则与 SHA）；
- 可选 LPLC2/LC4 入口。

每 step 记录：

- 发放前 `v`（step 开始）；
- leak 后 `v`；
- 突触输入总量；
- tonic、noise 抽样、inject；
- reset 前 `v_pre_threshold`；
- reset 后 `v`；
- `fired` 与 `1-v_pre_threshold` 阈值距离。

内存设计：只记录目标/中继细胞，不保存 166,700 全脑 v 历史；按 step 直接写二进制日志。若记录约 1,000 个细胞 × 400 步 × 约 10 个 float32 字段，约 16 MB 量级；若只用 415 个 KC/MBON，约 6–8 MB。
这低于当前约 115 MiB 的剩余内存余量，但**未经实测**；instrumentation 的 Python/日志开销对 900 s 的影响也 **UNVALIDATED**。必须先在小规模单元测试和短静态/合成步进中验证 step 等价性，不能直接用于正式 S3。

## 9. 下一阶段选择

**优先 B：先修正模型解释/验证驱动语义，不立即预注册新剂量实验。**

理由：

- 直接注入是 v 增量，不是实验电流；
- 阈值/重置/tonic/gain/noise 都是工程或手工参数；
- 现有 S3 数据没有 v/current 轨迹，无法区分阈值、reset、网络反馈；
- 用两档近似比例外推更强剂量会产生不可解释的人工刺激。

A（新的 KC 可驱动性/剂量实验）只有在完成电压轨迹/模型语义验证后，才可重新预注册；剂量必须先固定并论证，不得事后选取。

### 可证伪假设

- H1 阈值：多数无响应 KC 在阈值下，响应细胞接近阈值；若轨迹相反则拒绝。
- H2 重置：post-spike reset 限制速率；若轨迹/剂量关系不符则拒绝。
- H3 网络：未注入 KC 的负向变化来自共享网络；若隔离/轨迹不支持则拒绝。
- H4 时间聚合：响应是短瞬态被窗口掩盖；S4 分箱已不支持。

### 停止条件

- 没有可验证的电压/电流轨迹；
- 注入语义与实际生物驱动映射不一致；
- 超出 900 s / 2048 MiB；
- KC 饱和或剂量响应非单调；
- 未经新预注册的更强剂量外推。

## 10. S4 报告勘误

S4 Markdown 的 full-4064 KC 负向细胞数已从 `2734` 修正为 **`2737`**；other-3744 KC 保持 **`2734`**。

核对：

```text
positive: 311 + 835 = 1146
negative:   3 + 2734 = 2737
zero:       6 + 175 = 181
```

S4 JSON 数值和 S3 正式主终点未改。

## 11. 五个问题

1. **实际 LIF 语义是否与历史预注册描述一致？**
   在代码层面一致：S3 的 amount 确实作为 v 直接增量，公式与冻结量匹配；但这是模型语义，不是生物电流语义。

2. **哪些参数具有生物实验依据？**
   没有经过本数据集校准的阈值、reset、tonic、gain、noise、tau；W 只有连接组结构依据。源码明确说 tonic/gain/noise 是手调而非测量。

3. **S3 发放率数据能证明什么、不能证明什么？**
   能证明模型内直接电压注入使 matched KC 显著增加 +0.022858、广泛且非饱和；不能证明自然感觉驱动、真实电流等价、单突触因果、MBON 功能或学习。

4. **是否具备可靠的发放前电压/输入轨迹采集方案？**
   现有 API 不足；需要 instrumentation。小内存目标细胞日志方案可行但未实现、未验证，不能直接用于正式运行。

5. **下一次新实验是否值得预注册，还是先修正模型解释？**
   **先修正模型解释/验证驱动语义（B）**；不建议现在直接预注册更强刺激实验。

## 12. 边界

```text
NEW_FULL_BRAIN_AUTHORIZATION=False
TRAINING_AUTHORIZATION=False
REAL_SYNAPTIC_LEARNING_READY=NO
Level A/B/C/Biological=NO-GO
Robot/Maze/Dijkstra/DeltaW=NO-GO
```

未重跑 S3，未提高剂量，未训练，未修改安装的 flybrain 包，未执行完整脑采样。
