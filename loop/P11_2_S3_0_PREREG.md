# P11-2-S3-0 — KC→MBON 功能链因果实验预注册

基线：`8461d91`
状态：`PREREGISTRATION_COMPLETE; NOT_AUTHORIZED; NO_RUN`
`NEW_FULL_BRAIN_AUTHORIZATION=False`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
Level A/B/C/Biological=`NO-GO`

本文件只是设计/预注册；没有运行 FlyBrain.step、没有初始化完整 MaleCNS、没有创建 S3 授权或锁。

## 1. 冻结的既有结论

保留正式 S1R：

- 25/25，`RUN_COMPLETE`；
- entry `+0.066634727`，p `5.96e-8`；
- matched-320 KC `-0.001430625`，p `0.352361`，CI `[-0.004300078, +0.001516953]`；
- matched-95 MBON `-0.001760000`，p `0.438998`；
- 正式分级 `SENSORY_TRANSMISSION_NOT_DETECTED`。

保留 S2：320/320 matched KC 有静态两跳感觉通路；但有效感觉驱动未得到验证。
不得修改 S1R 结果、主终点或统计门槛。

## 2. 实验设计

- 网络：官方完整 MaleCNS 的合成 B（KC→KC-zero），原 LIF/噪声/权重语义。
- 全新 seeds：**201–225**（25）。
- 条件：**sham / kc_low / kc_primary**（每 seed 每条件 400 步）。
- 总步数：3 × 25 × 400 = **30,000**。
- 主窗口：`[100,300)`；刺激窗内每步注入，窗口外不注入。
- `sham`：不注入任何额外电流。
- `kc_low`：matched-320 KC，低剂量。
- `kc_primary`：matched-320 KC，唯一主要剂量。
- 不注入 sensory、MBON 或其他细胞。

### 2.1 靶点冻结

使用 S1R 的 matched-320 KC 全局索引；不重新选择、不按结果换靶点。

- 数量：320，唯一；
- target set SHA-256（i64 little-endian）：
  `6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385`
- 这 320 个也是 S2 中全部具有静态两跳感觉入口的 KC。

### 2.2 注入语义与剂量

`FlyBrain.step(inject=[(indices, amount)])` 直接把 `amount` 加到膜电位 `v`；它不是 nA 电流。

LIF：

```text
v <- exp(-dt/tau) v + gain*W@spikes + tonic + noise + inject
v >= 1 -> spike, reset 0
dt=0.020, tau=0.100, tonic=0.14
```

对每步固定注入 `a`，忽略噪声/阈值时：

```text
Delta_v_inf = a / (1 - exp(-dt/tau))
```

S1R 参考：`DV=0.20` → `a=0.03625384938440364`。

S3 冻结剂量：

| 条件 | Delta_v_inf | 每步注入量 |
|---|---:|---:|
| `kc_primary` | **0.20** | `0.03625384938440364` |
| `kc_low` | 0.10 | `0.01812692469220182` |

`kc_primary` 是唯一主要剂量；`kc_low` 只作剂量相关性的次级分析。
没有通过完整脑试跑选剂量；没有事后加剂量。

**饱和风险**：直接、同步驱动 320 个 KC 可能推到阈值/饱和。预注册的饱和门为：

- primary 的 matched KC 平均发放率 **< 0.85**；
- 且 per-cell primary 均值 ≥0.95 的 matched KC 比例 **≤0.10**。

任一失败 → `INCONCLUSIVE_KC_SATURATED`，不得用下游结果做功能结论。

## 3. 主终点与判别逻辑

### 3.1 KC 干预成功

- 指标：matched-320 KC 在 `[100,300)` 的 firing fraction，`kc_primary - sham`；
- 单位：seed；25 个配对 seed；
- 正向阈值：**+0.10**；
- 检验：paired exact sign-flip（双侧），alpha 0.05；
- 95% paired bootstrap 10,000（seed 20261019），下界 >0；
- 饱和门通过；
- 只有全部满足，才允许解释 MBON。

### 3.2 下游 MBON

仅在 KC 干预成功后解释：

- matched-95 MBON，`kc_primary - sham`；
- 正向阈值 **+0.05**；
- exact sign-flip p<0.05；bootstrap CI 下界 >0。

### 3.3 结论树

```text
KC effect < +0.10 或 p>=0.05 或 CI 含 0
    -> INCONCLUSIVE_KC_NOT_DRIVEN（不解释 MBON）

饱和门失败
    -> INCONCLUSIVE_KC_SATURATED（不解释 MBON）

KC 成功 + MBON 正向通过
    -> KC_MBON_FUNCTIONAL_UNDER_DIRECT_DRIVE
       （人工直接驱动下下游功能存在；自然感觉限制仍只是 supported）

KC 成功 + MBON 正向但 p/CI 不显著
    -> INCONCLUSIVE_DOWNSTREAM

KC 成功 + MBON 显著负向（<=-0.05, p<0.05, CI 上界<0）
    -> KC_MBON_NEGATIVE（负向/抑制性，不是正向功能链）

KC 成功 + MBON 非正非负显著
    -> KC_MBON_NOT_ACTIVATED（人工驱动下未激活，不能外推自然）
```

### 3.4 多重比较

- 主分析是层级门控：只有 KC 成功才检验 MBON；不做“换终点直到成功”。
- `kc_low` 的 KC/MBON 两个次级终点用 BH 校正；不能替代 primary。
- 细胞级结果只作描述性分析，不做细胞级 p 值挑选。

## 4. 资源估计

| 项目 | 值 |
|---|---:|
| S1R 参考 | 20,000 步 / 275.591 s / 1943.16 MiB |
| S3 | 30,000 步 / 3 条件 × 25 seeds × 400 |
| 线性时间估计 | **413.387 s** |
| 峰值估计 | **~1948.16 MiB**（S1R + 保守 5 MiB） |
| 硬上限 | 900 s / 2048 MiB |
| 估计余量 | ~99.84 MiB |
| 结论 | `RESOURCE_PLAUSIBLE_TIGHT`，`RESOURCE_VERIFIED=NO` |

**是否值得申请一次独立 S3 运行？**
**YES，但有条件**：估计在硬上限内，但 2 GiB 余量很小，且线性时间/峰值只是静态估计，不是已验证资源安全。申请前仍需资源 preflight；不得自动提高上限或重试。

## 5. 风险矩阵

| 风险 | 处理 | 级别 |
|---|---|---|
| KC 剂量太弱 | low/primary 预注册；KC 门失败→INCONCLUSIVE | 中 |
| KC 饱和/天花板 | 均值<0.85 + ≥0.95 比例≤0.10；失败→INCONCLUSIVE | 高 |
| 人工同步强刺激造成虚假功能结论 | sham、25 配对 seeds、双剂量、KC 门；只声称人工驱动下功能 | 高 |
| 非特异网络效应 | sham + 固定靶点；不做自然/学习外推 | 中 |
| 时间/内存超限 | 900/2048、单次、无重试/无上限提升 | 高 |
| 多重比较 | 层级 KC→MBON；低剂量 BH；不做细胞挑选 | 中 |
| MBON 结果模糊 | `INCONCLUSIVE_DOWNSTREAM`，不声称功能 | 中 |

## 6. 必须判为 INCONCLUSIVE 的结果

- KC 未达到 +0.10 或 p≥0.05 或 CI 含 0；
- KC 饱和门失败；
- MBON 正向但 p/CI 不显著，或既非正向也非负向显著；
- 缺失 seed/NPZ、完整性失败、资源超限/超时；
- 任何未达到预注册阈值的下游结果。

## 7. 五个问题

1. **靶点和电流如何提前冻结？**
   使用 S1R matched-320 的固定全局索引（SHA-256 已记录）；主剂量 `ΔV∞=0.20`（0.0362538494/步），低剂量 `ΔV∞=0.10`（0.0181269247/步）。不试跑选剂量。

2. **哪项结果真正区分上下游？**
   只有“KC 干预成功 + MBON 正向/负向/不激活”的层级结果能区分：KC 成功而 MBON 不响应支持下游功能受限；KC 成功且 MBON 响应支持人工驱动下下游功能存在。

3. **如何避免人工强刺激造成虚假功能性结论？**
   sham + 配对 seeds、低/主双剂量、KC 成功门、饱和门、显式限定“人工同步直接驱动”，不声称自然通路或学习已成立。

4. **在硬上限下是否值得申请一次 S3？**
   值得申请一次，但只能是 `RESOURCE_PLAUSIBLE_TIGHT` 的 30,000 步设计；约 413 s / 1948 MiB，余量约 100 MiB。资源未验证，必须 preflight，失败不补跑、不提高上限。

5. **哪些结果必须判 INCONCLUSIVE？**
   KC 未驱动、KC 饱和、MBON 模糊/不显著、缺失数据、资源/超时失败，以及任何不满足预注册门的退出。

## 8. 边界

- 本轮未运行任何完整脑、未创建 S3 授权/锁、未训练、未更新 ΔW、未运行 Dijkstra/迷宫/B/C/实体车。
- 不修改历史 S1R 数据/授权锁、官方连接组、冻结脚本、P1/readout/tags 或三个既有无关 dirty 文件。
- 候选干预只证明人工直接驱动下的功能，不外推自然感觉通路、学习或生物功能。
