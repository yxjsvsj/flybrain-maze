# P11-2-S5-3 — 小范围完整脑机制观测预注册

基线：`1855e91`
协议：`P11-2-S5-3/v1`
状态：`PREREGISTRATION_COMPLETE; NOT_AUTHORIZED; NO_RUN`

本实验只回答：S3 中 matched-320 KC 接受直接膜电位注入后，为什么群体发放率只增加 `+0.022858125`，未达到 `+0.10`。
不改变 S3 的正式结论：`INCONCLUSIVE_KC_NOT_DRIVEN`。

## 冻结设计

- 官方完整 MaleCNS；合成 B，仅删除 642,933 条 KC→KC 直接边。
- seeds **301–305**（5）。
- 条件：`sham`、`kc_low`、`kc_primary`。
- 每 seed 每条件 400 步；总 **6,000 步**、**15 组**。
- 窗口 `[100,300)`。
- KC 靶点：S1R matched-320，SHA-256：
  `6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385`
- MBON 靶点：matched-95，SHA-256：
  `2248dccc680981af16270f12f6365ca5832fab2a12bb5ace3b80cc73580b61a6`
- 联合 415 目标（KC 排序后接 MBON 排序）SHA-256：
  `942d87d267711edf943b0804ab9089ab23cec7532f017d10ebf81c3ec4c12567`
- 仅记录这 415 个目标；中继细胞暂不记录。
- `kc_low`：`0.01812692469220182`/步；`kc_primary`：`0.03625384938440364`/步；sham 无注入。
- 不注入 MBON/感觉细胞。
- CPU、batch=1；原 LIF/权重/噪声/reset。
- 900 s / 2048 MiB；一次运行；不重试；不提高上限。

## 冻结读数

全部为描述性，不建立新的生物功能显著性：

- sham/low/primary 的 KC pre-threshold v 与 threshold distance 分布；
- 距离阈值 `<0.05`、`<0.10`、`<0.20` 的细胞比例；
- 每步 synaptic（含 gain）、tonic、Bernoulli noise、inject；
- fired/reset 与 threshold distance 对应；
- KC/MBON 在 `[100,300)` 的发放率；
- pre/post/reset 与贡献分项闭合。

5 seeds、415 细胞、400 步不是大量独立 seed；细胞和时间步不得冒充独立样本。
不得从结果中挑选更强剂量，不声称生物突触传递或学习。

## 轨迹格式

`S5TRACE/1`，90-byte/record；字段与顺序：

```text
step, neuron,
v_start, v_leak, synaptic, tonic, noise, eye, inject,
v_pre_threshold, v_post_reset, threshold_distance,
fired, refractory_zeroed
```

每组 `{seed}_{condition}.bin` + `.bin.json` 头（目标顺序/哈希、字段、dtype、单位、记录数、SHA-256）。
目标顺序固定为联合 415 索引；不允许按响应筛选。

## 资源

单组 415 目标 ×400 步：

- 14.247894 MiB；写 writer-only 约 0.0409 s（S5-2 实测）。

15 组算术：

- 原始轨迹约 **213.718 MiB**；
- 备份后约 **427.437 MiB**；
- writer-only 时间约 **0.61 s**。

S3 实测 Job peak `1932.29 MiB`；新 InstrumentedFlyBrain 的额外峰值/时间、完整脑等价性均为 **UNVERIFIED**。
日志逐组流式写盘，不同时保留 15 组；完整脑真实资源不能由此声称已验证。

## 授权

当前：
```text
EXECUTION_AUTHORIZATION=NOT_AUTHORIZED
NEW_FULL_BRAIN_AUTHORIZATION=False
TRAINING_AUTHORIZATION=False
BIOLOGICAL_VALIDITY=UNVALIDATED
```

必须使用独立 S5-3 父入口、独立授权、独立一次性锁和命名 Windows Job；不得复用 S3/S1R 授权或锁。

## 解释边界

即使执行，结果只说明该完整脑模型在该直接电压刺激下的内部机制，不证明自然感觉传递、单突触因果、生物学习或 Level A/B/C。
