# P11-2-S5-3 — 完整脑机制观测授权模板

**STATUS: NOT_AUTHORIZED**
`NEW_FULL_BRAIN_AUTHORIZATION=False`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
`BIOLOGICAL_VALIDITY=UNVALIDATED`
Level A/B/C/Biological=`NO-GO`

本文件不是批准。当前没有 S5-3 `APPROVED` 授权，也没有 S5-3 生产锁。

## 冻结候选

- 官方完整 MaleCNS，合成 B，仅删除 642,933 条 KC→KC 直接连接。
- seeds 301–305（5）；条件 sham/kc_low/kc_primary；每 seed 每条件 400 步。
- 总 6,000 步、15 组。
- 窗口 `[100,300)`。
- KC 靶点 SHA `6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385`；
  MBON 靶点 SHA `2248dccc680981af16270f12f6365ca5832fab2a12bb5ace3b80cc73580b61a6`；
  联合 415 目标 SHA `942d87d267711edf943b0804ab9089ab23cec7532f017d10ebf81c3ec4c12567`。
- low `0.01812692469220182`，primary `0.03625384938440364`，sham 不注入。
- 只记录 matched-320 KC + matched-95 MBON，共 415；不记录中继细胞。
- 仅 CPU、batch=1；原 LIF/权重/噪声/reset。
- 900 s / 2048 MiB；一次运行；不重试；不提高上限。

## 科学目的

仅作描述性机制观测：阈值距离、pre-threshold 电压、突触/tonic/noise/inject 分量、发放/reset、KC/MBON 变化。
5 seeds、415 细胞、400 步不能作为大量独立样本；不得产生新的生物功能显著性结论，不得改变 S3 的 `INCONCLUSIVE_KC_NOT_DRIVEN`。

## 授权前必须满足

1. S5-3 合成验收通过，独立 commit/push。
2. experiment/tracer/flybrain source SHA、target hashes 与 freeze 一致。
3. `p11_2_s5_3_authorization.json` 仍为 `NOT_AUTHORIZED`；不得复用 S3/S1R 授权或锁。
4. 用户明确给出新的 S5-3 一次性批准文本，scope 与冻结方案一致。
5. 使用独立 S5-3 父入口和 2048 MiB 命名 Job；无自动重试。
6. 不接入 S1R/S3 历史父入口，不修改历史结果/归档。

## 当前决定

**不批准运行。**
本轮只完成预注册、代码和合成验收；官方完整脑动力学未执行。
