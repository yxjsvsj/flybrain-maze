# P11-2-S3 — 一次性运行授权模板

**STATUS: NOT_AUTHORIZED**
`NEW_S3_EXECUTION_AUTHORIZATION=False`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
Level A/B/C/Biological=`NO-GO`

本文件不是批准。当前没有 `loop/p11_2_s3_authorization.json` 的 `APPROVED` 记录，也没有 S3 生产消耗锁。

## 冻结实验

- 网络：官方完整 MaleCNS 合成 B，KC→KC-zero。
- 靶点：S1R matched-320 KC 全局索引；SHA-256 `6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385`。
- Seeds：201–225（25）。
- 条件：sham / kc_low / kc_primary。
- 每 seed 每条件 400 步；总 30,000 步；预计 75 NPZ。
- 窗口 `[100,300)`。
- primary amount：`0.03625384938440364`（ΔV∞=0.20）。
- low amount：`0.01812692469220182`（ΔV∞=0.10）。
- CPU；原始 LIF/噪声/连接权重；不训练；ΔW=0。
- 900 s / 2048 MiB；一次运行；不重试；不提高上限。

## 授权前必须满足

1. S3-1 静态实现/合成验收通过，独立 commit/push。
2. `p11_2_s3_script_freeze.json` 与 `p11_2_s3_experiment.py` SHA 完全一致
   （当前 `63154938b2d887cccfea7918ae27624e3c105ed5a86aa127b291d65239a2568e`）。
3. `p11_2_s3_authorization.json` 仍为 `NOT_AUTHORIZED`；不得复用 S1R 授权或锁。
4. 用户明确给出新的 S3 一次性批准文本；scope 必须与冻结方案完全一致。
5. 运行前资源 preflight；S3 静态估计：约 413.387 s / 1948.16 MiB，`RESOURCE_PLAUSIBLE_TIGHT`，不是已验证资源。
6. 失败/异常/超时/内存超限一律停止；不自动重跑、不补跑、不提高上限。

## 判别说明

- KC 正向 `+0.10` 门和饱和门未通过 → `INCONCLUSIVE`，不得解释 MBON。
- KC 成功但 MBON 未达到正向 `+0.05`、非显著、或未达到显著负向 `<=-0.05` → **统一 `INCONCLUSIVE_DOWNSTREAM`**，报告真实效应量/方向/p/CI，不得声称“MBON 功能不存在”。
- KC 成功且 MBON 正向门通过 → 仅说明**人工直接驱动下**的功能链；不能证明自然感觉传递、单突触专一因果或生物学习。

## 当前决定

**不批准运行。**
本阶段只完成生产代码、静态核查和合成验收；S3 未运行。
