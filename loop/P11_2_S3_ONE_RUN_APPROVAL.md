# P11-2-S3 — 一次性运行授权与执行记录

**STATUS: APPROVED; CONSUMED; RUN_COMPLETE (INCONCLUSIVE_KC_NOT_DRIVEN)**

`NEW_S3_EXECUTION_AUTHORIZATION=APPROVED_ONE_RUN; CONSUMED`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
Level A/B/C/Biological=`NO-GO`

## 授权

- 基线：`52423b2`
- 授权文件：`loop/p11_2_s3_authorization.json`
- 状态：`APPROVED`，`approved_by=user`
- checksum：`259da39ce09dff24c4f61b34a0a133e5bdd4f2d4b91407f7dfaf0e44cacdbad7`
- 子实验 freeze SHA：
  `63154938b2d887cccfea7918ae27624e3c105ed5a86aa127b291d65239a2568e`
- 一次性锁：
  `loop/p11_2_s3_authorization_locks/259da39ce09dff24c4f61b34a0a133e5bdd4f2d4b91407f7dfaf0e44cacdbad7.consumed.json`
- 第二次执行：`REFUSED / AUTHORIZATION_ALREADY_CONSUMED`

## 冻结方案

官方完整 MaleCNS 166,700 神经元；合成 B，删除 642,933 条 KC→KC；靶点 S1R matched-320 KC（SHA-256 `6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385`）；seeds 201–225；条件 sham/kc_low/kc_primary；每 seed 每条件 400 步；总 30,000 步；窗口 `[100,300)`；`kc_primary` `0.03625384938440364`，`kc_low` `0.01812692469220182`；CPU；900 s / 2048 MiB；一次运行、不重试。

## 执行结果

- 完成度：`25/25 seeds; 75/75 NPZ`
- 子进程：`CHILD_EXIT_OK`，exit `0`
- 墙钟：`444.561 s`（上限 900 s）
- Job peak：`1932.29 MiB`（上限 2048 MiB）
- 统计前/后 NPZ 哈希：通过；`post_stats_hashes_ok=True`
- report/run_meta/manifest：`RUN_COMPLETE`；科学判别：`INCONCLUSIVE_KC_NOT_DRIVEN`
- `scientific_go=False`，`downstream_interpretable=False`

### KC 干预

- primary−sham：`+0.022858125`
- exact p：`5.960464477539063e-08`
- 95% CI：`[+0.019925500, +0.025606891]`
- 预注册正向门槛：`+0.10` → **未达到**
- 饱和门：primary mean `0.2238375 < 0.85`；per-cell ≥0.95 比例 `0.0 ≤ 0.10` → 通过

### 下游 MBON

- primary−sham：`-0.000111579`
- exact p：`0.9374145269393921`
- 95% CI：`[-0.002854789, +0.002480000]`
- positive/negative 门：均未通过
- 因 KC 干预未达门槛：**MBON 不作机制解释**

### low dose（次级）

- KC：`+0.011631875`，p `5.960464477539063e-08`，CI `[+0.009444297, +0.013893125]`，BH q `1.19209e-07`
- MBON：`+0.001347368`，p `0.39802277088165283`，CI `[-0.001564263, +0.004465263]`，BH q `0.398022770882`

## 结论

运行、数据完整性和持久化均成功；但 **KC 直接干预只产生 +0.022858，未达到预注册 +0.10 成功门**，因此本次结果判定为 **INCONCLUSIVE_KC_NOT_DRIVEN**。
不能据此声称 KC→MBON 功能存在或不存在；也不能外推自然感觉传递、单突触专一因果或生物学习。没有重试、补跑、改剂量/靶点/seed/统计门槛。

## 归档

- 原始：`C:\Users\yang\AppData\Local\Temp\opencode\p11_2_s3_full\20261021T102823`
- 持久备份：`C:\AA_project\guoying_archive\p11_2_s3_run_20261021T102823`
- report：`loop/p11_2_s3_report.json`
- run_meta：`loop/p11_2_s3_run_meta.json`
- manifest：`loop/p11_2_s3_raw_manifest.json`
- 实验结束，未进入下一阶段。
