# P11-2-S3 — 一次性直接 KC 驱动实验结果

日期：2026-10-21
基线：`52423b2`
授权：`APPROVED_ONE_RUN; CONSUMED`
执行状态：`RUN_COMPLETE`
科学判别：`INCONCLUSIVE_KC_NOT_DRIVEN`

## 运行事实

| 项目 | 值 |
|---|---|
| seeds / NPZ | 25 / 75 |
| 子进程 | `CHILD_EXIT_OK` / exit 0 |
| 墙钟 | 444.561 s |
| Job peak | 1932.29 MiB |
| 统计前后哈希 | 通过 |
| report / run_meta / manifest | RUN_COMPLETE |
| 第二次执行 | refused / `AUTHORIZATION_ALREADY_CONSUMED` |

## KC primary−sham

- effect：`+0.022858125`
- exact p：`5.960464477539063e-08`
- 95% CI：`[+0.019925500, +0.025606891]`
- 预注册 +0.10 门：**未通过**
- 饱和门：mean `0.2238375` <0.85；fraction ≥0.95 `0.0` ≤0.10：**通过**

## MBON primary−sham

- effect：`-0.000111579`
- exact p：`0.9374145269393921`
- 95% CI：`[-0.002854789, +0.002480000]`
- 正向/显著负向门均未通过
- 因 KC 未达成功门：**INCONCLUSIVE_DOWNSTREAM，不作机制解释**

## low dose（次级）

- KC：`+0.011631875`，p `5.960464477539063e-08`，CI `[+0.009444297, +0.013893125]`，BH q `1.19209e-07`
- MBON：`+0.001347368`，p `0.39802277088165283`，CI `[-0.001564263, +0.004465263]`，BH q `0.398022770882`

## 解释

KC 电流注入确实产生了显著但很小的正向效应；未达到预注册 +0.10，故不能进入下游机制解释。
没有证据支持或否定“KC→MBON 功能不足”；本次只能判 **INCONCLUSIVE**。人工直接驱动不能证明自然感觉传递、单条突触因果或生物学习。

## 归档

- 原始：`C:\Users\yang\AppData\Local\Temp\opencode\p11_2_s3_full\20261021T102823`
- 备份：`C:\AA_project\guoying_archive\p11_2_s3_run_20261021T102823`
- report：`loop/p11_2_s3_report.json`
- run_meta：`loop/p11_2_s3_run_meta.json`
- manifest：`loop/p11_2_s3_raw_manifest.json`
- lock：`loop/p11_2_s3_authorization_locks/259da39ce09dff24c4f61b34a0a133e5bdd4f2d4b91407f7dfaf0e44cacdbad7.consumed.json`

实验结束，未进入下一阶段。
