# P11-2-S1R — 正式一次性运行结果

日期：2026-10-20
基线：`7124a36`
授权：`APPROVED_ONE_RUN; CONSUMED`
执行状态：`RUN_COMPLETE`
科学分级：`SENSORY_TRANSMISSION_NOT_DETECTED`

## 运行事实

| 项目 | 结果 |
|---|---|
| 完成度 | `25/25` |
| 子进程 | `CHILD_EXIT_OK` / exit `0` |
| 墙钟 | `275.591 s`（上限 900 s） |
| Job peak | `1943.16 MiB`（上限 2048 MiB） |
| 50 个 NPZ | 完整 |
| manifest 不可变文件 | 56 个，全部 SHA-256/大小匹配 |
| run_meta SHA-256 | `83ae91718d80f6253ba4f1b50d35c810eb87fdcff8d1344a6c1b1d7be970713c` |
| dataset SHA-256 | `b3aa4466eaf3516d218148b82b13b2514c903f0e07220ada88a2b14799667963` |
| 统计后哈希 | 通过 |
| 重试/补跑 | 无；第二次执行被一次性锁拒绝 |

## 科学结果

**Sensory entry（311 精确 LPLC2/LC4）**

- none：`0.012459164`
- stim：`0.079093891`
- stim−none：`+0.066634727`
- exact p：`5.960464477539063e-08`
- pass：`True`

**主终点（matched-320 KC，stim−none，[100,300)）**

- mean：`-0.001430625`
- exact p：`0.3523610830307007`
- 95% bootstrap CI：`[-0.004300078, 0.001516953]`
- +0.05 门槛：`False`
- 判定：未检测到正向 KC 感觉传递

**matched-95 MBON**

- mean：`-0.00176`
- exact p：`0.43899786472320557`
- 95% CI：`[-0.005720053, 0.002718263]`
- +0.05 门槛：`False`

**次级群体**

- full-4064 KC：none `0.351093307`，stim `0.349638189`
- full-97 MBON：none `0.460771134`，stim `0.459045361`
- 97 MBON per-cell：none `0.460771134`，stim `0.459045361`
- 低反馈 KC：2460 `1.0/1.0`，2650 `0.6274/0.6264`，3971 `0.9994/0.9986`（none/stim）
- 负向探索指标：均未达显著负向

## 结论

- 运行、资源、Job、50 NPZ 完整性和最终持久化均成功。
- 在冻结的 B KC→KC-zero 协议下，感觉入口有显著响应，但 matched-320 KC 与 matched-95 MBON 的 stim−none 未达到正向 +0.05/显著性门槛。
- 科学分级为 `SENSORY_TRANSMISSION_NOT_DETECTED`；不能据此声称学习、生物功能或 Level A/B/C 结论。
- 未训练、未更新真实 ΔW、未运行 Dijkstra/迷宫/B/C/实体控车。

## 结果路径

- 原始：`C:\Users\yang\AppData\Local\Temp\opencode\p11_2_s1r_full\20261020T160105`
- 备份：`C:\AA_project\guoying_archive\p11_2_s1r_run_20261020T160105`
- 报告：`loop/p11_2_s1r_report.json`
- 元数据：`loop/p11_2_s1r_run_meta.json`
- 清单：`loop/p11_2_s1r_raw_manifest.json`
- 一次性锁：`loop/p11_2_s1r_authorization_locks/b7f9def8d16a6161f34d45c888d8fbf47c90b53999481787bd8f686c49b362f5.consumed.json`

本实验已结束。未自动进入下一阶段。
