# P11-2-S1R — 一次性运行授权与执行记录

**STATUS: APPROVED; CONSUMED; RUN_COMPLETE**

`NEW_S1R_EXECUTION_AUTHORIZATION=APPROVED_ONE_RUN; CONSUMED`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
Level A/B/C/Biological=`NO-GO`

## 授权记录

- 基线 commit：`7124a36`
- 授权文件：`loop/p11_2_s1r_authorization.json`
- 授权状态：`APPROVED`，`approved_by=user`
- 授权 checksum：`b7f9def8d16a6161f34d45c888d8fbf47c90b53999481787bd8f686c49b362f5`
- 冻结脚本 SHA-256：`2a57790479e69f8aded1ccc37257d2b375d47de50255d06bb3580d61bc71ab93`
- 一次性锁：`loop/p11_2_s1r_authorization_locks/b7f9def8d16a6161f34d45c888d8fbf47c90b53999481787bd8f686c49b362f5.consumed.json`
- 第二次执行：`REFUSED / AUTHORIZATION_ALREADY_CONSUMED`（未启动子进程）

批准文本要点：完整 MaleCNS 合成 B、仅删除 642,933 条 KC→KC 连接、none/stim、seeds 101–125、每 seed 每条件 400 步、共 20,000 步；精确 LPLC2/LC4 311、dv=0.2、amount=0.036254、窗口 [100,300)、CPU only、900 s / 2048 MiB；一次运行，失败/异常/超时/内存超限停止，不自动重跑、不补跑、不提高上限、不改参数；不包含训练、真实 ΔW、Dijkstra、迷宫、Level B/C 或实体机器人控制。

## 冻结科学方案

| 项目 | 冻结值 |
|---|---|
| 网络 | B synthetic KC→KC-zero |
| 删除边 | 642,933 |
| 条件 | none / stim |
| seeds | 101–125（25） |
| 每 seed 每条件步数 | 400 |
| 总步数 | 20,000 |
| 感觉靶点 | 精确 LPLC2 + LC4，共 311 |
| dv | 0.2 |
| amount | 0.036254 |
| 主窗口 | [100,300) |
| 设备 | CPU only |
| LIF 参数 | 原参数冻结 |
| 训练 / ΔW | 不训练 / 0 |
| 统计 | exact sign-flip；bootstrap 10,000；主终点正向 +0.05 门槛 |

## 执行结果摘要

- 子进程：`CHILD_EXIT_OK`，exit `0`
- 完成度：`25/25`
- 墙钟：`275.591 s`（上限 900 s）
- Job peak：`1943.16 MiB`（上限 2048 MiB）
- 50 个原始 NPZ：完整；manifest 56 个不可变文件全部 SHA-256 匹配
- `run_meta` / `report` / `manifest`：均为 `RUN_COMPLETE`
- 统计后哈希校验：通过

科学结果：

- Sensory entry：none `0.012459164`，stim `0.079093891`，diff `+0.066634727`，exact p `5.960464477539063e-08`，pass=`True`
- 主终点（matched-320 KC，stim−none）：-0.001430625，exact p `0.3523610830307007`，95% CI `[-0.004300078, 0.001516953]`，+0.05 门槛 pass=`False`
- matched-95 MBON：-0.00176，exact p `0.43899786472320557`，95% CI `[-0.005720053, 0.002718263]`，pass=`False`
- 分级：`SENSORY_TRANSMISSION_NOT_DETECTED`
- 负向探索指标：均未达显著负向

结论：实验执行成功且完整性通过；在冻结协议下**未检测到正向 KC/MBON 感觉传递**。这不是运行/资源失败，也不支持训练或生物学习结论。未重跑、未补跑、未改参数。

## 原始结果

- 原始目录：`C:\Users\yang\AppData\Local\Temp\opencode\p11_2_s1r_full\20261020T160105`
- 持久备份：`C:\AA_project\guoying_archive\p11_2_s1r_run_20261020T160105`
- 报告：`loop/p11_2_s1r_report.json`
- 运行元数据：`loop/p11_2_s1r_run_meta.json`
- 原始清单：`loop/p11_2_s1r_raw_manifest.json`

本文件与 JSON 授权记录是本次一次运行的记录；一次性授权已消耗，禁止自动重试或直接进入下一阶段。
