# P11-2-S1R — 新一次性运行审批文件（模板）

**STATUS: NOT_AUTHORIZED**
`NEW_S1R_EXECUTION_AUTHORIZATION=False`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
Level A/B/C/Biological=`NO-GO`

本文件不是批准，也不包含可复用的旧 S1 授权。旧 S1 的一次性授权已消耗，不能复制其
`APPROVED` 状态、checksum 或 approval_text。只有用户在审阅 S1R-0 后明确写入新的
`p11_2_s1r_authorization.json`（`status=APPROVED`、`approved_by=user`、独立 checksum），
本文件才能被新的审批记录取代。

## 冻结科学方案（不得改变）

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

## 运行边界

- 只允许一次整体运行；**不自动重试、不补跑、不自动提高上限**。
- 硬上限：整体 **900 s（15 min）**、**2048 MiB（2 GiB）**。
- C1 实测 Job peak `1942.7 MiB`，余量约 `105.3 MiB`；S1/S1R 尚未实测，
  因此 `RESOURCE_VERIFIED=NO`，不得宣称资源必然足够。
- 只允许 `loop.p11_2_s1r_launcher.py` 作为生产父入口；它必须使用
  `python -u -m loop.p11_2_s1r_experiment --child <out_dir>`，cwd=仓库根。
- 未获新授权时，生产入口必须在 import/启动子进程之前 fail-closed。
- 禁止训练、真实 ΔW、Dijkstra、迷宫、B/C、实体控车、完整脑以外的越界实验。

## 新授权生效前必须满足

1. S1R-0 静态集成/合成回归通过，独立 commit/push。
2. `p11_2_s1r_script_freeze.json` 与 `p11_2_s1r_experiment.py` SHA 完全一致。
3. `p11_2_s1r_authorization.json` 仍为 `NOT_AUTHORIZED`；不得手工改成旧 S1 的内容。
4. 用户明确给出新的 S1R 一次性批准文本；新授权 scope 必须与上方冻结表完全一致。
5. 官方 `brain.npz`/`weights.npz`、P1-mapfix/readout/tags、历史 S1 失败现场均未被修改。
6. 运行前记录资源实测计划；运行后先持久化 run_meta，再做统计后处理。

## 当前决定

**不批准运行。**
本阶段只完成安全启动链集成、合成回归和审批准备；S1R 未运行，S1 仍为
`INCOMPLETE_NO_GO`。
