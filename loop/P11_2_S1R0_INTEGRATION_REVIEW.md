# P11-2-S1R-0 — 启动链集成审查

基线：`4823ba8`
状态：`S1R0_INTEGRATION_COMPLETE; NOT_AUTHORIZED; NO_RUN`
新增授权：`NEW_S1R_EXECUTION_AUTHORIZATION=False`
训练：`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`；Level A/B/C/Biological=`NO-GO`；A1/A2/A3/B=`NO-GO`

本轮只做静态集成、合成测试、失败现场备份和审批准备；没有运行 S1、S1R、完整 MaleCNS、
FlyBrain step、训练、ΔW、Dijkstra、迷宫、B/C 或实体控车。

## 1. 尚未集成的问题

- `p11_2_s1f_safe_job_launcher.py` 已支持 `module=`，且日志句柄、Job 顺序、状态语义已修复。
- 历史 `p11_2_s1_experiment.py` **仍然调用旧 `run_under_job()`**，并通过绝对脚本路径启动；
  该文件是历史失败记录的一部分，**不修改**。
- 旧 S1 授权、失败报告、6 个原始文件均保持原样；旧授权已消耗，不能复用或复制成新批准。

## 2. 新 S1R 生产链

```text
loop.p11_2_s1r_launcher
  -> load_auth()                  # 先鉴权；NOT_AUTHORIZED 立即 fail-closed
  -> load_freeze()                # 独立 S1R 冻结 SHA
  -> import lib loop.p11_2_s1r_experiment   # 仅在鉴权/冻结通过后
  -> official_hashes()
  -> run_under_job_safe(...)
       -> CreateProcessW:
            python -u -m loop.p11_2_s1r_experiment --child <out_dir>
            cwd = 仓库根
            env: PYTHONPATH=仓库根, PYTHONIOENCODING=utf-8
       -> CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread
       -> stdout/stderr/traceback -> experiment.log
       -> Job 2048 MiB / 900 s；单次；不重试
  -> persist run_meta.json（先于统计）
  -> validate 25 seeds / noise pairing
  -> 仅当 CHILD_EXIT_OK 且 25 seeds 完整时才 analyze + REPORT
```

关键点：

- 生产父进程 `p11_2_s1r_launcher.py` 只 import `loop.p11_2_s1f_safe_job_launcher.run_under_job_safe`；
  不含旧 `run_under_job()`。
- 子进程固定使用 `python -u -m`，cwd 固定仓库根；不再使用绝对脚本路径。
- 参数由 `subprocess.list2cmdline` 引用，含空格的 out_dir、cwd、日志路径均可用。
- 环境变量显式设置 `PYTHONPATH=ROOT`，避免 `loop` 包导入依赖当前目录。
- `run_meta` 在任何 `analyze`/统计之前持久化。
- `EXPERIMENT_SUCCEEDED` 需要：`CHILD_EXIT_OK` + `child_result` 存在 + 25 seeds + noise pairing 全 true；
  exit 0 但 24 seeds 仍为 `INCOMPLETE_NO_GO`。
- 未授权时 `p11_2_s1r_launcher --execute` 在 import/启动子进程之前返回 `REFUSED/NOT_AUTHORIZED`。

## 3. S1R 独立文件

| 文件 | 作用 | 状态 |
|---|---|---|
| `loop/p11_2_s1r_launcher.py` | 生产父入口/Job/元数据/报告 | 新增，未运行生产实验 |
| `loop/p11_2_s1r_experiment.py` | 独立 S1R 子进程入口；科学语义复制自冻结 S1 | 仅 import/compile；未运行 |
| `loop/p11_2_s1r_authorization.json` | 独立 S1R 授权 | `NOT_AUTHORIZED` |
| `loop/p11_2_s1r_script_freeze.json` | 独立 S1R 脚本冻结 | SHA 固定 |
| `loop/P11_2_S1R_ONE_RUN_APPROVAL.md` | 新授权模板 | `NOT_AUTHORIZED` |

`p11_2_s1r_experiment.py` 的 `--child` 在鉴权后才会 `from flybrain import FlyBrain`；
模块顶层只 import numpy/scipy/既有 C1 辅助函数，不加载完整脑。`--synthetic` 仍存在但本轮未调用。

## 4. 冻结科学方案

与 S1 完全一致，未改：

- B synthetic KC→KC-zero；删除 642,933 条边。
- none/stim 两条件；seeds 101–125；每 seed 每条件 400 步；总计 20,000 步。
- 精确 LPLC2/LC4 共 311；`dv=0.2`；`amount=0.036254`；主窗口 `[100,300)`。
- CPU only；原 LIF 参数；不训练；ΔW=0。
- exact sign-flip；bootstrap 10,000；主终点正向 +0.05 门槛。

本轮没有运行上述实验。

## 5. 资源与历史归档

- 未来 S1R 硬上限保持：整体 `900 s`、`2048 MiB`。
- C1 实测 Job peak `1942.7 MiB`，余量约 `105.3 MiB`；**S1/S1R 未实测**，
  `RESOURCE_VERIFIED=NO`，不得宣称资源必然足够。
- 历史 S1 失败现场保持不变：exit `1`、wall `0.82 s`、Job peak `521.05 MiB`、
  `experiment.log=0 B`、`0/25`、`INCOMPLETE_NO_GO`。
- 6 个原始文件的 SHA-256 已复核，备份到：
  `C:\AA_project\guoying_archive\p11_2_s1_failed_20261020T104624`
  （复制，不覆盖原文件；备份 hash 与 manifest 一致）。
- 官方 `brain.npz`/`weights.npz`、P1-mapfix=`db401f5`、readout/frozen 路径、tags 和三个
  既有无关 dirty 文件均保持；未改历史 S1 授权/报告/日志。

## 6. 合成回归

命令：

```powershell
.\.venv\Scripts\python.exe -m unittest loop.test_p11_2_s1r_launcher -v
```

结果：`Ran 16 tests ... OK`。覆盖：

1. 父进程真实引用 `run_under_job_safe`，不含旧 `run_under_job()`；
2. 子进程以 `python -u -m` 启动，cwd=临时包根（实际生产为仓库根）；
3. ROOT/包导入；含空格 out_dir/cwd 的参数引用；
4. stdout/stderr/traceback 保存；
5. Job 分配/子树约束；
6. exit1 → `CHILD_EXIT_NONZERO`；
7. exit0 但无产物 → `CHILD_OUTPUT_MISSING`；
8. exit0 但 24 seeds → `INCOMPLETE_NO_GO`，不判成功；
9. timeout / memory / invalid log → fail-closed；
10. 单次调用，counter=1；
11. run_meta 在 stats 回调之前存在；
12. 未授权生产入口 `--execute` 返回 `NOT_AUTHORIZED`，不启动子进程。

## 7. 结论

S1R-0 完成了“安全启动器 → 独立生产入口 → 独立模块 → 审批/冻结”的静态集成。
但本轮没有获得新授权，也没有运行 S1R；历史 S1 仍为 `INCOMPLETE_NO_GO`。
下一步只能提交新一次 S1R 授权申请；在用户明确批准并生成独立 `APPROVED` 授权与匹配 freeze
之前，生产入口必须继续 fail-closed。
