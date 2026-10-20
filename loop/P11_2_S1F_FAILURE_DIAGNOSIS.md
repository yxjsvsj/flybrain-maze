# P11-2-S1F — S1 失败根因诊断

日期：2026-10-20
状态：`DIAGNOSIS_COMPLETE; REPAIR_READY_FOR_NEW_APPROVAL`
S1 状态：`INCOMPLETE_NO_GO`（未改变）
新 S1 执行授权：`False`
训练授权：`False`
`REAL_SYNAPTIC_LEARNING_READY=NO`；Level A/B/C/Biological=`NO-GO`；A1/A2/A3/B=`NO-GO`

本文件只做离线诊断；没有运行 S1、没有初始化完整脑、没有调用 FlyBrain、没有重试、没有改官方数据/P1/readout/tags。

## 1. 失败场景（原始记录，只读）

| 项目 | 值 |
|---|---|
| raw 目录 | `C:\Users\yang\AppData\Local\Temp\opencode\p11_2_s1_full\20261020T104624` |
| 文件数 | 6（hash 全部匹配 manifest） |
| child exit code | `1` |
| 墙钟 | `0.82 s` |
| Job peak | `521.05 MiB` |
| `experiment.log` | `0 bytes` |
| `child_result.json` | 不存在 |
| 每 seed NPZ | `0` |
| 完成度 | `0/25` |
| 统计 | 不存在 |
| 重试 | 无 |
| Windows 事件/崩溃包 | `NOT_AVAILABLE`（未保留） |
| S1 report | `INCOMPLETE_NO_GO` |
| S1 脚本 SHA | `6d57f649bc62b20f0b0f04b8687f9d5c371e8633e6afe2c005bd1dcca239bc3f` |
| 授权 checksum | `d9087107c485cca57f983725a11423d65c3f2da0770b5cff7d1212424a948ab5` |

结论：S1 子进程确实退出（exit 1），但异常文本没有被观察/保留；不能从“0.82 s、521.05 MiB”推断已经加载完整脑或执行了任何 `FlyBrain.step`。

## 2. 七个问题

### Q1：S1 是否运行？

是，且只运行了一次：历史授权下的一次 A/B 运行；失败后未重试。S1 的一次性授权已消耗。

### Q2：是否已经初始化完整脑/FlyBrain？

**UNKNOWN，且不应推断。** 运行时间 `0.82 s`、Job peak `521.05 MiB` 与“失败发生在初始化/导入/早期路径”一致，但不足以证明加载完成或 step 次数。没有 traceback、没有 `child_result.json`，因此不能声称“完整脑已加载”“已经 step”“某个神经元响应为 0”。

### Q3：失败在哪里？

只能确定：子进程以 exit code `1` 结束；标准输出/错误没有进入 `experiment.log`；没有生成 `child_result.json` 或每 seed 结果。
**具体异常不能确认**（`HISTORICAL_EXACT_CAUSE=NOT_CONFIRMED`）。

### Q4：为什么日志是 0 字节？

历史 launcher 的 `CreateFileW` 把 `lpSecurityAttributes` 传为 `NULL`：

```text
None, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, None
```

即使 `CreateProcessW` 使用 `bInheritHandles=True` 和 `STARTF_USESTDHANDLES`，`NULL` security attributes 使日志句柄不可继承；子进程的 stdout/stderr 因此没有可写的有效句柄，traceback 丢失。
合成 A 已复现：历史 launcher 返回 `ok=True/reason='completed'`、exit 非 0（历史 S1 本身为 `1`）、`log_bytes=0`；修复后的 launcher 在同一合成 A 上记录到 `ModuleNotFoundError` traceback（`232 bytes`）。
判定：`LOG_INHERITANCE_DEFECT=CONFIRMED`。

### Q5：S1 脚本/launcher 有启动缺陷吗？

有，已合成确认：

- `loop/p11_2_s1_experiment.py` 顶部在设置 `ROOT`/`sys.path` 之前执行
  `from loop.p11_2_c1_experiment import ...`。
- 历史 launcher 以**绝对脚本路径**启动：`python -u "<abs script>"`。此时 `sys.path[0]` 是脚本所在目录 `loop\`，不是仓库根，因此 `loop.p11_2_c1_experiment` 不可导入。
- 合成 A（同样的“顶部包导入 + 绝对脚本路径”）得到 `ModuleNotFoundError`；B（`python -m s1f_pkg.main_m`，cwd=repo root）成功；C（显式 `PYTHONPATH`/`ROOT`）成功。

判定：`IMPORT_DEFECT=CONFIRMED_LAUNCHER_IMPORT_DEFECT`。
但这只是**合成复现**，不是历史 S1 的 traceback；不能单独证明历史 S1 就是因该 import 失败。

### Q6：是否因资源上限失败？

不是。Job peak `521.05 MiB < 2048 MiB`，墙钟 `0.82 s < 900 s`。没有触达内存/时间上限的证据。

### Q7：现有证据能否区分 import、日志、资源、模型异常？

不能完全区分。可确认两个**启动级缺陷**（import 路径、日志句柄继承）；无法确认历史异常文本。
因此：`HISTORICAL_EXACT_CAUSE=NOT_CONFIRMED`；S1 的模型结果（entry/KC/MBON/negative/resource）均为 **NOT MEASURED / NOT INFERRED**。

## 3. 合成 A/B/C 结果

| 路径 | 命令形态 | 结果 |
|---|---|---|
| A 历史 | `python -u <abs script>` | exit 非 0，`ModuleNotFoundError`，历史 log `0 B`（历史 S1 本身为 exit `1`） |
| A 安全 | 安全 launcher + 同一脚本 | `CHILD_EXIT_NONZERO`，log 含 traceback，`232 B` |
| B | `python -m pkg.main_m`，cwd=root | exit `0`，成功 |
| C | 显式 `PYTHONPATH=root` + 脚本 | exit `0`，成功 |

安全 launcher 的其他合成回归：

| 测试 | 结果 |
|---|---|
| stdout/stderr 捕获 | `CHILD_EXIT_OK`，日志含 `STDOUT_OK`/`STDERR_OK` |
| 未捕获 ImportError | `CHILD_EXIT_NONZERO`，日志含 traceback |
| exit 1 | `CHILD_EXIT_NONZERO` |
| 非法/不可继承日志句柄 | `CHILD_LOG_UNAVAILABLE`，不产生 artifact |
| Job assign 失败 | `CHILD_ASSIGN_FAILED`，不产生 artifact |
| timeout | `CHILD_TIMEOUT` |
| 显式内存失败 | `CHILD_MEMORY_LIMIT` |
| 子进程 crash | `CHILD_EXIT_NONZERO` |
| 必需 artifact | 缺失→`CHILD_OUTPUT_MISSING`；存在→`CHILD_EXIT_OK` |
| 子树内存失败 | 父进程仍 `CHILD_EXIT_OK`；孙进程 `MemoryError` 被 Job 限制，不被误记成父进程内存超限 |
| 重试 | 无（单次调用） |

## 4. 状态语义勘误

历史 `run_under_job` 的 `ok=True/reason='completed'` 只表示 **等待结束**，不表示 exit 0 或实验成功。S1 的正确拆分为：

```text
PROCESS_FINISHED=True
EXPERIMENT_SUCCEEDED=False
CHILD_EXIT_NONZERO=True
CHILD_OUTPUT_MISSING=True
CHILD_LOG_UNAVAILABLE=True
S1_STATUS=INCOMPLETE_NO_GO
```

## 5. 边界与结论

- S1 仍然是 `INCOMPLETE_NO_GO`；本次修复不改变 S1 结果，也不补跑。
- 不声称完整脑已加载、step 已执行、某个神经元/通路响应为 0。
- 不声称历史异常已经 100% 定位；两个启动级缺陷已合成确认，历史精确异常仍 `NOT_CONFIRMED`。
- 官方 `brain.npz`/`weights.npz` SHA 已复核；P1-mapfix=`db401f5`；readout/frozen 路径无改动；三个历史 dirty 文件保留；无 tag 变更。
- 新 S1 若要运行，必须是**新的显式授权**，使用修复后的 launcher 和 `python -m` 入口；不得自动重试、不得提高上限、不得把 S1F 当成 S1 结果。
