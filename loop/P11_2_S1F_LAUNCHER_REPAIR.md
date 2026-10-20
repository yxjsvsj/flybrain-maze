# P11-2-S1F — 安全 Job 启动器修复

日期：2026-10-20
状态：`REPAIR_READY_FOR_NEW_APPROVAL`
修复对象：`loop/p11_2_s1f_safe_job_launcher.py`
历史脚本：`loop/p11_2_s1_experiment.py`（未修改；SHA `6d57f649...`）
历史 launcher：`loop/p11_1c_full_brain_bench.py`（未修改）

本修复只覆盖启动/日志/状态语义；不是 S1 补跑，不改变 S1 的 `INCOMPLETE_NO_GO`。

## 1. 修复的三个问题

### 1.1 导入路径

**问题**：顶部 `from loop.p11_2_c1_experiment import ...` 早于 `ROOT`/`sys.path` 设置；绝对脚本路径启动时 `sys.path[0]=loop\`，包导入失败。

**修复/使用约束**：

- 未来 S1 子进程使用 `python -u -m loop.p11_2_s1_experiment ...`（cwd=仓库根），或
- 显式把仓库根加入 `PYTHONPATH`（合成 C 已验证），或
- 在脚本顶部先设置 `ROOT`/`sys.path` 再包导入（需另立脚本版本，不能改已冻结的 S1 脚本）。

本修复不修改冻结脚本；只保证修复后的 launcher 能记录失败 traceback。

### 1.2 日志句柄继承

**问题**：历史 `CreateFileW(..., NULL security attributes, ...)` 产生不可继承句柄；`bInheritHandles=True` 不足以让 stdout/stderr 可用，异常因此丢失。

**修复**：

- `SECURITY_ATTRIBUTES.bInheritHandle=True`，显式传给 `CreateFileW`；
- `GetHandleInformation` 校验 `HANDLE_FLAG_INHERIT`；不通过则 fail-closed；
- `STARTUPINFOEX` + `EXTENDED_STARTUPINFO_PRESENT`；
- `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` 只允许日志句柄（以及可继承的 stdin）；
- `CREATE_SUSPENDED` → `AssignProcessToJobObject` → `ResumeThread`，保证先入 Job 再执行；
- 失败不重试；返回明确 `child_status`。

### 1.3 状态语义

**问题**：`ok=True/reason='completed'` 被误读为成功；exit 1 仍显示 completed。

**修复**：

```text
PROCESS_FINISHED     = 子进程已创建并完成等待/终止
EXPERIMENT_SUCCEEDED = child_status == CHILD_EXIT_OK
CHILD_EXIT_OK        = exit 0 且（如有 required_artifact）artifact 存在
```

状态集合：

| status | 含义 |
|---|---|
| `CHILD_EXIT_OK` | exit 0；artifact（若有）存在 |
| `CHILD_EXIT_NONZERO` | 已运行但 exit != 0 |
| `CHILD_TIMEOUT` | 到达 wall cap，Job 已终止 |
| `CHILD_MEMORY_LIMIT` | 超过 mem cap；或未捕获 `MemoryError` 且 exit != 0 |
| `CHILD_OUTPUT_MISSING` | exit 0 但必需 artifact 缺失 |
| `CHILD_ASSIGN_FAILED` | Job assign/launch 失败；不产生结果 |
| `CHILD_LOG_UNAVAILABLE` | 日志句柄不可用/不可继承；fail-closed |

注意：带 `required_artifact` 的 `CHILD_EXIT_OK` 才代表实验产物完整；仅有 exit 0 不代表实验成功。

## 2. 合成回归结果

测试文件：`loop/test_p11_2_s1f_launcher_audit.py`
命令：

```powershell
.\.venv\Scripts\python.exe -m unittest loop.test_p11_2_s1f_launcher_audit -v
```

结果：`Ran 9 tests ... OK`。

覆盖：

- 历史 S1 raw 6 文件/hash/0 字节日志/无 child/无 seed 保留；
- 历史脚本“包导入早于路径设置”；
- 历史 `CreateFileW` 的 `NULL` security attributes；
- A（绝对脚本）失败、B（`-m`）成功、C（显式路径）成功；
- 历史 launcher 的 exit 非 0（历史 S1=1）+ `ok=True` 陷阱；
- 安全 launcher 的 stdout/stderr/ImportError/exit1/invalid log/assign fail/timeout/memory/crash/artifact/tree；
- 子树内存失败不误记父进程内存超限；
- S1 仍 `INCOMPLETE_NO_GO`，新 S1/训练授权仍为 `False`；
- 官方数据 hash、P1-mapfix、readout/frozen 路径、三个历史 dirty 文件。

## 3. 资源与未来 S1 约束

- S1F 合成测试 Job cap：`256 MiB`（仅测试）。
- 未来 S1 仍用历史批准的 `2 GiB` / `900 s`；**不得**因 S1F 提高上限。
- 未来 S1 必须：
  1. 新的显式运行授权（一次性）；
  2. 修复后的 launcher；
  3. `python -m` 或显式 `PYTHONPATH`；
  4. 单次运行，不自动重试；
  5. 运行前重新确认官方数据/P1/readout/tags 和失败场景；
  6. 运行后按 `PROCESS_FINISHED`/`EXPERIMENT_SUCCEEDED` 分别记录，禁止把“进程结束”写成“实验成功”。

## 4. 残余不确定性

- 历史 S1 的精确异常仍 `NOT_CONFIRMED`：日志为空且无 Windows 事件/崩溃包。
- 合成 A 证明“该启动方式会触发 import 失败”，但不证明历史 S1 必然因此失败。
- S1 的模型结果仍不可用；不得从 S1F 推断 entry/KC/MBON/negative/resource 或生物学习结论。
- 新 S1 运行授权仍为 `False`；本修复只达到 `REPAIR_READY_FOR_NEW_APPROVAL`。
