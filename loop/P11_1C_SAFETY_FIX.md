# P11-1c 安全加固与授权状态（**NOT_AUTHORIZED**）

**基线** `1afae98`。**本任务不构成 P11-1c 运行授权**；用户尚未批准 25-seed 完整脑实验。
仅完成**静态安全修复 + 合成测试**；**未启动完整脑实验**。

## 1. 授权状态
| 项 | 结果 |
|---|---|
| P11-1c 25-seed 实验授权 | **不存在** |
| `loop/p11_1c_authorization.json` | 缺失 → **NOT_AUTHORIZED** |
| 完整脑是否启动 | **否**（`full_brain_started=False`） |
| `Training Authorization` | **False** |

## 2. 安全修复
1. **启动竞态**：`CREATE_SUSPENDED` 创建子进程 → `AssignProcessToJobObject` → **成功后才 `ResumeThread`**；
   子进程在任何用户代码（含完整脑初始化）之前已被 Job 接管。
2. **分配失败**：`AssignProcessToJobObject`/`ResumeThread` 失败 → **立即 `TerminateProcess`**，返回 NO-GO；
   **绝不**继续运行。
3. **限制范围**：同时设置 `JOB_OBJECT_LIMIT_PROCESS_MEMORY`（进程级）与 `JOB_OBJECT_LIMIT_JOB_MEMORY`
   （Job 级），各 **2 GiB**；`KILL_ON_JOB_CLOSE`；派生进程继承 Job。
   报告同时记录 `PeakProcessMemoryUsed` 与 `PeakJobMemoryUsed`。
4. **全量哈希**：官方 brain/weights 使用**完整 64-hex SHA-256 相等**比较（非前缀）。
5. **单位统一**：使用 **GiB** 表示 2 GiB 上限（=2,147,483,648 B），内存报告统一为 **MiB**（`/1024²`）。
6. **无自动重试**：一次运行；失败不重跑。
7. **保留工作区**：三个既有无关改动保留、未覆盖、未纳入提交。
8. 未改变神经动力学、连接组或实验参数。

## 3. 合成安全测试（全部通过）
| 测试 | 结果 |
|---|---|
| 无 Job 时 512 MiB 分配（对照） | **ALLOC_OK**（rc 0） |
| 256 MiB Job 下 512 MiB 分配 | **MEMORY_ERROR**（rc 3） |
| 加入 Job 失败（暂停子进程） | **已终止**（assign_ok=False，exit 1） |
| 1 s 超时（10 s 子进程） | **已终止**（timeout=True，wall 1.006 s，exit 1） |
| 无自动重试 | **calls=1** |
| `all_ok` | **True** |

## 4. 未授权裁决
| 裁决 | 结果 |
|---|---|
| RESOURCE | **NOT_AUTHORIZED** |
| FULL_BRAIN_BASELINE | **INCONCLUSIVE**（未运行） |
| FULL_VS_SUBNET_DIFFERENCE | **INCONCLUSIVE**（未运行） |
| MODEL_ROOT_CAUSE | **NOT_ESTABLISHED** |
| Level A Training / B / C | **NO-GO / NO-GO / NO-GO** |
| Biological | **NO-GO** |
| **Training Authorization** | **False** |

## 5. 工作区保留
三个既有、无关改动**保留未动**：
- `hardware/tof5_client.py`
- `loop/diagnose.py`
- `loop/p10c1b_decoder_freeze.json`（CRLF）
**未覆盖、未重置、未纳入本次提交。**

## 6. 下一步
需用户**另行书面授权** P11-1c 的严格范围（完整 166,700/25,582,938；seeds 101–125；仅 none；每 seed 400 步；
共 10,000 步；原参数；W 只读、ΔW=0；2 GiB；15 min；仅一次）。获授权后可使用本加固脚本运行一次。
**不得**在未授权时运行完整脑、训练、更新 ΔW、Dijkstra、迷宫、B/C 或实体车。

**已停止。**
