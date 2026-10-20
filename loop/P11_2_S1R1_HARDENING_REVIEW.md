# P11-2-S1R-1 — 安全与统计加固

基线：`5cd7df5`
状态：`S1R1_HARDENING_COMPLETE; NOT_AUTHORIZED; NO_RUN`
`NEW_S1R_EXECUTION_AUTHORIZATION=False`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
Level A/B/C/Biological=`NO-GO`

本轮只做静态加固与合成测试；没有初始化完整 MaleCNS、没有运行 `FlyBrain.step`、没有启动正式 S1R、没有训练/ΔW/Dijkstra/迷宫/B/C/实体控车。

## 1. 一次性授权锁

实现位置：`loop/p11_2_s1r_launcher.py`

- 锁路径按新授权 checksum 派生：
  `loop/p11_2_s1r_authorization_locks/<auth_checksum>.consumed.json`
- 使用 `os.open(..., O_CREAT | O_EXCL | O_WRONLY)` 原子创建；只有创建成功才获得唯一执行权。
- 锁在任何子进程启动之前创建；写入失败、父进程崩溃、Job 失败、超时、内存失败、统计失败都不删除。
- 第二次执行（相同授权 checksum）直接 `AUTHORIZATION_ALREADY_CONSUMED`，不会自动恢复或重试。
- 新授权因 checksum 不同会派生新锁；旧锁永久保留，不能被“恢复”为可执行。
- 未获 `APPROVED` 时，入口在创建锁之前拒绝；当前无生产锁文件。

测试：`test_one_shot_lock_atomic_and_permanent` 验证首次成功、第二次拒绝、文件持久、状态为 `CONSUMED_STARTED`。

## 2. 数据完整性

实现位置：`loop/p11_2_s1r_experiment.py`

`validate_raw_dataset()` 严格检查：

- 恰好 `25 seeds × 2 conditions = 50` 个 `{seed}_{none|stim}.npz`；
- seeds 101–125 全部覆盖，none/stim 标签正确；
- 形状：`kc(400,4064)`、`mb(400,97)`、`lp(400,311)`、`kc_matched(400,320)`、`mb_matched(400,95)`；
- 所有数组有限、dtype 为数值；
- `amount=0.036254`、`stim_window=[100,300)`；
- `script_sha`、`auth_sha` 与冻结/授权一致；
- 全部文件 `w_hash` 一致；
- 每个 seed 的 none/stim `rng_before`/`rng_after` 配对一致；
- 每个 NPZ 的 SHA-256，及排序后的 dataset SHA-256；
- 缺失、额外、错误标签、错误形状、非有限值、噪声不配对都会使 `ok=False`。

父进程在统计前持久化 raw 验证/哈希；统计后再次核验哈希未变。
`EXPERIMENT_SUCCEEDED` 必须同时满足：`CHILD_EXIT_OK` + child_result 完整 25 seeds + raw 50 文件完整 + 哈希未变；**exit 0 或 25 条 seed 记录本身不算成功**。

测试：`test_raw_dataset_valid_50_files`、`test_raw_dataset_missing_and_extra`、`test_raw_dataset_shape_label_nonfinite_noise`、`test_raw_hash_mutation_detected`。

## 3. 入口统计修复

原问题：entry 的 `A_none`/`A_stim` 由 `mean(diff)+mean(none)` 之类推导，未分别计算两组真实均值。

修复：`entry_means(rows)` 分别计算
`none_mean`、`stim_mean`，再算 `diff = stim_mean - none_mean`；`analyze()` 使用该函数。

合成测试：`none=0.4`、`stim=0.5` → `diff=0.1`，`test_entry_means_real_none_stim` 通过。

## 4. 禁止绕过 Job

生产子进程 `loop.p11_2_s1r_experiment --child` 在鉴权/加载完整脑之前执行：

- `IsProcessInJob`；
- 通过父进程传入的命名 Job 名称 `OpenJobObjectW`；
- `QueryInformationJobObject` 验证 `JOB_OBJECT_LIMIT_PROCESS_MEMORY` 与
  `JOB_OBJECT_LIMIT_JOB_MEMORY`，且两者均精确为 `2048 MiB`；
- `JobObjectBasicProcessIdList` 验证当前 PID 确在该 Job 内。

环境变量 `P11_2_S1R_JOB_NAME` 只是名称提示，不是凭证；必须成功打开真实 Job 句柄、核验限制和 PID 成员。
普通环境变量/单独 `--child` 不能绕过。

合成验证：

- 无效命名 / 非受控环境：exit 9，`S1R_CHILD_JOB_GUARD_FAILED`；
- 1024 MiB 命名 Job：exit 9，`JOB_MEMORY_LIMIT_MISMATCH`；
- 2048 MiB 命名 Job：guard 通过，但因授权仍为 `NOT_AUTHORIZED` 在 `load_auth()` 失败；未加载完整脑。

## 5. 科学参数冻结

未改变：

- B synthetic KC→KC-zero；删除 642,933 条边；
- none/stim；seeds 101–125；每 seed 每条件 400 步；总 20,000 步；
- 精确 LPLC2/LC4 311；`dv=0.2`；`amount=0.036254`；窗口 `[100,300)`；
- CPU only；原 LIF；不训练；ΔW=0；
- exact sign-flip、bootstrap 10,000、主终点 +0.05。

## 6. 测试与归档

- `loop.test_p11_2_s1r_launcher`：**25/25 passed**；
- `loop.test_p11_2_s1f_launcher_audit`：**9/9 passed**；
- 总计 34 个合成/回归测试通过。
- 历史 S1 仍为 `INCOMPLETE_NO_GO`；6 个原始文件 hash 未变，备份仍为
  `C:\AA_project\guoying_archive\p11_2_s1_failed_20261020T104624`。
- 官方数据 hash、P1-mapfix=`db401f5`、readout/frozen 路径、tags、三个既有 dirty 文件均保持。
- 新 S1R freeze SHA：`4e518b57259e34d3577d2133a26dc5e0c3660bdf25bf9c82705950b8e65bac40`。
- 授权仍为 `NOT_AUTHORIZED`；无新 S1R 运行。

## 7. 资源与风险

- 未来仍限定整体 900 s / 2048 MiB；不自动提高上限。
- C1 实测 peak `1942.7 MiB`，余量约 `105.3 MiB`；S1/S1R 未实测，`RESOURCE_VERIFIED=NO`。
- 本轮的 Job guard 只会拒绝“非 2048 MiB 受控 Job”，不会提高资源上限，也不证明完整脑一定能在 2 GiB 内完成。
- 新一次 S1R 授权仍需用户明确批准；批准前生产入口必须继续 fail-closed。
