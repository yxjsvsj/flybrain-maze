# P11-2-S1R-1a — 正式执行前静态缺陷修复

基线：`b36b5db`
状态：`S1R1A_STATIC_FIXES_COMPLETE; NOT_AUTHORIZED; NO_RUN`
`NEW_S1R_EXECUTION_AUTHORIZATION=False`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
Level A/B/C/Biological=`NO-GO`

本轮只做静态修复、合成 NPZ、Windows 合成 Job 测试和回归；没有初始化完整 MaleCNS、没有调用 `FlyBrain.step`、没有启动正式 S1R。

## 1. RNG 哈希数据类型修复

原问题：`run_condition()` 保存的是 SHA-256 十六进制字符串，但 `validate_raw_dataset()` 用 `int(...)` 解析 `rng_before`/`rng_after`；真实数据可能抛 `ValueError` 并被误判为 `LOAD_FAILED`。

修复：

- 用 `_scalar_str()` 读取 `rng_before`/`rng_after`，不再转整数。
- 用 `re.fullmatch(r"[0-9a-fA-F]{64}")` 验证 64 位 SHA-256 十六进制格式；非法值记录 `RNG_BEFORE_INVALID` / `RNG_AFTER_INVALID`。
- 每个 seed 必须满足 `none.before == stim.before` 且 `none.after == stim.after`；不相等记录 `NOISE_PAIR_MISMATCH`。
- RNG 算法、seed、LIF 和所有科学参数未变。

测试：

- `test_rng_sha_hex_a_f_and_tamper` 用真实 64 字符、包含 `a–f` 的哈希构造 50 个 NPZ，验证通过。
- 篡改其中一个字符后，验证失败并报 `NOISE_PAIR_MISMATCH`。

## 2. Job 身份回退关闭

原问题：缺少 `P11_2_S1R_JOB_NAME` 时仍回退到 `QueryInformationJobObject(NULL)`，可能接受匿名/外层 Job。

修复：

- `--child` **必须**有非空 S1R 命名 Job；缺失直接 `JOB_NAME_MISSING`。
- 通过 `OpenJobObjectW` 打开真实命名 Job，不再有匿名回退。
- 验证：
  - `JOB_OBJECT_LIMIT_PROCESS_MEMORY` 与 `JOB_OBJECT_LIMIT_JOB_MEMORY` 均存在；
  - process/job 两个内存限制均精确为 2048 MiB；
  - `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` 存在；
  - `JobObjectBasicProcessIdList` 中确实包含当前 PID。
- 环境变量仅用于传递 Job 名称；必须打开句柄并核验限制/PID，不能单独作为凭证。
- 父入口仍通过 S1F 安全启动器创建命名 Job；子进程不能绕过父入口使用普通匿名 Job。

测试：

- 缺少名称：exit 9 / `JOB_NAME_MISSING`；
- 假名称：exit 9 / `NAMED_JOB_OPEN_FAILED` 或 `NOT_IN_JOB`；
- 错误 Job（存在但不含当前 PID）：exit 9 / `PID_NOT_IN_NAMED_JOB` 或 `NOT_IN_JOB`；
- 1024 MiB Job：exit 9 / `JOB_MEMORY_LIMIT_MISMATCH`；
- 2048 MiB 命名 Job：guard 通过，但因授权仍为 `NOT_AUTHORIZED` 在 `load_auth()` 失败；未加载完整脑。
- 源码回归确认 `KILL_ON_JOB_CLOSE_MISSING` 分支存在，且没有 `_query_extended(None)` 匿名回退。

## 3. child_result 严格验证

`validate_child_result()` 现在强制检查：

- `seeds` 恰为 101–125：无重复、无遗漏、无越界。
- 每条记录同时包含 `none` 和 `stim`。
- 每条 `none.finite`、`stim.finite` 均为 true。
- 每条 `noise_pair_ok` 均为 true，顶层 `noise_pairing_all` 也为 true。
- 官方数量：`n_neurons=166700`、`n_kc=4064`、`n_mb=97`、`n_kc_matched=320`、`n_mb_matched=95`、`n_lp=311`。
- `steps=400`、`device=cpu`、`batch=1`、`training=False`、`dW=0`。
- `amount=0.036254`、`stim_window=[100,300)`。
- `removed_edges=642933`、`nnz_B=24940005`、`KC_KC_nnz=0`。
- `B_hash` 是合法 64 字符 SHA-256，并与全部 NPZ 的唯一 `w_hash` 相等。
- 50 个 NPZ 继续通过原有 seeds/条件/形状/有限性/标签/噪声配对/哈希校验。
- 任一项失败：`INCOMPLETE_NO_GO`，不进入统计后处理。

测试 `test_strict_child_result_valid_and_counterexamples` 覆盖：

- 正常 25 条、真实格式 `B_hash` → 通过；
- 重复 seed → 失败；
- 缺少 `none` → 失败；
- `finite=false` → 失败；
- 错误 W_B 哈希 → `B_HASH_RAW_MISMATCH`；
- 缺少 `n_lp` → `PROTOCOL_MISMATCH:n_lp`。

## 4. 冻结与历史保留

- 独立 S1R 脚本新 SHA-256：
  `fe15653c35eb13b6849223052279d7b28f83c4c258417816711f89d76b44a5b9`
- `p11_2_s1r_script_freeze.json` 与新 SHA 一致。
- `p11_2_s1r_authorization.json` 仍为 `NOT_AUTHORIZED`；无生产授权锁。
- 历史 S1 仍为 `INCOMPLETE_NO_GO`；6 个原始文件哈希和备份未变。
- 官方数据、P1-mapfix/readout/tags、三个既有 dirty 文件保持。
- 科学参数完全不变：B KC→KC-zero、none/stim、25 seeds、每条件 400 步、20,000 步、精确 LPLC2/LC4 311、amount=0.036254、[100,300)、CPU、900 s / 2048 MiB。

## 5. 测试与结论

- `loop.test_p11_2_s1r_launcher`：**29/29 passed**
- `loop.test_p11_2_s1f_launcher_audit`：**9/9 passed**
- 合计 **38/38 passed**

最终状态：

```text
STATIC_HARDENING=GO
S1R_EXECUTION_NOW=NO-GO
S1R_AUTHORIZATION=NOT_AUTHORIZED
S1=INCOMPLETE_NO_GO
RESOURCE_VERIFIED=NO
```

可以进入新一次 S1R 授权申请；但未获用户明确批准前，生产入口必须继续 fail-closed，不得创建生产锁或启动正式实验。
