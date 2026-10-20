# P11-2-S1R-1b — 统计序列化与完整成功链修复

基线：`719cb2a`
状态：`S1R1B_POSTPROCESS_COMPLETE; NOT_AUTHORIZED; NO_RUN`
`READY_TO_REQUEST_ONE_S1R_RUN=YES`
`NEW_S1R_EXECUTION_AUTHORIZATION=False`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
Level A/B/C/Biological=`NO-GO`

本轮只做静态修复、合成 25×2 NPZ 后处理、Windows 合成 Job 回归；没有初始化完整 MaleCNS、没有调用 `FlyBrain.step`、没有正式 S1R。

## 1. NumPy JSON 序列化修复

`analyze()` 的 `per_seed` 不再包含以下 ndarray：

- `kc_cell_none` / `kc_cell_stim`
- `mb_cell_none` / `mb_cell_stim`
- `kc_matched_cell_none` / `kc_matched_cell_stim`
- `mb_matched_cell_none` / `mb_matched_cell_stim`

逐 seed 仅保留 JSON 标量、字符串、布尔值和必要短列表；`low_none`/`low_stim` 改为长度 3 的 float 列表。
完整细胞发放数组仍保存在 50 个原始 NPZ。群体 per-cell 汇总继续从当前 raw 数组计算，未丢失统计。

测试：

- `json.dumps(analyze(...))` 成功；
- 成功报告的 `per_seed` 不含上述数组键；
- `secondary.kc_cell_none` 等 per-cell 汇总仍存在。

## 2. 统一状态链

`analyze()` 返回 `status="ANALYSIS_COMPLETE"`。父进程统一状态：

| 阶段 | 状态 |
|---|---|
| 统计完成（尚未持久化） | `ANALYSIS_COMPLETE` |
| 25/25、50 NPZ、统计后哈希、report 与 manifest 成功落盘 | `RUN_COMPLETE` |
| `analyze` 或统计后哈希异常 | `POSTPROCESS_FAILED` / `INCOMPLETE_NO_GO` |
| report 或 manifest 写入失败 | `PERSISTENCE_FAILED` / `INCOMPLETE_NO_GO` |
| child/raw 不完整 | `INCOMPLETE_NO_GO` |

`run_meta` 初始为 `INCOMPLETE_NO_GO`；只有 report 与 manifest 均成功写入后，才写最终 `RUN_COMPLETE` run_meta。
报告/清单失败时会尽力用 `PERSISTENCE_FAILED` 失败报告覆盖，禁止虚假 `RUN_COMPLETE`。

## 3. 完整合成成功链

测试 fixture 使用 25 seeds × none/stim = 50 个真实维度 NPZ：

- shapes：`kc(400,4064)`、`mb(400,97)`、`lp(400,311)`、`kc_matched(400,320)`、`mb_matched(400,95)`；
- 真实格式 64 字符 SHA-256 RNG before/after；
- 真实格式 `w_hash`/`B_hash`；
- 严格 child_result 字段、seeds 101–125、finite、noise pairing；
- `ids.npz` 与完整原始 manifest。

端到端（生产后处理函数）：

```text
generate 50 NPZ
→ validate_raw_dataset
→ validate_child_result
→ analyze
→ json.dumps / report
→ run_meta
→ raw manifest
→ status consistency
```

成功结果：

```text
run_meta.status      = RUN_COMPLETE
report.status        = RUN_COMPLETE
manifest.status      = RUN_COMPLETE
s1r_status           = RUN_COMPLETE
experiment_succeeded = true
```

## 4. 失败路径合成测试

- **report 写入失败**：状态 `PERSISTENCE_FAILED`，`s1r_status=INCOMPLETE_NO_GO`；run_meta 不是 `RUN_COMPLETE`。
- **manifest 写入失败**：report/manifest/run_meta 均不出现最终 `RUN_COMPLETE`。
- **analyze 异常**：`POSTPROCESS_FAILED` / `INCOMPLETE_NO_GO`。
- **child/raw 不完整**：`INCOMPLETE_NO_GO`，不进入统计后处理。
- 所有失败路径 `experiment_succeeded=false`。

## 5. 冻结、历史与资源

- 独立 S1R 实验模块新 SHA-256：
  `9fbf0ab44d77adfdd9beb0a275b65181f5c3989820dd517dba22217870ae272b`
- freeze 与实验模块严格一致。
- 授权仍为 `NOT_AUTHORIZED`；无生产一次性锁。
- 历史 S1 仍为 `INCOMPLETE_NO_GO`；6 个原始文件与备份 hash 未变。
- 官方数据、P1-mapfix/readout/tags、三个既有 dirty 文件保持。
- 科学参数未变：B KC→KC-zero、none/stim、25 seeds、每条件 400 步、20,000 步、精确 LPLC2/LC4 311、amount=0.036254、[100,300)、CPU、900 s / 2048 MiB。
- C1 peak `1942.7 MiB`，余量 `105.3 MiB`；`RESOURCE_VERIFIED=NO`，不得宣称资源必然足够。

## 6. 测试与最终状态

- `loop.test_p11_2_s1r_launcher`：**34/34 passed**
- `loop.test_p11_2_s1f_launcher_audit`：**9/9 passed**
- 合计 **43/43 passed**

```text
STATIC_HARDENING=GO
READY_TO_REQUEST_ONE_S1R_RUN=YES
S1R_EXECUTION_NOW=NO-GO
S1R_AUTHORIZATION=NOT_AUTHORIZED
S1=INCOMPLETE_NO_GO
```

已具备提交新一次 S1R 一次性授权申请的条件。未获用户明确批准前，禁止创建生产授权锁或启动正式 S1R。
