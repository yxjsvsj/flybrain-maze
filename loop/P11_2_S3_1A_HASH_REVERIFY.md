# P11-2-S3-1a — 统计后原始数据哈希核验

基线：`c34103e`
状态：`S3_1A_HASH_REVERIFY_COMPLETE; NOT_AUTHORIZED; NO_RUN`
`READY_TO_REQUEST_ONE_S3_RUN=YES`

本轮只修复一个源码确认的缺陷，没有运行 FlyBrain.step、完整脑或正式 S3。

## 缺陷

`postprocess_and_persist()` 在 `exp.analyze()` 后直接设置 `post_stats_hashes_ok=True`，但没有重新计算统计前 75 个 NPZ 的 SHA-256。统计过程中若文件被修改，最终仍可能被错误标记为 `RUN_COMPLETE`。

## 修复

新增：

```text
loop.p11_2_s3_experiment.verify_raw_75_hashes(out_dir, files)
```

对 `validate_raw_75()` 返回的每个文件逐一检查：

- 文件存在；
- 字节大小严格一致；
- SHA-256 严格一致；
- 返回 `ok`、`mismatches`、`checked`。

`postprocess_and_persist()` 的顺序改为：

```text
validate_raw_75
→ analyze
→ verify_raw_75_hashes
→ 仅当全部匹配才允许 RUN_COMPLETE
→ report
→ manifest
→ final run_meta
```

不匹配时：

- `status = POSTPROCESS_FAILED`
- `run_status = INCOMPLETE_NO_GO`
- `experiment_succeeded = False`
- `scientific_go = False`
- `post_stats_hashes_ok = False`
- `post_stats_hash_mismatches` 记录具体文件
- 不自动重跑；不会出现虚假 `RUN_COMPLETE`

## 合成测试

新增 `test_post_stats_hash_tamper_rejects_run_complete`：

- 使用完整 25 seeds × 3 conditions 合成 fixture；
- 在 `analyze()` 返回后故意改写 `201_sham.npz`；
- 结果必须是 `POSTPROCESS_FAILED`，mismatch 包含 `201_sham.npz`；
- report/run_meta 不得出现 `RUN_COMPLETE`；
- `scientific_go=False`。

测试结果：**9/9 passed**（原有 8 项完整成功/失败/Job/持久化路径继续通过）。

## 冻结与状态

- S3 子实验脚本新 freeze SHA：
  `63154938b2d887cccfea7918ae27624e3c105ed5a86aa127b291d65239a2568e`
- 靶点、seeds、条件、剂量、窗口、统计门槛、900 s / 2048 MiB 均未变。
- S3 授权仍为 `NOT_AUTHORIZED`；无 S3 APPROVED 授权、无生产锁、无 S3 运行。
- 历史 S1R/S2、官方数据、P1/readout/tags 和三个既有无关 dirty 文件未修改。
- 资源仍为 `RESOURCE_PLAUSIBLE_TIGHT` / `RESOURCE_VERIFIED=NO`；未做完整脑校准。

## 结论

```text
STATIC_HASH_REVERIFY=GO
READY_TO_REQUEST_ONE_S3_RUN=YES
S3_EXECUTION_NOW=NO-GO
S3_AUTHORIZATION=NOT_AUTHORIZED
```

定点补丁完成。停止静态加固，等待用户明确批准一次 S3。
