# P11-2-S1R-1c — 两项定点勘误

基线：`8702272`
状态：`S1R1C_ERRATA_COMPLETE; NOT_AUTHORIZED; NO_RUN`
`READY_TO_REQUEST_ONE_S1R_RUN=YES`
`NEW_S1R_EXECUTION_AUTHORIZATION=False`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
Level A/B/C/Biological=`NO-GO`

本轮只做两项源码已确认缺陷的离线修复、真实维度合成 NPZ 回归；没有运行完整脑、FlyBrain.step 或正式 S1R。

## 1. 完整 97 MBON 的 none 数据源

修复：

```python
kcn = zn["kc"]; kcs = zs["kc"]; mbn = zn["mb"]; mbs = zs["mb"]
```

此前 `mbn = zs["mb"]`，导致完整 97 MBON 的 none 统计误用 stim 数据。
匹配 95 MBON 的 `mbmn = zn["mb_matched"]` / `mbms = zs["mb_matched"]` 未改。

非零差异合成回归（窗口 `[100,300)` 长度 200）：

- 完整 97 MBON：none `20/200=0.10`，stim `40/200=0.20`；
- 完整 97 MBON per-cell `mean_rate`：none `0.10`，stim `0.20`；
- 匹配 95 MBON：none `10/200=0.05`，stim `30/200=0.15`，配对差 `0.10`；
- `per_seed.full_mb_none/stim` 与 `secondary.full_97_mbon_none/stim` 均正确；
- 未使用全零数据。

主终点、刺激、统计门槛、seed、LIF、B 网络均未改变。

## 2. raw manifest 的 run_meta 哈希

问题：`_manifest()` 会把当前 `run_meta.json` 的旧哈希放进 `files`，而最终 `RUN_COMPLETE` run_meta 之后才写入，导致 listed hash 失效。

修复：

- `_manifest()` 的 `files` 只列不可变文件，明确排除 `run_meta.json`。
- 50 个 NPZ 的文件名、大小、逐文件 SHA-256 继续完整保留。
- manifest 用独立的 `run_meta_sha256` 记录**最终** run_meta 的字节哈希。
- run_meta 中不反向保存 manifest 哈希，避免循环引用。
- 合成成功后重新计算 manifest `files` 中每个文件：全部 SHA-256 与大小匹配。
- `run_meta_sha256` 与实际最终 run_meta 文件 SHA-256 匹配。
- `run_meta.json` 不出现在 `files` 中。

`PERSISTENCE_FAILED` / `POSTPROCESS_FAILED` / `INCOMPLETE_NO_GO` 路径保持：report/manifest 失败仍写入失败状态，最终 run_meta 不会虚假标记成功。

## 3. 冻结与历史

- 独立 S1R 实验模块新 SHA-256：
  `2a57790479e69f8aded1ccc37257d2b375d47de50255d06bb3580d61bc71ab93`
- `p11_2_s1r_script_freeze.json` 与新 SHA 一致。
- 授权仍为 `NOT_AUTHORIZED`；无生产锁。
- 历史 S1 仍为 `INCOMPLETE_NO_GO`；原始六文件与备份 hash 未变。
- 官方数据、P1-mapfix/readout/tags、三个既有 dirty 文件保持。
- 资源仍限定 `900 s / 2048 MiB`；C1 peak `1942.7 MiB`，余量 `105.3 MiB`；`RESOURCE_VERIFIED=NO`。

## 4. 测试与最终状态

- `loop.test_p11_2_s1r_launcher`：**36/36 passed**
- `loop.test_p11_2_s1f_launcher_audit`：**9/9 passed**
- 合计 **45/45 passed**

```text
STATIC_ERRATA=GO
READY_TO_REQUEST_ONE_S1R_RUN=YES
S1R_EXECUTION_NOW=NO-GO
S1R_AUTHORIZATION=NOT_AUTHORIZED
S1=INCOMPLETE_NO_GO
```

已停止静态加固，等待用户明确审批一次 S1R；未创建生产授权锁，未运行正式实验。
