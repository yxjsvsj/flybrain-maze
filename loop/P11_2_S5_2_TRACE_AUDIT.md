# P11-2-S5-2 — 轨迹记录准确性、二进制回读与资源预检

基线：`338ee94`
状态：`TRACE_CORRECTNESS_GO; IMPLEMENTATION_EQUIVALENCE_GO; CPU_BATCH1_ONLY; BIOLOGICAL_VALIDITY_UNVALIDATED`
未运行官方完整脑、未重跑 S3、未创建新正式授权。

## 1. 注入贡献修复

此前日志用 `_amount(amount)` 的第一个元素推断所有目标，无法保证非均匀/重叠注入正确。

现在：

- 在应用全部注入组前记录目标 v；
- 按原顺序逐组执行 `self.v[idx] += amount`；
- 注入贡献 = 应用后目标 v − 应用前目标 v；
- 支持多个注入组、目标重叠和重复目标；
- 不再逐目标 `np.isin()` 扫描注入索引；
- batch=1 下每神经元非均匀向量（size>1）明确抛出 `UNSUPPORTED_INJECT_AMOUNT`，且在修改 v/RNG/steps 之前拒绝。

合成测试覆盖三组重叠注入：

```text
group1 target 1: +0.030
group2 targets 1,2: +0.010
group3 targets 2,3: -0.005
=> inject log [0, +0.040, +0.005, -0.005]
```

与原版 `FlyBrain.step` 的 fired/v/RNG 逐 step 一致。

## 2. 支持范围

严格限制：

- NumPy CPU；
- batch=1；
- CUDA、其他 backend、batch>1 在构造函数或 step 明确拒绝；
- 安装的 `flybrain/brain.py` 未修改；
- 源码 SHA 仍为 `142ecfffd5d32e30655829e34b38b04024e49beae662bd078279e9610c789b0c`。

## 3. 二进制回读

`TraceWriter` / `read_trace` 格式：

- `S5TRACE/1`；
- 固定 **90-byte** record；
- 字段：`step, neuron, v_start, v_leak, synaptic, tonic, noise, eye, inject, v_pre_threshold, v_post_reset, threshold_distance, fired, refractory_zeroed`；
- `.bin` 数据 + `.bin.json` 头（目标顺序及哈希、字段、dtype、单位、记录数、二进制 SHA-256）。

`read_trace` 强制检查：

- 文件/头存在；
- protocol/format、字段、dtype、itemsize；
- target order 及 target_order SHA；
- record count 与文件字节数；
- 整个二进制 SHA-256；
- 每行 step、neuron 顺序；
- 返回按 `(step, target)` 排列的 structured rows。

测试逐行比较 `.bin` 回读与内存 `last_trace` 的 v_start、leak、synaptic、tonic、noise、eye、inject、pre-threshold、post-reset、threshold distance、fired、refractory。

拒绝：缺失文件/头、截断、SHA 变化、目标顺序不符。

## 4. 资源预检

只写 **一组**合成 400 步；没有创建 75 组输出。

实测（writer-only）：

| 目标数 | records | 文件 | 写时间 | tracemalloc 峰值 |
|---:|---:|---:|---:|---:|
| 415 | 166,000 | 14.25 MiB | 0.0409 s | 0.0838 MiB |
| 1415 | 566,000 | 48.58 MiB | 0.0974 s | 0.2640 MiB |

完整协议（75 组）仅作算术估计：

| 目标数 | 单组 | 75 组磁盘 | 备份×2 | writer-only 时间估算 |
|---:|---:|---:|---:|---:|
| 415 | 14.25 MiB | 1.0435 GiB | 2.0871 GiB | 3.07 s |
| 1415 | 48.58 MiB | 3.5581 GiB | 7.1162 GiB | 7.31 s |

这些是**日志 writer** 的测量/算术，不是完整脑实测：

- 未包含 instrumentation 与完整脑 step 的额外时间；
- 未测量完整脑进程 RSS/峰值；
- 不能据此宣称 900 秒或 2048 MiB 已安全；
- 未生成 75 组大规模输出。

## 5. 等价性

复用 12 神经元合成网络，原 S5-1 测试继续通过，并新增：

- 非均匀/重叠注入；
- 非均匀向量拒绝且状态不变；
- CPU/batch 边界；
- 二进制 round-trip 内容；
- 截断/SHA/目标顺序拒绝；
- 415/1415 单组 writer 压测。

结果：**14/14 tests passed**。

逐 step 检查：

- fired 完全一致；
- v 逐位一致；
- RNG bit_generator.state 一致；
- steps/last_spike 一致；
- logging 开/关不改变神经行为；
- 输入贡献闭合；
- 二进制内容与内存轨迹一致。

## 6. 生物真实性边界

- `TRACE_CORRECTNESS=GO`
- `IMPLEMENTATION_EQUIVALENCE=GO`
- `CPU_BATCH1_ONLY`
- `MODEL_MECHANISM_OBSERVABILITY=GO`（合成范围）
- `BIOLOGICAL_VALIDITY=UNVALIDATED`

本轮没有验证 LIF 阈值、tonic、gain、噪声或生物电流语义；记录器只让模型内部轨迹可审计。

## 7. 五个问题

1. **TRACE_CORRECTNESS 是否通过？** 通过。非均匀/重叠注入、round-trip、截断/SHA/顺序拒绝均通过。
2. **IMPLEMENTATION_EQUIVALENCE 是否仍通过？** 在 12 神经元合成 CPU batch=1 上通过；官方完整脑未测试。
3. **CPU/batch 边界是否严格执行？** 是，NumPy CPU batch=1；CUDA/其他/batch>1 明确拒绝。
4. **单组/完整协议资源口径是否正确？** 单组已实测；75 组只作算术估计；完整脑内存/时间未验证，不能声称安全。
5. **是否具备进入独立完整脑观测实验预注册设计的条件？** 具备**预注册设计**的工程基础；不具备直接执行完整脑观测的条件。仍需官方脑等价性/资源验证和新授权。

## 8. 边界

```text
NEW_FULL_BRAIN_AUTHORIZATION=False
TRAINING_AUTHORIZATION=False
REAL_SYNAPTIC_LEARNING_READY=NO
Level A/B/C/Biological=NO-GO
Robot/Maze/Dijkstra/DeltaW=NO-GO
BIOLOGICAL_VALIDITY=UNVALIDATED
```

未接入 S1R/S3 正式入口，未重跑历史实验，未修改旧结果和归档。
