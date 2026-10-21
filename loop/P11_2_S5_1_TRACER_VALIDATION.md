# P11-2-S5-1 — 小规模 LIF 轨迹记录器与逐步等价性验证

基线：`079da3f`
状态：`IMPLEMENTATION_EQUIVALENCE_GO; MODEL_MECHANISM_OBSERVABILITY_GO; BIOLOGICAL_VALIDITY_UNVALIDATED`
没有运行官方完整 MaleCNS、没有重跑 S3、没有新剂量实验。

## 1. 源码冻结

- 文件：`C:\AA_project\guoying\.venv\Lib\site-packages\flybrain\brain.py`
- SHA-256：`142ecfffd5d32e30655829e34b38b04024e49beae662bd078279e9610c789b0c`
- 与 S5-0 一致；不修改 site-packages。

## 2. 观测原型

新增 `loop/p11_2_s5_1_tracer.py`：

- `InstrumentedFlyBrain` 继承 `FlyBrain`，只支持 **CPU batch=1**；
- 完整保留 step 计算顺序：
  `synaptic*gain → leak → +tonic → Bernoulli noise → optional eye → inject → refractory → threshold → reset → state/step`；
- `TraceWriter` 以固定 90-byte 记录流式写 `.bin`，同时生成 `.bin.json` 头：
  - `target_indices` 顺序；
  - 字段/ dtype/单位；
  - 记录数、二进制 SHA-256。
- 只记录预指定目标，不保存完整脑 v 历史。

字段：

```text
step, neuron
v_start, v_leak, synaptic (with gain), tonic, noise, eye, inject
v_pre_threshold, v_post_reset, threshold_distance
fired, refractory_zeroed
```

单位均为 FlyBrain 模型单位；`v` 无量纲，不能标 mV/nA。

## 3. 逐步等价性

测试文件 `loop/test_p11_2_s5_1_tracer.py` 使用 **12 个神经元的合成网络**（非官方完整脑）：

- 正负符号稀疏权重；
- 无注入、low、primary 注入；
- Bernoulli 噪声；
- eye_drive；
- 发放/reset；
- 无发放、持续发放；
- 非零 refractory；
- logging 开/关。

结果：

- 测试 **7/7 passed**；
- 6 个独立等价性场景；
- `fired` 索引逐 step 完全一致；
- `v` 逐 step 逐位一致；
- RNG `bit_generator.state` 完全一致；
- `steps`、`last_spike` 一致；
- logging 开/关结果一致；
- 输入分项闭合：`v_start*decay + synaptic + tonic + noise + eye + inject` 与 `v_pre_threshold` 在 2e-6 内一致（非 refractory 细胞）；
- `v_post_reset = 0`（发放）或 `= v_pre`（未发放），`threshold_distance=1-v_pre`。

局限：仅 12 神经元、CPU、batch=1；未验证官方 166,700 或 CUDA。

## 4. 可观测性

本轮确实从 instrumented step 读取了：

- step 前 v；
- leak 后 v；
- 带 gain 的突触输入；
- tonic、实际 Bernoulli noise、eye、inject；
- 阈值前 v、reset 后 v、fired、阈值距离。

不是从 step 后的 `brain.v` 反推。
现有原版 API 本身无法提供这些；必须使用本记录器或等价 instrumentation。

## 5. 二进制格式与资源

字段固定 dtype；`TRACE_DTYPE.itemsize = 90 bytes`。

静态估计（只记录目标，不记录全脑 v）：

| 目标集合 | 400 步记录 |
|---|---:|
| matched 320 KC + 95 MBON（415） | ~14.25 MiB |
| 415 + 1000 个中继细胞 | ~34.33 MiB |
| 1415 个目标 | ~48.58 MiB |

这些是静态估计，不是完整脑实测；instrumentation 的时间开销和真实峰值内存 **UNVALIDATED**。
记录器尚未接入 S3 父入口，不能直接用于正式运行。

## 6. 生物真实性边界

严格区分：

- **IMPLEMENTATION_EQUIVALENCE=GO**：在小合成网络上，观测器与 FlyBrain.step 等价。
- **MODEL_MECHANISM_OBSERVABILITY=GO**：可以读取模型内部的 pre-reset v 和分项贡献。
- **BIOLOGICAL_VALIDITY=UNVALIDATED**：阈值/reset/tonic/gain/noise/tau 仍是工程/手工参数；W 仅是解剖连接组。观测器只让模型机制可见，不会把模型变成生物校准模型。

## 7. 五个问题

1. **与原版 FlyBrain.step 是否逐步等价？**
   在 12 神经元合成 CPU batch=1 测试中：是，v/fired/RNG/state 一致。官方完整脑未测试。

2. **是否真实取得 pre-reset v 和每项贡献？**
   是，instrumented step 直接记录；没有从 post-step v 反推。

3. **开启日志是否改变 RNG 和神经状态？**
   否，合成测试中 logging 开/关得到相同 v/RNG/fired/state。

4. **未来完整脑观测内存/时间风险？**
   静态估计 415 目标约 14.25 MiB，带 1000 中继约 34.33 MiB，1415 目标约 48.58 MiB；内存估算较低但时间开销和真实峰值 **未验证**。

5. **是否具备设计新独立观测实验的条件？**
   原型具备进一步合成验证条件，但**还不具备直接上完整脑**的条件；应先完成 instrumentation 的完整脑等价性/资源验证，并取得新的明确授权。当前仍优先 B：先修正模型解释。

## 8. 边界

```text
NEW_FULL_BRAIN_AUTHORIZATION=False
TRAINING_AUTHORIZATION=False
REAL_SYNAPTIC_LEARNING_READY=NO
Level A/B/C/Biological=NO-GO
Robot/Maze/Dijkstra/DeltaW=NO-GO
```

未重跑 S1R/S3，未提高剂量，未训练，未修改安装的 flybrain 包，未接入正式 S3。
