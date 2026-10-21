# P11-2-S5-3 — 机制观测生产链与合成验收

基线：`1855e91`
状态：`PREREGISTRATION_COMPLETE; SYNTHETIC_ACCEPTANCE_GO; TRACE_INTEGRITY_GO; RESOURCE_PLAUSIBLE; FULL_BRAIN_EQUIVALENCE_UNVERIFIED; NOT_AUTHORIZED; NO_RUN`

没有加载官方完整 MaleCNS，没有运行 S3，也没有创建 APPROVED 授权或生产锁。

## 1. 独立生产链

| 文件 | 作用 |
|---|---|
| `loop/p11_2_s5_3_experiment.py` | 独立 S5-3 子实验：415 目标轨迹、15 组 |
| `loop/p11_2_s5_3_launcher.py` | 默认拒绝父入口、命名 Job、一次性锁、报告 |
| `loop/p11_2_s5_3_authorization.json` | 独立授权模板 | `NOT_AUTHORIZED` |
| `loop/p11_2_s5_3_script_freeze.json` | experiment/tracer/source/target freeze |
| `loop/P11_2_S5_3_ONE_RUN_APPROVAL.md` | 新授权模板 | `NOT_AUTHORIZED` |

- 未接入 S1R/S3 父入口；未复用 S3/S1R 授权或锁。
- CPU、NumPy、batch=1；CUDA/其他 backend 由 tracer 拒绝。
- `CREATE_SUSPENDED → Job → Resume`；命名 Job 核验 2048 MiB、KILL_ON_JOB_CLOSE、PID 成员。
- 原子一次性锁；失败/超时/内存超限停止；不自动重试。
- 未授权入口在创建完整脑/Job 前拒绝。

## 2. 冻结目标与轨迹

- KC 320 + MBON 95 = **415**。
- KC SHA：`6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385`
- MBON SHA：`2248dccc680981af16270f12f6365ca5832fab2a12bb5ace3b80cc73580b61a6`
- 联合顺序 SHA：`942d87d267711edf943b0804ab9089ab23cec7532f017d10ebf81c3ec4c12567`
- 顺序：sorted matched KC（320）后接 sorted matched MBON（95）；tracer 不再全局重排目标。
- 15 组 = seeds 301–305 × sham/kc_low/kc_primary；每组 400 步。
- 只记录这 415 个目标；S5TRACE/1；90-byte records；不保存全脑 v 历史。
- 不注入 MBON/感觉细胞；不记录中继细胞。

## 3. 合成验收

`loop/test_p11_2_s5_3_launcher.py`：**6/6 passed**。

覆盖：

- 未授权生产入口拒绝；
- 合成一次性锁与第二次拒绝；
- 15 组 415 目标合成轨迹成功 + report/manifest/run_meta 完整性；
- 缺失、截断、SHA 篡改、目标顺序错误拒绝；
- report/manifest 持久化失败 → `PERSISTENCE_FAILED`；
- 合成 Job 异常退出、超时、内存限制。

另：S5-1/S5-2 tracer 回归 **15/15 passed**（含目标顺序保持、非均匀/重叠注入、round-trip、writer 压测）。

`_validate` 对每组逐行检查：

- target order / joint SHA；
- records、shape、finite、fired 0/1、threshold distance；
- `v_start*decay + synaptic + tonic + noise + eye + inject = v_pre_threshold`（非 refractory）；
- post-reset 与 fired 关系；
- trace header SHA 与 child metadata；
- 5 seeds 的三条件 RNG 配对。

## 4. 资源

S5-2 writer 实测（单组 415 目标 ×400 步）：

- 14.247894 MiB；writer-only 0.0409 s；tracemalloc 0.0838 MiB。

15 组算术：

- 轨迹原始：**213.718 MiB**；
- 备份后：**427.437 MiB**；
- writer-only 时间估算：**0.61 s**。

S3 实测 Job peak：`1932.29 MiB`。
新 InstrumentedFlyBrain 在官方脑内的额外时间/峰值/动力学等价性：**UNVERIFIED**。
未同时初始化两份完整脑；没有用静态/合成数据声称完整脑 900 s/2048 MiB 安全。

## 5. 5 个问题

1. **TRACE_CORRECTNESS 通过？** 合成 15 组/415 目标通过；缺失/截断/哈希/顺序均拒绝。
2. **IMPLEMENTATION_EQUIVALENCE 仍通过？** 小合成 CPU batch=1 通过；官方完整脑等价性 **UNVERIFIED**。
3. **CPU/batch 边界严格执行？** 是。
4. **资源口径正确？** 单组 writer 实测；15 组仅算术；完整脑额外资源未验证；`RESOURCE_PLAUSIBLE`，不能声称已安全。
5. **是否具备申请一次独立观测运行的条件？** **YES_CONDITIONAL**：预注册和合成验收完成，可提交新授权申请；执行前仍需新的明确批准，并接受 `FULL_BRAIN_EQUIVALENCE_UNVERIFIED` 与资源未实测的风险。

## 6. 最终判断

```text
PREREGISTRATION_COMPLETE
SYNTHETIC_ACCEPTANCE=GO
TRACE_INTEGRITY=GO
RESOURCE_PLAUSIBLE
FULL_BRAIN_EQUIVALENCE_UNVERIFIED
READY_TO_REQUEST_ONE_OBSERVATION_RUN=YES_CONDITIONAL
EXECUTION_AUTHORIZATION=NOT_AUTHORIZED
```

## 7. 边界

```text
NEW_FULL_BRAIN_AUTHORIZATION=False
NEW_S5_3_EXECUTION_AUTHORIZATION=False
TRAINING_AUTHORIZATION=False
REAL_SYNAPTIC_LEARNING_READY=NO
BIOLOGICAL_VALIDITY=UNVALIDATED
Level A/B/C/Biological=NO-GO
Robot/Maze/Dijkstra/DeltaW=NO-GO
```

未执行完整脑或 S3；未创建 APPROVED 授权/生产锁；未修改历史结果和归档。
