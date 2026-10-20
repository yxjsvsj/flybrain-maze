# P11-2-S3-1 — KC→MBON 直接驱动生产实现与静态验收

基线：`7305325`
状态：`S3_1_IMPLEMENTATION_COMPLETE; NOT_AUTHORIZED; NO_RUN`
`NEW_S3_EXECUTION_AUTHORIZATION=False`
`TRAINING_AUTHORIZATION=False`
`REAL_SYNAPTIC_LEARNING_READY=NO`
Level A/B/C/Biological=`NO-GO`

本轮只做生产代码、静态核查、合成 NPZ 和合成 Job 测试。没有运行 FlyBrain.step、没有初始化完整 MaleCNS、没有创建 S3 APPROVED 授权或生产锁。

## 1. 独立生产链

新增：

| 文件 | 作用 |
|---|---|
| `loop/p11_2_s3_launcher.py` | S3 父入口：授权/冻结/锁/Job/postprocess |
| `loop/p11_2_s3_experiment.py` | 独立 S3 子模块：KC 直接驱动、75 NPZ、分析 |
| `loop/p11_2_s3_authorization.json` | 独立 S3 授权模板 | `NOT_AUTHORIZED` |
| `loop/p11_2_s3_script_freeze.json` | 独立 S3 子模块冻结 |
| `loop/P11_2_S3_ONE_RUN_APPROVAL.md` | 授权模板 | `NOT_AUTHORIZED` |

安全链复用 S1R/S1F 已验证机制：

- `python -u -m loop.p11_2_s3_experiment --child <out_dir>`；
- `CREATE_SUSPENDED → AssignProcessToJobObject → ResumeThread`；
- 2048 MiB / 900 s；
- 可继承 stdout/stderr；
- 命名 Job 身份、KILL_ON_JOB_CLOSE、PID 成员核验；
- 原子一次性锁；失败不删除、不自动重试；
- 未授权时在创建 Job/完整脑之前 `REFUSED/NOT_AUTHORIZED`。

## 2. 注入 API 与靶点静态核验

检查 `flybrain/brain.py` 的实际实现：

- `step(inject=[(indices, amount)])` 把 `amount` 直接加到膜电位 `v`；
- **amount 是电压增量，不是 nA 电流**；
- indices 是完整脑的全局神经元索引；
- LIF：`v <- exp(-dt/tau)v + gain*W@spikes + tonic + noise + inject`，`v>=1` spike/reset。

冻结靶点：S1R matched-320 KC 全局索引；SHA-256：
`6e5f35f5662578fa06ec396973befc34bd53f171985fb8f752c3a8e37d350385`

剂量未改变：

- primary：ΔV∞=0.20，amount `0.03625384938440364`；
- low：ΔV∞=0.10，amount `0.01812692469220182`。

## 3. 75 NPZ 与统计

新增严格 `validate_raw_75()`：

- 25 seeds（201–225）× 3 条件（sham/kc_low/kc_primary）= **75 NPZ**；
- 形状：`kc(400,4064)`、`mb(400,97)`、`lp(400,311)`、`kc_matched(400,320)`、`mb_matched(400,95)`；
- 靶点全局索引与冻结 SHA、条件标签、注入量、时间窗；
- 三条件 RNG before/after 配对；
- finite、B 网络 hash、脚本 SHA、授权 checksum；
- 每文件 SHA-256、dataset hash；
- manifest 不把可更新的 run_meta 放进不可变 `files`；最终 run_meta 有独立真实字节 hash。

统计门未改变：

- KC：primary−sham ≥ `+0.10`、exact sign-flip p<0.05、bootstrap CI low>0；
- 饱和门：primary mean<0.85，per-cell mean≥0.95 比例≤0.10；
- MBON：仅 KC 成功后才解释，primary−sham ≥ `+0.05`、p<0.05、CI low>0；
- low dose：次级，BH 校正，不替代 primary。

### 判别树解释勘误

- KC 未驱动 / 饱和：`INCONCLUSIVE_KC_NOT_DRIVEN` / `INCONCLUSIVE_KC_SATURATED`；不解释 MBON。
- KC 成功 + MBON 正向门通过：`KC_MBON_FUNCTIONAL_UNDER_DIRECT_DRIVE`。
- KC 成功 + MBON 显著负向（≤−0.05, p<0.05, CI high<0）：`KC_MBON_NEGATIVE`。
- **其他所有 MBON 结果（小幅显著正向但 <+0.05、非显著、或未达显著负向）统一为 `INCONCLUSIVE_DOWNSTREAM`**，报告真实效应量、方向、p、CI；不得解释为“MBON 功能不存在”。
- 人工直接同步 KC 驱动只说明该合成 B 模型在该干预条件下的功能性；不能证明自然感觉传递、单条 KC→MBON 突触专一因果或生物学习。

## 4. 合成验收

`loop/test_p11_2_s3_launcher.py` 使用 25×3 合成 NPZ 和合成 Job，完整覆盖：

| 路径 | 结果 |
|---|---|
| 完整成功 | GO / `KC_MBON_FUNCTIONAL_UNDER_DIRECT_DRIVE` |
| KC 未达 +0.10 | NO-GO / `INCONCLUSIVE_KC_NOT_DRIVEN` |
| KC 饱和 | NO-GO / `INCONCLUSIVE_KC_SATURATED` |
| MBON 正向 | GO |
| MBON 显著负向 | NO-GO / `KC_MBON_NEGATIVE` |
| MBON 小幅显著正向 | INCONCLUSIVE / `INCONCLUSIVE_DOWNSTREAM` |
| MBON 非显著 | INCONCLUSIVE / `INCONCLUSIVE_DOWNSTREAM` |
| 缺 NPZ、重复 seed、哈希破坏 | NO-GO / `INCOMPLETE_NO_GO` |
| 异常退出/超时/内存超限 | NO-GO / `INCOMPLETE_NO_GO` |
| report/manifest 写入失败 | NO-GO / `PERSISTENCE_FAILED` |
| 第二次执行 | NO-GO / `AUTHORIZATION_ALREADY_CONSUMED` |
| 未授权生产入口 | NO-GO / `REFUSED_NOT_AUTHORIZED` |

结果为 **8/8 synthetic tests passed**。合成结果不是完整脑科学结果。

## 5. 资源

- S1R 实测：275.591 s / 1943.16 MiB。
- S3 静态估计：30,000 步 / 约 413.387 s / 约 1948.16 MiB。
- 上限：900 s / 2048 MiB。
- 状态：`RESOURCE_PLAUSIBLE_TIGHT`，`RESOURCE_VERIFIED=NO`。
- 没有用完整脑试跑校准内存/剂量；不得自动提高上限。

## 6. 五个问题

1. **生产入口是否完整实现？**
   是。父入口、子模块、NOT_AUTHORIZED 授权模板、独立 freeze、锁、命名 Job、75 NPZ 校验、统计/持久化链和合成测试均已实现。

2. **靶点 SHA 与注入语义是否严格核验？**
   是。320 个 matched KC 全局索引及 SHA 冻结；FlyBrain 源码确认 `inject` 是直接电压增量，不是 nA；剂量未改。

3. **75 NPZ、配对、统计和状态链是否通过合成测试？**
   是。25×3/75 NPZ、三条件 RNG 配对、KC/饱和/MBON 门、判别树、缺失/重复/哈希、异常、持久化失败均覆盖；8/8 通过。

4. **资源是否仍为 RESOURCE_PLAUSIBLE_TIGHT？**
   是。静态估计 413.387 s / 1948.16 MiB，余量约 100 MiB；未验证，不得写成已验证安全。

5. **是否具备申请全新一次性 S3 授权的条件？**
   是，静态验收完成；但当前 S3 授权为 `NOT_AUTHORIZED`，无生产锁，未运行 S3。若要运行，需用户明确批准新的一次性授权。

## 7. 边界

- 未修改历史 S1R/S2 数据、S1R 授权/锁、官方连接组、P1/readout/tags 或三个既有无关 dirty 文件。
- 未训练、未更新真实 ΔW、未运行 Dijkstra/迷宫/B/C/实体车。
- 没有创建 S3 APPROVED 授权或生产消耗锁。
