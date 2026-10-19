# P11-2-C1 资源与安全计划（**未运行完整脑**）

**状态：PLANNED；PLAUSIBLE_UNVERIFIED；FULL_BRAIN_NOT_RUN。**
`P11-2-C1 Execution Authorization = False`。

本文件为一次未来 A/B 实验（A intact、B KC→KC zero）的资源与安全计划。
本轮**不得**使用真实完整脑初始化或运行来验证预算；以下为历史测量 + 线性外推 + 合成测试。

---

## 1. 冻结实验规模

- 完整 MaleCNS：166,700 神经元。
- A intact / B KC→KC zero；**25 seeds（101–125）**；每 seed **400 步**；无刺激。
- 总步数：**20,000**（2 × 25 × 400）。
- 主要窗口：[100,300)。
- 整体上限：**900 s（15 min）/ 2048 MiB（2 GiB）**。
- 一次性；**不自动重跑**。

---

## 2. 时间预算

历史基准（P11-1c，25 seeds × 400 步 = 10,000 步，完整脑）：

- 初始化：**9.7534 s**；25 seed 运行和：**142.787 s**；
- 重建总 wall：**152.54 s**；
- 每步：152.54 / 10,000 = **15.254 ms**。

C1 外推（保守按 2× 历史，含两次 init 的等效）：

- 20,000 步 × 15.254 ms ≈ **305.08 s**；
- 15 min = 900 s；余量 **594.92 s**（66.1%）。

实际 A/B 在一个进程、一个 brain 实例中顺序执行，init 只发生一次，因此 305.08 s 是**保守上界**。

**结论：时间上可能在 15 min 内完成；未验证。**

---

## 3. 内存预算

历史基准（P11-1b，完整脑）：Job 峰值 **1497.87 MB**（P11-1c 的 Job 峰值因父进程崩溃丢失，未重跑；
不得用 P11-1c 推断，故采用 P11-1b 的资源基准）。

派生权重：

- 官方 W：25,582,938 非零；CSC 约 **205,330,308 B（195.82 MiB）**；
- `W_B`：24,940,005 非零；CSC 约 **200,186,844 B（190.91 MiB）**；
- 两者在 A/B 切换期间同时存在（同一 brain 实例，替换 CSC 数组）。

保守估计：

- 1497.87 + 190.91 ≈ **1688.78 MiB**；
- 2 GiB = 2048 MiB；余量 **359.22 MiB**（17.5%）。

**结论：2 GiB 内可能完成；未验证。** 若 Job 峰值 >2048 MiB → **NO_GO**，不重跑。

---

## 4. 安全框架

复用 P11-1c 已通过测试的 Windows 框架：

1. **CREATE_SUSPENDED** 创建子进程（尚未执行用户代码）；
2. 创建 Job Object，设置 **进程级 + Job 级 2 GiB** 限制，`KILL_ON_JOB_CLOSE`；
3. `AssignProcessToJobObject` 成功后才 `ResumeThread`；失败立即终止子进程；
4. 父进程统一 **15 min** 截止；超时 → `TerminateJobObject`；
5. 不自动重试；不因失败重启；
6. **子进程结果与 Job 峰值内存在后处理前持久化**：
   - 子进程写 `child_result.json` + 每 seed A/B NPZ；
   - 父进程在 `run_under_job` 返回后**立即**写 `run_meta.json`（`job_peaks`、wall、exit_code、
     timeout/mem_exceeded），**然后**才做任何比较/统计；
   - 若父进程在比较阶段崩溃，资源数据已在磁盘，不需要重跑；
7. 官方 `brain.npz` / `weights.npz` **完整 SHA-256** 前后相等比较；
8. 实验脚本 SHA-256 在运行前固定并写入 `run_meta.json`；
9. 原始 W 不变；结束时再次验证；
10. 3 个既有无关工作区改动（`hardware/tof5_client.py`、`loop/diagnose.py`、
    `loop/p10c1b_decoder_freeze.json`）**不清理、不覆盖、不提交**。

---

## 5. 合成测试（已运行，未用完整脑）

`p11_2_c1_preflight.py` / `test_p11_2_c1_preflight.py`：

- 无 Job 的 512 MB 分配成功（对照）；
- 256 MiB Job 下 512 MB 分配失败（内存限制）；
- 分配失败时子进程被终止（不恢复）；
- 超时（1 s）终止；
- 崩溃子进程 exit 3 被正确记录；
- 不自动重试（调用次数 1）；
- 合成 W_B 矩阵：只删除 KC→KC，其他边逐位不变，无新增，无重归一化；
- 合成噪声：同 seed 原始 RNG 逐位相等；不同 seed 原始 RNG 不同。

**这些是合成/静态测试；不是完整脑资源验证。**

---

## 6. 运行顺序（未来一次性）

1. 验证官方 brain/weights SHA-256 与 preflight 记录一致；
2. 验证脚本 SHA-256；
3. 加载 W（只读），内存派生 `W_B`；验证 642,933 删除、无新增、其他边不变；
4. 创建 Job（2 GiB，kill-on-close），`CREATE_SUSPENDED` 启动子进程，assign 后 resume；
5. 子进程：一个 `FlyBrain` 实例；对每个 seed：
   - `reset(seed)`；安装 A（官方 CSC）；400 步 A；记录 matched 320 / full 4,064 / MBON；
   - `reset(seed)`；安装 B（W_B CSC）；400 步 B；记录同样内容；
   - 每 seed 立即写 NPZ；
6. 子进程写 `child_result.json`；
7. 父进程立即写 `run_meta.json`（含 Job 峰值）；
8. 结束后再次验证官方 W SHA-256；
9. 后处理：仅读 NPZ/JSON；exact sign-flip + bootstrap；
10. 任何缺失/超时/超内存 → **INCONCLUSIVE/NO_GO**；不重跑。

---

## 7. 停止规则

- 超过 **900 s** 或 Job 峰值 **>2048 MiB** → 终止并 `NO_GO`；
- 子进程崩溃/非零退出 → 记录，不重跑；
- 任一 seed 缺失 → `INCONCLUSIVE`；
- 官方 W 哈希变化 → 立即停止，实验无效；
- 不调参、不替换 seed、不删除 seed 114、不增加 seed、不做第二次运行。

---

## 8. 结论

| 项目 | 结果 |
|---|---|
| 20,000 步时间 | **305.08 s（估算）< 900 s** |
| 内存 | **1688.78 MiB（估算）< 2048 MiB** |
| 资源状态 | **PLAUSIBLE_UNVERIFIED** |
| 完整脑本轮运行 | **NO** |
| 安全框架 | **复用并通过合成测试** |
| 执行授权 | **False** |

需用户单独书面批准一次 P11-2-C1；未批准前不运行。
