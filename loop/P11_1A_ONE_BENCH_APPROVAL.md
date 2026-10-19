# P11-1a 一次性完整脑无学习资源基准：审批申请（**未批准；未运行**）

**状态：NOT GRANTED；本轮未运行。** 本文件仅申请**一次**独立的完整脑资源基准。
**它不是 25-seed 模型有效性实验的授权**，也不授权训练/ΔW/B/C。

## 1. 目的
测量完整 MaleCNS 在**现有归一化 LIF**下的初始化时间、逐步耗时与峰值内存，
以决定 25-seed none-only 模型有效性实验是否满足 **15 min / 2 GB**。

## 2. 范围（一次、无学习）
- 完整 MaleCNS（N=166,700；nnz=25,582,938）；官方 W 只读；**ΔW=0**。
- **无刺激**（不注入 LPLC2/LC4；不运行 stim/recover；不运行 25 seeds）。
- 预热 **5 步**；正式 **10 / 50 / 100 步**；**一次**运行。
- 记录：初始化时间、逐步 median/p95、峰值进程内存、异常/非有限状态、Python/NumPy/SciPy/flybrain 版本、设备。
- **资源上限**：**15 min / 2 GB**；超限立即停止并报告 **INCONCLUSIVE**。
- 失败/超时**不得**重跑（一次性）；不得调整参数。

## 3. 明确不做
- 不运行 25-seed 模型有效性实验；不比较 full/subnet 统计；不运行 eligibility。
- 不训练、不更新真实 ΔW、不运行 Dijkstra、迷宫、B/C、实体小车。
- 不改变 dt/tau/gain/tonic/noise/归一化/W；不按结果调参。
- 不用合成矩阵性能替代完整脑实测。

## 4. 输出
- 初始化/逐步耗时、峰值内存、错误日志；
- 完整 SHA-256（brain/weights）与脚本 SHA-256；
- 明确结论：`RESOURCE_OK`（满足 15 min / 2 GB）或 `RESOURCE_NO_GO`（超限/失败）。

## 5. 审批后仍需
- 即使基准通过，25-seed 模型有效性实验**仍需另行书面授权**。
- `Phase 1 Execution Readiness` 在基准通过且获授权前保持 **NO-GO**。
- `Training Authorization=False`。

**请明确批准/拒绝本次一次性完整脑资源基准。未批准前不运行。**
