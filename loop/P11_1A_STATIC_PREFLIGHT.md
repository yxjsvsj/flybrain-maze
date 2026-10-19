# P11-1a 完整 MaleCNS 模型有效性实验：静态资源预检与范围冻结

**基线** `9471fe3`。**未初始化完整脑、未运行动力学、未训练、未更新 ΔW、未跑迷宫。**
本文件冻结 P11-1 阶段 1 的范围；**不授权执行**。

## 1. 范围冻结（阶段 1）
- **完整 MaleCNS vs 已归档的 23,218 神经元截断子网**。
- **仅 none 条件**；seeds **101–125**；每 seed **400 步**。
- 原 `dt=0.020`、`tau=0.100`、`gain=3.0`、`tonic=0.14`、`noise_hz=1.2`、`noise_amp=0.22`、
  逐突触后 L1 归一化**完全不变**；官方 W 只读；**ΔW=0**。
- **复用** P10-C2-A0 归档的 25 个 none 运行；**不重跑截断子网**、**不运行 stim/recover**、**不重新优化参数**。
- 完整脑计划量：**25 × 1 × 400 = 10,000 步**（不是 30,000 步）。
**本轮只冻结计划，不执行。**

## 2. 完整脑静态资源预检
| 项 | 值 |
|---|---|
| N / nnz | **166,700 / 25,582,938** |
| CSR/CSC dtype | float32 / int32 / int32 |
| CSR 数组（data+indices+indptr） | **205.33 MB** |
| CSC 数组 | **205.33 MB** |
| `brain.npz` 元数据数组 | **53.39 MB** |
| 权重文件 / 脑文件 | 205.33 MB / 53.40 MB |
| 状态数组（v / noise temp / partial×16 / current） | **13.34 MB** |
| **初始化峰值下界**（CSR+CSC+元数据+状态） | **477.39 MB** |
| 含权重文件缓冲 | 682.72 MB |
| **1.5× 开销上界估计** | **≈1,024 MB** |
- **未测量**：`load_npz`/`tocsc` 的临时副本、文件缓冲、NumPy/SciPy/Python 开销。
- **`resource_ready=False`**：静态数组远低于 2 GB，但**没有可信的实测峰值上界**。
- **2 GB 上限建议**：**保留 2 GB**；若基准实测峰值 >2 GB → **停止/INCONCLUSIVE**；
  **不得**自动提高上限，需另行批准。
- **为何 7.207× nnz 不能直接当倍率**：耗时受 fired 数量、缓存、固定 O(N) 累加与 Python 开销影响；
  峰值内存含 CSR/CSC 双副本、元数据、临时缓冲，非线性于 nnz。

## 3. 缩减为 none-only 的准确资源计划
- **10,000** 完整脑 LIF 步；**1 次 Brain 初始化 + 25 次 reset**。
- **静态核心估计**：10,000 × (1.3 ms × 7.207) ≈ **93.7 s ≈ 1.56 min**（**静态外推，非基准**）。
- 保存 KC/MBON 数组约 **166 MB**（仓库外）；**不计算资格迹**（none-only 基线比较不需要）。
- **不得**把该估计当作已验证的执行成本。

## 4. 统计与生物学可比性
固定报告：
1. 截断 320 KC vs 完整脑**相同 bodyId** 的 320 KC；
2. 完整脑**全部 4,064 KC** 单独报告；
3. 截断 95 MBON vs 完整脑匹配 95 MBON；
4. 完整脑**全部 97 MBON** 单独报告；
5. 发放占比、天花板比例、跨 seed 变化、启动瞬态。
**RNG 审计**：
- `same seed` **仅**表示相同初始 RNG 种子；完整脑 N=166,700 vs 子网 N=23,218，
  每步 `rng.random((N,B))` 抽样形状不同，**共同 bodyId 的逐神经元噪声序列不一致**。
- **`strict_noise_pairing=False`**：这是**共享 seed 标签的跨模型比较**，**不是**噪声配对比较。
- full/subnet 差异只能解释为**整体截断条件**，**不能**归因于某类外部抑制/兴奋。

## 5. 阶段 2 参数来源
- `gain=3.0/tonic=0.14/noise=1.2/0.22` = **flybrain 代码默认值/作者手工设置**；
- `gain=1.5/tonic=0.18` = **fly64 代码值**；
- **未找到**独立同行评审 DOI 或实测生理标定。
- **Phase 2 Parameter Provenance = INSUFFICIENT**；阶段 2 **Execution NO-GO**；不作为可运行依据。

## 6. P11-0 `dz` 勘误
- P11-0 的 `dz=-0.0224` 是 **跨细胞** `mean(per-cell means)/sd(per-cell means)`；
- **按独立 seed 配对的 pooled dz = -0.00316**；
- 两者不同；P11-0 的 `dz` **不应**称为 paired-by-seed Cohen's dz。已新增勘误，不覆盖历史。

## 7. 独立门控
| 门控 | 结果 |
|---|---|
| Historical P10-C2-A0 | **INCONCLUSIVE** |
| HIGH_BASELINE_PHENOTYPE | **SUPPORTED** |
| MODEL_ROOT_CAUSE | **NOT_ESTABLISHED** |
| Phase 1 Design Readiness | **GO** |
| Full-Brain Resource Readiness | **NO-GO** |
| Phase 1 Execution Readiness | **NO-GO** |
| Phase 2 Parameter Provenance | **INSUFFICIENT** |
| Phase 2 Execution Readiness | **NO-GO** |
| Level A Training / B / C | **NO-GO / NO-GO / NO-GO** |
| Biological | **NO-GO** |
| **Training Authorization** | **False** |

## 8. 所需授权
**先申请一次独立的完整脑小规模无学习资源基准**（见 `P11_1A_ONE_BENCH_APPROVAL.md`）。
该基准与 25-seed 模型有效性实验是**两项不同授权**；本文件不授权任何执行。
