# P2d-27 第一关报告：统计口径修正（**未进入第二关**）

**基线** `ebc308d`。仅修正统计实现；**未**改阈值、**未**改输入数据、**未**强行保留/排除任何 bodyId。
**未**跑动力学、**未**注入、**未**控车；**未改** P1、MaleCNS 权重、正式 readout、BASE 编码、tags。

## ⚠ 触发暂停条件（P2d-27 §7）
统计修正**改变了生物解剖有效性判定**：**PEN_a bodyId 20818 由 `invalid` → `valid`**（原 `ci=30.82°` → 新 `boot_circ_std=0.83°`）。
按 §7「若统计修正影响解剖门槛，报告并暂停，不进入第二关」——**在此暂停**，等待批准后再决定是否进入第二关。

## 1. `bootstrap_angle()` 修正（`loop/ring_spatial.py`）
- **旧实现错误**：`R = |mean_b (mean_n e^{iθ})|`——把每次 bootstrap 的**均值向量长度**一起平均（幅值加权），
  既高估了均值角的不确定性，又把该统计量**误命名为 `ci_half_deg`**。
- **新实现** `bootstrap_mean_angle()`：逐次算 bootstrap 样本的**圆均值角** `m_b`，再对 `{m_b}` 统计：
  - `boot_circ_std_deg = degrees(sqrt(-2 ln |mean_b e^{i m_b}|))` —— 即阈值 `UNC_MAX_DEG` 所门的
    **bootstrap 不确定性**（与旧阈值同义，但计算正确）；
  - `ci95_lo/hi/half_deg` —— **明确的 95% 置信区间**：`{m_b}` 上**最短包含 95% 的弧**的端点与半宽
    （`shortest_arc_ci`，`level=0.95`）。**不再把圆标准差叫 CI 半宽。**
- **与原始分布区分**：新增 `raw_circ_std_deg` = **原始突触角度分布**的圆标准差（分布宽窄），
  与 `boot_circ_std_deg`（均值角估计的稳定性）**分列**。
- 例（EPG 全体）：`raw_circ_std` 中位 **13.28°** vs `boot_circ_std` 中位 **0.200°**、`ci95_half` 中位 **0.39°**
  —— 「突触角分布很宽」与「圆均角很稳定」被清楚区分。

## 2. 置换检验 p 值（`loop/ring_validate.py`）
- 旧：`p = mean(null >= obs)` → 可能为 **0**。
- 新：**加一估计** `p = (1 + #{null >= obs}) / (1 + B)`（Phipson–Smyth）。

## 3. 有效性重判（原门槛 `UNC_MAX_DEG=30`，不重选）
| | 旧 | 新 |
|---|---|---|
| 有效细胞 | 87/88 | **88/88** |
| 无效 | PEN_a 20818 (`ci=30.82`) | **无** |
| EPG 有效 | 46/46 | **46/46（不变）** |
| 旧 `ci_half_deg` vs 新 `boot_circ_std_deg` | — | max\|Δ\|=**29.99**，mean\|Δ\|=**15.36**（旧值系统性偏大） |

- **46 个 EPG 的 bodyId、空间角序、PCA 坐标、原始突触数据全部不变**（已逐位核对：EPG 角 `old==new`）。
- 受影响者仅为**不确定性统计量**与**由它门控的 1 个 PEN_a 细胞**。

## 4. 合成测试（`loop/test_ring_spatial.py`，**21/21**）
覆盖并能**区分**：「宽分布/稳定均值」「小样本→不确定性增大」「±180° 环绕」「双峰」「CI 最短弧定义」。
`loop/test_ring_validate.py` **6/6**（真带状显著 vs 均匀不显著，含加一 p）。

## 5. 独立验证重算（不受影响，仅 p 口径更新）
| 组 | 观测 R_w 中位 | 零模型 R_w 中位 | p<0.05 比例 |
|---|---|---|---|
| PEN_a | 0.866 | 0.212 | 1.00 |
| PEN_b | 0.915 | 0.276 | 1.00 |
- 分半中位 0.2°，frac<20° = 1.00；EPG 46/46 仍全部可辨。**未**用验证数据反向调序。

## 6. 交付物（本轮）
- 统计修正：`loop/ring_spatial.py`、`loop/ring_validate.py`。
- 合成测试：`loop/test_ring_spatial.py`（21/21）、`loop/test_ring_validate.py`（6/6）。
- 重生成的证据表：`loop/p2d26f_cells.csv`（新列 `raw_circ_std_deg / boot_circ_std_deg / ci95_lo_deg / ci95_hi_deg / ci95_half_deg`）、
  `loop/p2d26f_validation.json`、`loop/p2d26f_ring.png`。

## 7. 结论与请求
- **PASS 门槛未改**（预设 `UNC_MAX_DEG=30`、覆盖/分半/带状阈值均未动）。
- 但统计修正**确实影响了逐细胞解剖有效性**（20818 翻转），触发 §7 暂停。
- **第二关（MaleCNS 状态保持预检）尚未运行。** 请确认：是否（a）认该修正不改变 EPG 环序主体、批准进入第二关，
  还是（b）先复核 20818 的翻转再做决定。
