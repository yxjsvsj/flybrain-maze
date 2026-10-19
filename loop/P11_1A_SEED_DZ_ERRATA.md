# P11-1a 补充：seed-level paired Cohen's dz 静态勘误

**基线** `272c81b`。**只读归档 P10-C2-A0**；未初始化/运行完整脑、未重跑 A0、未改 W/ΔW、未训练、未导航。
**本任务不是 P11-1b 运行授权**；完整脑资源基准仍 **NOT_AUTHORIZED**；`Training Authorization=False`。

## 1. 正确计算（25 个独立模拟重复）
对 seeds 101–125，每个 seed 先对 320 个 KC 求 `stim−none` 平均差 → 25 个独立差值：

`seed_mean[s] = mean_{320 KC}( rate_stim[100:300] − rate_none[100:300] )`

再计算：
- mean = **−2.0000037693534978×10⁻⁵**
- sample SD（ddof=1）= **4.58470662124455×10⁻⁴**
- **seed-level paired Cohen's dz = mean/SD = −0.04362337515962108**

（25 个 seed 的逐 seed 值见 `p11_1a_seed_dz.json`。）

## 2. 三个量必须区分
| 统计 | 值 | 定义 | 能否称为 seed-level paired dz |
|---|---|---|---|
| **seed-level paired Cohen's dz** | **−0.043623** | 每 seed 先对 320 KC 求均值；25 个 seed 间的 mean/SD | **是（唯一）** |
| cross-cell dz（P11-0） | **−0.022377** | `mean(per-cell means)/sd(per-cell means)` across 320 cells | **否** |
| pooled seed-cell 标准化比值（P11-1a） | **−0.003161** | `mean(all 8000 seed-cell diffs)/sd(all seed-cell diffs)` | **否** |

**勘误**：后两者**不得**冒充 seed-level paired Cohen's dz；历史 P11-0/P11-1a 报告**未修改**。

## 3. seed 处理
- **seed 114 保留**（index 13；值 **+0.000625**）。
- **未删除任何 seed**；**未挑选有利 seed**；25 个全部纳入。
- 未改变 P10-C2-A0 原始统计。

## 4. 安全与边界
- 归档 **ARCHIVED_VERIFIED**；官方 W/brain 全量 SHA-256 核验通过；P1 freeze 空。
- **未初始化完整脑**、**未运行任何动力学**、**未重跑 A0**。
- 完整脑资源基准仍 **NOT_AUTHORIZED**；本勘误不构成任何运行/训练授权。
- `Training Authorization=False`。

## 5. 输出
- `loop/P11_1A_SEED_DZ_ERRATA.md`（本文件）
- `loop/p11_1a_seed_dz.py`
- `loop/test_p11_1a_seed_dz.py`（22/22）
- `loop/p11_1a_seed_dz.json`

**已停止；不再重复创建审批文件。**
