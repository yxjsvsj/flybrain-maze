# P2d-26c：syn-points 等文件**实际数据可用性**核验（只核验；未重建/未注入）

**基线** `f93cdc0`。方法：**惰性 HTTP-range 读**（`loop/feather_range_probe.py`，测试 7/7）——
只 seek 读 footer/schema + 指定 batch 的字节，**未整包下载**（总流量 ≈ **6 MB**，非 13 GB）。全部按 **bodyId** 对应，不按类型名。

## 一、真实字段 / 覆盖 / 坐标 / 提取 / 缺口
| 文件（v1.0，bytes） | **已核实的真实字段** | **已核验目标细胞(46 EPG+42 PEN)** | 坐标 | 提取方式 | 剩余缺口 |
|---|---|---|---|---|---|
| `syn-points`（13,061,489,098；5455 batches×65536） | `x,y,z`(**int32**)、`kind`(**PreSyn/PostSyn**)、`conf`(float)、`sv`(int64)、**`body`(int64=bodyId)**、`compartment`(dict)、`major/primary/…_label`、`point_id`(uint64) | **0/88（本次未覆盖）**；batch0 含 22,595 个不同 body | **EM 体素** x/y/z | range 读 schema+小样（已）；**按 body 取需扫描或官方查询** | **未按 body 排序**→逐细胞需全扫(≈13 GB)；`compartment` 小样全为 `unknown`，覆盖率未知 |
| `syn-partners`（6,777,179,098；4759 batches） | `x/y/z_pre`、`body_pre`、`conf_pre`、`x/y/z_post`、`body_post`、`conf_post`、`primary_post` | **0/88（本次未覆盖）** | **EM 体素**（pre 与 post 各一套） | 同上 | 同上；含**配对**前后坐标 |
| `body-stats`（778,062,826；1349 batches） | `body, pre, post, status_fine, superclass, class, type, instance, downstream, synweight, rank` | 未查（可二分；按 body 有序迹象） | **无坐标** | range 读 schema+首行（已） | 无分区列；是"每 body×下游伙伴/权重/rank" |
| `EB` 网格（441,650） | **OBJ**：`v 49376.17 20697.27 17409.51`（顶点 float x/y/z） | — | **EM 体素**（float） | range 读头部（已） | **单位/轴序/原点与 syn-points 是否一致——未核实** |

**目标细胞**：本机 46 EPG / 20 PEN_a / 22 PEN_b 的 bodyId 已固定（P2d-24c/26a）；
在 `syn-points`/`syn-partners` 的**首个 batch** 中命中 **0/88** → 只能记 **"本次未覆盖"**，
**不能**据此判定完整数据没有它们（两文件**不是按 body 排序**，目标在后段）。

## 二、坐标对应（当前结论）
- **[代码事实]** `syn-points`/`syn-partners` 的 x/y/z 为 **int32 的 EM 体素坐标**；`brain.npz.positions`（=官方 `somaLocation`）**同为 EM 体素** → 二者**同源同框**（由 P2d-24c 的校验证明）。
- **[未核实]** **EB OBJ 网格**的顶点为 **float**（`v x y z`，量级 2e4–5e4）与突触同为 EM 体素框架**很可能一致**，但**单位/轴序/缩放/原点尚未核实**（需读一个已知投射细胞的突触 bbox 与 EB 网格 bbox 做同框交叉验证）→ 本轮**不做**。

## 三、下一次空间重建的**执行条件**（本轮只写条件，不执行）
1. **取目标突触**（二选一，需先声明所选路线）：
   - **(a) 官方定向查询**（优先，轻量，未测）：用 MaleCNS 官方接口按 bodyId 取该细胞的突触点/伙伴；
   - **(b) 全扫**（代价须先声明）：`syn-partners` **6.8 GB**（或 `syn-points` **13 GB**）**完整下载** → 磁盘 ≈ 6.8–13 GB、按 batch 流式过筛 88 个 body → 预计数分钟–数十分钟；**不自动进行**。
2. **核实同框**：读 **EB/EBr\*** 网格 bbox，与一个**已知 EB 投射细胞**的突触 bbox 对照（轴序/缩放/原点）；不一致则先解决框架问题。
3. **再**按 P2d-26b 的**预先固定方案**（EB 内突触比例→圆均极角→环序；独立证据：(a) EPG→PEN 固定偏移带状、(b) EBr\* 层相容）执行。

## 四、明确阻碍（不回到换谱算法/调阈值）
- **`syn-points`/`syn-partners` 未按 body 建索引** → 逐细胞取坐标要么**官方查询**（未测），要么**全扫 6.8–13 GB**（流量/磁盘/方式如上，**待批准**）。
- **EB 网格坐标框架与突触及 `positions` 的同框性未核实**。
- `compartment` 字段存在但在小样中全为 `unknown`，覆盖率未知。

**结论**：**数据可用性=可用但未打通**（字段/格式已核实，目标细胞的坐标尚**未实际取得**）。
按规则本轮**不执行** PCA/环重建/神经实验，**不换算法、不加人工连接、不放宽阈值**；
**保持不注入、不调 K、不控车**；未改 P1/权重/正式读出/BASE 编码/tags。
