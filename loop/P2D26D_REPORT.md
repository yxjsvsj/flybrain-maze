# P2d-26d：目标细胞坐标提取 + 坐标框架核验（以 `8b8b5d5` 为基线）

**本轮**：只做「取目标坐标」与「核验坐标框架」。**未做** PCA 环序 / 神经朝向 / 注入 / 动力学 / 控车；
**未改** P1、权重、正式读出、BASE 编码、tags。原始大文件**不提交 Git**。

---

## 1. 官方 neuPrint：**缺少认证**（结论：访问失败，**非数据不存在**）
- 数据集（固定）：**`male-cns:v1.0`**，服务器 `https://neuprint.janelia.org`。
- 查询时间：**2026-10-16T02:32:39Z**；筛选条件：`MATCH (n:Neuron) WHERE n.bodyId IN [<88 个已确认 bodyId>]`。
- 执行机凭据检查（**未配置**）：env `NEUPRINT_TOKEN`/`NEUPRINT_APPLICATION_CREDENTIALS`、`~/.neuprint`、`~/.config/neuprint`、`~/.cloudvolume/secrets/neuprint-secret.json` → **均为空**。
- 单次探测（**不重复匿名**）结果：**HTTP 401 Unauthorized**
  > `{"message":"authentication required: neuPrint now requires an API token for all data, including public datasets; log in at https://neuprint.janelia.org/account to get one"}`
- **明确报告：缺少认证**（no token）→ 本机**无法**用官方 API 取坐标。**不代表**这些细胞/突触在数据集中不存在。
- 已就绪：`loop/neuprint_fetch.py`（找到 token 即以小批核验→全 88 批量查询；无 token 时**明确报缺认证**并返回码 2）；`loop/p2d26d_targets.json`（冻结的 88 个原始 bodyId，**不按类型名重挑**）。

## 2. 数据来源元信息（GCS `flyem-male-cns` v1.0，本次读得）
| 对象 | 大小(B) | md5(base64) | updated(UTC) |
|---|---|---|---|
| `roisSmoothedDecimated/key/EB` | 441,650 | `T6KD85+/iBCxzzv5951dVA==` | 2026-05-05T17:20:26Z |
| `roisSmoothedDecimated/key/EBr1` | 150,586 | `n5BjkUHf+g8UncfOv3n2hQ==` | 2026-05-05T17:20:26Z |
| `roisSmoothedDecimated/key/EBr3d` | 267,156 | `gidM2l7CMrw2fGZu4vrTxQ==` | 2026-05-05T17:20:26Z |
| `syn-points-...-minconf-0.5.feather` | 13,061,489,098 | `xp0IdY3gdYIDXMiENXRJOg==` | 2026-06-03T13:56:31Z |
| `syn-partners-...-minconf-0.5.feather` | 6,777,179,098 | `WO/PcS+MTU3l8q1R6X3vdg==` | 2026-06-03T13:55:42Z |
| `body-stats-...-minconf-0.5.feather` | 778,062,826 | `QEwzScKFgBSOFoFeuZ84Kg==` | 2026-06-03T13:54:48Z |
- 网格来自 DVID 导出 `/node/:v1.0/roisSmoothedDecimated`；突触/统计来自同版本 `connectome-data/flat-connectome` → **同一 EM 体 v1.0**。

## 3. 坐标框架核验（**不只比数量级**；结论：**一致性成立但未证实同框**）
- **EB 网格**（本次下载，441,650 B，md5 与 GCS 一致）：OBJ 文本，`# OBJ file`，**3468 顶点 / 6712 面 / 3468 法线**；顶点为**浮点**（`is_integer_valued=False`，即平滑/抽稀后的曲面点）。
  - bbox：**x=[44313.6, 52648.2]  y=[16663.8, 23784.3]  z=[17399.5, 23873.8]**；extent ≈ **8335 × 7120 × 6474**。
- **syn-points**（range 抽样 batch 0 / 2727 / 5454，共 **6.24 MB**，未整包）：x=[5059, 89446] y=[6969, 58653] z=[4154, 134535]（int32）。
- **逐轴包含性**：EB 三轴均落在 syn-points 采样范围内（仅佐证）。
- **同框判据**（`loop/coord_frame.py` 明确定义）：需同时具备 (a) 显式来源/坐标系说明 与 (b) **单位/轴序/缩放/原点**无未知变换，方判 `conclusive=True`。
  - (a) 来源：**有**（同为 DVID v1.0，见上表）。
  - (b) 变换：**未取得**——OBJ 无缩放/单位声明，未与已知区做同框交叉验证。
  - → **`conclusive=False`**。**不得仅因范围重叠宣布同框**；**未**尝试任何"挑选最好看变换"的做法。

## 4. 交付物
- **目标坐标子集**：**空**（被缺认证阻断）。`loop/p2d26d_coverage.csv` 已给出 88 行的**表结构**，每行 `status=blocked:missing_auth`，其余列待 token 后填充（原始行数 / 唯一突触点数 / 前·后数量 / 坐标范围 / ROI）。
- **88 细胞覆盖表**：`loop/p2d26d_coverage.csv`（46 EPG + 20 PEN_a + 22 PEN_b）。
- **来源与坐标说明**：本文件 §1–§3。
- **测试**：`loop/test_neuprint_fetch.py`（**14/14**：目标清单 88、去重保 ROI、前/后分离计数、缺认证报错）、`loop/test_coord_frame.py`（**9/9**：OBJ 解析/bbox/逐轴包含/结论不只看重叠）。

## 5. 后备方案（**仅在查询受阻时提出；未获预算确认前不自动执行；不同时下两个大文件**）
**方案 A（官方，首选）**：在 `neuprint.janelia.org/account` 申请 token → 设 `NEUPRINT_TOKEN` → 运行 `python -m loop.neuprint_fetch`。**代价：≈0（仅查询流量）**。

**方案 B（若确实无认证/官方持续失败）**：下载 **仅 `syn-partners`**（**6.78 GB**，含配对前后坐标 + bodyId）流式筛 88 个 body。
- **流量**：≈ **6.8 GB**（单文件，避免与 13 GB 的 syn-points 同时下载）。
- **磁盘**：临时 **≈ 6.8–8 GB**（下载 + 输出子集），置于临时目录，**不提交 Git**。
- **断点续传**：GCS 支持 `Range`；用 HTTP `Range` 分块 + 记录已完成字节偏移（或 `curl -C -`、`gsutil -o GSUtil:resumable...`）续传；按 record batch 流式处理，**不整表入内存**。
- **处理**：逐个 record batch 过滤 `body_pre ∈ 88` 或 `body_post ∈ 88`，保留命中行；写小体量子集（预计 ≪100 MB）。
- **明确**：**需先获下载预算确认**；未确认前**不执行**。

---
**本轮边界遵守**：未做 PCA 环序、未算神经朝向、未注入、未跑动力学、未控车；未改 P1/权重/正式读出/BASE 编码/tags；未尝试匿名重复请求、未以访问失败判定"数据不存在"。
