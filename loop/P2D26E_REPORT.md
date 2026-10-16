# P2d-26e：官方 neuPrint 定向查询（方案 A）——取数完成 + 同框核验

**基线** `71a8a6a`。本轮：取得 88 个目标细胞的真实突触坐标，并**仅**核实 EB 网格与突触坐标是否同框。
**未**做 PCA/环序/神经朝向/注入/动力学/控车；**未改** P1、MaleCNS 权重、正式 readout、BASE 编码、tags。
**未启动方案 B**。token 存于本机 `~/.neuprint`（仓库之外，**未提交/未输出**）。

## 1. 取数结果：**88/88 returned_data（覆盖率 1.00）**
- 数据集 **`male-cns:v1.0`**；`p2d26d_targets.json` 的 88 个已确认 bodyId（46 EPG + 20 PEN_a + 22 PEN_b），**未按类型名重挑/替换**。
- **实际返回字段**（小批核对）：`['bodyId','type','roi','x','y','z','confidence','compartment','split']` → 映射 x/y/z/type/confidence/roi 全部命中，无缺列。
- 状态：**returned_data 88 / empty 0 / request_failed 0 / interrupted 0 / not_queried 0**。
- 突触计数（唯一突触点，前/后分离）：
  | 组 | 细胞 | 唯一突触点 | pre | post |
  |---|---|---|---|---|
  | EPG | 46 | 234,801 | 29,290 | 205,511 |
  | PEN_a(PEN1) | 20 | 54,296 | 10,999 | 43,297 |
  | PEN_b(PEN2) | 22 | 52,752 | 9,657 | 43,095 |
  | **合计** | **88** | **341,849** | 49,946 | 291,903 |
- 去重口径：同一突触的多重 ROI 行按 `(kind,x,y,z)` 合并为唯一突触点，**ROI 归属全部保留**（`roi` 字段为分号列表，如 `CRE(L);CentralBrain-unspecified;EB;FB;LAL(L);PB`）；置信度按 neuPrint 原样记录。

## 2. EB 网格 ↔ 突触坐标**同框核验**（**不是**"bbox 重叠即过"）
- 网格：`roisSmoothedDecimated/key/EB`（441,650 B；OBJ，3468 顶点 / 6712 面，**非水密**）；bbox **x=[44314,52648] y=[16664,23784] z=[17399,23874]**（EM 体素）。
- 靶点：**严格 `roi=='EB'`** 的突触点 **251,673 个**（88 细胞中带 EB 标注者）。
- 证据（`loop/eb_frame_check.py`，测试 7/7）：
  1. **来源**：网格与突触同为发布版本 **DVID v1.0**（显式来源说明）。
  2. **独立标注的 EB 点落入 EB 网格**：bbox 逐轴包含率 **= 1.0000**（整体；逐细胞亦 1.0），且 EB 点自身 bbox 与网格 bbox **紧贴**（x[44345,52574]⊂[44314,52648]，y、z 同理）。
  3. **错轴对照**：把 EB 点 (x,y) 交换后包含率 **= 0.0000** → 只有 **identity** 变换自洽。
  4. 佐证（非判据）：最近面法向有符号距离 ≈ **99.8%** EB 点在网格内，中位 ≈ −599 体素；非 EB 点仅 0.8% 落入。
- **结论：同框成立（identity 自洽，无需任何变换）**。**未**为得到漂亮环序而挑选坐标变换。
- **保留说明（诚实边界）**：网格非水密 → 精确"点-在-网格"计数有噪声，故结论基于 **bbox 包含 + 错轴对照 + 来源同源**，而非单纯 bbox 重叠；官方未声明网格单位/轴/原点的显式变换，属于"以 identity 自洽为据"的核实，非发布文书级证明。

## 3. 交付物
- `loop/p2d26d_coverage.csv`（**88 行**，每行：bodyId/group/status/raw_rows/unique_syn/pre/post/坐标范围/ROIs）。
- `loop/p2d26e_coords.npz`（**3.27 MB**，341,849 突触点的 bodyId/kind/xyz/conf/roi 编码）。原 `p2d26e_coords.csv`（17.6 MB）**可再生、不入库**。
- `loop/p2d26e_status.json`（88 细胞状态，供中断恢复）。
- 工具与测试：`loop/neuprint_fetch.py`（+ `test_neuprint_fetch.py` 21/21、`test_neuprint_fetch_e2e.py` 26/26）、`loop/eb_frame_check.py`（+ `test_eb_frame_check.py` 7/7）、`loop/coord_frame.py`（+ `test_coord_frame.py` 9/9）。

## 4. 未解决/待决
- 网格**单位/轴/原点的显式声明**官方缺失（现以 identity 自洽 + 同源为据）。
- PB/NO 网格仍不在 v1.0 `roisSmoothedDecimated`；`neuprint.roi` 为多点多重 ROI（已保留归属）。
- **下一轮（未做）**：以 EB 内突触的角向位置重建环序等——需另行批准。
