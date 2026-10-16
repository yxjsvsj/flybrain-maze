# P2d-26e：官方 neuPrint 定向查询（方案 A）——执行流程修复 + 离线测试

**基线** `71a8a6a`。本轮：修好并测试完整取数流程；**未**做 PCA/环序/神经朝向/注入/动力学/控车；
**未改** P1、MaleCNS 权重、正式 readout、BASE 编码、tags。**未启动方案 B（6.78 GB 下载）**。

## 1. 执行结果：**缺少认证 → 停止**（覆盖率 **0/88**，目标坐标**尚未取得**）
- 固定数据集 **`male-cns:v1.0`**；使用 `loop/p2d26d_targets.json` 的**88 个已确认 bodyId**（46 EPG + 20 PEN_a + 22 PEN_b），**未按类型名重挑/替换**。
- 执行机 token 检查：env `NEUPRINT_TOKEN`/`NEUPRINT_APPLICATION_CREDENTIALS`、`~/.neuprint`、`~/.config/neuprint`、`~/.cloudvolume/secrets/neuprint-secret.json` → **均未配置**。
- 程序行为：明确打印 `缺少认证`，**退出码 2**，**不发起任何匿名请求**，**未写入任何结果文件**。
- **不判定为"数据不存在"**。token 只从本机安全配置读取，**不输出/不上传/不写入仓库**。
- 实际覆盖率：**returned_data 0 / empty 0 / request_failed 0 / not_queried 88**。

## 2. 程序修复（`loop/neuprint_fetch.py`）
1. **补齐完整流程** `run()`：字段核对 → 小批验证 → **全部 88 个提取** → **落盘（覆盖表 + 坐标子集 + 状态 JSON）**；`main()` 不再只查前 3 个就退出。
2. **字段核对**：正式查询前用 `describe()`/`describe_columns()` 读**实际返回列**并逐列映射（x/y/z/type/confidence/roi）；**缺必需列即 `KeyError` 中止**（记下实际列名）。
3. **状态与 `aggregate()` 兼容**：`returned_data` / `empty` / `request_failed`（另含 `interrupted`/`not_queried`）；
   `aggregate()` 只对**带坐标**的行计数，**标记行绝不当作突触**（`raw_rows=0, unique_syn=0, bbox=None`）；数据存在时状态优先为 `returned_data`。
4. **绝不把缺失坐标当突触**：`handle()` 对返回行做坐标校验，整批无坐标 → 记 `request_failed(no_coords)`。
5. **去重保 ROI**：多 ROI 重复行按 `(kind,x,y,z)` 去重为唯一突触，但**保留全部 ROI 归属**；前/后**分别计数**。
6. **中断保留进度**：按 checkpoint 落盘；`KeyboardInterrupt` 时把当前及之后标为 `interrupted` 并**保存已完成进度**。
7. 结果**小体量**（`p2d26d_coverage.csv` / `p2d26e_coords.csv` / `p2d26e_status.json`）；**原始大文件与 token 不入库**。

## 3. 测试（离线，注入伪 `fetch_one`；无网络）
- `loop/test_neuprint_fetch.py` —— **21/21**（目标清单 88、缺认证守卫、标记行不计数、多 ROI 去重保 ROI、前/后分离、字段映射、缺必需列抛错）。
- `loop/test_neuprint_fetch_e2e.py` —— **26/26**：完整流程含 **部分请求失败 / 部分为空 / 多 ROI 重复 / 全部 88 目标保留 / 坐标持久化 / 中断后状态记录 / 字段核对失败中止 / 无坐标行不算突触**。

## 4. 未解决的坐标问题（阻塞，非本轮可解）
1. **缺 token**：需在 `neuprint.janelia.org/account` 取得 token 并配置到本机安全配置后重跑 `python -m loop.neuprint_fetch`，方可取得 88 个目标的真实突触坐标（前/后、唯一突触点、ROI、坐标、置信度）。
2. **坐标同框性未证实**（`loop/coord_frame.py` 判定 `conclusive=False`）：EB 网格与突触坐标**来源同为 DVID v1.0**，但 OBJ **未声明单位/轴序/缩放/原点**，尚无独立证据 → **不得凭 bbox 重叠宣布同框**，更**不得为得到漂亮环序而挑选坐标变换**。
3. 取得数据后，**只做**"EB 网格与突触坐标是否同框"的核实（含显式来源说明 + 目标细胞空间叠合检查）；不做环序/PCA。

## 5. 下一步（等待用户）
- 配置 token 后运行：`python -m loop.neuprint_fetch` → 生成 88 覆盖表与坐标子集 → 再做同框核实。
- **不启动方案 B**（6.78 GB `syn-partners` 流式筛选）除非另行批准。
