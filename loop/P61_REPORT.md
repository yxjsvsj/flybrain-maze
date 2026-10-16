# P6-1 报告：五路 ToF → MaleCNS 神经感觉接口独立可行性验证 → **NO-GO**

**基线** `9ff1997`。**仅验证五路感觉输入与神经响应**；不控虚拟/实体车；不训练；不引入 Dijkstra/地图/目标；
不改 P1/正式权重/正式 readout/BASE/tags；不查找第二候选。

## 第一步：核对当前真实实现（接口审计）
- 通路：`hardware/tof5.py`（L+70/FL+30/F0/FR−30/R−70）→ `loop/encode_tof.py::PhysicalToFEncoder`
  → `sense()`（VALID=mm/1000、NO_TARGET=max_range、TOO_NEAR=min）→ `update()`（新 `tof_seq` 才刷新，帧间**零阶保持**）
  → `_side()`（`size=min(wall_size/d,1.5)`、`growth=clip(Δsize/Δt_tof,0,max)`、`loom/chase/threat`）
  → `inject()`（loom/chase/threat ±fwd）。
- **关键发现**：`avoid_half = avoidance_fov_deg/2 = 60°` → `left={FL,F}`、`right={FR,F}`；**L(+70°)/R(−70°) 不进入神经避障通道**
  （`self.left/right` 掩码把 ±70 排除；只在建图 `valid_mask` 里用）。→ **外侧两路未接入脑**。
- 注入靶（`brain/neurons.py`，文献依据）：`loom→LPLC2`（looming）、`threat→LC4`（快速 looming/逃逸）、`chase→LC10a`（转向上游）。
- **基线引用**：P2（L50/R50、POP）与 P5（looming，NO-GO）的失败结论沿用，本轮**不重测已关闭方案**。

## 第二步：唯一五路神经感觉接口（新建独立模块）
`loop/p6_tof5_sensory.py`：**五路全部**接入，**逐路方位权重** `wL=clip(0.5+0.5·θ/90°,0,1)`、`wR=1−wL`
（不取左右最近值）：`loom_L=Σ loom_i·wL_i`、`loom_R=Σ loom_i·wR_i`（threat 同）→ 注入 `LPLC2_L/R`、`LC4_L/R`。
- **纯感觉**：不下发转向方向/目标角/转弯计时/回正命令；不注入 fwd。
- **披露**：本接口是**人工 ToF→神经感觉转换**（距离/变化→loom/threat），**不冒充**果蝇天然视网膜。
- 未修改冻结的 `loop/encode.py`/`loop/encode_tof.py`/正式 readout。

## 第三步：冻结测试
- 场景：每路 `near(0.30m) / away(2.50m) / approach(1.50→0.30m)` + `none`；左右镜像（L/FL vs R/FR）。
- 时序：**ToF 10Hz（帧间零阶保持）/ 脑 50Hz**；窗 PRE/STIM/POST=0.5/1.0/0.5s；seeds `[11..15]`；**有障碍条件减去同 seed 无遮挡对照**（同步窗差值）。
- 记录：5 路距离原值、状态、**逐路对注入的贡献**（`size/growth/loom/threat`）、神经元身份（LPLC2/LC4）与逐帧发放。
- 原 MaleCNS 权重/参数不变；编码方案/seed/门槛**运行前固定**（`K_SIG=3×` 对照每细胞 std）。

## 第四步：神经运动输出资格检查
- **不**复活 L50/R50、NS、EPG/PEN。下行候选按连接组**预先固定**：`LC4/LPLC2 → DN 直接靶`（bodyId 键，L 68 / R 70，来自 P3 `dn_target_set_by_body`）。
- 仅离线评估"左右区别 + 返回背景"，**不**因撤除后减弱就宣称会结束转弯。

## 结果（逐 seed）
| seed | 五路 near 生效（L/FL/F/FR/R） | DN_L(approach_L) | DN_R(approach_R) | DN 左右 OK |
|---|---|---|---|---|
|11|T/T/T/T/T|0.0150|0.0148|否|
|12|T/**F**/T/T/T|0.0156|0.0136|否|
|13|T/T/T/T/T|0.0169|0.0129|否|
|14|T/T/T/T/T|0.0175|0.0123|否|
|15|T/T/T/T/T|0.0181|0.0136|否|

- **五路感觉输入生效**：`all5_effective = 4/5` seed（seed12 的 FL 未过 3σ）→ **五路"确实生效"未完全达标**。
- **神经运动信号门槛不通过**：DN 对照差值仅 **0.013–0.018**，远低于 `3σ≈0.13–0.18` → **DN_lateral_ok = 0/5**。
- 即：注入确实驱动了 **LPLC2/LC4（感觉级）**，但**下行靶无可靠、左右可分的信号**；撤除后是否回背景无法作为"会结束转弯"的证据（仅记为下一关必要条件）。

## 结论：**NO-GO**
- 五路感觉输入**未全部达标**（4/5）+ **预注册神经信号门槛不通过（0/5）** → **不进入**虚拟车测试、**不**寻找第二候选。
- 与 P2/P5 一致：真实连接组具备解剖基质，但**当前 LIF 未把感觉注入转成可靠的下行运动信号**。
- 背景漂移/持续同向/跨 seed 符号混乱均**照实记为失败**，未用手写规则修正。

## 交付物
`loop/P61_REPORT.md`、`loop/p6_tof5_sensory.py`、`loop/test_p6_tof5_sensory.py`（**13/13**）、`loop/p61_result.json`、`loop/p6_raw.npz`（逐帧原始计数）。

**边界**：不接实体车/不上电；不改 P1/正式权重/正式 readout/BASE/tags；不训练/老师/Dijkstra。**已完成，停止，等待审批。**
