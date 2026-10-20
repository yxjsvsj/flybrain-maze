# P11-2-S0：完整脑感觉传递实验预注册（只设计，不运行）

**状态：PREFLIGHT COMPLETE；NO DYNAMICS。**
基线：`4f017c2`。本文件只做静态分析、源码审计、资源预检、统计预注册和合成测试；
**未运行** FlyBrain、未重跑 C1、未执行 S1、未训练、未更新 ΔW、未运行 Dijkstra/迷宫/B/C/实体车。

---

## 1. C1a 小型统计文字勘误（只新增说明）

- 完整 B 群体均值：**0.351093307**。
- 三个低反馈 KC 对总体均值的贡献：**0.000646358**。
- 其余 **4,061** KC 的**组内均值**：**0.350705836**。
- 其余 4,061 KC 折算到**全部 4,064 KC 总体均值**的贡献：
  `0.350705836 × 4061 / 4064 = **0.350446949**`。
- 校验：`0.000646358 + 0.350446949 = **0.351093307**`。

**组内均值 ≠ 总体贡献**。历史 C1a 报告未覆盖；本说明单独记录。

---

## 2. 冻结感觉入口

沿用 P10-C2-A0 已批准历史协议：

- **仅原始 `LPLC2` / `LC4` 神经元**；**精确 cell_type**，不是 `startswith("LC4")`。
  - LPLC2：**185**；LC4：**126**；合计 **311**。
  - `startswith("LC4")` 会误含 **LC40/41/43/44/45/46**，总数 **227**；**不得使用**。
- `dv=0.2`；`dt=0.020`；`tau=0.100`；
  `amount = dv*(1-exp(-dt/tau)) = **0.03625384938440364**`（报告值 **0.036254**）。
- **不是**旧实验的 per-step `amount=0.6`；`0.6` 不得当作 `dv=0.2`。
- 刺激窗口 **[100,300)**；仅该窗口；固定 target；不事后放大、不移动靶点。
- 不改变五 ToF 到感觉神经元的原始编码与分组；不新增连接；不直接刺激 KC/MBON；
  不注入方向/Dijkstra/地图/教师；不训练、不更新 ΔW。

源码核对：`p10c2_0a_gate.py` 的 `stimulus_audit()` 使用 `amount = dv*(1-exp(-dt/tau))`；
`p10c2_0b_adjudication.py` 使用 `DV=0.2`、`AMOUNT=0.036254`。S0 冻结该定义。

---

## 3. 静态可达性结论（详见 `P11_2_S0_PATH_AUDIT.md`）

- **LPLC2/LC4 → KC 直接边：0**。
- 到 KC 的**前馈入口是 2-hop**：**7,883** 条（7,783 正 / 100 负；signed sum **0.003759783**）。
- B 删除 KC→KC 后：**0** 条前馈路径经 KC 中间被删除；B 仍有 **7,883** 条；
  **前馈感觉路径保留**，KC→KC 递归被删除（642,933 条）。
- 直接 **LPLC2/LC4 → MBON：1** 条；2-hop → MBON：**2,528** 条。
- **KC→MBON：61,210** 条，全部为正；signed sum **56.178184509**；B 保留。
- 3-hop `source→KC→KC→MBON` 经 KC→KC 的路径：**0**（因没有 source→KC 直接边）。
- **解剖可达 ≠ 动力学可检测**；前馈 net 仅 0.00376，S1 可能检测不到；
  不得为产生反应而修改靶点、加桥接或放大刺激。

---

## 4. 未来单次实验（S1）预注册草案

- 完整 MaleCNS，CPU；**仅 B 合成网络**（KC→KC zero）。
- seeds **101–125**；每 seed **400 步**；条件 **none / stim**；**同一实验内配对**。
- 主窗口 **[100,300)**；原 LIF 参数固定；`W_B` 仅内存派生，不修改官方 `weights.npz`。
- 感觉：精确 LPLC2/LC4（311）；`dv=0.2`；`amount=0.036254`；[100,300)。
- ΔW=0；不计算学习资格迹；不修改真实突触；不直接刺激 KC/MBON。
- **不复用** C1-B 历史 none；同一实验内 none/stim 避免参数/随机流/初始化可比性假设。
- none/stim 每 seed 使用相同 seed、相同 `reset`、CPU、相同噪声流；stim 只加电压。
- 总步数 2×25×400 = **20,000**；15 min / 2 GiB；一次性；不自动重跑。

---

## 5. 统计门槛（已冻结）

1. **入口阳性对照**：精确 LPLC2/LC4（311）在 [100,300) 的发放比例，stim−none；
   exact two-sided sign-flip；要求 mean > 0 且 p < 0.05。失败 → `SENSORY_ENTRY_FAILED`。
2. **主终点**：匹配 320 KC 在 [100,300) 的 stim−none；seed 为独立单位（25 对）；
   exact sign-flip；95% paired bootstrap（10,000 次，seed 20261019）；
   **最低有意义效应 +0.05 绝对**；要求 mean ≥ +0.05、p < 0.05、CI 排除 0。
3. **完整 4,064 KC**：探索性 stim−none；不作为主因果终点。
4. **匹配 95 / 全部 97 MBON**：次级；匹配 95 的最低有意义效应 +0.05；全部 97 探索性。
5. **背景/天花板**：none 条件 matched-320/full-4064/MBON 的群体 rate 与 per-cell ≥0.85/≥0.95 细胞比例；
   若 none 已在天花板，无动态范围，实验无信息。
6. **三个低反馈 KC**：2460/109321、2650/117446、3971/545375 单独报告；**不作为总体代表**。
7. **无刺激漂移/假阳性**：none-only 的 matched-320 [100,200) vs [200,300)，exact sign-flip；
   报告 p 与效应；若 p<0.05 标记漂移。随机噪声由 within-seed paired、exact、25 seeds 控制；
   不把 cell-step 当独立样本；不事后换终点。
8. **失败条件**：
   - 入口无响应 → `SENSORY_ENTRY_FAILED`；
   - 入口有响应但 KC 主终点失败 → **`SENSORY_TRANSMISSION_NOT_DETECTED`**（不得升级为学习成功）；
   - KC 有响应但 MBON 无 → `KC_RESPONSE_WITHOUT_MBON_TRANSMISSION`；
   - KC+MBON 均通过 → `SENSORY_TRANSMISSION_DETECTED_IN_SYNTHETIC_B`（仅计算现象）；
   - 资源/缺失 → `RESOURCE_NO_GO` / `INCONCLUSIVE`。
- 所有 25 seeds 保留，含 seed 114；不挑选、不删除；不重跑。

---

## 6. 资源预检（不默认 2 GiB 足够）

- C1 实测 Job 峰值 **1942.7 MiB**；距 2 GiB 仅 **105.3 MiB**。
- S1 额外记录（每 seed 两条件，uint8）：**3,909,600 B ≈ 3.728 MiB**；
  加静态余量 2 MiB → **5.73 MiB**；
  静态估计峰值 **1948.43 MiB**，余量 **99.57 MiB**。
- 时间估计 **290.38 s** < 900 s；时间不是瓶颈。
- **`RESOURCE = RESOURCE_PLAUSIBLE_TIGHT`；`RESOURCE_VERIFIED = NO`**。
  2 GiB 不能称为已验证；S1 必须在 Job 下实测；若峰值 >2048 MiB 或 wall >900 s → `RESOURCE_NO_GO`。
- **不自动提高上限**；低内存逐 seed/条件立即保存 NPZ 后释放；不在内存保留 25-seed 全数组。
- 父进程在统计后处理前持久化 Job 峰值；沿用 `CREATE_SUSPENDED → Job Object → ResumeThread`。
- 记录脚本、源码、官方 W 全量 SHA-256；官方 W 只读。

---

## 7. 授权与科学边界

- **P11-2-C1 的一次性授权已消费**，不适用于 S0/S1。
- S0 **不生成自我批准配置**，不执行真实实验。
- S1 需要用户**单独书面批准**一次；`P11_2_S1_ONE_RUN_APPROVAL.md` 只是申请。
- 即使 S1 通过，也只支持**合成消融模型中的计算现象**，不代表原始完整 MaleCNS 已恢复生理功能。
- 真实 KC→MBON 学习未授权；DN 自主转向未验证；实车自动导航 **NO-GO**。
- `Training Authorization = False`；`REAL_SYNAPTIC_LEARNING_READY = NO`；
  Level A/B/C/Biological = **NO-GO**。

---

## 8. 结论

- **原感觉前馈路径保留**（2-hop via non-KC；B 未删除）；**KC→KC 递归已删除**；KC→MBON 保留。
- **值得申请一次 S1**（设计完整、主终点冻结），但**资源 tight**，且前馈 net 很小，可能得到
  `SENSORY_TRANSMISSION_NOT_DETECTED`；这不是学习或生物学结论。
- **2 GiB 不可称为验证通过**；只能称 `PLAUSIBLE_TIGHT`。
- 具体需要用户批准：一次 S1 的 B/no-stim-stim 配对、25 seeds、20,000 步、LPLC2/LC4 精确 311、
  `dv=0.2 amount=0.036254`、[100,300)、CPU、15 min/2 GiB、一次性、不自动重跑、
  不训练/ΔW/Dijkstra/迷宫/B/C/实体车。

---

## 9. 交付

- `loop/P11_2_S0_SENSORY_PREREG.md`（本文件）
- `loop/P11_2_S0_PATH_AUDIT.md`
- `loop/p11_2_s0_preflight.py`
- `loop/test_p11_2_s0_preflight.py`
- `loop/p11_2_s0_report.json`
- `loop/P11_2_S1_ONE_RUN_APPROVAL.md`

历史 C1/C1a 报告与原始数据保留，未覆盖；本轮未运行新动力学。
