# P11-1a 阶段 2 参数来源审查 + P11-0 `dz` 勘误（只读）

## 1. 阶段 2 候选值与来源
| 参数 | 候选 | 来源 | 级别 |
|---|---|---|---|
| gain | 1.5 / 3.0 | fly64/model.py（代码值）；flybrain/brain.py 默认值 3.0 | **作者手工设置/代码默认** |
| tonic | 0.10 / 0.14 / 0.18 | flybrain 默认 0.14；fly64 0.18 | **作者手工设置/代码默认** |
| noise_hz | 0.6 / 1.2 | flybrain 默认 1.2 | **作者手工设置/代码默认** |
| noise_amp | 0.22 | flybrain 默认 | **作者手工设置/代码默认** |
- **未找到**独立同行评审论文、DOI 或实测生理标定值；`brain.py` 仅写
  “Dynamics follow ornata/fly (fly64/model.py)”与“fly64 used tonic 0.18, gain 1.5 / inject.py showed tonic 0.14, gain 3.0”。
- **不能**把“文献提过相似模型”当作参数已被验证；这些是**代码默认/手工值**，不是独立校准。
- **Phase 2 Parameter Provenance = INSUFFICIENT**。
- **Phase 2 Execution = NO-GO**；参数敏感性保留为**待设计假设**。
- **禁止**根据 P10-C2-A0 的 KC 结果选择更有利的 gain/tonic/noise；不得声称“敏感性=生理定标”。

## 2. P11-0 `dz` 勘误
| 统计 | 值 | 定义 |
|---|---|---|
| P11-0 报告的 `dz` | **−0.022377** | `mean(per-cell means) / sd(per-cell means)`（**跨细胞**） |
| **pooled paired dz** | **−0.003161** | `mean(all paired diffs) / sd(all paired diffs)`（按 seed 配对） |
| per-cell paired dz | 未定义/不可用 | 部分 KC 跨 seed 方差为 0（272 个全零） |
- **勘误**：P11-0 的 `dz` 是**跨细胞标准化均值**，**不是**按独立 seed 配对的 Cohen's dz；两者不可混用。
- 历史 P11-0 文件**未修改**；以本文件记录差异。

## 3. 结论
- 阶段 2 参数来源**不足以支持执行**；不得自动批准运行。
- 阶段 1 与完整脑资源基准仍是独立事项；`Training Authorization=False`。
