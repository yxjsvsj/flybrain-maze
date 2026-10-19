# P10 文献勘误：Hige et al. 2015 引用（独立勘误，**不静默覆盖历史**）

## 错误
| 错误项 | 内容 | 出现位置 |
|---|---|---|
| **DOI 错误** | `10.1016/j.neuron.2014.12.045`（**非**该论文） | `loop/P100_REPORT.md`（P10-0）、`loop/p10_design.py` |
| **卷/页错误** | `Neuron 85:1341` | `loop/P100_REPORT.md`、`loop/p10_design.py`、`loop/P100C_DAN_COMPARTMENT_REVIEW.md`、`loop/p100c_dan_compartment.py` |

## 正确出处
**Hige et al. (2015), “Heterosynaptic Plasticity Underlies Aversive Olfactory Learning in Drosophila”, *Neuron* 88(5):985–998.**
**DOI：`10.1016/j.neuron.2015.11.003`**

## 受影响文件（共 4 个）
1. `loop/P100_REPORT.md` — 错误 DOI + 错误卷/页。
2. `loop/p10_design.py` — 错误 DOI + 错误卷/页。
3. `loop/P100C_DAN_COMPARTMENT_REVIEW.md` — 错误卷/页（无 DOI）。
4. `loop/p100c_dan_compartment.py` — 错误卷/页（无 DOI）。

## 处理政策
- **本文件为独立勘误记录**；**未**改写上述历史文件，**未**覆盖冻结结果。
- 引用层面的结论（一般 KC→MBON 可塑性 / 区室化 DA 调制为文献支持）**不因勘误而升级**；
  具体区室归属与“该 MaleCNS 突触受调制”的断言仍须以**正确出处 + 同源数据**重新核验。
- `g_unit` 未定标、可变目标不可辨识、功能调制未验证等阻碍**保持不变**。

## 可核验性
- 正确 DOI 由任务方给定并在此固定；错误 DOI 与错误卷/页已在上表逐文件列出。
- 未使用无法核验的文献或虚构实验数据。
