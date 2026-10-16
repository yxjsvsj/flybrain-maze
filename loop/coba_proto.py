"""P9-1a：**COBA-LIF 数值原型**（最小、单位一致、可验证）。**只测合成小网络**；不跑真实全脑、不接车。

单位约定（一致）：C[pF]、g[nS]、E[mV]、t[ms]、I[pA]；`C dV/dt` = pF·mV/ms = **pA** ✓；`tau = C/g` = pF/nS = **ms** ✓。
- 突触释放**只来自 presynaptic spike 事件**（`W @ S_prev`），**非**复位前膜电位连续释放（不重包装已关闭的 P8 渐变释放）。
- 电导用**精确指数衰减** `g *= exp(-dt/tau_syn)`；膜电位用**指数欧拉**（对定常电导为精确解、无条件稳定），
  并提供**内部子步长**以对时变电导收敛（**不是** 20 ms 显式欧拉）。

**W 的三层区分（严禁混淆）**：
  ① 连接组**原始突触计数**；② `weights.npz` 的**逐突触后归一化相对权重**（L1≤1，**无单位**）；
  ③ **未知的绝对电导**。`W -> g_syn` 需额外定标：本原型用**声明的合成单位** `g_unit_nS`（**非** nS 实测定标），
  **禁止**把归一化 W 冒充 nS。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class CobaParams:
    """单位：C[pF] g[nS] E[mV] tau[ms]。缺生物定标者标注"Synthetic"。"""
    C_pF: float = 100.0          # Synthetic（膜电容）
    g_l_nS: float = 2.0          # Synthetic（漏电导）-> tau_m = C/g_l = 50 ms
    E_l_mV: float = -60.0        # 文献量级（静息）
    E_exc_mV: float = 0.0        # 文献量级（胆碱能反转）
    E_inh_mV: float = -75.0      # 文献量级（GluCl/GABA）
    V_th_mV: float = -50.0       # 文献量级
    V_reset_mV: float = -60.0    # Synthetic
    tau_syn_e_ms: float = 2.0    # 文献量级 1-5 ms
    tau_syn_i_ms: float = 10.0   # 文献量级 5-20 ms
    g_unit_nS: float = 1.0       # **未知**绝对电导尺度的合成占位（1 单位权重 -> 1 nS，非实测定标）


class CobaNetwork:
    """COBA-LIF 最小网络。W_exc/W_inh 为稀疏 `(post, pre)` 的**相对**权重（无单位）。"""

    def __init__(self, W_exc, W_inh, params: CobaParams, dt_ms: float, substeps: int = 1,
                 V0_mV: float | None = None):
        self.p = params
        self.dt = float(dt_ms)
        self.substeps = int(substeps)
        self.n = W_exc.shape[0]
        assert W_exc.shape == W_inh.shape == (self.n, self.n)
        self.We = sparse.csr_matrix(W_exc, dtype=np.float64)
        self.Wi = sparse.csr_matrix(W_inh, dtype=np.float64)
        self.reset(V0_mV)

    def reset(self, V0_mV: float | None = None):
        v0 = self.p.E_l_mV if V0_mV is None else float(V0_mV)
        self.V = np.full(self.n, v0, np.float64)
        self.g_exc = np.zeros(self.n, np.float64)      # nS
        self.g_inh = np.zeros(self.n, np.float64)      # nS
        self.S_prev = np.zeros(self.n, np.float64)     # 上一步 spike(0/1)
        self.V_peak = self.V.copy()                    # **子步级**峰值（用于瞬态可观测）
        self.t_ms = 0.0

    def step(self):
        p = self.p
        dts = self.dt / self.substeps
        decay_e = np.exp(-dts / p.tau_syn_e_ms)
        decay_i = np.exp(-dts / p.tau_syn_i_ms)
        fired = np.zeros(self.n, bool)
        for k in range(self.substeps):
            # ① 精确指数电导衰减
            self.g_exc *= decay_e
            self.g_inh *= decay_i
            # ② 突触输入：**仅**来自上一步的 presynaptic spike 事件（子步开始时施加一次）
            if k == 0 and self.S_prev.any():
                self.g_exc += p.g_unit_nS * (self.We @ self.S_prev)
                self.g_inh += p.g_unit_nS * (self.Wi @ self.S_prev)
            # ③ 膜电位：指数欧拉（对定常电导精确、无条件稳定）
            g_tot = p.g_l_nS + self.g_exc + self.g_inh
            V_inf = (p.g_l_nS * p.E_l_mV + self.g_exc * p.E_exc_mV + self.g_inh * p.E_inh_mV) / g_tot
            tau_eff = p.C_pF / g_tot                     # ms
            self.V = V_inf + (self.V - V_inf) * np.exp(-dts / tau_eff)
            np.maximum(self.V_peak, self.V, out=self.V_peak)   # 子步级峰值（瞬态可观测）
        # ④ 发放与复位
        fired = self.V >= p.V_th_mV
        self.V[fired] = p.V_reset_mV
        self.S_prev = fired.astype(np.float64)
        self.t_ms += self.dt
        return fired


def mirror_network(W, side: np.ndarray):
    """左右镜像：把 side=='L' 与 'R' 的索引对调，生成镜像连接（用于对称性测试）。"""
    idx = np.arange(W.shape[0])
    L = idx[side == "L"]; R = idx[side == "R"]
    perm = idx.copy()
    n = min(len(L), len(R))
    perm[L[:n]] = R[:n]; perm[R[:n]] = L[:n]
    return W[perm][:, perm], perm
