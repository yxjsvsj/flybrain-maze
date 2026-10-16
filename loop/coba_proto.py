"""P9-1a-b：**COBA-LIF 数值原型**（脉冲事件传递；逐子步发放）。**只测合成小网络**；不跑真实全脑、不接车。

单位（一致）：C[pF]、g[nS]、E[mV]、t[ms]、I[pA]；`C dV/dt`=pF·mV/ms=**pA**；`tau=C/g`=pF/nS=**ms**。

**突触传递 = 严格脉冲事件**（`g += g_unit·(W @ S)`，`S` 为上一**子步**的发放计数）。
**本引擎不使用任何连续/渐变释放**（不重用已关闭的 P8 假设，**不提"复位前 V 释放"**）。

**因果时序（每个内部子步固定顺序）**：
  ① `g *= exp(-dts/tau_syn)`（精确指数衰减）；
  ② `g += g_unit·(W @ S)`（`S` = **上一子步**的发放，含外部注入）；
  ③ 膜电位指数积分 `V <- V_inf + (V-V_inf)exp(-dts/tau_eff)`；
  ④ **在本子步检查阈值**：`V>=V_th -> 记录发放、V<-V_reset`；`S <- 本子步发放`（供下一子步/下一步）。
→ 同一物理刺激在 dt/substeps 变化下**收敛**（见测试）。

**W 三层区分**：①连接组**原始突触计数**；②`weights.npz` 的**逐突触后归一化相对权重**（L1≤1，无单位）；
③**未知的绝对电导**。`W->g_syn` 需额外定标；原型用**声明的合成单位** `g_unit_nS`，**禁止**把归一化 W 冒充 nS。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class CobaParams:
    """单位：C[pF] g[nS] E[mV] tau[ms]。缺生物定标者标注 Synthetic。"""
    C_pF: float = 100.0          # Synthetic
    g_l_nS: float = 2.0          # Synthetic (tau_m=C/g_l=50 ms)
    E_l_mV: float = -60.0        # 文献量级
    E_exc_mV: float = 0.0        # 文献量级
    E_inh_mV: float = -75.0      # 文献量级
    V_th_mV: float = -50.0       # 文献量级
    V_reset_mV: float = -60.0    # Synthetic
    tau_syn_e_ms: float = 2.0    # 文献量级 1-5 ms
    tau_syn_i_ms: float = 10.0   # 文献量级 5-20 ms
    g_unit_nS: float = 1.0       # **未知**绝对电导尺度的合成占位（非实测定标）


class CobaNetwork:
    """We/Wi 为稀疏 `(post, pre)` 的**相对**权重（无单位，须非负；符号已分流到两个矩阵）。"""

    def __init__(self, W_exc, W_inh, params: CobaParams, dt_ms: float, substeps: int = 1,
                 V0_mV: float | None = None):
        self.p = params
        self.dt = float(dt_ms)
        self.substeps = int(substeps)
        if self.dt <= 0 or self.substeps < 1:
            raise ValueError("dt>0 且 substeps>=1")
        if params.g_unit_nS <= 0:
            raise ValueError("拒绝：g_unit_nS 必须为正（负电导输入无效）")
        self.n = W_exc.shape[0]
        assert W_exc.shape == W_inh.shape == (self.n, self.n)
        We = sparse.csr_matrix(W_exc, dtype=np.float64)
        Wi = sparse.csr_matrix(W_inh, dtype=np.float64)
        if (We.data < 0).any() or (Wi.data < 0).any():
            raise ValueError("拒绝：We/Wi 不得含负权（符号须分流为两个非负矩阵）")
        self.We, self.Wi = We, Wi
        self.reset(V0_mV)

    def reset(self, V0_mV: float | None = None):
        v0 = self.p.E_l_mV if V0_mV is None else float(V0_mV)
        self.V = np.full(self.n, v0, np.float64)
        self.g_exc = np.zeros(self.n, np.float64)      # nS
        self.g_inh = np.zeros(self.n, np.float64)      # nS
        self.S_prev = np.zeros(self.n, np.float64)     # 上一步（末子步）发放计数
        self.V_peak = self.V.copy()
        self.V_min = self.V.copy()
        self.spikes = np.zeros(self.n, np.float64)     # 本步发放次数
        self.spike_log: list[tuple[int, float]] = []   # (neuron, t_ms) 发放时刻
        self.t_ms = 0.0

    def step(self):
        """推进一个外层 dt；**逐内部子步**检查阈值并及时复位、按顺序传递突触前脉冲。"""
        p = self.p
        dts = self.dt / self.substeps
        decay_e = np.exp(-dts / p.tau_syn_e_ms)
        decay_i = np.exp(-dts / p.tau_syn_i_ms)
        S = self.S_prev                                  # 进入本步时的突触前发放（含外部注入）
        count = np.zeros(self.n, np.float64)
        for k in range(self.substeps):
            # ① 精确指数电导衰减
            self.g_exc *= decay_e
            self.g_inh *= decay_i
            # ② 突触传递：仅事件（上一子步的发放）
            if S.any():
                self.g_exc += p.g_unit_nS * (self.We @ S)
                self.g_inh += p.g_unit_nS * (self.Wi @ S)
            # ③ 膜电位指数积分
            g_tot = p.g_l_nS + self.g_exc + self.g_inh
            V_inf = (p.g_l_nS * p.E_l_mV + self.g_exc * p.E_exc_mV + self.g_inh * p.E_inh_mV) / g_tot
            tau_eff = p.C_pF / g_tot
            self.V = V_inf + (self.V - V_inf) * np.exp(-dts / tau_eff)
            np.maximum(self.V_peak, self.V, out=self.V_peak)
            np.minimum(self.V_min, self.V, out=self.V_min)
            # ④ **本子步**阈值检查 + 及时复位；发放供下一子步
            fired = self.V >= p.V_th_mV
            if fired.any():
                idx = np.flatnonzero(fired)
                for i in idx:
                    self.spike_log.append((int(i), self.t_ms + (k + 1) * dts))
                self.V[fired] = p.V_reset_mV
                count += fired
                S = fired.astype(np.float64)
            else:
                S = np.zeros(self.n, np.float64)
        self.S_prev = S
        self.spikes = count
        self.t_ms += self.dt
        return count


def mirror_network(W, side: np.ndarray):
    """左右镜像：把 side=='L' 与 'R' 的索引对调，生成镜像连接（对称性测试）。"""
    idx = np.arange(W.shape[0])
    L = idx[side == "L"]; R = idx[side == "R"]
    perm = idx.copy()
    n = min(len(L), len(R))
    perm[L[:n]] = R[:n]; perm[R[:n]] = L[:n]
    return W[perm][:, perm], perm
