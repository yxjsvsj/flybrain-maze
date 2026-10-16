"""P9-1b1：COBA 真实连接组的**参数定标与计算可行性门控**（只读/合成；不跑真实全脑）。

文献定量依据（可核实；单位与适用细胞见下表）——用于**声明**参数协议，**不**按转向/导航调参。
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import tracemalloc

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 可核实的果蝇电生理定量依据（单位/细胞/适用性） ----
LIT = [
    {"param": "R_in (input resistance)", "value": "KC: 2.5±1 GΩ; PN: 598±69 MΩ; 胚胎中枢: 5.1±0.4 GΩ",
     "cells": "Kenyon cells(P. americana); 触角叶 PN; 胚胎中枢神经元",
     "paper": "J Neurophysiol 2009, 10.1152/jn.00183.2009; J Neurosci 29(19):6239, 10.1523/JNEUROSCI.1039-09.2009; J Neurosci 18(12):4673",
     "use": "g_l=1/R_in ≈ 0.2–1.7 nS（**逐类型不同**）"},
    {"param": "C_m (capacitance)", "value": "KC: 2.7±0.8 pF; 胚胎: 2.1±0.15 pF (0.7–4.6); 2025 Cell TableS2: 0.9–5 pF",
     "cells": "KC / 胚胎中枢 / 多类型(Drosophila)", "paper": "10.1152/jn.00183.2009; J Neurosci 18(12):4673; Cell 2025 (10.1016/j.cell.2025.11.040)",
     "use": "C ≈ 2–5 pF（小型中枢神经元量级）"},
    {"param": "tau_m = R_in*C", "value": "≈ 5–25 ms（由上二者推算）", "cells": "Drosophila 小型中枢神经元",
     "paper": "由上述 R_in×C 推算", "use": "膜时间常数**量级**（逐类型缺直接值）"},
    {"param": "V_rest / V_th", "value": "−53±9 / −47±6 mV (KC); PN V_rest≈−60", "cells": "KC / PN",
     "paper": "10.1152/jn.00183.2009; 10.1523/JNEUROSCI.1039-09.2009", "use": "E_l≈−60、V_th≈−47 mV"},
    {"param": "E_exc (nAChR)", "value": "≈0 mV", "cells": "胆碱能突触", "paper": "Groschner 2022 (10.1038/s41586-022-04428-3)",
     "use": "兴奋反转电位"},
    {"param": "E_inh (GABA/GluCl)", "value": "mIPSC 反转 ≈−67 mV (Cl⁻)；GluCl 分流", "cells": "GABA/GluCl 突触",
     "paper": "10.1152/jn.00183.2009; Groschner 2022", "use": "抑制反转电位"},
    {"param": "tau_syn_e (nAChR mEPSC decay)", "value": "≈1.35–2 ms", "cells": "KC / 培养胚胎神经元",
     "paper": "Lee&O'Dowd 1999 (PMC6782340); PMC6740836", "use": "EPSP 衰减常数"},
    {"param": "tau_syn_i (GABA mIPSC decay)", "value": "≈2.67 ms", "cells": "KC", "paper": "PMC6740836",
     "use": "IPSP 衰减常数"},
    {"param": "单通道/单位点电导", "value": "单 nAChR 通道 ≈121±15 pS；单位点 ~0.1 nS；量化 q≈1.05 pA",
     "cells": "KC / ORN→PN(Drosophila)", "paper": "PMC6740836; Gouwens&Wilson 2009 (10.1523/JNEUROSCI.1039-09.2009); Kazama&Wilson 2008",
     "use": "**单突触/单位点**电导量级——**非**归一化 W 的 g_unit"},
]

# ---- 唯一、事前声明的参数使用协议 ----
PROTOCOL = {
    "C_pF": {"value": "2–5", "class": "cross-cell-type extrapolation", "source": "LIT C_m", "unit": "pF"},
    "g_l_nS": {"value": "0.2–1.7", "class": "cross-cell-type extrapolation", "source": "1/R_in", "unit": "nS"},
    "E_l/Exc/Inh/Vth_mV": {"value": "-60 / 0 / -67 / -47", "class": "measured(range)", "source": "LIT", "unit": "mV"},
    "tau_syn_e/i_ms": {"value": "1.5 / 2.7", "class": "measured(range)", "source": "LIT", "unit": "ms"},
    "g_unit_nS_per_unit_weight": {"value": "**UNKNOWN**", "class": "MISSING", "source": "无（本数据集无逐连接电导；无 count->nS 换算）",
                                  "unit": "nS / 单位归一化权重", "rule": "**禁止**把单突触电导(0.1–数 nS)直接指定给归一化 W；**禁止**按转向/导航成功率调"},
}


def g_unit_blocker():
    return {
        "issue": "单个生物突触电导 ≠ 归一化 weights.npz 的 g_unit",
        "why": "W 是**逐突触后 L1 归一化的相对权重**（无单位）；连接组只给**突触计数**，未给逐连接电导；"
               "且**无 count->functional-efficacy 的可信换算**发表。",
        "consequence": "**科学 GO 未通过**：不能把 0.1–数 nS 直接赋给归一化 W，也不能用合成 g_unit 结果宣称生物学成功。",
        "needed": ["逐连接/逐类型电导测量，或本数据集/同源的计数->电导标定；",
                   "若不可得，只能以**声明约定+有界范围**做敏感性分析（非功能结论）"],
    }


def measure_convergence_and_perf(N=166700, nnz=1000000, dt=20.0, substeps_list=(50, 100, 500, 2000), seed=1):
    """修正的性能/收敛实测：不同 substeps 的**合成网络**耗时、发放计数误差、峰值内存。"""
    from loop.coba_proto import CobaNetwork, CobaParams
    P = CobaParams(g_unit_nS=10.0)
    rng = np.random.default_rng(seed)
    W = sparse.csr_matrix((rng.random(nnz), (rng.integers(0, N, nnz), rng.integers(0, N, nnz))), shape=(N, N))
    # 固定物理刺激：1% 神经元在首步发放（脉冲事件）
    drive = np.zeros(N); drive[rng.integers(0, N, N // 100)] = 1.0
    ref = None
    out = []
    for sub in substeps_list:
        net = CobaNetwork(W, sparse.csr_matrix((N, N)), P, dt, sub)
        net.S_prev[:] = drive
        gc.collect(); tracemalloc.start()
        t0 = time.perf_counter()
        cnt = net.step()
        per = (time.perf_counter() - t0) * 1000
        cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        total = float(cnt.sum())
        if ref is None or sub == max(substeps_list):
            pass
        out.append({"substeps": sub, "dts_ms": dt / sub, "ms_per_step": round(per, 1),
                    "spikes": int(round(total)), "peak_mem_MB": round(peak / 1e6, 2)})
    ref_spikes = out[-1]["spikes"]
    for r in out:
        r["spike_err_vs_ref"] = abs(r["spikes"] - ref_spikes) / max(1, ref_spikes)
    Wmem = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1e6
    return {"N": N, "nnz": nnz, "dt": dt, "rows": out, "W_mem_MB": round(Wmem, 2),
            "note": "substeps=50 **未达** 强耦合链所需精度；参考=最大 substeps；不外推为真实全脑实测速度"}


def decision(perf=None):
    eng = {
        "real_W_mapping": "PASS（P9-1b0：W=We−Wi 精确、方向/计数/绝对权和守恒）",
        "compute_numeric_error": "可控（指数积分+子步；但精度要求 substeps≥数百）",
        "resources": (perf or {}).get("note", "CPU: substeps=500 时单步成本高；全脑 CPU 不可承受（需 CUDA 或子步高效方案）；小规模子网可行"),
        "verdict": "GO（**仅限小规模真实子网络**；且须满足证据化 substeps 精度）",
    }
    sci = {
        "membrane_params": "文献量级可得（R_in/C/τ_m/V/E_rev/τ_syn 有跨类型范围）",
        "absolute_conductance": "**g_unit 未通过**（无 count->nS 换算；单突触电导≠归一化 W 的 g_unit）",
        "verdict": "NO-GO（科学参数未通过）",
    }
    both = eng["verdict"].startswith("GO") and sci["verdict"].startswith("GO")
    return {"engineering": eng, "science": sci,
            "both_go_for_next_round": bool(both),
            "action": ("若两者均 GO 才建议申请小规模真实网络功能实验；"
                       "否则**报告阻碍并停止**，不得用合成参数结果宣称生物学成功。"
                       if not both else "可申请小规模真实网络功能实验")}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    perf = measure_convergence_and_perf()
    print("[P9-1b1] perf/convergence (synthetic):")
    for r in perf["rows"]:
        print(f"  substeps={r['substeps']:5d} dts={r['dts_ms']:.4f}ms {r['ms_per_step']:8.1f} ms/step "
              f"spikes={r['spikes']:6d} err={r['spike_err_vs_ref']:.3f} peak_mem={r['peak_mem_MB']}MB")
    print(f"  W_mem={perf['W_mem_MB']} MB; {perf['note']}")
    d = decision(perf)
    print("[P9-1b1] decision:", d["engineering"]["verdict"], "|", d["science"]["verdict"],
          "| next-round:", d["both_go_for_next_round"])
    json.dump({"literature": LIT, "protocol": PROTOCOL, "g_unit_blocker": g_unit_blocker(),
               "perf": perf, "decision": d},
              open(os.path.join(HERE, "p91b1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote p91b1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
