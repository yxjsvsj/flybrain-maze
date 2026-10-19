"""P10-0b 回归：two_hop 语义修正（合成稀疏图，手工可验证）+ 完整链 DP + DAN/裁决。

合成图仅用于**验证算法方向/计数/正负抵消/零权**；真实数字由 `loop.p100b_path_audit` 从官方 W 复算。
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.sparse as sp

from loop.p100_evidence import two_hop
from loop.p100b_path_audit import chain_layers, dp_chain, status

RESULTS = []


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def make_D(n, edges):
    rows = [r for r, _, _ in edges]
    cols = [c for _, c, _ in edges]
    vals = [v for _, _, v in edges]
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return {"W": W, "ids": np.arange(n), "cell_type": np.array(["x"] * n, dtype=object),
            "side": np.array(["?"] * n, dtype=object)}


def test_two_hop_synthetic():
    # 0 -> 1 -> 2 (w 2*3=6) 且 0 -> 3 -> 2 (w -2*1=-2)；4 为零权边
    D = make_D(6, [(1, 0, 2.0), (2, 1, 3.0), (3, 0, -2.0), (2, 3, 1.0), (4, 0, 0.0), (2, 4, 5.0)])
    r = two_hop(D, [0], [2])
    check("synthetic n_intermediate = 2", r["n_intermediate"] == 2, str(r["n_intermediate"]))
    check("synthetic n_paths = 2 (not intermediates)", r["n_paths"] == 2, str(r["n_paths"]))
    check("synthetic signed_product = 2*3 + (-2)*1 = 4", abs(r["signed_product"] - 4.0) < 1e-9, str(r["signed_product"]))
    check("synthetic abs_product = 6 + 2 = 8", abs(r["abs_product"] - 8.0) < 1e-9, str(r["abs_product"]))
    check("synthetic sum_w_in = 2 + (-2) = 0", abs(r["sum_w_in"] - 0.0) < 1e-9, str(r["sum_w_in"]))
    check("synthetic sum_w_out = 3 + 1 = 4", abs(r["sum_w_out"] - 4.0) < 1e-9, str(r["sum_w_out"]))
    # 方向：反向 (2 -> 0) 不应有中间神经元
    r2 = two_hop(D, [2], [0])
    check("synthetic reversed direction = 0 intermediates", r2["n_intermediate"] == 0, str(r2["n_intermediate"]))
    # 零权边 (4) 不计入路径
    check("synthetic zero-weight edge excluded", r["n_paths"] == 2)


def test_dp_synthetic_single_path():
    # 单链 0->1->2->3->4->5，权重 2,3,4,5,6 => 乘积 720
    D = make_D(6, [(1, 0, 2.0), (2, 1, 3.0), (3, 2, 4.0), (4, 3, 5.0), (5, 4, 6.0)])
    layers = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
    r = dp_chain(D, layers)
    check("dp synthetic n_paths_walk = 1", abs(r["n_paths_walk"] - 1.0) < 1e-9, str(r["n_paths_walk"]))
    check("dp synthetic signed_product = 720", abs(r["signed_product"] - 720.0) < 1e-6, str(r["signed_product"]))
    check("dp synthetic abs_product = 720", abs(r["abs_product"] - 720.0) < 1e-6, str(r["abs_product"]))


def test_real_chain():
    S = status()
    check("real chain layers = [LP,*,KC,MBON,*,DN]", S["layer_sizes"][0] > 0 and S["layer_sizes"][2] > 0
          and S["layer_sizes"][3] > 0 and S["layer_sizes"][5] > 0, str(S["layer_sizes"]))
    check("real structural path EXISTS", S["structural_path_exists"] is True)
    check("real bottleneck is X->KC", S["bottleneck_layer"] == "X->KC", str(S["bottleneck_layer"]))
    check("real DAN->KC anatomy present", S["dan"]["anatomy"]["DAN_to_KC"]["edges"] > 0)
    check("real DAN compartment evidence MISSING", "MISSING" in S["dan"]["compartment_level"])
    check("verdict structural=exists", S["verdict"]["structural_path"] == "存在")
    check("verdict modulatory=insufficient", "不足" in S["verdict"]["modulatory_evidence"])
    check("verdict functional unverified", "未验证" in S["verdict"]["functional_transmission"])
    check("verdict g_unit unresolved", "g_unit" in S["verdict"]["biological_calibration"])
    check("verdict NO-GO (go=False)", S["verdict"]["go"] is False)


def main() -> int:
    print("=========== p100b_path_audit ===========")
    test_two_hop_synthetic()
    test_dp_synthetic_single_path()
    test_real_chain()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
