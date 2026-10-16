"""P3-1 回归：visual_motor_audit 的解剖审计纯函数（by_type/ipsi_contra/dn_target_set）+ 实据不变式。"""
from __future__ import annotations

import sys

from loop.visual_motor_audit import (STA_VPN, by_type, dn_target_set, dn_target_set_by_body,
                                    indirect_paths, ipsi_contra, load_data, vpn_dn_edges)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== visual_motor_audit ===========")
    # 合成边：验证 ipsi/contra 与 by_type
    ed = [{"dn_type": "D1", "vpn_type": "V1", "vpn_side": "L", "dn_side": "L", "w": 3.0,
           "dn": 1, "vpn": 10},
          {"dn_type": "D1", "vpn_type": "V1", "vpn_side": "L", "dn_side": "R", "w": 1.0,
           "dn": 2, "vpn": 11}]
    ic = ipsi_contra(ed)
    check("ipsi_contra fraction", abs(ic["V1"]["ipsi"] - 0.75) < 1e-9)
    check("by_type sums weight", by_type(ed)[("D1", "V1")] == 4.0)

    # 实据（只读）：解剖路径存在且 ipsi 主导
    D = load_data()
    edges = vpn_dn_edges(D)
    check("has direct VPN->DN edges", len(edges) > 1000)
    icr = ipsi_contra(edges)
    check("LC4 ipsilateral-dominant (>0.9)", icr.get("LC4", {}).get("ipsi", 0) > 0.9)
    check("LPLC2 ipsilateral-dominant (>0.9)", icr.get("LPLC2", {}).get("ipsi", 0) > 0.9)
    check("frozen STA_VPN", STA_VPN == ("LC4", "LPLC4", "LPLC2"))
    tgt = dn_target_set(D)
    sides = {v["side"] for v in tgt.values()}
    check("DN target set non-empty with L&R", len(tgt) > 10 and {"L", "R"} <= sides)

    # 修正1：bodyId 为键保留同型多细胞 -> 数量应多于 by_type
    tgtb = dn_target_set_by_body(D)
    check("by-body targets >= by-type (no collapse)", len(tgtb) >= len(tgt))
    check("by-body sides both present", {"L", "R"} <= {v["side"] for v in tgtb.values()})

    # 修正2：二跳方向 = W[post,pre]，出权带符号
    ip = indirect_paths(D, STA_VPN, list(tgt.keys()), top=5)
    check("indirect paths non-empty", len(ip) > 0)
    check("indirect reports signed out weight", all("out_w_signed" in r and "out_w_abs" in r for r in ip))

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
