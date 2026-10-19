"""P11-0 回归：归档只读、高基线分布、KC 细胞级、资格迹、源码审计、裁决边界。不运行神经动力学。"""
from __future__ import annotations

import json
import os
import sys

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p11_0 offline diagnosis ===========")
    d = json.load(open(os.path.join(HERE, "p11_0_report.json"), encoding="utf-8"))
    a = d["archive"]
    if a.get("status") != "ARCHIVED_VERIFIED":
        check("RAW_DATA_UNAVAILABLE marked", a.get("status") == "RAW_DATA_UNAVAILABLE")
        v = d["verdicts"]
        check("still NO-GO", v["P10C2_A0_Canonical"] == "INCONCLUSIVE" and v["Training_Authorization"] is False)
    else:
        check("archive verified", a["status"] == "ARCHIVED_VERIFIED")
        check("76 files / 75 npz / records", a["integrity"]["file_count"] == 76
              and a["integrity"]["npz"] == 75 and a["integrity"]["records"])
        check("no archive missing/mismatch", not a["integrity"]["missing"] and not a["integrity"]["mismatch"])
        kc_cond = d["high_baseline"]["KC"]["none"]
        kc = kc_cond["stim"]
        check("KC mean ~0.9118", abs(kc["population_mean"]["mean"] - 0.9118) < 1e-3,
              str(kc["population_mean"]["mean"]))
        check("KC median ~0.96", abs(kc["population_mean"]["median"] - 0.96) < 1e-6)
        check("KC >=0.85 fraction high (>0.5)", kc["frac_ge_0.85"] > 0.5, str(kc["frac_ge_0.85"]))
        check("KC top-10% share low (<0.5)", kc_cond["concentration"]["top10_share_mean"] < 0.5)
        check("KC most cells needed for 80% (>50%)", kc_cond["concentration"]["n_cells_80pct_mean"] > 160)
        mb = d["high_baseline"]["MBON"]["none"]["stim"]
        check("MBON mean ~0.8815", abs(mb["population_mean"]["mean"] - 0.8815) < 1e-3)
        check("MBON >=0.85 fraction high", mb["frac_ge_0.85"] > 0.5)
        c = d["kc_cell_level"]
        check("KC 320 cells", c["n"] == 320)
        check("KC positive 0 / near-zero 319 / negative 1",
              c["positive"] == 0 and c["near_zero"] == 319 and c["negative"] == 1)
        check("KC BH-FDR 0 significant", c["bh_fdr"]["n_significant"] == 0)
        check("KC net small", abs(c["net"]) < 0.01)
        e = d["eligibility_distribution"]
        check("eligibility 4533 edges", e["n_edges"] == 4533)
        check("eligibility positive/negative broad", e["paired_change"]["positive"] > 500
              and e["paired_change"]["negative"] > 500)
        check("eligibility BH-FDR 0 significant", e["bh_fdr"]["n_significant"] == 0)
        check("binary fraction ~0.97 (not amplitude)", 0.9 < e["final_binary_fraction"]["none_mean"] < 1.0)
        s = d["source_audit"]
        check("model params recorded", s["confirmed_model_params"]["dt"] == 0.02
              and s["confirmed_model_params"]["gain"] == 3.0 and s["confirmed_model_params"]["tonic"] == 0.14)
        check("truncation recorded", s["truncation"]["lost_incoming_edges"] > 4_000_000)
        check("LP direct 0 / 2-hop 56", s["lplc2_lc4_to_kc"]["direct_edges"] == 0
              and s["lplc2_lc4_to_kc"]["two_hop_intermediates"] == 56)
        check("root cause NOT_ESTABLISHED", s["root_cause"] == "NOT_ESTABLISHED")
    v = d["verdicts"]
    check("P10C2-A0 Entrance PASS preserved", v["P10C2_A0_Entrance"] == "PASS")
    check("KC NOT_DETECTED preserved", v["P10C2_A0_KC_Main_Endpoint"] == "NOT_DETECTED")
    check("Ceiling FAIL preserved", v["P10C2_A0_Ceiling"] == "FAIL")
    check("Canonical INCONCLUSIVE preserved", v["P10C2_A0_Canonical"] == "INCONCLUSIVE")
    check("HIGH_BASELINE_PHENOTYPE SUPPORTED", v["HIGH_BASELINE_PHENOTYPE"] == "SUPPORTED")
    check("CELL_LEVEL_HETEROGENEITY NOT_OBSERVED", v["CELL_LEVEL_HETEROGENEITY"] == "NOT_OBSERVED")
    check("MODEL_ROOT_CAUSE NOT_ESTABLISHED", v["MODEL_ROOT_CAUSE"] == "NOT_ESTABLISHED")
    check("new model-validity experiment only a proposal", v["worth_new_preregistered_model_validity_experiment"] is True)
    check("Level A/B/C/Biological NO-GO", v["Level_A_Training"] == "NO-GO" and v["Level_B"] == "NO-GO"
          and v["Level_C"] == "NO-GO" and v["Biological"] == "NO-GO")
    check("training not authorized", v["Training_Authorization"] is False)
    # no new dynamics in the diagnosis module
    src = open(os.path.join(HERE, "p11_0_offline_diagnosis.py"), encoding="utf-8").read()
    check("no FlyBrain.step / flybrain import", "FlyBrain.step" not in src and "from flybrain" not in src)
    # historical A0 report unchanged
    r = json.load(open(os.path.join(HERE, "p10c2_a0_report.json"), encoding="utf-8"))
    check("historical A0 state retained", r["state"] == "INCONCLUSIVE" and r["training_authorized"] is False)
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
