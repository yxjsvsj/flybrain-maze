"""P11-1a 回归：完整脑静态内存、none-only 计划、RNG 配对限制、参数来源、dz 勘误、门控。静态。"""
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
    print("=========== p11_1a resource audit ===========")
    d = json.load(open(os.path.join(HERE, "p11_1a_report.json"), encoding="utf-8"))
    check("archive verified", d["archive"]["status"] == "ARCHIVED_VERIFIED")
    check("official hashes OK", d["hashes"]["ok"] is True)
    check("brain hash full", d["hashes"]["files"]["brain.npz"].startswith("cc9bd1ecd00bd703"))
    check("weights hash full", d["hashes"]["files"]["weights.npz"].startswith("c29919aa44069a27"))
    f = d["full_brain"]
    check("full N/nnz", f["csr"]["shape"] == [166700, 166700] and f["csr"]["nnz"] == 25582938)
    check("CSR float32/int32", f["csr"]["dtype"] == "float32" and f["csr"]["indices_dtype"] == "int32")
    check("CSR bytes 205.33 MB", abs(f["csr"]["total_bytes"] / 1e6 - 205.33) < 0.01)
    check("metadata 53.39 MB", abs(f["metadata_bytes"] / 1e6 - 53.39) < 0.01)
    b = d["memory_budget"]
    check("init lower ~477 MB", abs(b["init_peak_lower_MB"] - 477.39) < 1.0)
    check("upper ~1024 MB", abs(b["upper_bound_with_1.5x_overhead_MB"] - 1024.08) < 1.0)
    check("resource NOT ready (unmeasured)", b["resource_ready"] is False)
    check("benchmark required", b["benchmark_required"] is True)
    check("2 GB cap: keep, no auto-raise", "keep 2 GB" in b["recommendation_2GB"] and "do not auto-raise" in b["recommendation_2GB"])
    p = d["none_only_plan"]
    check("none-only 10,000 steps", p["planned_full_brain_steps"] == 10000)
    check("25 seeds x 400 x 1 condition", p["conditions"] == ["none"] and len(p["seeds"]) == 25 and p["steps_per_seed"] == 400)
    check("static core ~1.56 min", abs(p["static_core_minutes"] - 1.56) < 0.05)
    check("static extrapolation caveat", "NOT a measured" in p["static_extrapolation_note"])
    check("eligibility skipped", "not needed" in p["eligibility"])
    check("benchmark separate auth", p["benchmark_separate_authorization"] is True and p["executed"] is False)
    r = d["rng_audit"]
    check("same seed only initial RNG", r["same_seed_is_only_initial_rng"] is True)
    check("noise pairing NOT strict", r["strict_noise_pairing"] is False)
    check("comparison is between-model", "between-model" in r["comparison_type"])
    check("only truncation-condition interpretation", "overall truncation condition" in r["can_only_interpret_as"])
    pr = d["phase2_provenance"]
    check("provenance INSUFFICIENT", pr["provenance_status"] == "INSUFFICIENT")
    check("no DOI recorded", pr["peer_reviewed_doi"] is None)
    check("code defaults distinguished", "hand-set" in pr["flybrain_code_defaults"]["source"])
    check("phase2 execution not ready", pr["phase2_execution_ready"] is False)
    check("no selection by A0", pr["no_selection_by_A0_result"] is True)
    dz = d["p11_0_dz_errata"]
    check("P11-0 dz is cross-cell", abs(dz["p11_0_reported_dz"] + 0.022377) < 1e-4)
    check("pooled paired dz ~ -0.00316", abs(dz["pooled_paired_dz"] + 0.003161) < 1e-4)
    check("dz mislabel recorded", dz["errata_recorded"] is True and "NOT a paired-by-seed" in dz["mislabel"])
    g = d["gates"]
    check("P10-C2-A0 INCONCLUSIVE", g["Historical_P10C2_A0"] == "INCONCLUSIVE")
    check("HIGH_BASELINE SUPPORTED", g["HIGH_BASELINE_PHENOTYPE"] == "SUPPORTED")
    check("ROOT_CAUSE NOT_ESTABLISHED", g["MODEL_ROOT_CAUSE"] == "NOT_ESTABLISHED")
    check("Phase1 Design GO", g["Phase1_Design_Readiness"] == "GO")
    check("Full-brain Resource NO-GO", g["Full_Brain_Resource_Readiness"] == "NO-GO")
    check("Phase1 Execution NO-GO", g["Phase1_Execution_Readiness"] == "NO-GO")
    check("Phase2 Provenance INSUFFICIENT", g["Phase2_Parameter_Provenance"] == "INSUFFICIENT")
    check("Phase2 Execution NO-GO", g["Phase2_Execution_Readiness"] == "NO-GO")
    check("Level A/B/C/Biological NO-GO", g["Level_A_Training"] == "NO-GO" and g["Level_B"] == "NO-GO"
          and g["Level_C"] == "NO-GO" and g["Biological"] == "NO-GO")
    check("training False", g["Training_Authorization"] is False)
    src = open(os.path.join(HERE, "p11_1a_resource_audit.py"), encoding="utf-8").read()
    check("no FlyBrain.step / flybrain import", "FlyBrain.step" not in src and "from flybrain" not in src)
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
