"""P11-1d 回归：归档哈希、KC 全饱和核验（含 3 个零）、bodyId 映射、复算、来源、模型诊断。只读。"""
from __future__ import annotations

import json
import os
import subprocess
import sys

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p11_1d offline verify ===========")
    d = json.load(open(os.path.join(HERE, "p11_1d_report.json"), encoding="utf-8"))
    a = d["archive"]
    check("archive verified", a["status"] == "ARCHIVED_VERIFIED")
    check("29 files / 26 npz", a["file_count"] == 29 and a["npz"] == 26)
    check("all per-file hashes match (source+backup)", a["all_match"] is True)
    check("no missing/extra/mismatch", not a["missing"] and not a["extra"] and not a["mismatch"])
    k = d["kc_saturation"]
    check("KC verdict NOT_EXACTLY_FULL", k["verdict"] == "NOT_EXACTLY_FULL")
    check("data extraction VERIFIED", k["data_extraction_verdict"] == "VERIFIED")
    check("shapes/binary/totals pass", k["checks"]["shapes"] and k["checks"]["binary"] and k["checks"]["totals_match"])
    check("full 4064 NOT all ones every step", k["all_4064_every_step"] is False)
    check("matched 320 all ones every step", k["matched_320_every_step"] is True)
    check("3 zeros total", k["zeros_total"] == 3)
    check("affected seeds 104/114/115", sorted(x["seed"] for x in k["affected_seeds"]) == [104, 114, 115])
    check("each affected seed 1 zero / 1 cell / 1 step", all(x["zeros"] == 1 and x["cells_with_zero"] == 1
          and x["steps_not_all"] == 1 for x in k["affected_seeds"]))
    check("matched zeros = 0", k["matched_zeros_total"] == 0)
    check("min per-cell fraction ~0.995", abs(k["min_per_cell_fraction"] - 0.995) < 1e-6)
    check("first full step range 17-24", k["first_full_step"]["min"] == 17 and k["first_full_step"]["max"] == 24)
    check("full all-ones does NOT persist for affected seeds", k["all_persist_to_400"] is False)
    check("startup bins 0.1463 / 1.0", abs(k["bins"]["0-20"] - 0.146263) < 1e-5 and k["bins"]["40-60"] == 1.0)

    b = d["bodyid_mapping"]
    check("bodyId mapping VERIFIED", b["verdict"] == "VERIFIED")
    check("all bodyId checks pass", all(b["checks"].values()))
    check("kc 4064 / mbon 97 / matched 320/95", b["checks"]["kc_count_4064"] and b["checks"]["mbon_count_97"]
          and b["checks"]["matched_counts_320_95"])

    c = d["comparison_recompute"]
    check("comparison REPRODUCED", c["reproduced"] is True)
    check("diff +0.0882", abs(c["difference_mean"] - 0.088201) < 1e-5)
    check("CI [0.0599,0.1416]", abs(c["difference_ci"]["ci_low"] - 0.05989751) < 1e-5
          and abs(c["difference_ci"]["ci_high"] - 0.14159824) < 1e-5)
    check("seed 114 retained (+0.712)", abs(c["seed_114_diff"] - 0.711781) < 1e-4)
    check("25 per-seed diffs", len(c["per_seed_diff"]) == 25)
    check("noise pairing limitation stated", "NOT paired" in c["noise_pairing"])
    check("full 95 MBON ~0.9327", abs(c["full_95_MBON_mean"] - 0.932691) < 1e-5)
    check("subnet 95 MBON ~0.8815", abs(c["subnet_95_MBON_mean"] - 0.881472) < 1e-5)

    p = d["provenance"]
    check("run commit 7978938", p["run_commit"].startswith("7978938"))
    check("result commit f4176df", p["result_commit"] == "f4176df")
    check("source provenance PARTIAL", p["source_provenance"] == "PARTIAL")
    check("failed launch record present", p["launch_failure_record"] is True)
    check("RESOURCE INCONCLUSIVE", p["resource"].startswith("INCONCLUSIVE"))

    diag = d["model_diagnosis"]
    check("source-confirmed mechanisms", diag["status"]["source_confirmed_mechanisms"] is True)
    check("root cause NOT_ESTABLISHED", diag["status"]["causal_root_cause"] == "NOT_ESTABLISHED")

    v = d["verdicts"]
    check("ARCHIVE VERIFIED", v["ARCHIVE"] == "VERIFIED")
    check("KC NOT_EXACTLY_FULL", v["KC_FULL_SATURATION"] == "NOT_EXACTLY_FULL")
    check("DATA_EXTRACTION VERIFIED", v["DATA_EXTRACTION"] == "VERIFIED")
    check("BODYID VERIFIED", v["BODYID_MAPPING"] == "VERIFIED")
    check("FULL_VS_SUBNET REPRODUCED", v["FULL_VS_SUBNET_DIFFERENCE"] == "REPRODUCED")
    check("SOURCE PROVENANCE PARTIAL", v["SOURCE_PROVENANCE"] == "PARTIAL")
    check("RESOURCE INCONCLUSIVE", v["RESOURCE"] == "INCONCLUSIVE")
    check("MODEL_ROOT_CAUSE NOT_ESTABLISHED", v["MODEL_ROOT_CAUSE"] == "NOT_ESTABLISHED")
    check("REAL_SYNAPTIC_LEARNING_READY NO", v["REAL_SYNAPTIC_LEARNING_READY"] == "NO")
    check("training False", v["Training_Authorization"] is False)
    check("Level A/B/C/Biological NO-GO", v["Level_A_Training"] == "NO-GO" and v["Level_B"] == "NO-GO"
          and v["Level_C"] == "NO-GO" and v["Biological"] == "NO-GO")

    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout
        check("raw full npz not tracked", "101.npz" not in tracked and "125.npz" not in tracked)
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30).stdout
        check("pre-existing dirty files preserved", all(x in st for x in
              ("hardware/tof5_client.py", "loop/diagnose.py", "loop/p10c1b_decoder_freeze.json")))
        r = subprocess.run(["git", "diff", "--stat", "P1-mapfix", "--", "nav", "config.py",
                            "loop/decode.py", "loop/encode.py", "loop/run.py"],
                           cwd=ROOT, capture_output=True, text=True, timeout=30)
        check("P1 freeze empty", r.returncode == 0 and r.stdout.strip() == "")
    except Exception:                            # noqa: BLE001
        check("git checks", False)
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
