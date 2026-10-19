"""P10-C2-A0 回归：归档哈希、离线复算、来源审计、最终 NO-GO。仅读文件，不运行神经动力学。"""
from __future__ import annotations

import json
import os
import subprocess
import sys

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p10c2_a0 offline verify ===========")
    man = json.load(open(os.path.join(HERE, "p10c2_a0_archive_manifest.json"), encoding="utf-8"))
    close = json.load(open(os.path.join(HERE, "p10c2_a0_closeout.json"), encoding="utf-8"))
    rep = json.load(open(os.path.join(HERE, "p10c2_a0_report.json"), encoding="utf-8"))
    a = man["archive"]
    if a.get("status") != "ARCHIVED_VERIFIED":
        check("raw data missing/incomplete marked", a.get("status") in ("RAW_DATA_MISSING", "RAW_DATA_INCOMPLETE"))
        v = close["verdicts"]
        check("still NO-GO", v["Canonical_A0"] == "INCONCLUSIVE" and v["Training_Authorization"] is False)
    else:
        check("archive verified", a["status"] == "ARCHIVED_VERIFIED")
        check("76 files (75 npz + records)", a["file_count"] == 76 and a["npz_count"] == 75 and a["records_present"])
        check("no missing/mismatch/extra", not a["source_missing"] and not a["source_mismatch"] and not a["source_extra"])
        check("all per-file hashes match (source+backup)", a["all_hashes_match"] is True)
        check("manifest total matches report", a["archive_total_sha256"] == rep["raw_manifest_sha256"])
        off = close["offline"]
        check("offline checks all pass", all(off["checks"][k] for k in
              ("conditions_ok", "steps_ok", "shapes_ok", "meta_ok", "records_rates_match")))
        check("75 runs / 25 seeds", off["n_runs"] == 75 and off["n_seeds"] == 25)
        check("4533 real edges", off["checks"]["edges"] == 4533)
        p_exact = off["entrance"]["p_exact"]
        check("entrance p_exact unrounded = 2/2^25", abs(p_exact - 2 / (2 ** 25)) < 1e-18, repr(p_exact))
        check("entrance p not reported as 0", p_exact > 0 and off["entrance"]["p_display_round6"] == 0.0)
        check("KC p exact ~0.83416", abs(off["kc"]["p_exact"] - 0.8341633) < 1e-6, repr(off["kc"]["p_exact"]))
        check("recomputed matches report (display rounding noted)",
              all(v is True for k, v in off["matches_report"].items() if k != "display_rounding"))
        check("entrance PASS", close["verdicts"]["Entrance"] == "PASS")
        check("KC NOT_DETECTED", close["verdicts"]["KC_Main_Endpoint"] == "NOT_DETECTED")
        check("ceiling FAIL", close["verdicts"]["Ceiling"] == "FAIL")
        check("eligibility PENDING", close["verdicts"]["Eligibility_Meaningful_Threshold"] == "PENDING_APPROVAL")
        check("canonical INCONCLUSIVE", close["verdicts"]["Canonical_A0"] == "INCONCLUSIVE")
        prov = close["provenance"]
        check("script hash matches current", prov["script_hash_matches_current"] is True)
        check("run commit distinguished from result commit", prov["run_commit"].startswith("4835fd8")
              and prov["result_commit_added_script"] == "b11c74c")
        r = prov["reset_audit"]
        check("reset resets rng/v/fired/steps", r["source_has_rng_seed"] and r["source_has_v_zero"]
              and r["source_has_fired_empty"] and r["source_has_steps_zero"])
        check("reset per condition", r["reset_called_per_condition"] is True)
        e = prov["eligibility_audit"]
        check("eligibility strictly causal", e["strictly_causal"] is True)
        check("eligibility uses real edges", e["uses_real_KC_MBON_edges"] is True and e["n_edges_in_arrays"] == 4533)
        check("approval Mode A recorded", prov["approval"]["status"] == "APPROVED" and prov["approval"]["mode"] == "A")
    check("Level A Training NO-GO", close["verdicts"]["Level_A_Training"] == "NO-GO")
    check("Level B/C NO-GO", close["verdicts"]["Level_B"] == "NO-GO" and close["verdicts"]["Level_C"] == "NO-GO")
    check("Biological NO-GO", close["verdicts"]["Biological"] == "NO-GO")
    check("training not authorized", close["verdicts"]["Training_Authorization"] is False)
    # raw data not tracked in Git
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=os.path.dirname(HERE), capture_output=True, text=True, timeout=30).stdout
        check("raw A0 npz not tracked", "101_none.npz" not in tracked and "125_recover.npz" not in tracked)
    except Exception:                            # noqa: BLE001
        check("git ls-files available", False)
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
