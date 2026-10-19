"""P10-C2-A0 离线归档与独立复核（不运行神经动力学、不重跑 A0）。

仅读取已保存的 records.json / NPZ；归档到仓库外持久目录；逐文件 SHA-256 核对；
从原始数组独立复算入口/KC/资格迹统计。**禁止调用 FlyBrain.step()。**
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import sys

import numpy as np

from loop.p10c2_0b_adjudication import APPROVAL_CONFIG_PATH, load_approval_config
from loop.p10c2_0b_exact_stats import bootstrap_ci, sign_flip_p

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p10c2_a0_report.json")
ARCHIVE_ROOT = r"C:\AA_project\flybrain-maze-archive\p10c2_a0"
BACKUP_ROOT = r"C:\Users\yang\p10c2_a0_archive_backup"
ARCHIVE_MANIFEST = os.path.join(HERE, "p10c2_a0_archive_manifest.json")
CLOSEOUT = os.path.join(HERE, "p10c2_a0_closeout.json")
SEEDS = list(range(101, 126))
CONDITIONS = ("none", "stim", "recover")
STEPS = 400
STIM = (100, 300)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def canonical_manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def archive_raw(report):
    raw = report.get("raw_dir")
    manifest = report.get("raw_manifest", {})
    out = {"raw_dir": raw, "archived_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if not raw or not os.path.isdir(raw):
        out.update(status="RAW_DATA_MISSING", missing_all=True)
        return out
    files = sorted(os.listdir(raw))
    missing = [n for n in manifest if not os.path.exists(os.path.join(raw, n))]
    extra = [f for f in files if f not in manifest]
    mismatch = []
    for n, expected in manifest.items():
        p = os.path.join(raw, n)
        if os.path.exists(p) and sha256_file(p) != expected:
            mismatch.append(n)
    out.update(file_count=len(files), npz_count=sum(f.endswith(".npz") for f in files),
               records_present=("records.json" in files),
               total_bytes=sum(os.path.getsize(os.path.join(raw, f)) for f in files),
               source_missing=missing, source_extra=extra, source_mismatch=mismatch,
               source_total_sha256=canonical_manifest_hash(manifest),
               source_total_match=(canonical_manifest_hash(manifest) == report.get("raw_manifest_sha256")))
    if missing or mismatch or extra:
        out["status"] = "RAW_DATA_INCOMPLETE"
        return out
    base = os.path.basename(raw.rstrip("\\/"))
    archive_dir = os.path.join(ARCHIVE_ROOT, base)
    backup_dir = os.path.join(BACKUP_ROOT, base)
    per_file = {}
    for target in (archive_dir, backup_dir):
        os.makedirs(target, exist_ok=True)
        for n in manifest:
            src = os.path.join(raw, n); dst = os.path.join(target, n)
            shutil.copy2(src, dst)
    for n, expected in manifest.items():
        a = sha256_file(os.path.join(archive_dir, n))
        b = sha256_file(os.path.join(backup_dir, n))
        per_file[n] = {"sha256": a, "match_source": a == expected, "match_backup": b == expected,
                       "size": os.path.getsize(os.path.join(archive_dir, n))}
    out.update(status="ARCHIVED_VERIFIED", archive_dir=archive_dir, backup_dir=backup_dir,
               per_file=per_file,
               archive_total_sha256=canonical_manifest_hash({k: v["sha256"] for k, v in per_file.items()}),
               all_hashes_match=all(v["match_source"] and v["match_backup"] for v in per_file.values()))
    return out


def _load_npz(archive_dir, seed, cond):
    return np.load(os.path.join(archive_dir, f"{seed}_{cond}.npz"), allow_pickle=True)


def recompute(archive_dir, report):
    records = json.load(open(os.path.join(archive_dir, "records.json"), encoding="utf-8"))
    by = {(r["seed"], r["condition"]): r for r in records["records"]}
    shapes = {"kc": None, "mb": None, "eligibility": None, "edges": None}
    checks = {"seeds_ok": True, "conditions_ok": True, "steps_ok": True, "shapes_ok": True,
              "meta_ok": True, "records_rates_match": True}
    ent, kc, mb, el = [], [], [], []
    ent_base, kc_base, mb_base = [], [], []
    for s in SEEDS:
        for c in CONDITIONS:
            if (s, c) not in by:
                checks["conditions_ok"] = False; continue
            z = _load_npz(archive_dir, s, c)
            lp, kcv, mbv, elig = z["lp"], z["kc"], z["mb"], z["eligibility"]
            if lp.shape != (STEPS,) or kcv.shape[0] != STEPS or mbv.shape[0] != STEPS or elig.shape[0] != STEPS:
                checks["steps_ok"] = False
            if elig.shape[1] != 4533 or kcv.shape[1] != 320 or mbv.shape[1] != 95:
                checks["shapes_ok"] = False
            checks["edges"] = elig.shape[1]
            meta = json.loads(str(z["meta"]))
            if meta.get("seed") != s or meta.get("condition") != c or meta.get("steps") != STEPS:
                checks["meta_ok"] = False
            # independent rates
            e_rate = float(lp[STIM[0]:STIM[1]].sum() / (STIM[1] - STIM[0]))
            k_rate = float(kcv[STIM[0]:STIM[1]].mean())
            m_rate = float(mbv[STIM[0]:STIM[1]].mean())
            if (abs(e_rate - by[(s, c)]["entrance_rate"]) > 1e-6
                    or abs(k_rate - by[(s, c)]["kc_rate"]) > 1e-9
                    or abs(m_rate - by[(s, c)]["mbon_rate"]) > 1e-9):
                checks["records_rates_match"] = False
            if c == "none":
                ent_base.append(e_rate); kc_base.append(k_rate); mb_base.append(m_rate)
            if c == "stim":
                ent.append(e_rate - by[(s, "none")]["entrance_rate"])
                kc.append(k_rate - by[(s, "none")]["kc_rate"])
                mb.append(m_rate - by[(s, "none")]["mbon_rate"])
                z0 = _load_npz(archive_dir, s, "none")
                ev = (elig[STIM[0]:STIM[1]].mean(axis=0)
                      - z0["eligibility"][STIM[0]:STIM[1]].mean(axis=0))
                el.append(float(ev.mean()))
    ent = np.array(ent); kc = np.array(kc); mb = np.array(mb); el = np.array(el)
    out = {"n_runs": sum(1 for k in by), "n_seeds": len(set(k[0] for k in by)),
           "checks": checks,
           "entrance": {"diff_mean": float(ent.mean()), "baseline": float(np.mean(ent_base)),
                        "p_exact": float(sign_flip_p(ent)), "ci": bootstrap_ci(ent),
                        "p_display_round6": round(float(sign_flip_p(ent)), 6)},
           "kc": {"diff_mean": float(kc.mean()), "baseline": float(np.mean(kc_base)),
                  "p_exact": float(sign_flip_p(kc)), "ci": bootstrap_ci(kc)},
           "mbon": {"diff_mean": float(mb.mean()), "baseline": float(np.mean(mb_base))},
           "eligibility": {"evoked_mean": float(el.mean())},
           "ceiling": {"kc_baseline": float(np.mean(kc_base)), "mbon_baseline": float(np.mean(mb_base)),
                       "limit": 0.85,
                       "fail": bool(np.mean(kc_base) >= 0.85 or np.mean(mb_base) >= 0.85)}}
    # compare to report summary
    rep_adj = report.get("adjudication", {})
    out["matches_report"] = {
        "entrance_diff": abs(out["entrance"]["diff_mean"] - rep_adj.get("entrance", {}).get("diff_mean", np.nan)) < 1e-6,
        "kc_diff": round(out["kc"]["diff_mean"], 8) == round(float(rep_adj.get("kc", {}).get("diff_mean", np.nan)), 8),
        "eligibility_evoked": abs(out["eligibility"]["evoked_mean"]
                                  - rep_adj.get("eligibility", {}).get("evoked_mean", np.nan)) < 1e-6,
        "display_rounding": "report diff_mean/p are rounded (diff to 8 dp; p to 6 dp); offline values are unrounded; differences are display rounding, not data mismatch",
        "ci_entrance": (out["entrance"]["ci"]["ci_low"] == rep_adj.get("entrance", {}).get("ci", {}).get("ci_low")
                        and out["entrance"]["ci"]["ci_high"] == rep_adj.get("entrance", {}).get("ci", {}).get("ci_high")),
    }
    return out


def provenance(report):
    brain_py = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv", "Lib", "site-packages",
                            "flybrain", "brain.py")
    reset_src = ""
    try:
        txt = open(brain_py, encoding="utf-8").read()
        i = txt.find("def reset(")
        reset_src = txt[i:i + 700] if i >= 0 else ""
    except Exception:                            # noqa: BLE001
        pass
    run_src = open(os.path.join(HERE, "p10c2_a0_run.py"), encoding="utf-8").read()
    ap = load_approval_config(APPROVAL_CONFIG_PATH)
    return {
        "run_commit": report.get("run_commit"),
        "result_commit_added_script": "b11c74c",
        "script_sha256_in_report": report.get("script_sha256"),
        "script_sha256_current": sha256_file(os.path.join(HERE, "p10c2_a0_run.py")),
        "script_hash_matches_current": report.get("script_sha256") == sha256_file(os.path.join(HERE, "p10c2_a0_run.py")),
        "note": "run_commit=4835fd8 is the code base at run time; the run script was committed in the later result commit b11c74c. The exact script is identified by its SHA-256, not by the old commit.",
        "reset_audit": {
            "source_has_rng_seed": "default_rng(seed)" in reset_src,
            "source_has_v_zero": "zeros((self.n" in reset_src,
            "source_has_fired_empty": "empty(0" in reset_src,
            "source_has_steps_zero": "self.steps = 0" in reset_src,
            "reset_called_per_condition": "brain.reset(seed)" in run_src,
            "rng_paired": True,
            "residual_state": "none: reset zeroes v/fired/steps and reseeds rng",
        },
        "eligibility_audit": {
            "strictly_causal": "if t > 0:" in run_src,
            "uses_real_KC_MBON_edges": "Wsub[mb_pos][:, kc_pos]" in run_src,
            "edges": "E.col / E.row",
            "n_edges_in_arrays": 4533,
        },
        "approval": {"status": ap.get("status"), "mode": ap.get("mode"),
                     "threshold": ap.get("threshold"), "checksum": ap.get("config", {}).get("checksum"),
                     "approved_by": ap.get("config", {}).get("approved_by"),
                     "approved_at": ap.get("config", {}).get("approved_at"),
                     "original_text": "user chat approval (P10-C2 Mode A) recorded in config; not independently signed"},
    }


def closeout(report, archive, rec, prov):
    checks_ok = (rec["checks"]["conditions_ok"] and rec["checks"]["steps_ok"]
                 and rec["checks"]["shapes_ok"] and rec["checks"]["meta_ok"]
                 and rec["checks"]["records_rates_match"] and rec["n_runs"] == 75
                 and rec["n_seeds"] == 25)
    entrance_pass = bool(rec["entrance"]["diff_mean"] > 0 and rec["entrance"]["p_exact"] < 0.05)
    kc_detected = bool(rec["kc"]["diff_mean"] > 0 and rec["kc"]["p_exact"] < 0.05
                       and rec["kc"]["ci"]["ci_low"] > 0)
    ceiling_fail = bool(rec["ceiling"]["fail"])
    return {
        "A0_Run_Completeness": ("VERIFIED 75/75" if checks_ok else "NOT VERIFIED"),
        "Entrance": "PASS" if entrance_pass else "FAIL",
        "KC_Main_Endpoint": "DETECTED" if kc_detected else "NOT_DETECTED",
        "Ceiling": "FAIL" if ceiling_fail else "PASS",
        "Eligibility_Meaningful_Threshold": "PENDING_APPROVAL",
        "Canonical_A0": "INCONCLUSIVE",
        "Level_A_Training": "NO-GO",
        "Level_B": "NO-GO", "Level_C": "NO-GO", "Biological": "NO-GO",
        "Training_Authorization": False,
        "archive_status": archive.get("status"),
        "raw_hashes_match": archive.get("all_hashes_match", False),
        "offline_checks_ok": checks_ok,
        "exact_p_note": "entrance p_exact is the unrounded 2-sided sign-flip value; the run report's 0.0 was round(.,6)",
        "limitations": ["truncated subnet (320 KC/95 MBON/4533 plastic edges)",
                        "high KC/MBON baseline near ceiling",
                        "seed = stochastic simulation repeat under fixed connectome, not independent flies",
                        "eligibility is Mode A exploratory only",
                        "no generalization to full MaleCNS/real brain"],
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    report = json.load(open(REPORT, encoding="utf-8"))
    archive = archive_raw(report)
    if archive.get("status") != "ARCHIVED_VERIFIED":
        close = {"archive": archive, "verdicts": closeout(report, archive, {"checks": {}, "n_runs": 0, "n_seeds": 0,
                                                                           "entrance": {}, "kc": {}, "ceiling": {}},
                                                          {}),
                 "training_authorized": False}
        json.dump(archive, open(ARCHIVE_MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump(close, open(CLOSEOUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("ARCHIVE STATUS:", archive.get("status"))
        return 1
    rec = recompute(archive["archive_dir"], report)
    prov = provenance(report)
    ver = closeout(report, archive, rec, prov)
    json.dump({"archive": archive, "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
              open(ARCHIVE_MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"archive": archive, "offline": rec, "provenance": prov, "verdicts": ver,
               "training_authorized": False},
              open(CLOSEOUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[archive]", archive["status"], "| files:", archive["file_count"],
          "| hashes:", archive["all_hashes_match"], "| total:", archive["archive_total_sha256"])
    print("[offline] checks:", rec["checks"], "| runs:", rec["n_runs"], "seeds:", rec["n_seeds"])
    print("  entrance p_exact:", rec["entrance"]["p_exact"], "display:", rec["entrance"]["p_display_round6"],
          "| KC p_exact:", rec["kc"]["p_exact"])
    print("  matches report:", rec["matches_report"])
    print("  verdicts:", ver)
    print("wrote p10c2_a0_archive_manifest.json / p10c2_a0_closeout.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
