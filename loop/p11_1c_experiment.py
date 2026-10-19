"""P11-1c：完整 MaleCNS 25-seed none-only 模型有效性实验（**已获用户书面授权，一次**）。

子进程：完整脑 + 25 seeds × 400 步无刺激；逐 seed 写仓库外 NPZ。父进程：校验授权/哈希/安全自检 ->
Job(2 GiB, kill-on-close, 暂停启动) -> 监控 15 min -> 与归档截断子网比较。不训练、ΔW=0。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
AUTH_PATH = os.path.join(HERE, "p11_1c_authorization.json")
REPORT = os.path.join(HERE, "p11_1c_experiment_report.json")
MANIFEST = os.path.join(HERE, "p11_1c_full_manifest.json")
ARCHIVE_MANIFEST = os.path.join(HERE, "p10c2_a0_archive_manifest.json")
SEEDS = list(range(101, 126))
STEPS = 400
STIM = (100, 300)
KNOWN = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}
MEM_CAP_GIB = 2.0
MEM_CAP_BYTES = int(MEM_CAP_GIB * 1024 ** 3)
WALL_CAP_S = 900.0

from loop.p11_1c_full_brain_bench import (KNOWN as KNOWN_SAFETY, MEM_CAP_BYTES as SAFETY_BYTES,
                                          _mib, _sha256_file, run_under_job, safety_selftest,
                                          official_hashes)


def canonical_checksum(cfg):
    body = {k: v for k, v in cfg.items() if k != "checksum"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_auth():
    if not os.path.exists(AUTH_PATH):
        return {"granted": False, "reason": "authorization file missing"}
    cfg = json.load(open(AUTH_PATH, encoding="utf-8"))
    if cfg.get("checksum") != canonical_checksum(cfg):
        return {"granted": False, "reason": "authorization checksum mismatch", "config": cfg}
    sc = cfg.get("scope", {})
    exp_params = {"dt": 0.02, "tau": 0.1, "gain": 3.0, "tonic": 0.14, "noise_hz": 1.2, "noise_amp": 0.22}
    ok = (cfg.get("protocol") == "P11-1c/v1" and cfg.get("status") == "APPROVED"
          and sc.get("n_neurons") == 166700 and sc.get("nnz") == 25582938
          and sc.get("seeds") == SEEDS and sc.get("conditions") == ["none"]
          and sc.get("steps") == STEPS and sc.get("total_steps") == 10000
          and sc.get("params") == exp_params and sc.get("W_readonly") is True
          and sc.get("dW") == 0 and sc.get("learning") is False
          and sc.get("mem_cap_GiB") == 2.0 and sc.get("wall_cap_min") == 15
          and sc.get("one_run") is True and sc.get("no_training") is True
          and sc.get("no_dijkstra") is True and sc.get("no_B_C") is True and sc.get("no_robot") is True)
    return {"granted": bool(ok), "reason": None if ok else "authorization scope mismatch", "config": cfg}


def precompute_matched_ids(out_dir):
    from loop.p10c1b_lif_benchmark import build_subnet
    sub = build_subnet()
    D, L = sub["D"], sub["layers"]
    kc_ids = [int(x) for x in D["ids"][L[2]]]
    mb_ids = [int(x) for x in D["ids"][L[3]]]
    if len(kc_ids) != 320 or len(mb_ids) != 95:
        raise RuntimeError(f"matched id count mismatch: {len(kc_ids)}/{len(mb_ids)}")
    with open(os.path.join(out_dir, "matched_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"kc": kc_ids, "mbon": mb_ids}, f)
    del sub
    return kc_ids, mb_ids


def child_experiment(out_dir, matched_ids_path):
    import numpy as np
    from flybrain import FlyBrain
    try:
        meta_brain = np.load(os.path.join(DATA_DIR, "brain.npz"), allow_pickle=True)
        ids = meta_brain["ids"]
        matched = json.load(open(matched_ids_path, encoding="utf-8"))
        t0 = time.perf_counter()
        brain = FlyBrain(data=DATA_DIR, device="cpu", batch=1, seed=101, dt=0.020)
        init_s = time.perf_counter() - t0
        kc_idx = np.flatnonzero(np.char.startswith(brain.cell_type.astype(str), "KC"))
        mb_idx = np.flatnonzero(np.char.startswith(brain.cell_type.astype(str), "MBON"))
        id_to_pos = {int(b): i for i, b in enumerate(ids)}
        kc_matched = np.array([id_to_pos[int(b)] for b in matched["kc"]], dtype=np.int64)
        mb_matched = np.array([id_to_pos[int(b)] for b in matched["mbon"]], dtype=np.int64)
        np.savez(os.path.join(out_dir, "ids.npz"), ids=ids, kc_idx=kc_idx, mb_idx=mb_idx,
                 kc_matched=kc_matched, mb_matched=mb_matched)
        kc_map = np.full(len(ids), -1, np.int64); kc_map[kc_idx] = np.arange(len(kc_idx))
        mb_map = np.full(len(ids), -1, np.int64); mb_map[mb_idx] = np.arange(len(mb_idx))
        results = []
        for seed in SEEDS:
            t_seed = time.perf_counter()
            brain.reset(seed)
            kc = np.zeros((STEPS, len(kc_idx)), np.uint8)
            mb = np.zeros((STEPS, len(mb_idx)), np.uint8)
            finite = True
            for t in range(STEPS):
                fired = np.asarray(brain.step())
                if not np.isfinite(brain.v).all():
                    finite = False
                fk = fired[kc_map[fired] >= 0]
                fm = fired[mb_map[fired] >= 0]
                kc[t, kc_map[fk]] = 1
                mb[t, mb_map[fm]] = 1
            np.savez_compressed(os.path.join(out_dir, f"{seed}.npz"), kc=kc, mb=mb,
                                meta=json.dumps({"seed": seed, "condition": "none", "steps": STEPS,
                                                 "params": {"dt": brain.dt, "tau": brain.tau, "gain": brain.gain,
                                                            "tonic": brain.tonic, "noise_hz": brain.noise_hz,
                                                            "noise_amp": brain.noise_amp}}))
            results.append({"seed": seed, "wall_s": round(time.perf_counter() - t_seed, 3),
                            "finite": bool(finite),
                            "kc_fired_mean": round(float(kc.sum(axis=1).mean()), 2),
                            "mb_fired_mean": round(float(mb.sum(axis=1).mean()), 2)})
        res = {"ok": True, "init_s": round(init_s, 4), "n_neurons": int(brain.n),
               "nnz": int(len(brain.indices)), "n_KC": int(len(kc_idx)), "n_MBON": int(len(mb_idx)),
               "n_matched_KC": int(len(kc_matched)), "n_matched_MBON": int(len(mb_matched)),
               "seeds": results, "device": "cpu",
               "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "scipy": __import__("scipy").__version__,
                            "flybrain": getattr(__import__("flybrain"), "__version__", "0.1.0")}}
        with open(os.path.join(out_dir, "child_result.json"), "w", encoding="utf-8") as f:
            json.dump(res, f)
        return 0
    except Exception as e:                       # noqa: BLE001
        with open(os.path.join(out_dir, "child_result.json"), "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": repr(e)[:400]}, f)
        return 5


def _fraction(arr):
    return arr[STIM[0]:STIM[1]].mean(axis=0)


def _stats(arr):
    f = _fraction(arr)
    return {"mean": round(float(f.mean()), 6), "median": round(float(np.median(f)), 6),
            "ge_0.85": round(float((f >= 0.85).mean()), 6),
            "ge_0.95": round(float((f >= 0.95).mean()), 6)}


def _bins(arr, nb=20):
    return [round(float(arr[b].mean()), 6) for b in np.array_split(np.arange(STEPS), nb)]


def _agg(stats_list):
    import numpy as np
    means = [x["mean"] for x in stats_list]
    return {"per_seed_mean": [round(float(x), 6) for x in means],
            "mean": round(float(np.mean(means)), 6),
            "sd": round(float(np.std(means, ddof=1)), 6),
            "ge_0.85": round(float(np.mean([x["ge_0.85"] for x in stats_list])), 6),
            "ge_0.95": round(float(np.mean([x["ge_0.95"] for x in stats_list])), 6),
            "median": round(float(np.median(means)), 6)}


def compare(out_dir, matched, archive_dir):
    import numpy as np
    from loop.p10c2_0b_exact_stats import bootstrap_ci
    ids = np.load(os.path.join(out_dir, "ids.npz"), allow_pickle=True)
    kc_idx = ids["kc_idx"]; mb_idx = ids["mb_idx"]
    kc_matched = np.searchsorted(kc_idx, ids["kc_matched"])
    mb_matched = np.searchsorted(mb_idx, ids["mb_matched"])
    f_full_kc, f_full_mb, f_m_kc, f_m_mb = [], [], [], []
    f_sub_kc, f_sub_mb = [], []
    kc_bins_full, kc_bins_sub, mb_bins_full, mb_bins_sub = [], [], [], []
    for s in SEEDS:
        z = np.load(os.path.join(out_dir, f"{s}.npz"), allow_pickle=True)
        kc = z["kc"]; mb = z["mb"]
        f_full_kc.append(_stats(kc)); f_full_mb.append(_stats(mb))
        f_m_kc.append(_stats(kc[:, kc_matched])); f_m_mb.append(_stats(mb[:, mb_matched]))
        kc_bins_full.append(_bins(kc)); mb_bins_full.append(_bins(mb))
        za = np.load(os.path.join(archive_dir, f"{s}_none.npz"), allow_pickle=True)
        f_sub_kc.append(_stats(za["kc"])); f_sub_mb.append(_stats(za["mb"]))
        kc_bins_sub.append(_bins(za["kc"])); mb_bins_sub.append(_bins(za["mb"]))
    full_kc = _agg(f_full_kc); full_mb = _agg(f_full_mb)
    m_kc = _agg(f_m_kc); m_mb = _agg(f_m_mb)
    sub_kc = _agg(f_sub_kc); sub_mb = _agg(f_sub_mb)
    d = np.array(m_kc["per_seed_mean"]) - np.array(sub_kc["per_seed_mean"])
    ci = bootstrap_ci(d)
    return {
        "full_4064_KC": full_kc, "full_matched_320_KC": m_kc, "subnet_320_KC": sub_kc,
        "full_97_MBON": full_mb, "full_matched_95_MBON": m_mb, "subnet_95_MBON": sub_mb,
        "difference_matched_320_KC": {"per_seed": [round(float(x), 6) for x in d],
                                      "mean": round(float(d.mean()), 6),
                                      "sd": round(float(d.std(ddof=1)), 6), "ci": ci},
        "startup_bins_KC": {"full_mean": np.mean(kc_bins_full, axis=0).round(6).tolist(),
                            "subnet_mean": np.mean(kc_bins_sub, axis=0).round(6).tolist()},
        "startup_bins_MBON": {"full_mean": np.mean(mb_bins_full, axis=0).round(6).tolist(),
                              "subnet_mean": np.mean(mb_bins_sub, axis=0).round(6).tolist()},
        "noise_pairing": "same seed label only; full vs subnet N differ -> per-neuron noise NOT paired",
    }


def build_manifest(out_dir):
    import hashlib
    man = {}
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            man[name] = {"sha256": _sha256_file(p), "bytes": os.path.getsize(p)}
    return man


def compare_only(out_dir):
    """Offline post-processing of a completed child run (no new dynamics)."""
    child = json.load(open(os.path.join(out_dir, "child_result.json"), encoding="utf-8"))
    n = len(child.get("seeds", [])) if child.get("ok") else 0
    arch = json.load(open(ARCHIVE_MANIFEST, encoding="utf-8"))
    archive_dir = arch["archive"]["archive_dir"]
    cmp = compare(out_dir, None, archive_dir)
    reconstructed_wall = round(float(child.get("init_s", 0))
                               + sum(float(x.get("wall_s", 0)) for x in child.get("seeds", [])), 2)
    full_mean = cmp["full_matched_320_KC"]["mean"]
    baseline = "HIGH" if full_mean >= 0.85 else ("NOT_HIGH" if full_mean < 0.85 else "INCONCLUSIVE")
    d = cmp["difference_matched_320_KC"]; ci = d["ci"]
    observed = (ci["ci_low"] > 0 or ci["ci_high"] < 0) and abs(d["mean"]) >= 0.05
    man = build_manifest(out_dir)
    auth = load_auth()
    report = {
        "protocol": "P11-1c/v1", "status": "INCONCLUSIVE",
        "authorization": auth,
        "authorization_text": (auth.get("config") or {}).get("authorization_text"),
        "status_note": ("child completed 25/25; parent post-processing crashed before writing the report, "
                        "so the Job peak-memory metric was lost. Per the one-run rule the child was NOT rerun. "
                        "Science comparison below is from the saved per-seed arrays."),
        "run_commit": os.popen("git rev-parse HEAD").read().strip(),
        "script_sha256": _sha256_file(os.path.abspath(__file__)),
        "hashes": official_hashes(), "hashes_ok": official_hashes()["ok"],
        "SAFETY_PRECHECK": "PASS",
        "safety_note": "the child was launched only after the synthetic safety selftest passed; the 2 GiB Job limit was assigned before resume",
        "RESOURCE": "INCONCLUSIVE",
        "resource_note": "child completed; Job peak committed memory was not captured before the parent crash; NOT re-run",
        "COMPLETENESS": f"{n}/25",
        "reconstructed_child_wall_s": reconstructed_wall,
        "wall_cap_s": WALL_CAP_S, "mem_cap_GiB": MEM_CAP_GIB,
        "child": child, "comparison": cmp, "raw_dir": out_dir,
        "manifest": man, "manifest_sha256": hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest(),
        "FULL_BRAIN_BASELINE": baseline,
        "FULL_VS_SUBNET_DIFFERENCE": ("OBSERVED" if observed else "NOT_DETECTED"),
        "MODEL_ROOT_CAUSE": "NOT_ESTABLISHED",
        "verdicts": {"SAFETY_PRECHECK": "PASS", "RESOURCE": "INCONCLUSIVE",
                     "COMPLETENESS": f"{n}/25", "FULL_BRAIN_BASELINE": baseline,
                     "FULL_VS_SUBNET_DIFFERENCE": ("OBSERVED" if observed else "NOT_DETECTED"),
                     "MODEL_ROOT_CAUSE": "NOT_ESTABLISHED",
                     "Training_Authorization": False, "Level_A_Training": "NO-GO",
                     "Level_B": "NO-GO", "Level_C": "NO-GO", "Biological": "NO-GO"},
        "training_authorized": False,
    }
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(man, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[P11-1c compare-only] completeness:", report["COMPLETENESS"],
          "| reconstructed wall s:", reconstructed_wall)
    print("  full matched 320 KC mean:", cmp["full_matched_320_KC"]["mean"],
          "| subnet 320 KC mean:", cmp["subnet_320_KC"]["mean"])
    print("  diff:", d["mean"], "CI:", ci, "| baseline:", baseline,
          "| difference:", report["FULL_VS_SUBNET_DIFFERENCE"])
    print("wrote", REPORT)
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    if "--experiment" in sys.argv:
        i = sys.argv.index("--experiment")
        return child_experiment(sys.argv[i + 1], sys.argv[i + 2])
    if "--compare-only" in sys.argv:
        return compare_only(sys.argv[sys.argv.index("--compare-only") + 1])
    if "--self-test-safety" in sys.argv:
        s = safety_selftest(); print(json.dumps(s, ensure_ascii=False, indent=1))
        return 0 if s["all_ok"] else 1
    # production
    if os.path.exists(REPORT):
        print("REFUSING: experiment report already exists; one run only.")
        return 2
    auth = load_auth()
    hashes = official_hashes()
    report = {"protocol": "P11-1c/v1", "authorization": auth,
              "authorization_text": (auth.get("config") or {}).get("authorization_text"),
              "run_commit": os.popen("git rev-parse HEAD").read().strip(),
              "script_sha256": _sha256_file(os.path.abspath(__file__)),
              "hashes": hashes, "hashes_ok": hashes["ok"],
              "seeds": SEEDS, "steps": STEPS, "conditions": ["none"],
              "training_authorized": False}
    if not auth["granted"]:
        report["status"] = "NOT_AUTHORIZED"
        json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("NOT_AUTHORIZED:", auth.get("reason")); return 1
    if not hashes["ok"]:
        report["status"] = "SAFETY_NO_GO"; report["reason"] = "official full SHA-256 mismatch"
        json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("SAFETY_NO_GO: hash mismatch"); return 1
    selftest = safety_selftest()
    report["SAFETY_PRECHECK"] = "PASS" if selftest.get("all_ok") else "FAIL"
    report["safety_selftest"] = selftest
    if not selftest.get("all_ok"):
        report["status"] = "SAFETY_NO_GO"
        report["RESOURCE"] = "NO_GO"
        json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("SAFETY_NO_GO: synthetic resource isolation tests failed; full brain NOT started")
        return 1
    out_dir = os.path.join(tempfile.gettempdir(), "opencode", "p11_1c_full",
                           time.strftime("%Y%m%dT%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    matched_path = os.path.join(out_dir, "matched_ids.json")
    kc_ids, mb_ids = precompute_matched_ids(out_dir)
    report["matched_ids"] = {"n_KC": len(kc_ids), "n_MBON": len(mb_ids)}
    log_path = os.path.join(out_dir, "experiment.log")
    t0 = time.perf_counter()
    r = run_under_job(MEM_CAP_BYTES, ["--experiment", out_dir, matched_path], WALL_CAP_S, 2048, log_path, script_path=os.path.abspath(__file__))
    wall = round(time.perf_counter() - t0, 3)
    report["resource_run"] = r
    report["wall_s"] = wall
    child_path = os.path.join(out_dir, "child_result.json")
    child = json.load(open(child_path, encoding="utf-8")) if os.path.exists(child_path) else None
    report["child"] = child
    n_done = len(child.get("seeds", [])) if child and child.get("ok") else 0
    report["COMPLETENESS"] = f"{n_done}/25"
    ok_run = (r.get("ok") and not r.get("timeout") and not r.get("mem_exceeded")
              and r.get("exit_code") == 0 and child and child.get("ok") and n_done == 25)
    report["RESOURCE"] = "OK" if ok_run else "NO_GO"
    if not ok_run:
        report["status"] = "INCONCLUSIVE"
        report["reason"] = ("timeout" if r.get("timeout") else "memory cap" if r.get("mem_exceeded")
                            else "child failed/partial")
        json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("INCONCLUSIVE:", report["reason"], "| completeness:", report["COMPLETENESS"]); return 1
    arch = json.load(open(ARCHIVE_MANIFEST, encoding="utf-8"))
    archive_dir = arch["archive"]["archive_dir"]
    cmp = compare(out_dir, (kc_ids, mb_ids), archive_dir)
    report["comparison"] = cmp
    full_mean = cmp["full_matched_320_KC"]["mean"]
    report["FULL_BRAIN_BASELINE"] = ("HIGH" if full_mean >= 0.85
                                     else ("NOT_HIGH" if full_mean < 0.85 else "INCONCLUSIVE"))
    d = cmp["difference_matched_320_KC"]
    ci = d["ci"]
    observed = (ci["ci_low"] > 0 or ci["ci_high"] < 0) and abs(d["mean"]) >= 0.05
    report["FULL_VS_SUBNET_DIFFERENCE"] = ("OBSERVED" if observed else "NOT_DETECTED")
    report["MODEL_ROOT_CAUSE"] = "NOT_ESTABLISHED"
    report["raw_dir"] = out_dir
    report["manifest"] = build_manifest(out_dir)
    report["manifest_sha256"] = hashlib.sha256(json.dumps(report["manifest"], sort_keys=True).encode()).hexdigest()
    report["status"] = "COMPLETED"
    report["verdicts"] = {"SAFETY_PRECHECK": report["SAFETY_PRECHECK"],
                          "RESOURCE": report["RESOURCE"], "COMPLETENESS": report["COMPLETENESS"],
                          "FULL_BRAIN_BASELINE": report["FULL_BRAIN_BASELINE"],
                          "FULL_VS_SUBNET_DIFFERENCE": report["FULL_VS_SUBNET_DIFFERENCE"],
                          "MODEL_ROOT_CAUSE": "NOT_ESTABLISHED",
                          "Training_Authorization": False, "Level_A_Training": "NO-GO",
                          "Level_B": "NO-GO", "Level_C": "NO-GO", "Biological": "NO-GO"}
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(report["manifest"], open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[P11-1c] status:", report["status"], "| completeness:", report["COMPLETENESS"])
    print("  SAFETY:", report["SAFETY_PRECHECK"], "| RESOURCE:", report["RESOURCE"],
          "| wall_s:", wall, "| job_peak:", r.get("job_peaks"))
    print("  full matched 320 KC mean:", cmp["full_matched_320_KC"]["mean"],
          "| subnet 320 KC mean:", cmp["subnet_320_KC"]["mean"],
          "| diff:", d["mean"], "CI:", ci)
    print("  FULL_BRAIN_BASELINE:", report["FULL_BRAIN_BASELINE"],
          "| FULL_VS_SUBNET_DIFFERENCE:", report["FULL_VS_SUBNET_DIFFERENCE"])
    print("  raw_dir:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
