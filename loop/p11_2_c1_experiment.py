"""P11-2-C1：完整 MaleCNS 的 KC→KC 反馈删除因果实验（A intact / B zero）。

一次性 A/B；25 seeds × 400 步；无刺激；20,000 步；15 min / 2 GiB；CPU。
父进程负责 Job 安全、run_meta 持久化和统计；子进程负责完整脑、W_B 和 A/B 运行。

本脚本只在获得一次明确授权后执行。`--synthetic` 仅运行小型合成矩阵/小型 FlyBrain，
不加载完整脑。所有完整脑权重派生在子进程 Job 内完成。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
EXP_REPORT = os.path.join(HERE, "p11_1c_experiment_report.json")
FULL_MANIFEST = os.path.join(HERE, "p11_1c_full_manifest.json")
AUTH_PATH = os.path.join(HERE, "p11_2_c1_authorization.json")
FREEZE_PATH = os.path.join(HERE, "p11_2_c1_script_freeze.json")
REPORT = os.path.join(HERE, "p11_2_c1_report.json")
RUN_META = os.path.join(HERE, "p11_2_c1_run_meta.json")
RAW_MANIFEST = os.path.join(HERE, "p11_2_c1_raw_manifest.json")

SEEDS = list(range(101, 126))
STEPS = 400
STIM = (100, 300)
CONDITIONS = ["A", "B"]
TOTAL_STEPS = 2 * len(SEEDS) * STEPS
WALL_CAP_S = 900.0
MEM_CAP_MIB = 2048
MEM_CAP_BYTES = MEM_CAP_MIB * 1024 ** 2
MIN_EFFECT = 0.10
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20261019
LOW_FEEDBACK = [2460, 2650, 3971]
OFFICIAL = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}
W_NNZ = 25582938
W_B_NNZ = 24940005
KC_KC_NNZ = 642933
REMOVED_SUM = 2275.627441
P11_1B_PEAK_MB = 1497.87
P11_1B_PEAK_MIB = 1428.48
W_B_MIB = 190.91


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _scope():
    return {
        "n_neurons": 166700, "conditions": ["intact", "kc_kc_zero"],
        "seeds": SEEDS, "steps": STEPS, "total_steps": TOTAL_STEPS,
        "window": list(STIM), "stim": "none", "device": "cpu",
        "wall_cap_s": WALL_CAP_S, "mem_cap_MiB": MEM_CAP_MIB,
        "one_run": True, "auto_retry": False, "training": False, "dW": 0,
    }


def auth_payload(approval_text, scope=None):
    return {"protocol": "P11-2-C1/v1", "status": "APPROVED", "approved_by": "user",
            "approval_text": approval_text, "scope": scope or _scope()}


def auth_file(approval_text, scope=None):
    p = auth_payload(approval_text, scope)
    p["checksum"] = _sha_bytes(_canon(p))
    return p


def load_auth():
    a = json.load(open(AUTH_PATH, encoding="utf-8"))
    checksum = a.pop("checksum", None)
    if checksum != _sha_bytes(_canon(a)):
        raise RuntimeError("AUTHORIZATION_CHECKSUM_MISMATCH")
    a["checksum"] = checksum
    if a.get("protocol") != "P11-2-C1/v1" or a.get("status") != "APPROVED" or a.get("approved_by") != "user":
        raise RuntimeError("AUTHORIZATION_NOT_APPROVED")
    if a.get("scope") != _scope():
        raise RuntimeError("AUTHORIZATION_SCOPE_MISMATCH")
    return a


def load_freeze():
    f = json.load(open(FREEZE_PATH, encoding="utf-8"))
    if os.path.abspath(f.get("script", "")) != os.path.abspath(__file__):
        raise RuntimeError("FREEZE_SCRIPT_PATH_MISMATCH")
    if f.get("sha256") != _sha256(__file__):
        raise RuntimeError("SCRIPT_FREEZE_MISMATCH")
    return f


def load_archive():
    rep = json.load(open(EXP_REPORT, encoding="utf-8"))
    raw = rep["raw_dir"]
    man = json.load(open(FULL_MANIFEST, encoding="utf-8"))
    if not raw or not os.path.isdir(raw):
        return {"ok": False, "status": "OFFLINE_RECONSTRUCTION_UNVERIFIED", "reason": "raw dir missing"}
    names = sorted(os.listdir(raw))
    missing = [n for n in man if not os.path.exists(os.path.join(raw, n))]
    mismatch = [n for n in man if os.path.exists(os.path.join(raw, n)) and _sha256(os.path.join(raw, n)) != man[n]["sha256"]]
    extra = [n for n in names if n not in man]
    return {"ok": not (missing or mismatch or extra), "status": "VERIFIED" if not (missing or mismatch or extra) else "OFFLINE_RECONSTRUCTION_UNVERIFIED",
            "raw_dir": raw, "file_count": len(names), "missing": missing, "mismatch": mismatch, "extra": extra}


def official_hashes():
    files = {k: _sha256(os.path.join(DATA_DIR, k)) for k in OFFICIAL}
    return {"ok": all(files[k] == v for k, v in OFFICIAL.items()), "files": files}


# ---------------------------------------------------------------- connection fields
def build_W_B_csc(indptr, indices, weights, is_kc):
    """CSC column filter: remove only edges with pre in KC and post in KC.

    This avoids a full COO duplicate; only one new CSC ternary is allocated.
    """
    n = len(indptr) - 1
    indptr = np.asarray(indptr); indices = np.asarray(indices); weights = np.asarray(weights)
    keep_counts = np.empty(n, np.int64)
    removed_edges = 0; removed_sum = 0.0
    for j in range(n):
        s, e = int(indptr[j]), int(indptr[j + 1])
        if is_kc[j]:
            post = indices[s:e]
            keep = ~is_kc[post]
            c = int(keep.sum())
            keep_counts[j] = c
            removed_edges += (e - s) - c
            if c != (e - s):
                removed_sum += float(weights[s:e][~keep].sum(dtype=np.float64))
        else:
            keep_counts[j] = e - s
    nnz = int(keep_counts.sum())
    new_indptr = np.empty(n + 1, dtype=indptr.dtype); new_indptr[0] = 0
    new_indices = np.empty(nnz, dtype=indices.dtype)
    new_weights = np.empty(nnz, dtype=weights.dtype)
    w = 0
    for j in range(n):
        s, e = int(indptr[j]), int(indptr[j + 1])
        post = indices[s:e]; dat = weights[s:e]
        if is_kc[j]:
            keep = ~is_kc[post]
            c = int(keep.sum())
            new_indices[w:w + c] = post[keep]
            new_weights[w:w + c] = dat[keep]
        else:
            c = e - s
            new_indices[w:w + c] = post
            new_weights[w:w + c] = dat
        w += c
        new_indptr[j + 1] = w
    if w != nnz:
        raise RuntimeError("W_B length mismatch")
    return (new_indptr, new_indices, new_weights), removed_edges, removed_sum


def verify_W_B_fields(A, B, is_kc, kc_idx):
    ai, ax, aw = A; bi, bx, bw = B
    n = len(ai) - 1
    if len(bi) != n + 1 or bi[-1] != len(bx) or len(bx) != len(bw):
        return {"ok": False, "reason": "shape"}
    removed = int(ai[-1] - bi[-1])
    expected_removed = 0
    for j in np.asarray(kc_idx):
        s, e = int(ai[j]), int(ai[j + 1])
        expected_removed += int(is_kc[ax[s:e]].sum())
    if removed != expected_removed:
        return {"ok": False, "reason": "removed_count", "removed": removed,
                "expected_removed": expected_removed}
    # only KC posts can lose incoming edges; verify no KC->KC in B and other columns exact
    bad_kc = 0; bad_other = 0
    for j in range(n):
        s, e = int(ai[j]), int(ai[j + 1]); bs, be = int(bi[j]), int(bi[j + 1])
        apost = ax[s:e]; adat = aw[s:e]
        bpost = bx[bs:be]; bdat = bw[bs:be]
        if is_kc[j]:
            if np.any(is_kc[bpost]):
                bad_kc += 1
            keep = ~is_kc[apost]
            if not (np.array_equal(bpost, apost[keep]) and np.array_equal(bdat, adat[keep])):
                bad_other += 1
        else:
            if not (np.array_equal(bpost, apost) and np.array_equal(bdat, adat)):
                bad_other += 1
    sub = np.zeros(n, bool)
    for j in np.asarray(kc_idx):
        s, e = int(bi[j]), int(bi[j + 1])
        sub[bx[s:e]] = True
    return {"ok": bool(bad_kc == 0 and bad_other == 0 and not np.any(sub[np.asarray(kc_idx)])),
            "removed": removed, "expected_removed": expected_removed,
            "bad_kc_columns": bad_kc, "bad_other_columns": bad_other,
            "B_KC_KC_nnz": int(sub[np.asarray(kc_idx)].sum()),
            "bitwise_other_columns": bool(bad_other == 0)}


def install_weights(brain, fields):
    brain.indptr, brain.indices, brain.weights = fields


def field_switch_probe(brain, A, B, fired, kc_idx, is_kc):
    """Use the installed FlyBrain.synaptic_input() itself, not a reimplementation."""
    install_weights(brain, A)
    ca = brain.synaptic_input(np.asarray(fired)).ravel()
    install_weights(brain, B)
    cb = brain.synaptic_input(np.asarray(fired)).ravel()
    install_weights(brain, A)
    cr = brain.synaptic_input(np.asarray(fired)).ravel()
    d = ca - cb
    non_kc = ~is_kc
    return {
        "A_restore_exact": bool(np.array_equal(ca, cr)),
        "non_KC_unchanged": bool(np.allclose(d[non_kc], 0.0, atol=1e-6)),
        "KC_posts_changed": int((np.abs(d[np.asarray(kc_idx)]) > 1e-9).sum()),
        "max_abs_diff": float(np.max(np.abs(d))) if len(d) else 0.0,
    }


def rng_hash(brain):
    st = brain.rng.bit_generator.state
    return _sha_bytes(json.dumps(st, sort_keys=True, default=str).encode())


def run_condition(brain, seed, steps, kc_idx, mb_idx, kc_map, mb_map, fields, condition, out_dir,
                  script_sha, auth_sha, w_hash):
    install_weights(brain, fields)
    brain.reset(seed)
    before = rng_hash(brain)
    kc = np.zeros((steps, len(kc_idx)), np.uint8)
    mb = np.zeros((steps, len(mb_idx)), np.uint8)
    finite = True
    t0 = time.perf_counter()
    for t in range(steps):
        fired = np.asarray(brain.step())
        if not np.isfinite(brain.v).all():
            finite = False
        fk = fired[kc_map[fired] >= 0]
        fm = fired[mb_map[fired] >= 0]
        kc[t, kc_map[fk]] = 1
        mb[t, mb_map[fm]] = 1
    wall = time.perf_counter() - t0
    after = rng_hash(brain)
    return {"kc": kc, "mb": mb, "before": before, "after": after, "finite": finite,
            "wall_s": round(wall, 4), "condition": condition, "seed": seed,
            "script_sha": script_sha, "auth_sha": auth_sha, "w_hash": w_hash}


def run_child(data_dir, ids_path, matched_path, out_dir, seeds, steps, validate=True,
              expected=(4064, 97, 320, 95)):
    from flybrain import FlyBrain
    script_sha = _sha256(__file__)
    if validate:
        auth = load_auth(); freeze = load_freeze(); hashes = official_hashes()
        if not hashes["ok"]:
            raise RuntimeError("OFFICIAL_HASH_MISMATCH")
        if freeze.get("sha256") != script_sha:
            raise RuntimeError("SCRIPT_FREEZE_MISMATCH")
        auth_sha = auth["checksum"]
    else:
        auth_sha = "SYNTHETIC"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    brain = FlyBrain(data=data_dir, device="cpu", batch=1, seed=101, dt=0.020)
    init_s = time.perf_counter() - t0
    meta = np.load(os.path.join(data_dir, "brain.npz"), allow_pickle=True)
    ids = meta["ids"]
    ct = brain.cell_type.astype(str)
    kc_idx = np.flatnonzero(np.char.startswith(ct, "KC"))
    mb_idx = np.flatnonzero(np.char.startswith(ct, "MBON"))
    z = np.load(ids_path, allow_pickle=True)
    kc_ref = z["kc_idx"]; mb_ref = z["mb_idx"]
    kc_m_global = z["kc_matched"]; mb_m_global = z["mb_matched"]
    if not (np.array_equal(kc_idx, kc_ref) and np.array_equal(mb_idx, mb_ref)):
        raise RuntimeError("KC/MBON INDEX_MISMATCH")
    kc_m = np.searchsorted(kc_idx, kc_m_global)
    mb_m = np.searchsorted(mb_idx, mb_m_global)
    if not (np.array_equal(kc_idx[kc_m], kc_m_global) and np.array_equal(mb_idx[mb_m], mb_m_global)):
        raise RuntimeError("MATCHED_MAPPING_MISMATCH")
    n_kc, n_mb, n_km, n_mm = expected
    if not (len(kc_idx) == n_kc and len(mb_idx) == n_mb and len(kc_m) == n_km and len(mb_m) == n_mm):
        raise RuntimeError(f"EXPECTED_COUNTS_MISMATCH {len(kc_idx)}/{len(mb_idx)}/{len(kc_m)}/{len(mb_m)}")
    # persist mapping in the raw dir immediately
    np.savez(os.path.join(out_dir, "ids.npz"), ids=ids, kc_idx=kc_idx, mb_idx=mb_idx,
             kc_matched=kc_m_global, mb_matched=mb_m_global)
    if os.path.exists(matched_path):
        with open(matched_path, "rb") as f:
            mb = f.read()
        with open(os.path.join(out_dir, "matched_ids.json"), "wb") as f:
            f.write(mb)
    # connection fields
    A = (brain.indptr, brain.indices, brain.weights)
    is_kc = np.zeros(brain.n, bool); is_kc[kc_idx] = True
    B, removed, removed_sum = build_W_B_csc(*A, is_kc)
    vf = verify_W_B_fields(A, B, is_kc, kc_idx)
    if not vf["ok"]:
        raise RuntimeError(f"W_B_VERIFY_FAIL {vf}")
    if n_kc == 4064 and (removed != KC_KC_NNZ or len(B[1]) != W_B_NNZ):
        raise RuntimeError(f"W_B_FULL_COUNT_MISMATCH removed={removed} nnz={len(B[1])}")
    # field-switch integration check using the installed FlyBrain.synaptic_input
    probe = np.concatenate([kc_idx[:10], mb_idx[:2]])
    fs = field_switch_probe(brain, A, B, probe, kc_idx, is_kc)
    if not (fs["A_restore_exact"] and fs["non_KC_unchanged"]):
        raise RuntimeError(f"CODE_INTEGRATION_NO_GO {fs}")
    A_hash = _sha_bytes(A[0].tobytes() + A[1].tobytes() + A[2].tobytes())
    B_hash = _sha_bytes(B[0].tobytes() + B[1].tobytes() + B[2].tobytes())
    kc_map = np.full(brain.n, -1, np.int64); kc_map[kc_idx] = np.arange(len(kc_idx))
    mb_map = np.full(brain.n, -1, np.int64); mb_map[mb_idx] = np.arange(len(mb_idx))
    results = []; all_pairs = True
    for seed in seeds:
        rA = run_condition(brain, seed, steps, kc_idx, mb_idx, kc_map, mb_map, A, "A", out_dir,
                           script_sha, auth_sha, A_hash)
        kcA = rA["kc"]; mbA = rA["mb"]
        np.savez_compressed(os.path.join(out_dir, f"{seed}_A.npz"), kc=kcA, mb=mbA,
                            kc_matched=kcA[:, kc_m], mb_matched=mbA[:, mb_m],
                            seed=seed, condition="A", script_sha=script_sha, auth_sha=auth_sha,
                            w_hash=A_hash, rng_before=rA["before"], rng_after=rA["after"])
        rB = run_condition(brain, seed, steps, kc_idx, mb_idx, kc_map, mb_map, B, "B", out_dir,
                           script_sha, auth_sha, B_hash)
        kcB = rB["kc"]; mbB = rB["mb"]
        np.savez_compressed(os.path.join(out_dir, f"{seed}_B.npz"), kc=kcB, mb=mbB,
                            kc_matched=kcB[:, kc_m], mb_matched=mbB[:, mb_m],
                            seed=seed, condition="B", script_sha=script_sha, auth_sha=auth_sha,
                            w_hash=B_hash, rng_before=rB["before"], rng_after=rB["after"])
        pair = bool(rA["before"] == rB["before"] and rA["after"] == rB["after"])
        all_pairs = all_pairs and pair
        results.append({"seed": seed, "A_wall_s": rA["wall_s"], "B_wall_s": rB["wall_s"],
                        "A_finite": rA["finite"], "B_finite": rB["finite"],
                        "noise_pair_ok": pair, "A_rng_before": rA["before"], "B_rng_before": rB["before"],
                        "A_rng_after": rA["after"], "B_rng_after": rB["after"]})
        install_weights(brain, A)
    res = {
        "ok": True, "init_s": round(init_s, 4), "n_neurons": int(brain.n),
        "nnz_A": int(len(A[1])), "nnz_B": int(len(B[1])), "removed_edges": int(removed),
        "removed_sum": round(float(removed_sum), 6), "KC_KC_nnz": int(vf["B_KC_KC_nnz"]),
        "A_hash": A_hash, "B_hash": B_hash, "script_sha256": script_sha, "auth_sha256": auth_sha,
        "field_switch": fs, "W_B_verify": vf, "seeds": results,
        "noise_pairing_all": bool(all_pairs), "device": "cpu", "batch": 1, "steps": steps,
        "versions": _versions(),
    }
    with open(os.path.join(out_dir, "child_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def _versions():
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "flybrain": getattr(__import__("flybrain"), "__version__", "0.1.0")}


# ---------------------------------------------------------------- analysis
def _frac(arr, lo, hi):
    return float(arr[lo:hi].mean()) if arr.size else float("nan")


def _bins(arr, nb=20):
    return [round(float(x.mean()), 6) for x in np.array_split(arr, nb, axis=0)]


def _longest_runs(arr):
    # arr: steps x cells; per-cell longest consecutive 1 run
    out = np.zeros(arr.shape[1], np.int32)
    for j in range(arr.shape[1]):
        best = cur = 0
        for v in arr[:, j]:
            cur = cur + 1 if v else 0
            if cur > best:
                best = cur
        out[j] = best
    return out


def _stats(kc, mb):
    return {
        "kc_full_mean": round(float(kc.mean()), 6),
        "kc_full_ge085": round(float((kc >= 0.85).mean()), 6),
        "kc_full_ge095": round(float((kc >= 0.95).mean()), 6),
        "mb_full_mean": round(float(mb.mean()), 6),
        "kc_bins": _bins(kc),
        "mb_bins": _bins(mb),
        "kc_longest_run_max": int(_longest_runs(kc).max()) if kc.size else None,
        "kc_longest_run_mean": round(float(_longest_runs(kc).mean()), 4) if kc.size else None,
        "kc_cells_ge085": int((kc.mean(axis=0) >= 0.85).sum()),
        "kc_cells_ge095": int((kc.mean(axis=0) >= 0.95).sum()),
        "kc_cell_fraction_ge085": round(float((kc.mean(axis=0) >= 0.85).mean()), 6),
        "low_feedback": {str(j): round(float(kc[:, j].mean()), 6) for j in LOW_FEEDBACK},
    }


def _matched(kc_m, mb_m):
    return {
        "kc_matched_mean": round(float(kc_m.mean()), 6),
        "kc_matched_ge085": round(float((kc_m >= 0.85).mean()), 6),
        "kc_matched_ge095": round(float((kc_m >= 0.95).mean()), 6),
        "mb_matched_mean": round(float(mb_m.mean()), 6),
        "kc_matched_bins": _bins(kc_m),
        "mb_matched_bins": _bins(mb_m),
        "kc_matched_longest_run_max": int(_longest_runs(kc_m).max()) if kc_m.size else None,
    }


def _bootstrap_ci(d, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"ci_low": round(float(np.percentile(m, 2.5)), 6),
            "ci_high": round(float(np.percentile(m, 97.5)), 6), "n": n, "seed": seed}


def _exact_signflip(d):
    from loop.p11_2_c1_preflight import exact_signflip_p
    return float(exact_signflip_p(d))


def analyze(out_dir):
    ids = np.load(os.path.join(out_dir, "ids.npz"), allow_pickle=True)
    low = LOW_FEEDBACK
    rows = []
    for s in SEEDS:
        za = np.load(os.path.join(out_dir, f"{s}_A.npz"), allow_pickle=True)
        zb = np.load(os.path.join(out_dir, f"{s}_B.npz"), allow_pickle=True)
        kcA = za["kc"]; mbA = za["mb"]; kcmA = za["kc_matched"]; mbmA = za["mb_matched"]
        kcB = zb["kc"]; mbB = zb["mb"]; kcmB = zb["kc_matched"]; mbmB = zb["mb_matched"]
        A = _stats(kcA, mbA); B = _stats(kcB, mbB)
        Am = _matched(kcmA, mbmA); Bm = _matched(kcmB, mbmB)
        rows.append({"seed": s, "A": {**A, **Am}, "B": {**B, **Bm},
                     "d_matched": round(Bm["kc_matched_mean"] - Am["kc_matched_mean"], 6),
                     "noise_pair_ok": bool(za["rng_before"] == zb["rng_before"]
                                           and za["rng_after"] == zb["rng_after"])})
    d = np.array([x["d_matched"] for x in rows], float)
    p = _exact_signflip(d); ci = _bootstrap_ci(d)
    mean = float(d.mean()); sd = float(d.std(ddof=1))
    if mean <= -MIN_EFFECT and p < 0.05:
        verdict = "CAUSAL_CONTRIBUTION_SUPPORTED"
    elif mean <= -MIN_EFFECT:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NOT_SUPPORTED"
    Amean = float(np.mean([x["A"]["kc_matched_mean"] for x in rows]))
    Bmean = float(np.mean([x["B"]["kc_matched_mean"] for x in rows]))
    lowA = {str(j): round(float(np.mean([x["A"]["low_feedback"][str(j)] for x in rows])), 6) for j in low}
    lowB = {str(j): round(float(np.mean([x["B"]["low_feedback"][str(j)] for x in rows])), 6) for j in low}
    fullA = float(np.mean([x["A"]["kc_full_mean"] for x in rows]))
    fullB = float(np.mean([x["B"]["kc_full_mean"] for x in rows]))
    satB = int(round(np.mean([x["B"]["kc_cells_ge085"] for x in rows])))
    fracB = float(np.mean([x["B"]["kc_cell_fraction_ge085"] for x in rows]))
    return {
        "protocol": "P11-2-C1/v1", "completeness": f"{len(rows)}/25",
        "per_seed": rows,
        "primary": {"endpoint": "matched 320 KC firing fraction [100,300), per seed",
                    "A_mean": round(Amean, 6), "B_mean": round(Bmean, 6),
                    "effect_mean": round(mean, 6), "effect_sd": round(sd, 6),
                    "effect_median": round(float(np.median(d)), 6),
                    "diffs": [round(float(x), 6) for x in d],
                    "exact_signflip_p": p, "ci": ci,
                    "min_meaningful_effect": MIN_EFFECT,
                    "verdict": verdict},
        "secondary": {"full_4064_A_mean": round(fullA, 6), "full_4064_B_mean": round(fullB, 6),
                      "full_4064_A_cells_ge085": int(round(np.mean([x["A"]["kc_cells_ge085"] for x in rows]))),
                      "full_4064_B_cells_ge085": satB,
                      "full_4064_B_cell_fraction_ge085": round(fracB, 6),
                      "full_4064_B_cells_ge095": int(round(np.mean([x["B"]["kc_cells_ge095"] for x in rows]))),
                      "low_feedback_A": lowA, "low_feedback_B": lowB,
                      "mbon_full_A": round(float(np.mean([x["A"]["mb_full_mean"] for x in rows])), 6),
                      "mbon_full_B": round(float(np.mean([x["B"]["mb_full_mean"] for x in rows])), 6),
                      "mbon_matched_A": round(float(np.mean([x["A"]["mb_matched_mean"] for x in rows])), 6),
                      "mbon_matched_B": round(float(np.mean([x["B"]["mb_matched_mean"] for x in rows])), 6)},
        "interpretation": "model-internal causal contribution only; no biological mechanism or learning claim",
        "remaining_saturated": {"matched_320_B": round(Bmean, 6),
                                "full_4064_B": round(fullB, 6),
                                "full_4064_B_cells_ge085": satB,
                                "full_4064_B_cell_fraction_ge085": round(fracB, 6),
                                "full_4064_B_cells_ge095": int(round(np.mean([x["B"]["kc_cells_ge095"] for x in rows]))),
                                "low_feedback_B": lowB},
        "sensory_readiness": ("YES_FOR_SEPARATE_PROPOSAL" if verdict == "CAUSAL_CONTRIBUTION_SUPPORTED"
                              else "NO_INVESTIGATE_NON_KC"),
        "gate": {"Training_Authorization": False, "REAL_SYNAPTIC_LEARNING_READY": "NO",
                 "Level_A/B/C/Biological": "NO-GO", "A1_A2_A3/B": "NO-GO",
                 "DeltaW": 0, "Dijkstra": "NO", "Maze": "NO", "B/C": "NO", "Robot": "NO"},
    }


def _manifest(out_dir, extra):
    files = []
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            files.append({"name": name, "sha256": _sha256(p), "size": os.path.getsize(p)})
    return {"protocol": "P11-2-C1/v1", "out_dir": out_dir, "files": files, **extra}


def synthetic_tiny():
    """Small synthetic FlyBrain + CSC field switch; no full brain."""
    from flybrain import FlyBrain
    tmp = tempfile.mkdtemp(prefix="p11_2_c1_tiny_")
    data = os.path.join(tmp, "data"); os.makedirs(data)
    n = 4
    dense = np.zeros((n, n), np.float32)
    dense[0, 1] = 0.5; dense[1, 0] = 0.25
    dense[0, 2] = 0.3; dense[2, 0] = 0.4
    dense[2, 3] = 0.6; dense[3, 2] = 0.7
    ids = np.array([100, 101, 102, 103])
    np.savez(os.path.join(data, "brain.npz"), ids=ids, visual=np.array([], np.int64),
             azimuth=np.zeros(n, np.float32), cell_type=np.array(["KC", "KC", "MBON", "nonKC"]),
             side=np.array(["L", "R", "L", "R"]))
    sp.save_npz(os.path.join(data, "weights.npz"), sp.csr_matrix(dense))
    np.savez(os.path.join(tmp, "ids.npz"), ids=ids, kc_idx=np.array([0, 1]), mb_idx=np.array([2]),
             kc_matched=np.array([0, 1]), mb_matched=np.array([2]))
    with open(os.path.join(tmp, "matched_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"kc": [100, 101], "mbon": [102]}, f)
    out = os.path.join(tmp, "out"); os.makedirs(out)
    res = run_child(data, os.path.join(tmp, "ids.npz"), os.path.join(tmp, "matched_ids.json"),
                    out, [101, 102], 5, validate=False, expected=(2, 1, 2, 1))
    return {"tmp": tmp, "out": out, "child": res, "ok": bool(res["ok"] and res["noise_pairing_all"]
                                                              and res["removed_edges"] == 2
                                                              and res["field_switch"]["A_restore_exact"]
                                                              and res["field_switch"]["non_KC_unchanged"])}


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--synthetic" in argv:
        r = synthetic_tiny(); print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
        return 0 if r["ok"] else 2
    if "--child" in argv:
        i = argv.index("--child"); out_dir = os.path.abspath(argv[i + 1])
        repo = load_archive(); auth = load_auth(); freeze = load_freeze()
        res = run_child(DATA_DIR, os.path.join(repo["raw_dir"], "ids.npz"),
                        os.path.join(repo["raw_dir"], "matched_ids.json"), out_dir,
                        SEEDS, STEPS, validate=True)
        print("[child] ok", res["ok"], "| seeds", len(res["seeds"]),
              "| removed", res["removed_edges"], "| noise pairs", res["noise_pairing_all"])
        return 0 if res["ok"] else 3
    if "--postprocess" in argv:
        i = argv.index("--postprocess"); out_dir = os.path.abspath(argv[i + 1])
        meta = json.load(open(os.path.join(out_dir, "run_meta.json"), encoding="utf-8"))
        child = json.load(open(os.path.join(out_dir, "child_result.json"), encoding="utf-8"))
        rep = analyze(out_dir); rep["run_meta"] = meta; rep["child"] = child
        json.dump(rep, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[postprocess] wrote", REPORT); return 0
    # parent: one-time Job-controlled run
    from loop.p11_1c_full_brain_bench import run_under_job
    auth = load_auth(); freeze = load_freeze(); hashes = official_hashes()
    if not hashes["ok"]:
        print("OFFICIAL_HASH_MISMATCH"); return 4
    archive = load_archive()
    if not archive.get("ok"):
        print("ARCHIVE_UNVERIFIED", archive); return 4
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.join(tempfile.gettempdir(), "opencode", "p11_2_c1_full", stamp)
    os.makedirs(out_dir, exist_ok=True)
    for name in ("ids.npz", "matched_ids.json"):
        with open(os.path.join(archive["raw_dir"], name), "rb") as f:
            b = f.read()
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(b)
    with open(os.path.join(out_dir, "authorization.json"), "w", encoding="utf-8") as f:
        json.dump(auth, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "script_freeze.json"), "w", encoding="utf-8") as f:
        json.dump(freeze, f, ensure_ascii=False, indent=1)
    log = os.path.join(out_dir, "experiment.log")
    t0 = time.perf_counter()
    run = run_under_job(MEM_CAP_BYTES, ["--child", out_dir], WALL_CAP_S, MEM_CAP_MIB, log,
                        script_path=os.path.abspath(__file__))
    meta = {
        "protocol": "P11-2-C1/v1", "status": "RUN_COMPLETE" if run.get("ok") else "RUN_FAILED",
        "out_dir": out_dir, "run_wall_s": run.get("wall_s"),
        "job_peaks": run.get("job_peaks"), "peak_working_set_MiB": run.get("peak_working_set_MiB"),
        "job_peak_MiB": (run.get("job_peaks") or {}).get("PeakJobMemoryUsed_MiB"),
        "exit_code": run.get("exit_code"), "timeout": run.get("timeout"),
        "mem_exceeded": run.get("mem_exceeded"), "reason": run.get("reason"),
        "script_sha256": _sha256(__file__), "authorization_sha256": auth["checksum"],
        "freeze_sha256": _sha256(FREEZE_PATH), "official_hashes": hashes,
        "archive_status": archive["status"], "log_tail": run.get("log_tail", "")[-2000:],
        "created": stamp,
    }
    # persist resource data BEFORE any post-processing
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    with open(RUN_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    child_path = os.path.join(out_dir, "child_result.json")
    child = json.load(open(child_path, encoding="utf-8")) if os.path.exists(child_path) else None
    if run.get("ok") and child and len(child.get("seeds", [])) == len(SEEDS) and child.get("noise_pairing_all"):
        rep = analyze(out_dir); rep["run_meta"] = meta; rep["child"] = child
        rep["complete_25_pairs"] = True
        rep["gate"]["P11_2_C1_EXECUTION_AUTHORIZATION"] = "APPROVED_ONE_RUN; CONSUMED"
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        manifest = _manifest(out_dir, {"official_hashes": hashes, "script_sha256": meta["script_sha256"],
                                       "authorization_sha256": auth["checksum"],
                                       "run_meta_sha256": _sha256(os.path.join(out_dir, "run_meta.json"))})
        with open(RAW_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        print("[P11-2-C1] complete:", rep["completeness"], "| A", rep["primary"]["A_mean"],
              "| B", rep["primary"]["B_mean"], "| d", rep["primary"]["effect_mean"],
              "| p", rep["primary"]["exact_signflip_p"], "| CI", rep["primary"]["ci"])
        print("  job peak MiB:", meta["job_peak_MiB"], "| wall s:", meta["run_wall_s"])
        print("  verdict:", rep["primary"]["verdict"], "| sensory:", rep["sensory_readiness"])
        return 0
    n_done = len(child.get("seeds", [])) if child else 0
    rep = {"protocol": "P11-2-C1/v1", "status": "INCOMPLETE_NO_GO",
           "completeness": f"{n_done}/25", "complete_25_pairs": False,
           "run_meta": meta, "child": child,
           "reason": "resource/timeout/crash/missing seed; one-run rule: no rerun",
           "gate": {"P11_2_C1_EXECUTION_AUTHORIZATION": "APPROVED_ONE_RUN; CONSUMED",
                    "Training_Authorization": False, "REAL_SYNAPTIC_LEARNING_READY": "NO",
                    "Level_A/B/C/Biological": "NO-GO", "A1_A2_A3/B": "NO-GO"}}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    manifest = _manifest(out_dir, {"official_hashes": hashes, "script_sha256": meta["script_sha256"],
                                   "authorization_sha256": auth["checksum"],
                                   "run_meta_sha256": _sha256(os.path.join(out_dir, "run_meta.json")),
                                   "status": "INCOMPLETE_NO_GO"})
    with open(RAW_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("[P11-2-C1] incomplete/failed; no rerun. run:", run)
    if child:
        print("  child seeds:", len(child.get("seeds", [])), "| noise:", child.get("noise_pairing_all"))
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
