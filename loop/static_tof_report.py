"""P2d-7 离线分析工具：真实静态 ToF × 神经读出的可复现报告（无硬件、无 Bridge）。

读 `loop/physical_static_diag.py` 产出的场景 CSV，输出：
  - 单场景统计（帧数/时长、五路状态比例、VALID 距离、size/growth/chase、
    brain_turn_raw；含"排除前 N 秒"对照）；
  - 场景间比较（以 A_open 为基线：ΔFR / ΔFL / ΔSYM / baseline_drift）；
  - 刺激纯度检查（B/C 是否纯 FR-only / FL-only；D 是否对称）。

约定：brain_turn_raw **正值=偏左转，负值=偏右转**。仅用 `status==VALID` 的距离；
NO_TARGET 不作为距离。缺列/空文件明确报错，不静默补零。

用法：
  python -m loop.static_tof_report --a A.csv --b B.csv --c C.csv --d D.csv \
      --a-again A2.csv [--exclude-head-s 2.0] [--json out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys

CH = ["L", "FL", "F", "FR", "R"]
MASK_CH = ["FL", "F", "FR"]                       # 进神经避障掩码的通道
REQUIRED = (["t", "brain_turn_raw"]
            + [f"{c}_mm" for c in CH] + [f"{c}_status" for c in CH]
            + [f"{k}_{s}" for s in ("L", "R") for k in ("size", "growth", "chase")])


class ReportError(RuntimeError):
    pass


# --------------------------------------------------------------------------
def load_rows(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        raise ReportError(f"{path}: {exc}") from exc
    if not rows:
        raise ReportError(f"{path}: empty CSV")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise ReportError(f"{path}: missing columns {missing}")
    return rows


def _num(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if v in (None, ""):
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def _stats(xs):
    if not xs:
        return None
    return {"n": len(xs), "mean": st.fmean(xs),
            "std": (st.pstdev(xs) if len(xs) > 1 else 0.0),
            "min": min(xs), "max": max(xs)}


def _valid_ratio(rows, ch):
    if not rows:
        return 0.0
    v = sum(1 for r in rows if (r.get(f"{ch}_status") or "") == "VALID")
    return v / len(rows)


def exclude_head(rows, head_s):
    if not rows or head_s <= 0:
        return rows
    ts = _num(rows, "t")
    if not ts:
        return rows
    t0 = min(ts)
    tail = [r for r in rows if r.get("t") not in (None, "")
            and (float(r["t"]) - t0) >= head_s]
    return tail or rows


def summarize(rows):
    n = len(rows)
    ts = _num(rows, "t")
    chans = {}
    for c in CH:
        sts = [(r.get(f"{c}_status") or "") for r in rows]
        cnt = {k: sts.count(k) for k in ("VALID", "NO_TARGET", "TOO_NEAR", "IO_ERROR")}
        vdist = [float(r[f"{c}_mm"]) for r in rows
                 if (r.get(f"{c}_status") == "VALID" and r.get(f"{c}_mm") not in (None, ""))]
        chans[c] = {"status_counts": cnt, "valid_ratio": (cnt["VALID"] / n if n else 0.0),
                    "dist_mm": _stats(vdist)}
    sides = {f"{k}_{s}": _stats(_num(rows, f"{k}_{s}"))
             for s in ("L", "R") for k in ("size", "growth", "chase")}
    bt = _num(rows, "brain_turn_raw")
    bts = _stats(bt)
    if bts is not None:
        bts["pos_frac"] = sum(1 for x in bt if x > 0) / len(bt)
    return {"n_frames": n, "duration_s": ((max(ts) - min(ts)) if ts else None),
            "channels": chans, "sides": sides, "brain_turn_raw": bts}


def purity(rows, target):
    """target: 'FR' | 'FL' | ('FL','FR')。返回纯度判定。"""
    tgts = [target] if isinstance(target, str) else list(target)
    others = [c for c in MASK_CH if c not in tgts]
    vr = {c: _valid_ratio(rows, c) for c in CH}
    tgt_ok = all(vr[c] >= 0.5 for c in tgts)
    oth_ok = all(vr[c] <= 0.15 for c in others)
    info = {"target": tgts, "other_mask": others, "valid_ratio": vr}
    if not tgt_ok:
        info["label"] = "EMPTY_TARGET"
    elif oth_ok:
        info["label"] = "pure"
    else:
        info["label"] = "MIXED"
    return info


def symmetry(rows):
    fl = _num(rows, "FL_mm") if _valid_ratio(rows, "FL") > 0 else []
    fr = _num(rows, "FR_mm") if _valid_ratio(rows, "FR") > 0 else []
    mfl = st.fmean(fl) if fl else None
    mfr = st.fmean(fr) if fr else None
    if mfl and mfr:
        rel = abs(mfl - mfr) / (0.5 * (mfl + mfr))
        return {"FL_mean_mm": mfl, "FR_mean_mm": mfr, "rel_asym": rel,
                "symmetric": rel <= 0.15}
    return {"FL_mean_mm": mfl, "FR_mean_mm": mfr, "rel_asym": None, "symmetric": False}


def compare(base_mean, b_mean, c_mean, d_mean, a2_mean):
    return {"dFR": (b_mean - base_mean), "dFL": (c_mean - base_mean),
            "dSYM": (d_mean - base_mean), "baseline_drift": (a2_mean - base_mean)}


# --------------------------------------------------------------------------
def _bt_mean(rows):
    s = summarize(rows)["brain_turn_raw"]
    return None if s is None else s["mean"]


def _fmt(s, unit=""):
    if s is None:
        return "—"
    u = unit
    return (f"n={s['n']} mean={s['mean']:.3f}{u} std={s['std']:.3f}{u} "
            f"min={s['min']:.3f} max={s['max']:.3f}")


def print_scene(name, rows, head_s):
    print(f"\n===== {name}  (n={len(rows)}) =====")
    full = summarize(rows)
    tail = summarize(exclude_head(rows, head_s))
    print(f"  frames={full['n_frames']} duration={full['duration_s']:.2f}s  "
          f"(exclude first {head_s:g}s: n={tail['n_frames']})")
    for c in CH:
        cc = full["channels"][c]
        sc = cc["status_counts"]
        print(f"  {c:2s}: VALID {cc['valid_ratio']*100:4.0f}%  "
              f"[V{sc['VALID']} NT{sc['NO_TARGET']} TN{sc['TOO_NEAR']} IO{sc['IO_ERROR']}]  "
              f"dist(mm)={_fmt(cc['dist_mm'])}")
    for side in ("L", "R"):
        vals = "  ".join(f"{k}_{side}={_fmt(full['sides'][f'{k}_{side}'])}"
                         for k in ("size", "growth", "chase"))
        print(f"  {vals}")
    bt, bt_t = full["brain_turn_raw"], tail["brain_turn_raw"]
    print(f"  brain_turn_raw FULL: {_fmt(bt)}  pos_frac={bt['pos_frac']:.2f}" if bt else "  brain_turn_raw: —")
    if bt_t:
        print(f"  brain_turn_raw (excl head): mean={bt_t['mean']:.4f} std={bt_t['std']:.4f}")
    return full, tail


def main(argv=None) -> int:
    try:                                        # Windows GBK 控制台：避免非 GBK 字符报错
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                           # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="P2d-7 static ToF report")
    ap.add_argument("--a", required=True, help="A_open CSV")
    ap.add_argument("--b", default=None, help="B_FR_500 CSV")
    ap.add_argument("--c", default=None, help="C_FL_500 CSV")
    ap.add_argument("--d", default=None, help="D_symmetric CSV")
    ap.add_argument("--a-again", default=None, help="A_open_again CSV")
    ap.add_argument("--exclude-head-s", type=float, default=2.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    scenes = {}
    for name, path in (("A_open", args.a), ("B_FR_500", args.b), ("C_FL_500", args.c),
                       ("D_symmetric", args.d), ("A_open_again", args.a_again)):
        if path:
            scenes[name] = load_rows(path)

    out = {"scenes": {}}
    for name, rows in scenes.items():
        full, tail = print_scene(name, rows, args.exclude_head_s)
        out["scenes"][name] = {"full": full, "excl_head": tail}

    # 刺激纯度
    print("\n===== 刺激纯度 =====")
    pure = {}
    if "B_FR_500" in scenes:
        pure["B_FR_500"] = purity(scenes["B_FR_500"], "FR")
    if "C_FL_500" in scenes:
        pure["C_FL_500"] = purity(scenes["C_FL_500"], "FL")
    if "D_symmetric" in scenes:
        pure["D_symmetric"] = purity(scenes["D_symmetric"], ("FL", "FR"))
        pure["D_symmetric"]["symmetry"] = symmetry(scenes["D_symmetric"])
    for k, v in pure.items():
        extra = ""
        if "symmetry" in v:
            extra = f" symmetry={v['symmetry']}"
        print(f"  {k}: {v['label']}  valid_ratio={ {c: round(r,2) for c,r in v['valid_ratio'].items()} }{extra}")
    out["purity"] = pure

    # 场景间比较
    bt = {n: _bt_mean(r) for n, r in scenes.items()}
    print("\n===== 场景间比较（baseline=A_open；正=偏左，负=偏右）=====")
    for n in ("A_open", "B_FR_500", "C_FL_500", "D_symmetric", "A_open_again"):
        if n in bt and bt[n] is not None:
            print(f"  {n:14s} brain_turn_raw mean={bt[n]:+.4f}")
    cmp_ = None
    if bt.get("A_open") is not None:
        cmp_ = compare(bt["A_open"], bt.get("B_FR_500", bt["A_open"]),
                       bt.get("C_FL_500", bt["A_open"]), bt.get("D_symmetric", bt["A_open"]),
                       bt.get("A_open_again", bt["A_open"]))
        print(f"  ΔFR(B-A)={cmp_['dFR']:+.4f}  ΔFL(C-A)={cmp_['dFL']:+.4f}  "
              f"ΔSYM(D-A)={cmp_['dSYM']:+.4f}  baseline_drift(A2-A)={cmp_['baseline_drift']:+.4f}")
    out["compare"] = cmp_

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
        print(f"\n[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
