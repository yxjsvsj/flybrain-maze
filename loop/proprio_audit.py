"""P2d-24 第一关：在**实际** MaleCNS 连接组里寻找"自运动/本体感觉/视觉运动/运动副本"输入通路。

只读。判据区分：真实解剖/文献依据 / 连接组中实际可定位的神经元 / 人工推断功能。
不把 L50/R50、DNa02、电机输出当作"自运动感觉输入"；不凭名字像运动神经就宣称找到反馈回路。
"""
from __future__ import annotations

import sys

import numpy as np

from loop.run import make_setup

KEYWORDS = {
    "自运动/本体": ["proprio", "mechano", "chordoton", "campaniform", "haltere", "stretch",
                    "Johnston", "hair_plate", "hairplate"],
    "视觉运动(精确)": ["T4", "T5"],
    "efference/副本": ["efference", "corollary", "reafferen"],
    "平衡棒/前庭": ["vestib", "gyro"],
}
SENSORY_KW = "sensory"


def group_by(idx, key):
    d = {}
    for i in idx:
        d.setdefault(key[i], 0)
        d[key[i]] += 1
    return d


def downstream(csc, src, depth=2, max_n=400000):
    indptr, indices = csc
    cur = set(int(i) for i in src)
    reached = set(cur)
    for _ in range(depth):
        nxt = set()
        for j in cur:
            for e in range(indptr[j], indptr[j + 1]):
                t = int(indices[e])
                if t not in reached:
                    nxt.add(t)
                    reached.add(t)
                    if len(reached) > max_n:
                        return reached, True
        cur = nxt
    return reached, False


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device="cpu", seed=64,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    b = brain._b
    ct = np.array([str(x) for x in b.cell_type])
    sc = np.array([str(x) for x in b.superclass]) if b.superclass is not None else None
    side = np.array([str(x) for x in b.side])
    print(f"[P2d-24] MaleCNS: n={b.n}  cell_type 种类={len(set(ct.tolist()))}  "
          f"superclass 种类={len(set(sc.tolist())) if sc is not None else 0}")
    print("  superclass 计数(前 20):",
          sorted(group_by(range(b.n), sc).items(), key=lambda kv: -kv[1])[:20] if sc is not None else "-")
    dn = set(np.asarray(brain.cells(["descending_neuron"])).tolist())
    sc_low = np.char.lower(sc.astype(str)) if sc is not None else None
    motor = set(np.flatnonzero(np.char.find(sc_low, "motor") >= 0).tolist()) if sc is not None else set()
    print(f"  descending_neuron n={len(dn)}  motor-superclass n={len(motor)}")
    print(f"  sensory_input={b.sensory_input}: 17937 类神经元入突触被置 0（仅 sensory superclass）")
    csc = (b.indptr, b.indices)

    # superclass 采样（看是否有 ol_sensory/visual 的运动检测器、vnc_sensory 的本体感受器）
    if sc is not None:
        print("\n== 相关 superclass 的 cell_type 采样 ==")
        for s in ("ol_sensory", "visual_projection", "vnc_sensory", "sensory_ascending",
                  "visual_centrifugal"):
            idx = np.flatnonzero(sc == s)
            types = sorted(set(ct[idx].tolist()))
            print(f"  {s:18s} n={len(idx):5d} 类型数={len(types):4d} 采样={types[:12]}")
        # 精确 T4/T5（视觉运动/optic-flow 检测器）
        t45 = [t for t in set(ct.tolist()) if t[:2] in ("T4", "T5") and (len(t) == 2 or t[2].isdigit() is False)]
        t45 = [t for t in set(ct.tolist()) if t.startswith(("T4", "T5"))]
        if t45:
            idx = np.flatnonzero(np.isin(ct, t45)).tolist()
            ds, _ = downstream(csc, idx, depth=2)
            print(f"\n== 精确 T4/T5: 类型={sorted(t45)[:10]} n={len(idx)} 2跳到DN={len(ds & dn)} "
                  f"到motor={len(ds & motor)}")
        else:
            print("\n== 精确 T4/T5: 无命中（连接组无该命名）")
        # vnc_sensory -> vnc_motor（本体/机械感 -> 运动）
        vs = np.flatnonzero(sc == "vnc_sensory").tolist()
        ds, _ = downstream(csc, vs, depth=2)
        print(f"== vnc_sensory n={len(vs)} 2跳下游={len(ds)} 到DN={len(ds & dn)} 到motor={len(ds & motor)}")

    print("\n== 候选通路关键词命中 ==")
    csc = (b.indptr, b.indices)
    found_any = False
    for label, kws in KEYWORDS.items():
        hits = [t for t in set(ct.tolist()) if any(k.lower() in t.lower() for k in kws)]
        if not hits:
            print(f"  [{label}] 无命中")
            continue
        for t in hits:
            idx = np.flatnonzero(ct == t).tolist()
            s = sc[idx[0]] if sc is not None else "?"
            is_sens = SENSORY_KW in str(s).lower()
            ds, _ = downstream(csc, idx, depth=2)
            to_dn = len(ds & dn)
            to_motor = len(ds & motor) if motor else 0
            found_any = found_any or (to_dn > 0)
            print(f"  [{label}] {t:24s} n={len(idx):4d} superclass={s:16s} "
                  f"sensory_input受影响={is_sens}  2跳下游={len(ds)}  到DN={to_dn} 到motor={to_motor}")
    if not found_any:
        print("  → 未发现任何候选在 2 跳内到达 descending_neuron")

    # 现有"间接"感觉类：整体 sensory superclass 是否 2 跳可达 DN
    if sc is not None:
        s_all = np.flatnonzero(np.char.find(np.char.lower(sc.astype(str)), SENSORY_KW) >= 0)
        ds, _ = downstream(csc, s_all.tolist(), depth=2, max_n=800000)
        print(f"\n[sensory 全体] n={len(s_all)}  2跳下游={len(ds)}  到DN={len(ds & dn)}  "
              f"到motor={len(ds & motor) if motor else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
