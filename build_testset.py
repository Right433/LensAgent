"""
build_testset.py  —  构建 200 条评估 test set
输出: /gz-data/testset.json

Question 类型（共 5 类，各 40 条）：
  A. in_domain     —— 数据库有精确匹配（FOV/Fno/aper 完全命中）
  B. out_of_domain —— 参数组合数据库没有，考验泛化检索
  C. specific      —— 具体需求：同时指定 FOV/Fno/aper/RMS 目标
  D. range         —— 范围需求：给区间而非精确值
  E. partial       —— 缺省需求：只给部分参数（1~2个），其余不限

评价指标：
  i.   hit_rate       —— 与用户输入相符（FOV/Fno 在容差内 = hit）
  ii.  norm_error_sum —— 各指标相对误差归一化加权和
         = W_FOV*|Δfov/fov| + W_FNUM*|Δfnum/fnum| + W_APER*|Δaper/aper|
  iii. final_score    —— 误差总和 + 归一化RMS（越小越好）
         = norm_error_sum + rms / RMS_SCALE
"""

import json, random, pickle
from pathlib import Path
from collections import Counter

FAISS_DIR = "/gz-data/faiss_index"
OUTPUT    = "/gz-data/testset.json"
random.seed(42)

W_FOV, W_FNUM, W_APER = 0.4, 0.4, 0.2
RMS_SCALE = 0.1   # mm，量纲对齐

def load_lenses():
    pkl = Path(FAISS_DIR) / "lenses.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"找不到 {pkl}，请先运行 build_rag.py")
    with open(pkl, "rb") as f:
        lenses = pickle.load(f)
    for i, l in enumerate(lenses):
        l["lens_idx"] = i
    return lenses

def _gt(lens):
    return {
        "lens_idx":  lens.get("lens_idx"),
        "source":    Path(lens.get("source", "")).name,
        "fov":       lens.get("fov"),
        "fnum":      lens.get("fnum"),
        "aper":      lens.get("aper"),
        "calc_effl": lens.get("calc_effl"),
        "calc_totr": lens.get("calc_totr"),
        "calc_rms":  lens.get("calc_rms"),
        "calc_yimg": lens.get("calc_yimg"),
        "fit":       lens.get("fit"),
    }

def _valid(lens):
    return all(lens.get(k) is not None
               for k in ("fov", "fnum", "aper", "calc_rms"))

def get_param_sets(lenses):
    fovs  = sorted(set(l["fov"]  for l in lenses if l.get("fov")  is not None))
    fnums = sorted(set(l["fnum"] for l in lenses if l.get("fnum") is not None))
    apers = sorted(set(l["aper"] for l in lenses if l.get("aper") is not None))
    existing = {(l["fov"], l["fnum"]) for l in lenses
                if l.get("fov") and l.get("fnum")}
    return fovs, fnums, apers, existing

# ── A. in_domain ──────────────────────────────────────────────────────────────
def gen_A(lenses, n=40):
    pool = [l for l in lenses if _valid(l)]
    samples = random.sample(pool, min(n, len(pool)))
    items = []
    for lens in samples:
        fov, fnum, aper = lens["fov"], lens["fnum"], lens["aper"]
        templates = [
            f"我需要一个 FOV={fov}度、F/{fnum}、入瞳径{aper}mm 的镜头",
            f"找 FOV {fov}° 光圈 F/{fnum} 口径 {aper}mm 的镜头方案",
            f"有没有视场角 {fov} 度、F 数 {fnum}、通光口径 {aper}mm 的设计",
            f"FOV={fov}度 Fno={fnum} 入瞳径={aper}mm 推荐哪个镜头",
            f"查找满足 FOV {fov}° F/{fnum} 入瞳 {aper}mm 的镜头",
        ]
        items.append({
            "id": f"A_{len(items):03d}", "type": "in_domain",
            "query": random.choice(templates),
            "constraints": {"fov": fov, "fnum": fnum, "aper": aper},
            "ground_truth": _gt(lens),
            "eval_config": {
                "has_fov": True, "has_fnum": True, "has_aper": True,
                "tol_fov": 1.0, "tol_fnum": 0.1, "tol_aper": 1.0,
            },
        })
    return items[:n]

# ── B. out_of_domain ──────────────────────────────────────────────────────────
def gen_B(lenses, n=40):
    fovs, fnums, apers, existing = get_param_sets(lenses)
    ood_fovs  = [f for f in [5,8,12,15,18,22,45,55,65,75,85,100,120] if f not in fovs]
    ood_fnums = [f for f in [1.2,1.4,1.6,1.8,3.0,3.5,4.5,5.0,6.3,8.0,11.0] if f not in fnums]
    ood_fovs  = ood_fovs  + list(fovs[:5])
    ood_fnums = ood_fnums + list(fnums[-3:])

    items, attempts = [], 0
    while len(items) < n and attempts < 10000:
        attempts += 1
        fov  = random.choice(ood_fovs)
        fnum = random.choice(ood_fnums)
        aper = random.choice([4,6,8,10,12,15,18,20,25,30])
        if (fov, fnum) in existing:
            continue
        templates = [
            f"需要 FOV={fov}度、F/{fnum} 的镜头，找最接近的方案",
            f"找视场角 {fov}° 光圈 F{fnum} 入瞳径 {aper}mm 的镜头",
            f"有没有 FOV {fov} 度 Fno={fnum} 的设计可以参考",
            f"FOV={fov}度 F/{fnum} 口径 {aper}mm 最接近的镜头是什么",
        ]
        items.append({
            "id": f"B_{len(items):03d}", "type": "out_of_domain",
            "query": random.choice(templates),
            "constraints": {"fov": fov, "fnum": fnum, "aper": aper},
            "ground_truth": None,
            "eval_config": {
                "has_fov": True, "has_fnum": True, "has_aper": True,
                "tol_fov": 5.0, "tol_fnum": 0.5, "tol_aper": 3.0,
            },
        })
    return items[:n]

# ── C. specific ───────────────────────────────────────────────────────────────
def gen_C(lenses, n=40):
    pool = [l for l in lenses if _valid(l)]
    samples = random.sample(pool, min(n * 3, len(pool)))
    items = []
    for lens in samples:
        if len(items) >= n:
            break
        fov, fnum, aper = lens["fov"], lens["fnum"], lens["aper"]
        rms  = lens["calc_rms"]
        effl = lens.get("calc_effl")
        emphasis = random.choice(["rms", "effl", "both"])
        if emphasis == "rms":
            add = f"要求近轴 RMS 小于 {rms*1.5:.3f}mm"
        elif emphasis == "effl" and effl:
            add = f"焦距约 {effl:.1f}mm"
        else:
            add = (f"焦距约 {effl:.1f}mm，RMS 小于 {rms*1.5:.3f}mm"
                   if effl else f"RMS 越小越好")
        templates = [
            f"需要 FOV={fov}度 F/{fnum} 入瞳径{aper}mm 的镜头，{add}",
            f"找 FOV={fov}° Fno={fnum} 口径{aper}mm 的方案，{add}",
            f"设计要求：视场角{fov}度、光圈F/{fnum}、入瞳{aper}mm，{add}",
            f"FOV {fov}度 F/{fnum} {aper}mm口径，{add}，推荐最优方案",
        ]
        items.append({
            "id": f"C_{len(items):03d}", "type": "specific",
            "query": random.choice(templates),
            "constraints": {"fov": fov, "fnum": fnum, "aper": aper,
                            "rms_target": round(rms * 1.5, 4)},
            "ground_truth": _gt(lens),
            "eval_config": {
                "has_fov": True, "has_fnum": True, "has_aper": True,
                "tol_fov": 1.0, "tol_fnum": 0.15, "tol_aper": 1.0,
                "rms_target": round(rms * 1.5, 4),
            },
        })
    return items[:n]

# ── D. range ──────────────────────────────────────────────────────────────────
def gen_D(lenses, n=40):
    fovs, fnums, _, _ = get_param_sets(lenses)
    fov_pairs  = [(a,b) for a in fovs for b in fovs if 2 <= b-a <= 15][:200]
    fnum_pairs = [(a,b) for a in fnums for b in fnums if 0.2 <= b-a <= 1.5][:200]

    items, attempts = [], 0
    while len(items) < n and attempts < 5000:
        attempts += 1
        if not fov_pairs or not fnum_pairs:
            break
        fov_lo, fov_hi   = random.choice(fov_pairs)
        fnum_lo, fnum_hi = random.choice(fnum_pairs)
        candidates = [l for l in lenses if _valid(l)
                      and fov_lo <= l["fov"]  <= fov_hi
                      and fnum_lo <= l["fnum"] <= fnum_hi]
        if not candidates:
            continue
        best = min(candidates, key=lambda l: l["calc_rms"])
        templates = [
            f"FOV 在 {fov_lo}~{fov_hi} 度之间，F/{fnum_lo} 到 F/{fnum_hi}，RMS 尽量小",
            f"视场角 {fov_lo}°~{fov_hi}°，光圈 F{fnum_lo}~F{fnum_hi}，找性能最好的",
            f"需要 FOV {fov_lo}-{fov_hi}度、Fno {fnum_lo}-{fnum_hi} 的镜头",
            f"FOV={fov_lo}到{fov_hi}度 F数{fnum_lo}到{fnum_hi} 有哪些方案",
            f"视场 {fov_lo}~{fov_hi}度 光圈 F/{fnum_lo}~F/{fnum_hi} 推荐 RMS 最小的",
        ]
        items.append({
            "id": f"D_{len(items):03d}", "type": "range",
            "query": random.choice(templates),
            "constraints": {"fov_min": fov_lo, "fov_max": fov_hi,
                            "fnum_min": fnum_lo, "fnum_max": fnum_hi},
            "ground_truth": _gt(best),
            "eval_config": {
                "has_fov": True, "has_fnum": True, "has_aper": False,
                "tol_fov": (fov_hi-fov_lo)/2,
                "tol_fnum": (fnum_hi-fnum_lo)/2,
                "tol_aper": 999,
                "fov_range": [fov_lo, fov_hi],
                "fnum_range": [fnum_lo, fnum_hi],
            },
        })
    return items[:n]

# ── E. partial ────────────────────────────────────────────────────────────────
def gen_E(lenses, n=40):
    pool = [l for l in lenses if _valid(l)]
    items, attempts = [], 0
    while len(items) < n and attempts < 5000:
        attempts += 1
        lens = random.choice(pool)
        fov, fnum, aper = lens["fov"], lens["fnum"], lens["aper"]
        mode = random.choice(["fov_only", "fnum_only", "fov_fnum", "fov_aper"])

        if mode == "fov_only":
            query = random.choice([
                f"FOV={fov}度的镜头，其他参数不限，找 RMS 最小的",
                f"视场角 {fov}° 的镜头有哪些推荐",
                f"全视场 {fov} 度的镜头设计方案",
            ])
            constraints = {"fov": fov}
            cfg = {"has_fov": True, "has_fnum": False, "has_aper": False,
                   "tol_fov": 1.0, "tol_fnum": 999, "tol_aper": 999}
        elif mode == "fnum_only":
            query = random.choice([
                f"F/{fnum} 的镜头，视场和口径不限，RMS 越小越好",
                f"光圈 F{fnum} 的镜头方案推荐",
                f"Fno={fnum} 有哪些可用镜头",
            ])
            constraints = {"fnum": fnum}
            cfg = {"has_fov": False, "has_fnum": True, "has_aper": False,
                   "tol_fov": 999, "tol_fnum": 0.1, "tol_aper": 999}
        elif mode == "fov_fnum":
            query = random.choice([
                f"FOV={fov}度、F/{fnum} 的镜头，口径不限，RMS 尽量小",
                f"视场角 {fov}° 光圈 F{fnum} 的方案有哪些",
                f"找 FOV {fov}度 F/{fnum} 性能最好的镜头",
            ])
            constraints = {"fov": fov, "fnum": fnum}
            cfg = {"has_fov": True, "has_fnum": True, "has_aper": False,
                   "tol_fov": 1.5, "tol_fnum": 0.2, "tol_aper": 999}
        else:
            query = random.choice([
                f"FOV={fov}度、入瞳径{aper}mm 的镜头，光圈不限，RMS 越小越好",
                f"视场角 {fov}° 口径 {aper}mm 的镜头方案",
                f"FOV {fov}度 通光口径 {aper}mm 有哪些推荐",
            ])
            constraints = {"fov": fov, "aper": aper}
            cfg = {"has_fov": True, "has_fnum": False, "has_aper": True,
                   "tol_fov": 1.5, "tol_fnum": 999, "tol_aper": 1.5}

        items.append({
            "id": f"E_{len(items):03d}", "type": "partial",
            "query": query,
            "constraints": constraints,
            "ground_truth": _gt(lens),
            "eval_config": cfg,
        })
    return items[:n]

# ══════════════════════════════════════════════════════════════════════════════
# 评分函数（供 eval.py 调用）
# ══════════════════════════════════════════════════════════════════════════════
def score_response(item: dict, agent_response: dict) -> dict:
    """
    三层评分：
      i.   hit            —— 与用户输入相符程度
      ii.  norm_error_sum —— 各指标误差归一化加权和
      iii. final_score    —— norm_error_sum + rms/RMS_SCALE
    """
    cfg  = item.get("eval_config", {})
    cons = item.get("constraints", {})

    t_fov  = cons.get("fov")  or cons.get("fov_min")
    t_fnum = cons.get("fnum") or cons.get("fnum_min")
    t_aper = cons.get("aper")

    has_fov  = cfg.get("has_fov",  bool(t_fov))
    has_fnum = cfg.get("has_fnum", bool(t_fnum))
    has_aper = cfg.get("has_aper", bool(t_aper))
    tol_fov  = cfg.get("tol_fov",  2.0)
    tol_fnum = cfg.get("tol_fnum", 0.3)
    tol_aper = cfg.get("tol_aper", 2.0)

    retrieved = agent_response.get("retrieved_lenses", [])
    if not retrieved:
        return {"hit": False, "fov_rel_err": 1.0, "fnum_rel_err": 1.0,
                "aper_rel_err": 1.0, "norm_error_sum": 1.0,
                "final_score": 1.0 + 999/RMS_SCALE, "rms": 999.0}

    best = min(retrieved, key=lambda x: x.get("calc_rms") or 999)

    def rel_err(pred, target):
        if not target:
            return 0.0
        return abs((pred or 0) - target) / abs(target)

    fov_err  = rel_err(best.get("fov"),  t_fov)  if has_fov  else 0.0
    fnum_err = rel_err(best.get("fnum"), t_fnum) if has_fnum else 0.0
    aper_err = rel_err(best.get("aper"), t_aper) if has_aper else 0.0
    rms      = best.get("calc_rms") or 999.0

    norm_sum = W_FOV * fov_err + W_FNUM * fnum_err + W_APER * aper_err
    final    = norm_sum + rms / RMS_SCALE

    hit = True
    if has_fov  and t_fov  is not None:
        hit &= abs((best.get("fov")  or 999) - t_fov)  <= tol_fov
    if has_fnum and t_fnum is not None:
        hit &= abs((best.get("fnum") or 999) - t_fnum) <= tol_fnum
    if has_aper and t_aper is not None:
        hit &= abs((best.get("aper") or 999) - t_aper) <= tol_aper

    return {
        "hit":            hit,
        "fov_rel_err":    round(fov_err,  4),
        "fnum_rel_err":   round(fnum_err, 4),
        "aper_rel_err":   round(aper_err, 4),
        "norm_error_sum": round(norm_sum, 4),
        "final_score":    round(final,    4),
        "rms":            round(rms,      6),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("加载镜头库…")
    lenses = load_lenses()
    valid  = [l for l in lenses if _valid(l)]
    print(f"共 {len(lenses)} 条，追迹成功 {len(valid)} 条")

    fovs, fnums, apers, _ = get_param_sets(valid)
    print(f"FOV:  {min(fovs):.1f}~{max(fovs):.1f}度  ({len(fovs)} 种)")
    print(f"Fno:  {min(fnums):.2f}~{max(fnums):.2f}  ({len(fnums)} 种)")
    print(f"Aper: {min(apers):.1f}~{max(apers):.1f}mm ({len(apers)} 种)")

    print("\n生成 test set…")
    testset  = []
    testset += gen_A(lenses, n=40)
    testset += gen_B(lenses, n=40)
    testset += gen_C(lenses, n=40)
    testset += gen_D(lenses, n=40)
    testset += gen_E(lenses, n=40)

    random.shuffle(testset)
    for i, item in enumerate(testset):
        item["global_id"] = i

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)

    cnt = Counter(item["type"] for item in testset)
    print(f"\n✅ {OUTPUT}  共 {len(testset)} 条")
    for t, n in sorted(cnt.items()):
        print(f"   {t:12s}: {n:3d} 条")

    print("\n── 样例 ──")
    for t in ["in_domain","out_of_domain","specific","range","partial"]:
        ex = next((x for x in testset if x["type"] == t), None)
        if ex:
            print(f"\n[{t}] {ex['query']}")
            print(f"  constraints: {ex['constraints']}")
