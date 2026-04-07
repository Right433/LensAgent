"""
build_testset.py  —  构建 200 条评估 test set
输出: /gz-data/testset.json

Question 类型（共 4 类，各 50 条）：
  A. in-domain     —— 数据库里有完全匹配的镜头（FOV/Fno 精确命中）
  B. out-of-domain —— 参数组合数据库没有，考验泛化
  C. 范围需求       —— "FOV 在 30~40 度之间，RMS 尽量小"
  D. 缺省需求       —— 只给部分参数，其余不限

评价指标（每条 query 附带 ground truth）：
  i.  与用户输入的相符程度（FOV误差、Fno误差）
  ii. 各指标误差归一化后求和
  iii.误差总和 + RMS（越小越好）
"""

import json, random, pickle
from pathlib import Path
from collections import Counter

FAISS_DIR = "/gz-data/faiss_index"
OUTPUT    = "/gz-data/testset.json"
random.seed(42)

# ── 加载镜头库 ────────────────────────────────────────────────────────────────
def load_lenses():
    pkl = Path(FAISS_DIR) / "lenses.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"找不到 {pkl}，请先运行 build_rag.py")
    with open(pkl, "rb") as f:
        return pickle.load(f)

def _make_ground_truth(lens):
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

# ── A. in-domain ──────────────────────────────────────────────────────────────
def gen_A_in_domain(lenses, n=50):
    samples = random.sample(lenses, min(n, len(lenses)))
    items = []
    for lens in samples:
        fov  = lens.get("fov")
        fnum = lens.get("fnum")
        aper = lens.get("aper")
        rms  = lens.get("calc_rms")
        templates = [
            f"我需要一个 FOV={fov}度、F/{fnum} 的镜头，RMS 尽量小",
            f"找 FOV {fov} 度、光圈 F/{fnum}、入瞳径 {aper}mm 的镜头",
            f"有没有视场角 {fov}°、F 数为 {fnum} 的镜头方案",
            f"查找 FOV={fov}度 Fno={fnum} 近轴 RMS 最优的设计",
            f"需要全视场 {fov} 度、通光口径 {aper}mm、F/{fnum} 的镜头",
        ]
        items.append({
            "id":           f"A_{len(items):03d}",
            "type":         "in_domain",
            "query":        random.choice(templates),
            "constraints":  {"fov": fov, "fnum": fnum, "aper": aper},
            "ground_truth": _make_ground_truth(lens),
            "eval_metric":  {"fov_tolerance": 2.0, "fnum_tolerance": 0.2, "rms_ref": rms},
        })
    return items

# ── B. out-of-domain ──────────────────────────────────────────────────────────
def gen_B_out_of_domain(lenses, n=50):
    fovs  = [20, 25, 28, 30, 40, 45, 50, 60, 70, 90]
    fnums = [1.4, 1.8, 2.0, 2.5, 3.5, 4.0, 5.6, 8.0]
    apers = [6, 8, 10, 15, 20, 25, 30]
    existing = {(l.get("fov"), l.get("fnum")) for l in lenses}

    items, attempts = [], 0
    while len(items) < n and attempts < 5000:
        attempts += 1
        fov  = random.choice(fovs)
        fnum = random.choice(fnums)
        aper = random.choice(apers)
        if (fov, fnum) in existing:
            continue
        templates = [
            f"需要 FOV={fov}度、F/{fnum} 的镜头",
            f"找视场角 {fov} 度、光圈 F{fnum} 的设计方案",
            f"有没有 FOV {fov}°、入瞳径 {aper}mm、Fno={fnum} 的镜头",
            f"FOV={fov}度 F/{fnum} 口径 {aper}mm 镜头推荐",
        ]
        items.append({
            "id":           f"B_{len(items):03d}",
            "type":         "out_of_domain",
            "query":        random.choice(templates),
            "constraints":  {"fov": fov, "fnum": fnum, "aper": aper},
            "ground_truth": None,
            "eval_metric":  {"fov_tolerance": 3.0, "fnum_tolerance": 0.3, "rms_ref": None},
        })
    return items

# ── C. 范围需求 ───────────────────────────────────────────────────────────────
def gen_C_range(lenses, n=50):
    items = []
    while len(items) < n:
        sample = random.sample(lenses, 5)
        fovs  = sorted([l.get("fov")  for l in sample if l.get("fov")  is not None])
        fnums = sorted([l.get("fnum") for l in sample if l.get("fnum") is not None])
        if not fovs or not fnums or fovs[0] == fovs[-1]:
            continue
        fov_lo, fov_hi   = fovs[0],  fovs[-1]
        fnum_lo, fnum_hi = fnums[0], fnums[-1]
        templates = [
            f"FOV 在 {fov_lo}~{fov_hi} 度之间，F/{fnum_lo} 到 F/{fnum_hi}，RMS 尽量小",
            f"视场角 {fov_lo} 到 {fov_hi} 度，光圈 F{fnum_lo}~F{fnum_hi}，找性能最好的",
            f"需要 FOV {fov_lo}°~{fov_hi}°、Fno {fnum_lo}~{fnum_hi} 的镜头，优先 RMS 小",
            f"FOV={fov_lo}-{fov_hi}度 F数{fnum_lo}-{fnum_hi} 列出 RMS 最小的方案",
        ]
        items.append({
            "id":           f"C_{len(items):03d}",
            "type":         "range",
            "query":        random.choice(templates),
            "constraints":  {"fov_min": fov_lo, "fov_max": fov_hi,
                             "fnum_min": fnum_lo, "fnum_max": fnum_hi},
            "ground_truth": None,
            "eval_metric":  {"fov_range": [fov_lo, fov_hi],
                             "fnum_range": [fnum_lo, fnum_hi], "rms_ref": None},
        })
    return items[:n]

# ── D. 缺省需求 ───────────────────────────────────────────────────────────────
def gen_D_partial(lenses, n=50):
    items = []
    while len(items) < n:
        lens = random.choice(lenses)
        fov  = lens.get("fov")
        fnum = lens.get("fnum")
        aper = lens.get("aper")
        rms  = lens.get("calc_rms")
        if fov is None:
            continue

        drop = random.sample(["fnum", "aper"], k=random.randint(1, 2))

        if "fnum" in drop and "aper" not in drop:
            query = random.choice([
                f"FOV={fov}度、入瞳径{aper}mm 的镜头，光圈不限，RMS 越小越好",
                f"视场角 {fov}° 口径 {aper}mm 的镜头方案有哪些",
            ])
            constraints = {"fov": fov, "aper": aper}
        elif "aper" in drop and "fnum" not in drop:
            query = random.choice([
                f"FOV={fov}度、F/{fnum} 的镜头，口径不限",
                f"视场角 {fov}° 光圈 F{fnum} 的方案，RMS 尽量小",
            ])
            constraints = {"fov": fov, "fnum": fnum}
        else:
            query = random.choice([
                f"FOV={fov}度的镜头，其他参数不限，找 RMS 最小的",
                f"视场角 {fov}° 的镜头设计方案",
                f"全视场 {fov} 度的镜头有哪些推荐",
            ])
            constraints = {"fov": fov}

        items.append({
            "id":           f"D_{len(items):03d}",
            "type":         "partial",
            "query":        query,
            "constraints":  constraints,
            "ground_truth": _make_ground_truth(lens),
            "eval_metric":  {"fov_tolerance": 3.0, "rms_ref": rms},
        })
    return items[:n]

# ── 评估函数（供 eval.py 调用）────────────────────────────────────────────────
def score_response(item: dict, agent_response: dict) -> dict:
    """
    对 Agent 返回的结果打分。

    agent_response 格式:
        {
          "retrieved_lenses": [
              {"lens_idx": int, "fov": float, "fnum": float, "calc_rms": float},
              ...
          ],
          "answer": str   # Agent 最终输出文本
        }

    返回:
        {
          "fov_error":  float,  # 相对误差
          "fnum_error": float,  # 相对误差
          "rms":        float,  # 推荐镜头的近轴 RMS
          "norm_sum":   float,  # fov_err + fnum_err + rms（越小越好）
          "hit":        bool,   # 是否在容差范围内命中
        }
    """
    c        = item.get("constraints", {})
    t_fov    = c.get("fov")
    t_fnum   = c.get("fnum")
    tol_fov  = item["eval_metric"].get("fov_tolerance",  2.0)
    tol_fnum = item["eval_metric"].get("fnum_tolerance", 0.3)

    retrieved = agent_response.get("retrieved_lenses", [])
    if not retrieved:
        return {"fov_error": 999, "fnum_error": 999, "rms": 999,
                "norm_sum": 999, "hit": False}

    best = min(retrieved, key=lambda x: x.get("calc_rms") or 999)

    fov_err  = abs(best.get("fov",  0) - t_fov)  / max(t_fov,  1) if t_fov  else 0.0
    fnum_err = abs(best.get("fnum", 0) - t_fnum) / max(t_fnum, 1) if t_fnum else 0.0
    rms      = best.get("calc_rms") or 999.0
    norm_sum = fov_err + fnum_err + rms

    hit = (
        (t_fov  is None or abs(best.get("fov",  999) - t_fov)  <= tol_fov) and
        (t_fnum is None or abs(best.get("fnum", 999) - t_fnum) <= tol_fnum)
    )

    return {
        "fov_error":  round(fov_err,  4),
        "fnum_error": round(fnum_err, 4),
        "rms":        round(rms,      6),
        "norm_sum":   round(norm_sum, 4),
        "hit":        hit,
    }

# ── 主程序 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("加载镜头库…")
    lenses = load_lenses()
    for i, l in enumerate(lenses):
        l["lens_idx"] = i
    print(f"共 {len(lenses)} 条镜头")

    testset  = []
    testset += gen_A_in_domain(lenses,     n=50)
    testset += gen_B_out_of_domain(lenses, n=50)
    testset += gen_C_range(lenses,         n=50)
    testset += gen_D_partial(lenses,       n=50)

    random.shuffle(testset)
    for i, item in enumerate(testset):
        item["global_id"] = i

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ test set 已生成: {OUTPUT}  共 {len(testset)} 条")
    cnt = Counter(item["type"] for item in testset)
    for t, n in sorted(cnt.items()):
        print(f"   {t:15s}: {n} 条")
