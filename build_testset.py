import pickle, random, json
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)

with open("/gz-data/faiss_index/lenses.pkl", "rb") as f:
    ALL_LENSES = pickle.load(f)
print(f"总镜头数: {len(ALL_LENSES)}")

# ── 实际参数值 ────────────────────────────────────────────────────────────
APER_VALS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]   # 数据库仅有这6种

fov_vals  = sorted(set(l["fov"]  for l in ALL_LENSES if l.get("fov")  is not None))
fnum_vals = sorted(set(round(l["fnum"], 1) for l in ALL_LENSES if l.get("fnum") is not None))

print(f"FOV  范围: {min(fov_vals)}° ~ {max(fov_vals)}°, 共{len(fov_vals)}种")
print(f"Fnum 范围: F/{min(fnum_vals)} ~ F/{max(fnum_vals)}, 共{len(fnum_vals)}种")
print(f"Aper 可选: {APER_VALS}")

combo_index = defaultdict(list)
for i, l in enumerate(ALL_LENSES):
    key = (l.get("fov"), round(l["fnum"], 1) if l.get("fnum") else None)
    combo_index[key].append(i)
in_domain_combos = [(k, v) for k, v in combo_index.items() if None not in k]

# ── 设计口吻模板 ──────────────────────────────────────────────────────────
def design_query(fov, fnum, aper, rms_req=None):
    fnum = round(fnum, 1)
    templates = [
        f"帮我设计一个FOV={fov}度、F/{fnum}、口径{aper}mm的镜头",
        f"我需要设计一款视场角{fov}度、光圈F/{fnum}、入瞳口径{aper}mm的镜头",
        f"设计一个FOV={fov}度、F数为{fnum}、通光口径{aper}mm的光学系统",
        f"请帮我设计FOV={fov}度 F/{fnum} 口径{aper}mm的镜头方案",
        f"我要设计一款{fov}度视场、F/{fnum}、口径{aper}mm的镜头",
    ]
    q = random.choice(templates)
    if rms_req:
        q += f"，要求RMS小于{rms_req}mm"
    return q

def design_query_range(fov_lo, fov_hi, fnum, aper):
    fnum = round(fnum, 1)
    templates = [
        f"帮我设计一个视场角在{fov_lo}到{fov_hi}度之间、F/{fnum}、口径{aper}mm的镜头",
        f"设计FOV={fov_lo}~{fov_hi}度、F/{fnum}的镜头，入瞳口径{aper}mm",
        f"我需要一款{fov_lo}-{fov_hi}度视场角、光圈F/{fnum}的镜头方案，口径{aper}mm",
    ]
    return random.choice(templates)

def design_query_single(param, val):
    val = round(val, 1) if isinstance(val, float) else val
    templates = {
        "fov":  [
            f"帮我设计一个视场角{val}度的镜头",
            f"我需要设计一款FOV={val}度的光学系统",
            f"设计一个{val}度视场的镜头，成像质量尽量好",
        ],
        "fnum": [
            f"帮我设计一个F/{val}的镜头",
            f"我需要一款F数为{val}的镜头方案",
            f"设计一个光圈F/{val}的光学系统，RMS越小越好",
        ],
        "aper": [
            f"帮我设计一个入瞳口径{val}mm的镜头",
            f"设计一款通光口径{val}mm的光学系统",
            f"我需要口径{val}mm的镜头方案",
        ],
        "semantic": [
            "帮我设计一款大视场镜头",
            "我需要设计一个小F数大光圈镜头",
            "帮我设计一款长焦镜头，视场角尽量小",
            "设计一个高分辨率小视场镜头",
        ],
    }
    return random.choice(templates.get(param, templates["semantic"]))

# ── 1. in-domain（60条）────────────────────────────────────────────────────
def make_in_domain(n=60):
    samples = random.sample(in_domain_combos, min(n, len(in_domain_combos)))
    items = []
    for (fov, fnum), lens_ids in samples:
        aper = random.choice(APER_VALS)
        items.append({
            "id":    len(items) + 1,
            "type":  "in-domain",
            "query": design_query(fov, fnum, aper),
            "ground_truth": {
                "fov":  fov,
                "fnum": round(fnum, 1),
                "aper": aper,
                "candidate_lens_ids": lens_ids[:10],
            },
            "note": "精确参数，数据库中存在"
        })
    return items

# ── 2. out-of-domain（40条）──────────────────────────────────────────────
def make_out_of_domain(n=40):
    existing = set(combo_index.keys())
    ood = []
    for fov in [3, 7, 8, 12, 17, 22, 27, 37, 42, 47, 60, 70, 80, 90]:
        for fnum in [1.2, 1.5, 3.5, 7.0, 8.0, 11.0, 14.0, 16.0]:
            fnum_r = round(fnum, 1)
            if (fov, fnum_r) not in existing:
                ood.append((fov, fnum_r))
    samples = random.sample(ood, min(n, len(ood)))
    items = []
    for fov, fnum in samples:
        aper = random.choice(APER_VALS)
        items.append({
            "id":    len(items) + 1,
            "type":  "out-of-domain",
            "query": design_query(fov, fnum, aper),
            "ground_truth": {
                "fov":  fov,
                "fnum": fnum,
                "aper": aper,
                "candidate_lens_ids": [],
                "nearest_fov":  min(fov_vals,  key=lambda x: abs(x - fov)),
                "nearest_fnum": min(fnum_vals, key=lambda x: abs(x - fnum)),
            },
            "note": "参数不在数据库中，期望返回最近邻并说明"
        })
    return items

# ── 3. 范围需求（40条）───────────────────────────────────────────────────
def make_range_query(n=40):
    items = []
    for _ in range(n):
        fov   = random.choice(fov_vals)
        fnum  = round(random.choice(fnum_vals), 1)
        aper  = random.choice(APER_VALS)
        delta = random.choice([5, 10, 15])
        fov_lo = max(1, fov - delta)
        fov_hi = fov + delta
        valid_ids = [i for i, l in enumerate(ALL_LENSES)
                     if l.get("fov") is not None and fov_lo <= l["fov"] <= fov_hi][:20]
        items.append({
            "id":    len(items) + 1,
            "type":  "范围需求",
            "query": design_query_range(fov_lo, fov_hi, fnum, aper),
            "ground_truth": {
                "fov_range": [fov_lo, fov_hi],
                "fnum":      fnum,
                "aper":      aper,
                "valid_lens_ids": valid_ids,
            },
            "note": f"FOV范围 [{fov_lo}, {fov_hi}]"
        })
    return items

# ── 4. 具体需求（40条）───────────────────────────────────────────────────
def make_specific(n=40):
    items = []
    for _ in range(n):
        lens = ALL_LENSES[random.randint(0, len(ALL_LENSES)-1)]
        fov  = lens.get("fov",  30)
        fnum = round(lens.get("fnum", 2.8), 1)
        aper = random.choice(APER_VALS)
        rms_thr = random.choice([0.05, 0.1, 0.15, 0.3])
        best = sorted(
            [l for l in ALL_LENSES
             if l.get("fov") == fov and round(l.get("fnum",0),1) == fnum
             and l.get("calc_rms") is not None],
            key=lambda l: l["calc_rms"]
        )[:5]
        items.append({
            "id":    len(items) + 1,
            "type":  "具体需求",
            "query": design_query(fov, fnum, aper, rms_req=rms_thr),
            "ground_truth": {
                "fov": fov, "fnum": fnum, "aper": aper,
                "rms_threshold": rms_thr,
                "best_rms_lenses": [ALL_LENSES.index(l) for l in best],
            },
            "note": "多参数 + RMS约束"
        })
    return items

# ── 5. 缺省需求（20条）───────────────────────────────────────────────────
def make_default(n=20):
    items = []
    param_choices = (
        [("fov",  v) for v in random.sample(fov_vals, 6)] +
        [("fnum", v) for v in random.sample(fnum_vals, 6)] +
        [("aper", v) for v in APER_VALS] +
        [("semantic", None)] * 2
    )
    random.shuffle(param_choices)
    for param, val in param_choices[:n]:
        q = design_query_single(param, val) if val else design_query_single("semantic", None)
        items.append({
            "id":   len(items) + 1,
            "type": "缺省需求",
            "query": q,
            "ground_truth": {
                "known_param": param,
                "known_value": round(val, 1) if isinstance(val, float) else val,
            },
            "note": "单参数或语义查询"
        })
    return items

# ── 合并 + 打乱 + 重新编号 ────────────────────────────────────────────────
all_items = (
    make_in_domain(60) +
    make_out_of_domain(40) +
    make_range_query(40) +
    make_specific(40) +
    make_default(20)
)
random.shuffle(all_items)
for i, item in enumerate(all_items):
    item["id"] = i + 1

out = Path("/gz-data/testset_200.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(all_items, f, ensure_ascii=False, indent=2)

from collections import Counter
type_cnt = Counter(item["type"] for item in all_items)
print(f"\n✅ 测试集已保存: {out}")
print(f"总数: {len(all_items)} 条")
for t, c in type_cnt.items():
    print(f"  {t}: {c} 条")

# 打印样例
print("\n=== 各类型样例 ===")
shown = set()
for item in all_items:
    if item["type"] not in shown:
        print(f"[{item['type']}] {item['query']}")
        shown.add(item["type"])
    if len(shown) == 5:
        break

