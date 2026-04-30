

import argparse, json, math, random, os, glob, time, sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. Benchmark 用例生成
#    30% in-domain:  FOV ∈ [25,70]°, F# ∈ [1.8,4.0]  （训练库常见范围）
#    70% OOD:        FOV 或 F# 超出以上范围
#    包含: 纯数值描述、纯文字描述（fuzzy）、带 y_target 的焦距指定、多镜片要求
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)

def _rms_budget(fov: float, fnum: float) -> float:
    """按经验给出合理的 RMS 目标（mm）。F# 小/FOV 大 → 放松目标"""
    base = 0.04
    if fnum <= 1.4:   base = 0.08
    elif fnum <= 2.0: base = 0.05
    if fov >= 70:     base = max(base, 0.06)
    elif fov >= 50:   base = max(base, 0.04)
    return round(base, 3)

# In-domain 范围（与训练库重叠）
_ID_FOV   = (25, 70)
_ID_FNUM  = (1.8, 4.0)

# OOD 扩展范围
_OOD_CONFIGS = [
    # 极大 FOV
    {"fov_range": (80, 120), "fnum_range": (2.0, 5.6), "tag": "wide_fov"},
    # 极小 FOV（长焦）
    {"fov_range": (3, 20),   "fnum_range": (4.0, 8.0), "tag": "tele"},
    # 极大光圈
    {"fov_range": (20, 50),  "fnum_range": (0.9, 1.4), "tag": "fast_lens"},
    # 极小 F# + 宽 FOV
    {"fov_range": (50, 90),  "fnum_range": (1.2, 1.8), "tag": "fast_wide"},
    # 超长焦
    {"fov_range": (1, 5),    "fnum_range": (8.0, 16.0), "tag": "super_tele"},
    # 宽 FOV 大光圈（手机镜头风格）
    {"fov_range": (70, 90),  "fnum_range": (1.5, 2.2), "tag": "mobile_style"},
    # 医疗/内窥（小 FOV，小 F#，短总长）
    {"fov_range": (55, 75),  "fnum_range": (1.6, 2.4), "tag": "endoscope"},
]

# 文字描述模板（fuzzy query，缺省参数由 agent 补全）
_FUZZY_TEMPLATES = [
    "设计一个天文摄影镜头，要求色差小，像质好",
    "设计一个用于安防监控的广角镜头，FOV尽量大",
    "需要一个手机摄像头镜头，要超薄，尽量小的RMS",
    "设计一个人像摄影镜头，大光圈，背景虚化好",
    "需要一个工业检测用定焦镜头，畸变小，FOV 30度",
    "设计一个车载摄像头广角镜头，FOV至少100度",
    "需要一个显微镜物镜，高倍率，成像清晰",
    "设计一套用于无人机的轻型光学系统，宽视角",
]

def _gen_numeric_case(fov, fnum, tag, effl=None, n_elem=None, rms_tgt=None):
    """生成一条数值描述的测试用例"""
    rms = rms_tgt or _rms_budget(fov, fnum)
    q = f"设计FOV={fov:.0f}度 F/{fnum:.1f}的镜头，要求RMS < {rms}mm"
    if effl:
        q += f"，有效焦距约{effl:.0f}mm"
    if n_elem:
        q += f"，使用{n_elem}片镜片"
    return {
        "id": None,  # 填写时赋值
        "tag": tag,
        "query": q,
        "target": {"fov": fov, "fnum": fnum, "rms_target": rms,
                   "effl": effl, "n_elements": n_elem},
        "split": None,  # in_domain / ood
    }


def generate_benchmark(n_total=100, out_path="/gz-data/benchmark_100.json"):
    n_id  = int(n_total * 0.30)  # 30 in-domain
    n_ood = n_total - n_id       # 70 OOD

    cases = []

    # ── In-domain ──
    for _ in range(n_id):
        fov  = round(random.uniform(*_ID_FOV), 1)
        fnum = round(random.choice([1.8, 2.0, 2.4, 2.8, 3.5, 4.0]), 1)
        effl = None
        if random.random() < 0.3:
            effl = round(random.uniform(15, 80), 1)
        n_elem = None
        if random.random() < 0.25:
            n_elem = random.choice([4, 5, 6])
        c = _gen_numeric_case(fov, fnum, "in_domain", effl, n_elem)
        c["split"] = "in_domain"
        cases.append(c)

    # ── OOD numeric ──
    n_ood_numeric = n_ood - len(_FUZZY_TEMPLATES)
    per_cfg = max(1, n_ood_numeric // len(_OOD_CONFIGS))
    for cfg in _OOD_CONFIGS:
        for _ in range(per_cfg):
            fov  = round(random.uniform(*cfg["fov_range"]), 1)
            fnum = round(random.uniform(*cfg["fnum_range"]) * 2) / 2  # 步进 0.5
            fnum = max(0.9, fnum)
            c = _gen_numeric_case(fov, fnum, cfg["tag"])
            c["split"] = "ood"
            cases.append(c)

    # ── OOD fuzzy ──
    for tmpl in _FUZZY_TEMPLATES:
        cases.append({
            "id": None,
            "tag": "fuzzy",
            "query": tmpl,
            "target": {},   # fuzzy: 目标由 agent 补全
            "split": "ood",
        })

    # 截断 / 补到 n_total
    random.shuffle(cases)
    cases = cases[:n_total]

    # 编号
    for i, c in enumerate(cases):
        c["id"] = f"BM{i+1:03d}"

    # 统计
    n_id_actual  = sum(1 for c in cases if c["split"] == "in_domain")
    n_ood_actual = sum(1 for c in cases if c["split"] == "ood")
    print(f"生成 {len(cases)} 条：in-domain={n_id_actual}  OOD={n_ood_actual}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"✓ 写入 {out_path}")
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# 2. Eval 跑现有 20 条测试集
#    · 调用 run_agent(query)
#    · 收集: pass/fail, RMS, EFFL, F#, TOTR, layout.png 路径, zmx 路径
#    · 写 CSV + JSON 汇总报告
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(cases_path: str, session_dir: str, dry_run: bool = False):
    """
    对 cases_path 里的每条用例运行 run_agent 并汇总指标。

    session_dir: 所有输出文件（layout_*.png, snapshots_*.json, zmx）的根目录。
    dry_run: True 时只打印 query，不真正调用 agent（测试用）。
    """
    import csv

    sys.path.insert(0, "/gz-data")
    import agent_zemax as az

    # 设置 session 目录，让 agent 的快照和图片落到正确位置
    az._SESSION_DIR = session_dir
    os.makedirs(session_dir, exist_ok=True)

    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    csv_rows = []

    for idx, case in enumerate(cases):
        cid   = case.get("id", f"C{idx+1:03d}")
        query = case["query"]
        tag   = case.get("tag", "?")
        split = case.get("split", "?")
        tgt   = case.get("target", {})

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(cases)}] {cid} ({tag}/{split})")
        print(f"Query: {query}")

        if dry_run:
            print("  [DRY RUN] 跳过")
            continue

        # 清理每轮状态
        az._SEARCH_COUNT.clear()
        az._MODIFY_COUNT.clear()
        az._OPTIMIZE_STALL.clear()
        az._LENS_BACKUP.clear()
        az._LENS_BACKUP_PRE_ZEMAX.clear()
        az._PHASE_SNAPSHOTS.clear()
        az._INTERPRET_CALLED = False
        az._INTERPRET_RESULT = {}

        t0 = time.time()
        try:
            output = az.run_agent(query)
        except Exception as e:
            output = f"CRASH: {e}"
        elapsed = round(time.time() - t0, 1)

        # ── 从 output 提取 Final Answer 字段 ──
        import re
        fa_match = re.search(
            r'镜头#(\d+)\s*\|\s*FOV=([^\s|]+)\s*\|\s*F/([^\s|]+)\s*\|\s*RMS=([^\s|]+)',
            output or "")
        fa_lens  = fa_match.group(1) if fa_match else None
        fa_fov   = fa_match.group(2) if fa_match else None
        fa_fnum  = fa_match.group(3) if fa_match else None
        fa_rms   = fa_match.group(4) if fa_match else None
        passed   = "达标✓" in (output or "")

        # ── 找生成的图片和 zmx ──
        layout_files = sorted(glob.glob(f"{session_dir}/layout_*.png"))
        spot_files   = sorted(glob.glob(f"{session_dir}/spot_*.png"))
        zmx_files    = sorted(glob.glob(f"C:\\zemax_results\\*.zmx")) if os.name == "nt" else []
        snap_files   = sorted(glob.glob(f"{session_dir}/snapshots_*.json"))

        row = {
            "id": cid, "tag": tag, "split": split,
            "query": query[:80],
            "passed": passed,
            "lens_idx": fa_lens,
            "final_fov": fa_fov,
            "final_fnum": fa_fnum,
            "final_rms": fa_rms,
            "target_rms": tgt.get("rms_target"),
            "elapsed_s": elapsed,
            "layout_png": layout_files[-1] if layout_files else None,
            "spot_png":   spot_files[-1]   if spot_files else None,
            "zmx":        zmx_files[-1]    if zmx_files else None,
            "snapshots":  snap_files[-1]   if snap_files else None,
            "output_tail": (output or "")[-200:],
        }
        results.append(row)
        csv_rows.append(row)

        # 打印摘要
        _pass_str = "✅ 达标" if passed else "❌ 未达标"
        print(f"  {_pass_str}  RMS={fa_rms}  elapsed={elapsed}s")
        if row["layout_png"]:
            print(f"  Layout: {row['layout_png']}")
        if row["zmx"]:
            print(f"  ZMX:    {row['zmx']}")

    # ── 写 JSON ──
    summary_json = f"{session_dir}/eval_results.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ JSON 汇总: {summary_json}")

    # ── 写 CSV ──
    if results:
        summary_csv = f"{session_dir}/eval_results.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"✓ CSV  汇总: {summary_csv}")

    # ── 打印统计 ──
    n_pass = sum(1 for r in results if r["passed"])
    n_tot  = len(results)
    n_id_p = sum(1 for r in results if r["split"] == "in_domain" and r["passed"])
    n_id   = sum(1 for r in results if r["split"] == "in_domain")
    n_ood_p= sum(1 for r in results if r["split"] == "ood"       and r["passed"])
    n_ood  = sum(1 for r in results if r["split"] == "ood")
    print(f"\n{'='*40}")
    print(f"总达标率:      {n_pass}/{n_tot}  ({100*n_pass/max(n_tot,1):.1f}%)")
    if n_id:
        print(f"In-domain 率: {n_id_p}/{n_id}   ({100*n_id_p/n_id:.1f}%)")
    if n_ood:
        print(f"OOD 达标率:   {n_ood_p}/{n_ood}  ({100*n_ood_p/max(n_ood,1):.1f}%)")

    # ── SCP 命令提示 ──
    print(f"\n  scp root@<server_ip>:{session_dir}/*.png .")
    print(f"  scp root@<server_ip>:{summary_csv} .")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen",  action="store_true", help="生成 benchmark")
    ap.add_argument("--eval", action="store_true", help="跑 eval")
    ap.add_argument("--n",    type=int, default=100, help="benchmark 用例数")
    ap.add_argument("--out",  default="/gz-data/benchmark_100.json")
    ap.add_argument("--cases",default="/gz-data/ood_20cases.json")
    ap.add_argument("--session_dir", default="/gz-data/results/eval_run")
    ap.add_argument("--dry_run", action="store_true", help="只打印 query，不调 agent")
    args = ap.parse_args()

    if args.gen:
        generate_benchmark(n_total=args.n, out_path=args.out)

    if args.eval:
        run_eval(args.cases, args.session_dir, dry_run=args.dry_run)

    if not args.gen and not args.eval:
        ap.print_help()
