cat > /gz-data/eval.py << 'PYEOF'
#!/usr/bin/env python3
"""
eval.py
    python eval.py                              # 全量
    python eval.py --n 5                        # 前 5 条
    python eval.py --resume                     # 断点续跑
    python eval.py --plot-only                  # 只重画图
    python eval.py --shard 0 --total-shards 4  # 并发分片
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np

DATA_DIR   = Path("/gz-data")
TESTSET    = DATA_DIR / "testset.json"
RESULT_F   = DATA_DIR / "eval_results.jsonl"
SUMMARY_F  = DATA_DIR / "eval_summary.json"
REPORT_DIR = DATA_DIR / "eval_report"
REPORT_DIR.mkdir(exist_ok=True)

FOV_RANGE  = (4.0,  59.0)
FNUM_RANGE = (1.6,  6.0)
APER_RANGE = (2.0,  12.0)
RMS_MAX    = 0.5
TOL_FOV, TOL_FNUM, TOL_APER = 5.0, 0.5, 3.0

from agent import run_agent

def parse_prediction(text: str) -> dict:
    fa = re.search(r"Final Answer[::](.*)", text, re.DOTALL | re.IGNORECASE)
    t = fa.group(1) if fa else text
    pred = {}
    fov_all = re.findall(r"FOV[=:][^0-9]*([0-9]+\.?[0-9]*)", t, re.IGNORECASE)
    if fov_all:
        pred["fov"] = float(fov_all[-1])
    fn_all = re.findall(r"F/([0-9]+\.?[0-9]*)", t, re.IGNORECASE)
    if fn_all:
        pred["fnum"] = float(fn_all[-1])
    rm = re.search(r"RMS[^0-9]*([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)\s*mm", t, re.IGNORECASE)
    if rm:
        pred["rms"] = float(rm.group(1))
    return pred

def norm_err(val, pred, lo, hi):
    if val is None or pred is None:
        return None
    return abs(val - pred) / (hi - lo)

def score_item(item: dict, pred: dict) -> dict:
    c = item["constraints"]
    s = {}
    if "fov" in c and "fov" in pred:
        s["fov_err"] = norm_err(c["fov"], pred["fov"], *FOV_RANGE)
    elif "fov_min" in c and "fov" in pred:
        s["fov_err"] = norm_err((c["fov_min"]+c["fov_max"])/2, pred["fov"], *FOV_RANGE)
    if "fnum" in c and "fnum" in pred:
        s["fnum_err"] = norm_err(c["fnum"], pred["fnum"], *FNUM_RANGE)
    elif "fnum_min" in c and "fnum" in pred:
        s["fnum_err"] = norm_err((c["fnum_min"]+c["fnum_max"])/2, pred["fnum"], *FNUM_RANGE)
    if c.get("aper", 0) > 0 and "aper" in pred:
        s["aper_err"] = norm_err(c["aper"], pred["aper"], *APER_RANGE)
    if "rms" in pred:
        s["rms_pred"] = pred["rms"]
        s["rms_norm"] = min(pred["rms"] / RMS_MAX, 1.0)
    errs = [v for k, v in s.items() if k.endswith("_err") and v is not None]
    s["norm_sum"] = sum(errs) / len(errs) if errs else None
    if not pred:
        return {**s, "hit": 0}
    hit = True
    if "fov"  in c and "fov"  in pred: hit &= abs(c["fov"]  - pred["fov"])  <= TOL_FOV
    if "fnum" in c and "fnum" in pred: hit &= abs(c["fnum"] - pred["fnum"]) <= TOL_FNUM
    if c.get("aper", 0) > 0 and "aper" in pred:
        hit &= abs(c["aper"] - pred["aper"]) <= TOL_APER
    if "rms_target" in c and "rms" in pred:
        hit &= pred["rms"] <= c["rms_target"]
    if "fov_min" in c and "fov" in pred:
        hit &= c["fov_min"] <= pred["fov"] <= c["fov_max"]
    if "fnum_min" in c and "fnum" in pred:
        hit &= c["fnum_min"] <= pred["fnum"] <= c["fnum_max"]
    s["hit"] = int(hit)
    return s

def plot_results(results: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TYPES  = ["in_domain", "out_of_domain", "partial", "range", "specific"]
    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    def avg(key):
        out = {}
        for t in TYPES:
            vals = [r["scores"][key] for r in results
                    if r["type"] == t and r["scores"].get(key) is not None]
            out[t] = float(np.mean(vals)) if vals else 0.0
        return out
    metrics = [
        ("fov_err",  "FOV Norm Error"),
        ("fnum_err", "Fnum Norm Error"),
        ("norm_sum", "Overall Norm Error"),
        ("hit",      "Hit Rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LensAgent Baseline Evaluation", fontsize=14, fontweight="bold")
    for ax, (key, title) in zip(axes.flat, metrics):
        d = avg(key)
        bars = ax.bar(d.keys(), d.values(), color=COLORS, width=0.5)
        ax.set_title(title)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, d.values()):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = REPORT_DIR / "baseline_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 图表已保存: {out}")

def summarize(results):
    TYPES = ["in_domain", "out_of_domain", "partial", "range", "specific"]
    summary = {"total": len(results), "by_type": {}}
    for t in TYPES:
        sub = [r for r in results if r["type"] == t]
        if not sub:
            continue
        hits  = [r["scores"].get("hit", 0) for r in sub]
        norms = [r["scores"]["norm_sum"] for r in sub if r["scores"].get("norm_sum") is not None]
        rmss  = [r["scores"]["rms_pred"] for r in sub if r["scores"].get("rms_pred") is not None]
        summary["by_type"][t] = {
            "n":            len(sub),
            "hit_rate":     round(float(np.mean(hits)), 3),
            "avg_norm_sum": round(float(np.mean(norms)), 4) if norms else None,
            "avg_rms":      round(float(np.mean(rmss)),  4) if rmss  else None,
        }
    overall = float(np.mean([r["scores"].get("hit", 0) for r in results]))
    summary["overall_hit_rate"] = round(overall, 3)
    return summary, overall

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",            type=int,  default=None)
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--plot-only",    action="store_true")
    p.add_argument("--shard",        type=int,  default=None, help="分片索引 0-based")
    p.add_argument("--total-shards", type=int,  default=4,    help="总分片数")
    p.add_argument("--merge",        action="store_true",     help="合并所有分片结果")
    args = p.parse_args()

    # ── merge 模式：合并分片结果并生成汇总 ──────────────────────────────
    if args.merge:
        results = []
        for i in range(args.total_shards):
            f = DATA_DIR / f"eval_results_shard{i}.jsonl"
            if f.exists():
                for line in f.read_text().splitlines():
                    if line.strip():
                        results.append(json.loads(line))
        print(f"合并 {len(results)} 条结果")
        RESULT_F.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
            encoding="utf-8"
        )
        summary, overall = summarize(results)
        SUMMARY_F.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"总命中率: {overall:.1%}  ({len(results)} 条)")
        for t, s in summary["by_type"].items():
            print(f"  {t:15s}: hit={s['hit_rate']:.1%}  norm_sum={s['avg_norm_sum']}  rms={s['avg_rms']}")
        plot_results(results)
        return

    # ── plot-only 模式 ───────────────────────────────────────────────────
    if args.plot_only:
        results = [json.loads(l) for l in RESULT_F.read_text().splitlines() if l.strip()]
        plot_results(results)
        return

    # ── 确定输出文件和数据范围 ───────────────────────────────────────────
    items = json.loads(TESTSET.read_text(encoding="utf-8"))
    if args.n:
        items = items[:args.n]

    if args.shard is not None:
        # 分片：每个 shard 取自己的那段
        items = [x for i, x in enumerate(items) if i % args.total_shards == args.shard]
        out_file = DATA_DIR / f"eval_results_shard{args.shard}.jsonl"
        print(f"[Shard {args.shard}/{args.total_shards}] {len(items)} 条")
    else:
        out_file = RESULT_F

    # 断点续跑
    done_ids = set()
    if (args.resume or args.shard is not None) and out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])
        print(f"已完成 {len(done_ids)} 条，跳过")

    results = []
    mode = "a" if (args.resume or args.shard is not None) else "w"
    with open(out_file, mode, encoding="utf-8") as fout:
        for i, item in enumerate(items):
            iid = item.get("id", i)
            if iid in done_ids:
                continue

            q = item["query"]
            print(f"[{i+1}/{len(items)}] [{item['type']:15s}] {q[:55]}")
            t0 = time.time()
            try:
                response = run_agent(q)
                status   = "ok"
            except Exception as e:
                response = f"ERROR: {e}"
                status   = "error"
                print(f"  ❌ {e}")
            elapsed = round(time.time() - t0, 1)

            pred   = parse_prediction(response)
            scores = score_item(item, pred)

            row = {
                "id":       iid,
                "type":     item["type"],
                "query":    q,
                "response": response[:800],
                "pred":     pred,
                "scores":   scores,
                "status":   status,
                "elapsed":  elapsed,
            }
            results.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            ns = scores.get("norm_sum")
            ns_str = f"{ns:.3f}" if ns is not None else "-"
            hit_icon = "✅" if scores.get("hit") else "❌"
            print(f"  {hit_icon}  pred={pred}  norm_sum={ns_str}  [{elapsed}s]")

    # 非分片模式才自动汇总
    if args.shard is None:
        if not results:
            results = [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]
        summary, overall = summarize(results)
        SUMMARY_F.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n总命中率: {overall:.1%}  ({len(results)} 条)")
        for t, s in summary["by_type"].items():
            print(f"  {t:15s}: hit={s['hit_rate']:.1%}  norm_sum={s['avg_norm_sum']}  rms={s['avg_rms']}")
        plot_results(results)
    else:
        print(f"[Shard {args.shard}] 完成 {len(results)} 条 → {out_file}")

if __name__ == "__main__":
    main()
PYEOF
echo "✅ eval.py 写入完成"
