"""
eval.py     —  跑 Agent baseline 并打分
用法:
    python eval.py                        # 跑全部 200 条
    python eval.py --n 20                 # 只跑前 20 条（快速验证）
    python eval.py --type in_domain       # 只跑某类
输出:
    /gz-data/eval_results.jsonl   每条 query 的详细结果
    /gz-data/eval_summary.json    汇总指标
"""

import json, sys, argparse, time
from pathlib import Path

# ── 复用 agent.py 的所有组件 ─────────────────────────────────────────────────
sys.path.insert(0, "/gz-data")
from agent import load_rag, build_agent, ALL_LENSES, VS
import agent as _agent_mod

from build_testset import score_response

TESTSET_PATH = "/gz-data/testset.json"
RESULTS_PATH = "/gz-data/eval_results.jsonl"
SUMMARY_PATH = "/gz-data/eval_summary.json"

# ── 解析 Agent 输出，提取候选镜头列表 ─────────────────────────────────────────
def parse_agent_output(output: str, agent_output: dict) -> dict:
    """
    从 AgentExecutor 的完整输出里提取候选镜头信息。
    intermediate_steps 里有每次工具调用的输入输出。
    """
    retrieved = []

    # 从 intermediate_steps 里找 lens_search / rank_by_rms 的返回结果
    for action, observation in agent_output.get("intermediate_steps", []):
        tool_name = getattr(action, "tool", "")
        if tool_name in ("lens_search", "rank_by_rms"):
            # 解析观察结果里的镜头信息
            for line in str(observation).split("\n"):
                if "镜头#" not in line:
                    continue
                try:
                    # 格式: [N] 镜头#IDX | FOV=X° | F/Y | ...
                    idx  = int(line.split("镜头#")[1].split()[0].rstrip("|"))
                    fov  = float(line.split("FOV=")[1].split("°")[0]) if "FOV=" in line else None
                    fnum = float(line.split("F/")[1].split()[0].rstrip("|")) if "F/" in line else None
                    rms_s = line.split("近轴RMS=")[1].split("mm")[0] if "近轴RMS=" in line else None
                    rms  = float(rms_s) if rms_s else None

                    if idx < len(_agent_mod.ALL_LENSES):
                        l = _agent_mod.ALL_LENSES[idx]
                        retrieved.append({
                            "lens_idx": idx,
                            "fov":      l.get("fov"),
                            "fnum":     l.get("fnum"),
                            "aper":     l.get("aper"),
                            "calc_rms": l.get("calc_rms"),
                            "calc_effl":l.get("calc_effl"),
                        })
                except Exception:
                    continue

    # 去重（同一 lens_idx 只保留一次）
    seen = set()
    unique = []
    for r in retrieved:
        if r["lens_idx"] not in seen:
            seen.add(r["lens_idx"])
            unique.append(r)

    return {
        "retrieved_lenses": unique,
        "answer": agent_output.get("output", ""),
    }

# ── 主评估循环 ────────────────────────────────────────────────────────────────
def run_eval(items: list, executor) -> list:
    results = []
    for i, item in enumerate(items):
        print(f"\n[{i+1}/{len(items)}] {item['id']} | {item['type']}")
        print(f"  Q: {item['query']}")

        t0 = time.time()
        try:
            raw = executor.invoke(
                {"input": item["query"]},
                return_only_outputs=False,
            )
            elapsed = time.time() - t0
            agent_resp = parse_agent_output(item["query"], raw)
            scores     = score_response(item, agent_resp)
            status     = "ok"
        except Exception as e:
            elapsed    = time.time() - t0
            agent_resp = {"retrieved_lenses": [], "answer": f"ERROR: {e}"}
            scores     = {"fov_error": 999, "fnum_error": 999, "rms": 999,
                          "norm_sum": 999, "hit": False}
            status     = "error"
            print(f"  ❌ {e}")

        result = {
            **item,
            "agent_answer":     agent_resp["answer"],
            "retrieved_lenses": agent_resp["retrieved_lenses"],
            "scores":           scores,
            "elapsed_s":        round(elapsed, 2),
            "status":           status,
        }
        results.append(result)

        print(f"  hit={scores['hit']} | RMS={scores['rms']:.4f} | "
              f"norm_sum={scores['norm_sum']:.4f} | {elapsed:.1f}s")

        # 实时写入，防止中途崩溃丢数据
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return results

# ── 汇总统计 ──────────────────────────────────────────────────────────────────
def summarize(results: list) -> dict:
    from collections import defaultdict
    import statistics

    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    def _stats(items):
        scores = [r["scores"] for r in items if r["status"] == "ok"]
        if not scores:
            return {}
        hits     = [s["hit"]      for s in scores]
        norms    = [s["norm_sum"] for s in scores if s["norm_sum"] < 999]
        rms_list = [s["rms"]      for s in scores if s["rms"]      < 999]
        return {
            "n":              len(items),
            "hit_rate":       round(sum(hits) / len(hits), 4) if hits else 0,
            "avg_norm_sum":   round(statistics.mean(norms),    4) if norms    else 999,
            "avg_rms":        round(statistics.mean(rms_list), 6) if rms_list else 999,
            "median_rms":     round(statistics.median(rms_list), 6) if rms_list else 999,
            "avg_elapsed_s":  round(statistics.mean([r["elapsed_s"] for r in items]), 2),
            "error_count":    sum(1 for r in items if r["status"] == "error"),
        }

    summary = {
        "total":   len(results),
        "overall": _stats(results),
        "by_type": {t: _stats(items) for t, items in by_type.items()},
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int,  default=None, help="只跑前 N 条")
    parser.add_argument("--type", type=str,  default=None,
                        help="只跑某类: in_domain / out_of_domain / range / partial")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已有结果（从 eval_results.jsonl 读已完成 id）")
    args = parser.parse_args()

    # 加载 RAG + Agent
    _agent_mod.VS, _agent_mod.ALL_LENSES = load_rag()
    executor = build_agent()

    # 加载 testset
    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)

    # 过滤
    if args.type:
        testset = [x for x in testset if x["type"] == args.type]
    if args.resume and Path(RESULTS_PATH).exists():
        done_ids = set()
        with open(RESULTS_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        testset = [x for x in testset if x["id"] not in done_ids]
        print(f"Resume: 跳过 {len(done_ids)} 条已完成，剩余 {len(testset)} 条")
    if args.n:
        testset = testset[:args.n]

    print(f"评估 {len(testset)} 条 query…")

    # 清空或追加
    if not args.resume:
        open(RESULTS_PATH, "w").close()

    results  = run_eval(testset, executor)
    summary  = summarize(results)

    print("\n" + "="*50)
    print("评估完成")
    print(f"  总计:     {summary['total']} 条")
    o = summary["overall"]
    print(f"  命中率:   {o.get('hit_rate', 0):.2%}")
    print(f"  平均RMS:  {o.get('avg_rms', 999):.4f} mm")
    print(f"  avg_norm: {o.get('avg_norm_sum', 999):.4f}")
    print(f"\n详细结果: {RESULTS_PATH}")
    print(f"汇总结果: {SUMMARY_PATH}")
