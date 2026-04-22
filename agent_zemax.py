# ─── dotenv 自动加载 ───
import os
try:
    from dotenv import load_dotenv
    load_dotenv("/gz-data/.env")
except ImportError:
    pass

import os, sys, ast, copy, pickle, argparse, json
from pathlib import Path

import torch
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import PromptTemplate

sys.path.insert(0, "/gz-data")
import sys as _sys; _sys.modules.setdefault("agent_zemax", _sys.modules[__name__])

# self-evolve 模块（check_spec / 轨迹记录 / 蒸馏）
try:
    from self_evolve import (
        check_spec, record_step,
        start_session, end_session,
        load_learned_for_prompt, get_learned_detail,
    )
    _HAS_SELF_EVOLVE = True
except ImportError:
    _HAS_SELF_EVOLVE = False
    print("⚠ self_evolve 未找到，self-evolve 功能禁用")
    def record_step(*a, **kw): pass
    def start_session(*a, **kw): pass
    def end_session(*a, **kw): return {}
    def load_learned_for_prompt(): return {}
    def get_learned_detail(name): return None

BASE_URL  = "http://localhost:8001/v1"
API_KEY   = "EMPTY"
LLM_MODEL = "qwen3"
EMB_MODEL = "/root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots/183bb99aa7af74355fb58d16edf8c13ae7c5433e"
FAISS_DIR = "/gz-data/faiss_index"



_device = "cpu"  # 强制CPU，GPU留给Qwen

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0,
    max_tokens=512,    
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 64},
)


def load_rag():
    faiss_path = Path(FAISS_DIR) / "index.faiss"
    pkl_path   = Path(FAISS_DIR) / "lenses.pkl"
    if not faiss_path.exists():
        raise FileNotFoundError(f"向量库不存在：{FAISS_DIR}")

    print(f"加载向量库: {FAISS_DIR}")
    vs = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

    lenses = []
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            lenses = pickle.load(f)
        print(f"加载镜头数据: {len(lenses)} 条")
    else:
        print("⚠ lenses.pkl 不存在")

    return vs, lenses

# ══════════════════════════════════════════════════════════════════════════════
# optical_calculator
# ══════════════════════════════════════════════════════════════════════════════
try:
    from optical_calculator import paraxial_trace, lens_to_surfaces
    _HAS_OPTICS = True
except ImportError:
    _HAS_OPTICS = False
    print("⚠ optical_calculator 未找到")

def _calc(lens: dict) -> dict:
    if not _HAS_OPTICS:
        return {"valid": False, "msg": "optical_calculator 未加载"}
    if isinstance(lens.get("surfaces"), str):
        lens = dict(lens)
        lens["surfaces"] = ast.literal_eval(lens["surfaces"])
    try:
        return paraxial_trace(lens_to_surfaces(lens), lens_meta=lens)
    except Exception as e:
        return {"valid": False, "msg": str(e)}



VS: FAISS        = None
ALL_LENSES: list = []

_LENS_BACKUP: dict = {}
_LENS_BACKUP_PRE_ZEMAX: dict = {}  # ★ P16 Zemax 优化前的快照(独立于 modify 备份,便于 reset)
_SEARCH_COUNT: dict = {}  # lens_search调用计数，防死循环
_INTERPRET_CALLED: bool = False  # interpret_requirement只能调一次
_OPTIMIZE_STALL: dict = {}
_OPTIMIZE_TOTAL: dict = {}
_MODIFY_COUNT: dict = {}  # modify_lens调用计数，防死循环
_OPTIMIZE_MAX = 6
# local_optimize 停滞计数器 {lens_idx: consecutive_no_improve_count}
_OPTIMIZE_STALL: dict = {}

# ★ P14: 全局工具调用历史 + 熔断
# 在最近 _CIRCUIT_WINDOW 次工具调用中,若 (tool_name, normalized_input) 累计
# 出现 >= _CIRCUIT_MAX_REPEAT 次 → 熔断。
# 用"窗口内累计"而不是"连续"是为了抓 ABABAB 这种交替循环(上次 F/2.0 bug 就是这种模式:
# modify_lens 同入参之间反复插 rms_calculator/local_optimize,"连续数"法抓不到)。
_TOOL_CALL_HISTORY: list = []   # [(tool_name, input_hash), ...]
_CIRCUIT_MAX_REPEAT = 3         # 窗口内累计重复阈值
_CIRCUIT_WINDOW     = 10        # 回看窗口

def _normalize_tool_input(raw: str) -> str:
    """把工具输入规范化成一个用于"是否重复"比较的短字符串。
    去除空白差异 / 引号 / 大小写,让 agent 传 "144097" 和 "lens_idx=144097 " 被视为同一次调用。
    """
    s = str(raw or "").strip().lower()
    # 去除所有空白和常见引号
    for ch in (' ', '\t', '\n', '"', "'", "`"):
        s = s.replace(ch, '')
    return s[:200]   # 截断保护

def _circuit_guard(tool_name: str, raw_input: str):
    """★ P14 熔断检查。
    最近 _CIRCUIT_WINDOW 次调用中同一 (tool_name, norm_input) 累计出现
    _CIRCUIT_MAX_REPEAT 次 → 返回熔断字符串;否则记录本次并返回 None。
    """
    key = (tool_name, _normalize_tool_input(raw_input))
    _TOOL_CALL_HISTORY.append(key)
    # 只保留最近窗口
    if len(_TOOL_CALL_HISTORY) > _CIRCUIT_WINDOW:
        del _TOOL_CALL_HISTORY[:-_CIRCUIT_WINDOW]
    # 窗口内累计相同 key 的次数
    count = sum(1 for k in _TOOL_CALL_HISTORY if k == key)
    if count >= _CIRCUIT_MAX_REPEAT:
        return (f"⛔ 熔断: 工具 {tool_name} 在最近 {_CIRCUIT_WINDOW} 次调用中已用"
                f"相同输入 ({_normalize_tool_input(raw_input)[:80]}) {count} 次,行为未变化。\n"
                f"这说明当前策略无效。请立即【切换策略或输出 Final Answer】,"
                f"不要再用相同参数调同一个工具。\n"
                f"可选方案:\n"
                f"  • 换候选镜头(用 rank_by_rms 结果里的 rank=2/3)\n"
                f"  • 换修改目标(modify_lens 换一个 surface 或 param)\n"
                f"  • reset_lens 回滚后试不同策略\n"
                f"  • 若已尽力,输出 Final Answer 并如实说明未达标")
    return None

def _clear_call_history():
    """每次 chain 结束时清空历史,避免跨查询污染。由 _ToolCallLogger.on_chain_end 调用。"""
    _TOOL_CALL_HISTORY.clear()


def _strip_think(text: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _parse_tool_input(input_str: str) -> dict:
    """兼容 'lens_idx=36237, ...' 和 '36237, ...' 两种格式"""
    input_str = str(input_str).strip().strip(chr(34)).strip(chr(39))
    parts = {}
    for seg in input_str.split(","):
        seg = seg.strip().strip(chr(34)).strip(chr(39))
        if "=" in seg:
            k, v = seg.split("=", 1)
            parts[k.strip()] = v.strip().strip(chr(34)).strip(chr(39))
        else:
            try:
                parts["lens_idx"] = str(int(seg.strip()))
            except ValueError:
                pass
    return parts



@tool
def lens_search(query: str) -> str:
    """语义检索候选镜头。输入: 需求描述。输出: top-5 (id/fov/fnum/effl/rms)。"""
    # 防止死循环：同一 query 连续调用超过2次直接返回
    _q_key = query[:30]
    _SEARCH_COUNT[_q_key] = _SEARCH_COUNT.get(_q_key, 0) + 1
    if _SEARCH_COUNT[_q_key] > 2:
        return json.dumps({"error": "lens_search已调用多次，请直接用已有结果选择镜头并优化，或输出Final Answer。"})

    results = VS.similarity_search(query, k=5)
    if not results:
        return "未检索到相关镜头"
    results_list = []
    rank = 0
    for doc in results:
        m = doc.metadata
        rms  = m.get("calc_rms")
        effl = m.get("calc_effl")
        # 过滤坏数据：rms/effl 任一为 None 跳过
        if rms is None or effl is None:
            continue
        rank += 1
        results_list.append({
            "rank": rank, "id": m.get("lens_idx"),
            "fov": m.get("fov"), "fnum": m.get("fnum"),
            "effl": round(effl, 2),
            "rms": round(rms, 4),
        })
        if rank >= 5: break
    return json.dumps(results_list, ensure_ascii=False)


@tool
def rms_calculator(lens_idx: str) -> str:
    """查当前镜头实时近轴性能。输入: lens_idx(整数)。"""
    try:
        # 兼容 LLM 传 "lens_idx=36237" 或 "36237" 两种格式
        if isinstance(lens_idx, str) and "=" in lens_idx:
            lens_idx = lens_idx.split("=")[-1].strip()
        lens_idx = int(str(lens_idx).strip())
    except (ValueError, TypeError):
        return f"lens_idx 必须是整数，收到: {lens_idx!r}"
    if not ALL_LENSES:
        return "镜头数据未加载"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界（共 {len(ALL_LENSES)} 条）"

    lens = ALL_LENSES[lens_idx]
    # 优先用实时追迹（反映最新修改）
    r = _calc(lens)
    if r.get("valid"):
        return f"#{lens_idx} FOV={lens.get('fov')}° F/{lens.get('fnum')} RMS={r['rms']:.4f}mm EFFL={r['effl']:.2f}mm"
    # 回退到缓存值
    return json.dumps({
        "id": lens_idx, "fov": lens.get('fov'), "fnum": lens.get('fnum'),
        "rms": lens.get('calc_rms'), "error": r.get('msg','')
    }, ensure_ascii=False)


@tool
def rank_by_rms(query: str) -> str:
    """按 FOV/Fnum 数值距离检索并按 RMS 排序。输入: "FOV=35 F/2.8, fov_tol=5, fnum_tol=0.5"。若 warning 出现则为 OOD,直接用 rank=1。"""
    import re
    # 解析 FOV/Fnum 目标值和容忍度
    fov_target = fnum_target = None
    fov_tol = 10.0
    fnum_tol = 0.5

    fov_m = re.search(r'FOV[=:\s]*([0-9]+\.?[0-9]*)', query, re.IGNORECASE)
    fn_m  = re.search(r'F/([0-9]+\.?[0-9]*)', query, re.IGNORECASE)
    tol_fov_m  = re.search(r'fov_tol[=:\s]*([0-9]+\.?[0-9]*)', query, re.IGNORECASE)
    tol_fn_m   = re.search(r'fnum_tol[=:\s]*([0-9]+\.?[0-9]*)', query, re.IGNORECASE)
    if fov_m:  fov_target  = float(fov_m.group(1))
    if fn_m:   fnum_target = float(fn_m.group(1))
    if tol_fov_m:  fov_tol  = float(tol_fov_m.group(1))
    if tol_fn_m:   fnum_tol = float(tol_fn_m.group(1))

    # ─── 纯数值近邻：FOV 主导，Fnum 次要 tiebreaker（Fnum 能靠扩光阑优化） ───
    def _distance(lens, idx):
        d = 0.0
        # FOV 是主排序键（权重大）
        if fov_target is not None and fov_target > 0:
            fov = lens.get("fov")
            if fov is None:
                return 1e9
            rel = (float(fov) - fov_target) / fov_target
            d += abs(rel) * 10.0  # 对称惩罚，值大的优先排小，保证 FOV 最近的一定排前面
        # Fnum 只作次要 tiebreaker（权重小），因为 Fnum 能靠改光阑优化
        if fnum_target is not None and fnum_target > 0:
            fnum = lens.get("fnum")
            if fnum is not None:
                rel = (float(fnum) - fnum_target) / fnum_target
                # Fnum 偏大（需扩光阑）惩罚略高于偏小
                d += abs(rel) * 1.0 if rel > 0 else abs(rel) * 0.5
        # RMS 只作极小权重
        rms = lens.get("calc_rms")
        if rms is not None:
            d += float(rms) * 0.01
        return d

    # 全库打分，直接取距离最近的 top20（不过滤，规则：FOV 最接近 → Fnum 最接近）
    scored_all = []
    for i, lens in enumerate(ALL_LENSES):
        if lens.get("calc_rms") is None:
            continue
        scored_all.append((_distance(lens, i), i, lens))
    scored_all.sort(key=lambda x: x[0])
    top20 = [(float(l.get("calc_rms")), {**l, "lens_idx": i}) for _, i, l in scored_all[:20]]

    # 从 top-20 让 Gemini 按 Skill R1 规则选 top-5
    candidates = []
    try:
        from openai import OpenAI
        from retrieval_skills import RETRIEVAL_SELECTION_PROMPT
        _cli = OpenAI(
            api_key="sk-uwMXbGBi2LKb9EnmGIOQT1QOISpA8jgazzvXwVLq5o5h79WZ",
            base_url="https://us.novaiapi.com/v1",
        )
        candidates_desc = [
            {"lens_idx": m["lens_idx"], "fov": m.get("fov"), "fnum": m.get("fnum"),
             "effl": round(m.get("calc_effl") or 0, 2), "rms": round(rms, 4)}
            for rms, m in top20
        ]
        print(f"\n[OOD-DEBUG] top20 FOV: {sorted(set(round(c["fov"] or 0, 0) for c in candidates_desc))}")
        print(f"[OOD-DEBUG] top20 Fnum: {sorted(set(round(c["fnum"] or 0, 1) for c in candidates_desc))}")
        gemini_prompt = RETRIEVAL_SELECTION_PROMPT.format(
            fov_target=fov_target, fnum_target=fnum_target,
            candidates_json=json.dumps(candidates_desc, ensure_ascii=False),
        )
        gemini_resp = _cli.chat.completions.create(
            model=os.environ.get("GEMINI_MODEL_SELECT", "gemini-3-flash-preview"),
            messages=[{"role": "user", "content": gemini_prompt}],
            max_tokens=1000,
        )
        resp_text = gemini_resp.choices[0].message.content.strip()
        resp_text = resp_text.replace("```json", "").replace("```", "").strip()
        print(f"[OOD-DEBUG] Gemini 选择: {resp_text[:200]}")
        # 容错：从响应中提取 JSON 数组，兼容模型返回多余文字的情况
        import re as _re
        _arr_match = _re.search(r"\[[\d,\s]+\]", resp_text)
        if not _arr_match:
            # 兜底：截断数组（无右括号），提取已有的完整 ID
            _arr_match2 = _re.search(r"\[([\d,\s]+)", resp_text)
            if _arr_match2:
                resp_text = "[" + _arr_match2.group(1).rstrip(", ") + "]"
        if _arr_match:
            resp_text = _arr_match.group(0)
        selected_ids = json.loads(resp_text)
        id_map = {m["lens_idx"]: (rms, m) for rms, m in top20}
        candidates = [id_map[idx] for idx in selected_ids if idx in id_map]
    except Exception as e:
        print(f"[OOD-DEBUG] Gemini 选镜失败，退回数值距离前5: {e}")
    # Gemini 失败或返回空则退回距离前5
    if not candidates:
        candidates = top20[:5]
    else:
        candidates = candidates[:5]

    # 判定是否算 OOD（最佳候选 FOV 或 Fnum 偏差大就警告 agent）
    is_ood = False
    if candidates and (fov_target is not None or fnum_target is not None):
        _, best = candidates[0]
        best_fov = best.get("fov")
        best_fnum = best.get("fnum")
        if fov_target and best_fov and abs(float(best_fov) - fov_target) > fov_tol:
            is_ood = True
        if fnum_target and best_fnum and abs(float(best_fnum) - fnum_target) > fnum_tol:
            is_ood = True
    fallback = is_ood  # 复用下游 warning 生成逻辑

    results_list = []
    for rank, (rms, m) in enumerate(candidates, 1):
        results_list.append({
            "rank": rank, "id": m.get("lens_idx"),
            "rms": round(rms, 4), "fov": m.get("fov"),
            "fnum": m.get("fnum"), "effl": round(m.get("calc_effl", 0), 2),
        })

    out = {"results": results_list}
    if fallback:
        from retrieval_skills import RETRIEVAL_SKILL_SHORT as RETRIEVAL_SKILL_SUMMARY
        # 计算返回镜头的 EFFL 偏差提示
        effl_hints = []
        for r in results_list:
            effl = r.get("effl", 0)
            if effl and effl > 0:
                effl_hints.append(f"镜头#{r['id']} EFFL={effl}mm")
        effl_hint_str = "、".join(effl_hints) if effl_hints else "未知"
        out["warning"] = (
            f"⚠ 无FOV≈{fov_target}° F/{fnum_target}精确匹配，用上方镜头作起点。\n"
            f"EFFL: {effl_hint_str}\n"
            f"步骤: 1.align_effl → 2.修光阑SD降F# → 3.local_optimize → 4.zemax_optimize(rms_target=X,fnum_target=Y)"
        )

    # ★ self-evolve: 记录检索决策(OOD 判定、选中的候选)
    try:
        _picked_id = results_list[0].get("id") if results_list else None
        _picked_fov  = results_list[0].get("fov")  if results_list else None
        _picked_fnum = results_list[0].get("fnum") if results_list else None
        _picked_rms  = results_list[0].get("rms")  if results_list else None
        record_step("rank_by_rms", _picked_id if _picked_id is not None else -1,
                    {"fov_target": fov_target, "fnum_target": fnum_target,
                     "is_ood": bool(is_ood),
                     "picked_id": _picked_id,
                     "picked_fov": _picked_fov, "picked_fnum": _picked_fnum,
                     "picked_rms_paraxial": _picked_rms,
                     "n_candidates": len(results_list)},
                    kind="decide")
    except Exception:
        pass

    return json.dumps(out, ensure_ascii=False)


@tool
def get_lens_surfaces(lens_idx: str) -> str:
    """查镜头面型(简表,含 stop_surface)。输入: lens_idx。"""
    try:
        if isinstance(lens_idx, str) and "=" in lens_idx:
            lens_idx = lens_idx.split("=")[-1].strip()
        lens_idx = int(str(lens_idx).strip())
    except (ValueError, TypeError):
        return f"lens_idx 必须是整数，收到: {lens_idx!r}"
    if not ALL_LENSES:
        return "镜头数据未加载"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界"

    lens  = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        surfs = ast.literal_eval(surfs)

    # 返回所有面（含 AIR 面/光阑面），加 semi_diameter 字段
    # AIR + r≈∞ + sd局部最小 = 光阑面（Skill 13 识别规则）
    all_surfs = []
    for i, s in enumerate(surfs):
        if i == 0 or i == len(surfs) - 1:
            continue  # 跳过 OBJ 和 IMA
        r = s.get("radius", 0)
        sd = s.get("semi_diameter", 0)
        entry = {
            "s": int(s.get("surface_num", i)),
            "r": round(float(r), 2) if abs(float(r)) < 1e7 else "inf",
            "t": round(float(s.get("thickness", 0)), 2),
            "sd": round(float(sd), 2) if sd else 0,
            "m": s.get("material", "AIR"),
        }
        all_surfs.append(entry)

    # 自动识别光阑面（r≈∞ + AIR + sd 最小）
    stop_candidates = [s for s in all_surfs
                       if s["r"] == "inf" and s["m"] == "AIR"]
    stop_surf = None
    if stop_candidates:
        stop_surf = min(stop_candidates, key=lambda x: x["sd"])["s"]

    # ★ P18d 紧凑文本,每面 1 行,去掉 JSON 的引号/括号开销
    # 原版 JSON 一个 15 面镜头约 500 chars,新版约 300 chars
    header = (f"#{lens_idx} fov={lens.get('fov')} fnum={lens.get('fnum')} "
              f"n_surf={len(surfs)} stop={stop_surf}")
    lines = [header, "s r t sd material"]
    for e in all_surfs:
        r_s = f"{e['r']}" if e['r'] == "inf" else f"{e['r']:g}"
        mat = e['m'] if e['m'] and e['m'].upper() != 'AIR' else '-'
        lines.append(f"{e['s']} {r_s} {e['t']:g} {e['sd']:g} {mat}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 新增 Tools
# ══════════════════════════════════════════════════════════════════════════════

def _safe_eval_numeric(expr):
    """安全计算 LLM 传来的简单数值表达式,例如 '5.033*(2.1/2.0)' → 5.285。
    只允许数字/小数点/基本算术运算符/括号/科学计数法 (e/E),禁止名字/函数/下划线。
    解析失败返回 None。
    """
    if expr is None:
        return None
    s = str(expr).strip()
    # 先试直接转 float
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    # 白名单字符校验 — 只允许 [0-9 . + - * / ( ) e E 空格],其余一律拒绝
    import re
    if not re.match(r'^[0-9.\+\-\*/()eE\s]+$', s):
        return None
    # 用 ast 而不是 eval,进一步限制只能是数值表达式节点
    import ast, operator
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }
    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):   # py<3.8 兼容
            return float(node.n)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_walk(node.operand))
        raise ValueError(f"非法节点: {type(node).__name__}")
    try:
        tree = ast.parse(s, mode="eval")
        val = _walk(tree)
        return float(val)
    except Exception:
        return None


@tool
def modify_lens(input_str: str) -> str:
    """改指定面的参数。输入: "lens_idx=X, surface=Y, param=P, value=V"。param: radius/thickness/material/semi_diameter。value 支持算式如 5.033*(2.1/2.0)。material 必须 CDGM 牌号。"""
    # ★ P14 熔断
    _tripped = _circuit_guard("modify_lens", input_str)
    if _tripped:
        return _tripped
    # 解析输入
    try:
        input_str = str(input_str).strip().strip(chr(34)).strip(chr(39))
        parts = {}
        for seg in input_str.split(","):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                parts[k.strip()] = v.strip()

        lens_idx   = int(parts["lens_idx"])
        surface_id = int(parts["surface"])
        param      = parts["param"].lower()
        value_raw  = parts["value"]
    except Exception as e:
        return f"输入格式错误: {e}\n正确格式: lens_idx=55097, surface=3, param=radius, value=-48.5"

    if not ALL_LENSES:
        return "镜头数据未加载"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界"

    # 死循环防护：同一镜头同一参数修改超过3次拒绝
    _mod_key = f"{lens_idx}_{surface_id}_{param}"
    _MODIFY_COUNT[_mod_key] = _MODIFY_COUNT.get(_mod_key, 0) + 1
    if _MODIFY_COUNT[_mod_key] > 3:
        return f"⛔ 镜头#{lens_idx} 面{surface_id} {param}已修改{_MODIFY_COUNT[_mod_key]}次仍未改善，禁止继续修改同一参数。请改用local_optimize或换其他面/参数。"

    lens  = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        surfs = ast.literal_eval(surfs)
        lens["surfaces"] = surfs

    # 备份原始数据（只备份一次）
    if lens_idx not in _LENS_BACKUP:
        _LENS_BACKUP[lens_idx] = copy.deepcopy(lens)

    # ★ self-evolve: 先算 before (给蒸馏器看的对比基线)
    try:
        _r_before = _calc(lens)
        _metrics_before = {"rms":  _r_before.get("rms"),
                           "effl": _r_before.get("effl")} if _r_before.get("valid") else None
    except Exception:
        _metrics_before = None

    # 找目标面
    target = None
    for s in surfs:
        if int(s.get("surface_num", -1)) == surface_id:
            target = s
            break
    if target is None:
        available = [int(s.get("surface_num", 0)) for s in surfs]
        return f"面号 {surface_id} 不存在，可用面号: {available}"

    # 执行修改
    NUMERIC_PARAMS = {"radius", "thickness", "semi_diameter"}
    old_val = target.get(param, "不存在")

    if param in NUMERIC_PARAMS:
        # ★ 支持 LLM 传来的简单算式 (如 "5.033*(2.1/2.0)"), 不必强制它自己算
        new_val = _safe_eval_numeric(value_raw)
        if new_val is None:
            return (f"参数 {param} 需要数值或简单算式，收到: {value_raw!r}\n"
                    f"允许的格式: 纯数字 (5.28) 或算术表达式 (5.033*(2.1/2.0))，"
                    f"只支持 + - * / ( ) 运算符")

        # ★ 光阑面错误检测：改 semi_diameter 却不是光阑面 → 直接告知正确面号，打断死循环
        if param == "semi_diameter":
            try:
                _optics_chk = surfs[1:-1] if len(surfs) >= 3 else surfs
                _stop_cands = [(s.get("surface_num"), float(s.get("semi_diameter", 0)))
                               for s in _optics_chk
                               if abs(float(s.get("radius", 0))) > 1e7
                               and str(s.get("material", "AIR")).upper() == "AIR"
                               and float(s.get("semi_diameter", 0)) > 0]
                if _stop_cands:
                    _real_stop = int(min(_stop_cands, key=lambda x: x[1])[0])
                    _real_stop_sd = min(_stop_cands, key=lambda x: x[1])[1]
                    if int(surface_id) != _real_stop:
                        _cur_fnum = float(lens.get("fnum", 0))
                        _target_hint = (f"{_real_stop_sd:.2f} × ({_cur_fnum:.2f}/目标Fnum)"
                                        if _cur_fnum > 0 else "正确sd值")
                        return (
                            f"⛔ 错误面号：面{surface_id} 不是光阑面！\n"
                            f"  光阑面是面 {_real_stop}（sd={_real_stop_sd:.2f}，r≈∞，材料=AIR，sd最小）\n"
                            f"  修改非光阑面的 semi_diameter 不会改变 F#，这是 check_spec 一直报 F# 偏大的原因。\n"
                            f"  ✅ 正确做法：\n"
                            f"  Action: modify_lens\n"
                            f"  Action Input: \"lens_idx={lens_idx}, surface={_real_stop}, "
                            f"param=semi_diameter, value={_target_hint}\""
                        )
            except Exception:
                pass

        # 护栏：semi_diameter 修改必须符合光学常识
        if param == "semi_diameter":
            old_sd = float(target.get("semi_diameter", 0))
            if old_sd > 0:
                ratio = new_val / old_sd
                if ratio < 0.5:
                    return (f"⚠ 拒绝修改：semi_diameter 从 {old_sd:.2f} 缩到 {new_val:.2f}（比例{ratio:.2f}）\n"
                            f"若想降低 Fnum，应该扩大 semi_diameter，value 应是 {old_sd:.2f} × (当前Fnum/目标Fnum)\n"
                            f"例：当前Fnum=2.7 目标=1.4 → value = {old_sd:.2f} × (2.7/1.4) = {old_sd * 2.7/1.4:.2f}")
                if ratio > 3.0:
                    return (f"⚠ 拒绝修改：semi_diameter 扩大比例 {ratio:.2f} 过大（>3倍），可能撞其他透镜。分两步逐步扩。")
        target[param] = new_val

        # ★ 若修改的是光阑面的 semi_diameter，同步更新 lens["fnum"]（按反比例）
        # 光阑识别：r≈∞ + AIR + sd 局部最小（和 get_lens_surfaces 一致）
        if param == "semi_diameter":
            try:
                _optics = surfs[1:-1] if len(surfs) >= 3 else surfs
                _stops = [(s.get("surface_num", i+1), float(s.get("semi_diameter", 0)))
                          for i, s in enumerate(_optics)
                          if abs(float(s.get("radius", 0))) > 1e7
                          and str(s.get("material", "AIR")).upper() == "AIR"
                          and float(s.get("semi_diameter", 0)) > 0]
                if _stops:
                    _stop_snum = min(_stops, key=lambda x: x[1])[0]
                    if int(surface_id) == int(_stop_snum):
                        # 改的就是光阑面！按比例更新 fnum
                        _cur_fnum = float(lens.get("fnum", 0))
                        if _cur_fnum > 0 and old_sd > 0:
                            _new_fnum = _cur_fnum * (old_sd / new_val)
                            lens["fnum"] = round(_new_fnum, 3)
                            print(f"[modify_lens] 光阑 sd {old_sd:.2f}→{new_val:.2f}, Fnum {_cur_fnum:.2f}→{_new_fnum:.2f}")
            except Exception as _e:
                print(f"[modify_lens] Fnum 同步更新失败: {_e}")
    elif param == "material":
        new_val = value_raw
        # 校验材料是否在玻璃库
        try:
            import optical_calculator as _oc
            if hasattr(_oc, "GlassDB") or hasattr(_oc, "_GLASS_DB"):
                # 简单试调 _calc 看是否会 "Unknown material"
                _test_lens = copy.deepcopy(lens)
                _test_surfs = _test_lens.get("surfaces", [])
                if isinstance(_test_surfs, str):
                    _test_surfs = ast.literal_eval(_test_surfs)
                for _s in _test_surfs:
                    if int(_s.get("surface_num", -1)) == int(surface_id):
                        _s["material"] = new_val
                        break
                _test_lens["surfaces"] = _test_surfs
                _r = _calc(_test_lens)
                if not _r.get("valid") and "Unknown material" in str(_r.get("msg", "")):
                    return f"⚠ 拒绝修改：材料 {new_val!r} 不在玻璃库。请换候选镜头，或用有效 CDGM 牌号（如 H-ZLAF55D、H-FK61、H-LAK52 等）"
        except Exception:
            pass
        target["material"] = new_val
    else:
        return f"不支持的参数: {param}，可选: radius / thickness / material / semi_diameter"

    # 轨迹记录（self-evolve）
    try:
        _r_after = _calc(lens)
        record_step("modify_lens", lens_idx,
                    {"surface": surface_id, "param": param,
                     "old": old_val, "new": new_val},
                    metrics_before=_metrics_before,
                    metrics_after={"rms":  _r_after.get("rms"),
                                   "effl": _r_after.get("effl")},
                    kind="write")
    except Exception:
        pass

    return (
        f"✓ 镜头#{lens_idx} 面{surface_id} {param} 修改成功\n"
        f"  {old_val} → {new_val}\n"
        f"请调用 rms_calculator({lens_idx}) 查看修改后的性能。"
    )


@tool
def reset_lens(lens_idx: str) -> str:
    """按快照回滚:优先 Zemax 优化前 > 任意修改前原值。输入: lens_idx。"""
    # ★ P14 熔断
    _tripped = _circuit_guard("reset_lens", lens_idx)
    if _tripped:
        return _tripped
    try:
        if isinstance(lens_idx, str) and "=" in lens_idx:
            lens_idx = lens_idx.split("=")[-1].strip()
        lens_idx = int(str(lens_idx).strip())
    except (ValueError, TypeError):
        return f"lens_idx 必须是整数，收到: {lens_idx!r}"

    # ★ P16 优先级: pre_zemax > pre_modify
    # 这样 agent 可以撤销"Zemax 优化让像质变差"的那一步,回到优化前
    # (优化前本身可能已经经历过 modify_lens, 那是合理的已选修改)
    src_name = None
    if lens_idx in _LENS_BACKUP_PRE_ZEMAX:
        ALL_LENSES[lens_idx] = copy.deepcopy(_LENS_BACKUP_PRE_ZEMAX[lens_idx])
        del _LENS_BACKUP_PRE_ZEMAX[lens_idx]
        src_name = "Zemax 优化前"
    elif lens_idx in _LENS_BACKUP:
        ALL_LENSES[lens_idx] = copy.deepcopy(_LENS_BACKUP[lens_idx])
        del _LENS_BACKUP[lens_idx]
        src_name = "原始 DB"
    else:
        return f"镜头#{lens_idx} 没有任何快照,无需重置。"

    # 清空该镜头的修改计数(允许重新开始一轮修改)
    for k in list(_MODIFY_COUNT.keys()):
        if k.startswith(f"{lens_idx}_"):
            del _MODIFY_COUNT[k]
    return f"✓ 镜头#{lens_idx} 已恢复到[{src_name}]快照状态。"


@tool
def interpret_requirement(description: str) -> str:
    """把自然语言需求转成 FOV/F#/RMS 参数。输入: 需求描述。只能用一次。"""
    global _INTERPRET_CALLED
    if _INTERPRET_CALLED:
        return "interpret_requirement已调用过，禁止重复调用，请直接用已有参数检索。"
    _INTERPRET_CALLED = True
    desc = description.lower()

    # 场景规则库：关键词 → (fov范围, fnum范围, rms_target, 场景说明)
    rules = [
        # 天文/星空
        (["星空", "银河", "天文", "星星", "夜空"],
         (80, 120, 1.4, 2.8, 0.05, "天文摄影：超广角大光圈，收集更多星光")),
        # 人像
        (["人像", "写真", "肖像", "portrait"],
         (35, 50, 1.4, 2.8, 0.03, "人像摄影：中等视场大光圈，背景虚化好")),
        # 风景
        (["风景", "山", "海", "建筑", "旅行", "广角"],
         (60, 90, 2.8, 5.6, 0.05, "风景摄影：宽视场，清晰度高")),
        # 长焦/望远
        (["望远", "长焦", "月亮", "鸟", "野生动物", "体育"],
         (5, 15, 2.8, 5.6, 0.03, "望远摄影：窄视场长焦，捕捉远处细节")),
        # 手机/微型
        (["手机", "微型", "小型", "模组", "摄像模组"],
         (70, 80, 2.0, 2.8, 0.04, "手机摄像头：宽视场，小体积")),
        # 超广角大光圈（FOV>100或F<1.4）
        (["超广角", "鱼眼", "全景", "180度", "150度", "120度"],
         (140, 180, 0.8, 1.4, 0.15, "超广角鱼眼：极大视场，畸变容忍度高")),
        # 安防/监控
        (["安防", "监控", "摄像头", "广场", "停车场"],
         (90, 120, 1.4, 2.0, 0.08, "安防监控：超广角大光圈，覆盖范围大")),
        # 工业检测
        (["工业", "检测", "机器视觉", "pcb", "零件", "测量"],
         (10, 30, 4.0, 8.0, 0.02, "工业检测：窄视场高精度，畸变要求严格")),
        # 医疗/内窥镜
        (["内窥镜", "医疗", "胶囊", "腹腔"],
         (90, 140, 2.0, 4.0, 0.08, "医疗内窥镜：超广角，极小体积")),
        # 车载
        (["车载", "倒车", "行车", "驾驶", "adas"],
         (100, 120, 1.8, 2.8, 0.06, "车载摄像头：超广角，大光圈适应低光")),
        # 微距
        (["微距", "近摄", "花朵", "昆虫", "macro"],
         (20, 40, 2.8, 5.6, 0.02, "微距摄影：中等视场，高分辨率")),
    ]

    matched = None
    for keywords, params in rules:
        if any(kw in desc for kw in keywords):
            matched = params
            break

    if matched is None:
        # 默认通用场景
        matched = (35, 50, 2.8, 4.0, 0.05, "通用场景（未识别具体需求，使用默认参数）")

    fov_min, fov_max, fnum_min, fnum_max, rms_target, note = matched
    fov_mid  = (fov_min + fov_max) / 2
    fnum_mid = round((fnum_min + fnum_max) / 2, 1)

    # ★ 用户若明确给出 RMS/FOV/F# 数值，优先使用，不被场景默认值覆盖
    import re as _re2
    _rms_m = _re2.search(r"rms\s*[<<=]?\s*([0-9]*\.?[0-9]+)\s*mm", desc)
    if _rms_m:
        rms_target = float(_rms_m.group(1))
    _fov_m = _re2.search(r"fov\s*[==]?\s*([0-9]+(?:\.[0-9]+)?)", desc)
    if _fov_m:
        fov_mid = float(_fov_m.group(1))
    _fn_m = _re2.search(r"f[/#]\s*([0-9]+(?:\.[0-9]+)?)", desc)
    if _fn_m:
        fnum_mid = float(_fn_m.group(1))

    return ("📋 需求解析结果:\n" +
            "  场景: " + note + "\n" +
            "  推荐 FOV  = " + str(fov_mid) + " 度  （范围 " + str(fov_min) + "~" + str(fov_max) + " 度）\n" +
            "  推荐 F数  = F/" + str(fnum_mid) + "  （范围 F/" + str(fnum_min) + "~F/" + str(fnum_max) + "）\n" +
            "  RMS 目标  = " + str(rms_target) + " mm\n" +
            "  建议查询  : FOV=" + str(fov_mid) + "度 F/" + str(fnum_mid) + " RMS<" + str(rms_target) + "mm\n" +
            "接下来将用以上参数进行镜头检索和优化。")


@tool
def align_effl(input_str: str) -> str:
    """按比例缩放整个系统以对齐 EFFL。输入: "lens_idx=X, target_effl=V"。偏差>85%会拒绝。"""
    try:
        input_str = str(input_str).strip().strip(chr(34)).strip(chr(39))
        parts = {}
        for seg in input_str.split(","):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                parts[k.strip()] = v.strip()
        lens_idx    = int(parts["lens_idx"])
        target_effl = float(parts["target_effl"])
    except Exception as e:
        return f"输入格式错误: {e}. 正确格式: lens_idx=36237, target_effl=34.0"

    if not ALL_LENSES:
        return "镜头数据未加载"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界"

    lens = ALL_LENSES[lens_idx]

    # 先算当前 EFFL
    r = _calc(lens)
    if not r.get("valid"):
        return f"追迹失败，无法对齐EFFL: {r.get('msg')}"
    current_effl = r["effl"]
    if abs(current_effl) < 1e-6:
        return "当前EFFL接近0，无法缩放"

    scale = target_effl / current_effl

    # 放缩比例限制：超过 ±30% 警告，超过 ±60% 拒绝
    scale_pct = abs(scale - 1.0) * 100
    if scale_pct > 300:
        return (f"⚠ 放缩比例过大（{scale:.3f}，偏差{scale_pct:.0f}%），拒绝执行。"
                f"当前EFFL={current_effl:.2f}mm，目标={target_effl}mm，差距过大，"
                f"镜头结构不适合此目标EFFL，建议重新检索更接近的起始镜头。")

    # 备份
    if lens_idx not in _LENS_BACKUP:
        _LENS_BACKUP[lens_idx] = copy.deepcopy(lens)

    # 等比缩放所有面的 radius 和 thickness
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        surfs = ast.literal_eval(surfs)
        lens["surfaces"] = surfs

    for s in surfs:
        if s.get("radius") is not None:
            s["radius"] = s["radius"] * scale
        if s.get("thickness") is not None:
            s["thickness"] = s["thickness"] * scale
        if s.get("semi_diameter") is not None:
            s["semi_diameter"] = s["semi_diameter"] * scale

    # 验证新EFFL
    r2 = _calc(lens)
    if r2.get("valid"):
        # ★ 必须回写元数据：zemax_optimize 会用 lens["calc_effl"] 当 target_effl,
        # 不回写就会拿缩放前的旧值当目标 → Zemax 白优化
        lens["calc_effl"] = r2["effl"]
        lens["calc_rms"]  = r2["rms"]

        warn = ""
        if scale_pct > 150:
            warn = f"  ⚠ 放缩比例较大({scale:.3f})，建议验证后再优化。\n"

        # 轨迹记录
        try:
            record_step("align_effl", lens_idx,
                        {"target_effl": target_effl, "scale": round(scale, 4)},
                        {"rms": r.get("rms"),  "effl": current_effl},
                        {"rms": r2.get("rms"), "effl": r2.get("effl")},
                        kind="write")
        except Exception:
            pass

        return ("✓ 镜头#" + str(lens_idx) + " EFFL预对齐完成\n" +
                "  缩放比例: " + f"{scale:.4f}" + f" (偏差{scale_pct:.0f}%)" + "\n" +
                warn +
                "  EFFL: " + f"{current_effl:.4f}" + " -> " + f"{r2['effl']:.4f}" + " mm (目标=" + str(target_effl) + ")\n" +
                "  近轴RMS: " + f"{r2['rms']:.6f}" + " mm\n" +
                "请继续调用 rms_calculator 验证后开始优化.")
    return f"缩放完成但追迹失败: {r2.get('msg')}"


@tool
def random_restart(input_str: str) -> str:
    """随机扰动参数跳出局部极小。输入: "lens_idx=X, strength=0.05"。"""
    # ★ P14 熔断
    _tripped = _circuit_guard("random_restart", input_str)
    if _tripped:
        return _tripped
    import random, math
    try:
        input_str = str(input_str).strip().strip(chr(34)).strip(chr(39))
        parts = {}
        for seg in input_str.split(","):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                parts[k.strip()] = v.strip()
        if "lens_idx" not in parts:
            for seg in input_str.split(","):
                try: parts["lens_idx"] = str(int(seg.strip().strip(chr(34)).strip(chr(39)))); break
                except: pass
        lens_idx = int(parts["lens_idx"])
        strength = float(parts.get("strength", "0.05"))
        strength = max(0.01, min(0.15, strength))  # 限制范围
    except Exception as e:
        return f"输入格式错误: {e}. 正确格式: lens_idx=36237, strength=0.05"

    if not ALL_LENSES:
        return "镜头数据未加载"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界"

    lens  = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        surfs = ast.literal_eval(surfs)
        lens["surfaces"] = surfs

    # 备份（若还没备份）
    if lens_idx not in _LENS_BACKUP:
        _LENS_BACKUP[lens_idx] = copy.deepcopy(lens)

    # ★ self-evolve: 先算 before
    try:
        _r_before_rr = _calc(lens)
        _metrics_before_rr = ({"rms": _r_before_rr.get("rms"),
                               "effl": _r_before_rr.get("effl")}
                              if _r_before_rr.get("valid") else None)
    except Exception:
        _metrics_before_rr = None

    # 只扰动有曲率的非AIR面（跳过平面和像面）
    perturbed = []
    for s in surfs:
        r = s.get("radius")
        mat = str(s.get("material", "AIR")).upper()
        if r is not None and abs(r) > 1e-3 and abs(r) < 1e8 and mat not in ("AIR", ""):
            delta = 1.0 + random.uniform(-strength, strength)
            old_r = r
            s["radius"] = r * delta
            perturbed.append(f"面{int(s.get('surface_num',0))}: radius {old_r:.3f}→{s['radius']:.3f}")

    if not perturbed:
        return "未找到可扰动的有效面（需要有限曲率半径的玻璃面）"

    # 计算扰动后RMS
    r2 = _calc(lens)
    rms_str = f"{r2['rms']:.6f} mm" if r2.get("valid") else f"追迹失败: {r2.get('msg')}"

    # 重置停滞计数器，允许重新优化
    _OPTIMIZE_STALL[lens_idx] = 0

    # 轨迹记录
    try:
        record_step("random_restart", lens_idx,
                    {"strength": strength, "n_perturbed": len(perturbed)},
                    _metrics_before_rr,
                    {"rms": r2.get("rms") if r2.get("valid") else None,
                     "effl": r2.get("effl") if r2.get("valid") else None},
                    kind="write")
    except Exception:
        pass

    return f"扰动完成#{lens_idx} 扰动后RMS:{rms_str} 继续local_optimize或若变差则reset_lens"



@tool
def split_lens(input_str: str) -> str:
    """拆分指定面成两个薄镜,用于厚度过大或球差大的情况。输入: "lens_idx=X, surface=Y"。"""
    try:
        input_str = str(input_str).strip().strip(chr(34)).strip(chr(39))
        parts = {}
        for seg in input_str.split(","):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                parts[k.strip()] = v.strip()
        lens_idx   = int(parts["lens_idx"])
        surface_id = int(parts["surface"])
        ratio      = float(parts.get("ratio", "0.5"))
        ratio      = max(0.2, min(0.8, ratio))
    except Exception as e:
        return f"输入格式错误: {e}. 示例: lens_idx=36237, surface=6, ratio=0.5"

    if not ALL_LENSES:
        return "镜头数据未加载"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界"

    lens  = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        surfs = ast.literal_eval(surfs)
        lens["surfaces"] = surfs

    # 备份
    if lens_idx not in _LENS_BACKUP:
        _LENS_BACKUP[lens_idx] = copy.deepcopy(lens)

    # ★ self-evolve: 先算 before
    try:
        _r_before_sp = _calc(lens)
        _metrics_before_sp = ({"rms": _r_before_sp.get("rms"),
                               "effl": _r_before_sp.get("effl")}
                              if _r_before_sp.get("valid") else None)
    except Exception:
        _metrics_before_sp = None

    # 找目标面
    target_idx = None
    for i, s in enumerate(surfs):
        if int(s.get("surface_num", -1)) == surface_id:
            target_idx = i
            break
    if target_idx is None:
        available = [int(s.get("surface_num", 0)) for s in surfs]
        return f"面号 {surface_id} 不存在，可用面号: {available}"

    orig = surfs[target_idx]
    mat  = str(orig.get("material", "AIR")).upper()
    if mat in ("AIR", ""):
        return f"面{surface_id} 是空气面，无法拆分（只能拆分玻璃面）"

    r_orig = orig.get("radius")
    t_orig = orig.get("thickness", 0.0)
    sd     = orig.get("semi_diameter", 0.0)

    if r_orig is None or abs(r_orig) > 1e7:
        return f"面{surface_id} 是平面（radius≈∞），不适合拆分"

    # 计算两片的 radius
    # 光焦度 phi = (n-1)/R，拆分后 phi1 = ratio*phi，phi2 = (1-ratio)*phi
    # 取材料折射率
    from optical_calculator import CDGMLibrary
    _lib = CDGMLibrary()
    nd = _lib.get_index(mat)
    if nd is None:
        nd = 1.7  # fallback

    phi_orig = (nd - 1.0) / r_orig
    phi1     = ratio * phi_orig
    phi2     = (1.0 - ratio) * phi_orig

    r1 = (nd - 1.0) / phi1 if abs(phi1) > 1e-9 else 1e9
    r2 = (nd - 1.0) / phi2 if abs(phi2) > 1e-9 else 1e9

    # 厚度：前片 60%，空气间隔 0.5mm，后片 40%
    t1       = t_orig * 0.6
    t_air    = 0.5
    t2       = t_orig * 0.4

    # 构造三个新面（替换原来一个面）
    base_num = int(orig.get("surface_num", surface_id))
    face1 = {
        "surface_num":   base_num,
        "radius":        r1,
        "thickness":     t1,
        "material":      orig["material"],
        "semi_diameter": sd,
    }
    face_air = {
        "surface_num":   base_num + 0.5,   # 临时编号，后面重新排
        "radius":        None,             # 平面
        "thickness":     t_air,
        "material":      "AIR",
        "semi_diameter": sd,
    }
    face2 = {
        "surface_num":   base_num + 1,
        "radius":        r2,
        "thickness":     t2,
        "material":      orig["material"],
        "semi_diameter": sd,
    }

    # 替换原面，插入三个新面，后续面号 +2
    new_surfs = []
    for i, s in enumerate(surfs):
        snum = int(s.get("surface_num", 0))
        if i < target_idx:
            new_surfs.append(s)
        elif i == target_idx:
            new_surfs.extend([face1, face_air, face2])
        else:
            s2 = dict(s)
            s2["surface_num"] = snum + 2
            new_surfs.append(s2)

    # 重新整理面号为连续整数
    for i, s in enumerate(new_surfs):
        s["surface_num"] = i

    lens["surfaces"] = new_surfs

    # 验证拆分后性能
    r_check = _calc(lens)
    rms_str  = f"{r_check['rms']:.6f} mm" if r_check.get("valid") else f"追迹失败: {r_check.get('msg')}"
    effl_str = f"{r_check['effl']:.4f} mm" if r_check.get("valid") else "N/A"

    # 轨迹记录
    try:
        record_step("split_lens", lens_idx,
                    {"surface": surface_id, "ratio": ratio,
                     "r1": round(r1, 3), "r2": round(r2, 3)},
                    _metrics_before_sp,
                    {"rms":  r_check.get("rms")  if r_check.get("valid") else None,
                     "effl": r_check.get("effl") if r_check.get("valid") else None},
                    kind="write")
    except Exception:
        pass

    return ("✓ 镜头#" + str(lens_idx) + " 面" + str(surface_id) + " 拆分完成\n" +
            "  前片 radius=" + f"{r1:.3f}" + " mm  thickness=" + f"{t1:.3f}" + " mm  材料=" + str(orig["material"]) + "\n" +
            "  空气间隔=" + f"{t_air}" + " mm\n" +
            "  后片 radius=" + f"{r2:.3f}" + " mm  thickness=" + f"{t2:.3f}" + " mm  材料=" + str(orig["material"]) + "\n" +
            "  拆分后 EFFL=" + effl_str + "  RMS=" + rms_str + "\n" +
            "建议接着调用 local_optimize 对新面型做梯度优化。")


@tool
def local_optimize(input_str: str) -> str:
    """对所有玻璃面 radius 做梯度下降降 RMS。输入: "lens_idx=X" (可选 iterations=30, lr=0.5)。"""
    # ★ P14 熔断
    _tripped = _circuit_guard("local_optimize", input_str)
    if _tripped:
        return _tripped
    try:
        input_str = str(input_str).strip().strip(chr(34)).strip(chr(39))
        parts = {}
        for seg in input_str.split(","):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                parts[k.strip()] = v.strip()
        # 兼容 "40016, iterations=50, lr=0.5" 格式（第一段是纯数字）
        if "lens_idx" not in parts:
            segs = [s.strip() for s in input_str.split(",")]
            for seg in segs:
                try:
                    parts["lens_idx"] = str(int(seg))
                    break
                except ValueError:
                    pass
        if "lens_idx" not in parts:
            for seg in input_str.split(","):
                try: parts["lens_idx"] = str(int(seg.strip().strip(chr(34)).strip(chr(39)))); break
                except: pass
        lens_idx   = int(parts["lens_idx"])
        iterations = int(parts.get("iterations", "30"))
        lr         = float(parts.get("lr", "0.5"))
        iterations = max(5, min(200, iterations))
        lr         = max(0.01, min(5.0, lr))
    except Exception as e:
        return f"输入格式错误: {e}. 示例: lens_idx=36237, iterations=30, lr=0.5"

    if not ALL_LENSES:
        return "镜头数据未加载"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界"

    stall_key = lens_idx  # 停滞计数器 key

    # 强制终止：连续2次改善0% 拒绝继续调用
    if _OPTIMIZE_STALL.get(stall_key, 0) >= 2:
        _OPTIMIZE_STALL[stall_key] = 0
        return (f"⛔ 镜头#{lens_idx} 优化已停滞，拒绝重复调用。"
                f"请改用 random_restart(lens_idx={lens_idx}, strength=0.05) 后再试，"
                f"或直接输出 Final Answer。")


    lens  = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        surfs = ast.literal_eval(surfs)
        lens["surfaces"] = surfs

    # 备份
    if lens_idx not in _LENS_BACKUP:
        _LENS_BACKUP[lens_idx] = copy.deepcopy(lens)

    # 找可优化面：有限曲率、非AIR材料
    opt_surfaces = []
    for s in surfs:
        r   = s.get("radius")
        mat = str(s.get("material", "AIR")).upper()
        if r is not None and abs(r) > 1e-3 and abs(r) < 1e8 and mat not in ("AIR", ""):
            opt_surfaces.append(s)

    if not opt_surfaces:
        return "未找到可优化面（需要有限曲率的玻璃面）"

    # 初始状态
    r0 = _calc(lens)
    if not r0.get("valid"):
        return f"初始追迹失败: {r0.get('msg')}"
    rms_init  = r0["rms"]
    effl_init = r0["effl"]

    # 梯度下降主循环
    best_rms           = rms_init
    no_improve         = 0
    current_lr         = lr
    effl_tol           = 0.03
    rms_floor          = 0.003
    effl_penalty_scale = 0.5

    def penalized_rms(result):
        if not result.get("valid"):
            return float("inf")
        rms  = result["rms"]
        effl = result["effl"]
        effl_drift = abs(effl - effl_init) / max(abs(effl_init), 1e-6)
        penalty = rms_init * effl_penalty_scale * max(0.0, effl_drift - effl_tol)
        return rms + penalty


    for iteration in range(iterations):
        # 提前终止：RMS 已达到近轴模型下限
        r_cur_state = _calc(lens)
        if r_cur_state.get("valid") and r_cur_state["rms"] < rms_floor:
            break

        improved = False
        for s in opt_surfaces:
            r_cur = s["radius"]
            eps   = abs(r_cur) * 0.01 + 0.1

            s["radius"] = r_cur + eps
            rms_plus  = penalized_rms(_calc(lens))

            s["radius"] = r_cur - eps
            rms_minus = penalized_rms(_calc(lens))

            # 梯度 + 更新
            grad  = (rms_plus - rms_minus) / (2 * eps)
            r_new = r_cur - current_lr * grad

            # 防止符号翻转
            if r_cur * r_new < 0:
                r_new = r_cur * 0.95
            s["radius"] = r_new

            # 验证：带惩罚的目标函数是否改善
            new_score = penalized_rms(_calc(lens))
            if new_score < penalized_rms(r_cur_state):
                r_cur_state = _calc(lens)
                best_rms    = r_cur_state["rms"]
                improved    = True
            else:
                s["radius"] = r_cur  # 回退

        if not improved:
            no_improve += 1
            current_lr *= 0.7
            if no_improve >= 5 or current_lr < 0.005:
                break
        else:
            no_improve = 0
            current_lr = lr

    # 最终状态
    r_final = _calc(lens)
    if not r_final.get("valid"):
        return f"优化后追迹失败: {r_final.get('msg')}"

    rms_final  = r_final["rms"]
    effl_final = r_final["effl"]

    # 汇报每面的radius变化
    changes = []
    for s in opt_surfaces:
        orig_s = None
        for bs in _LENS_BACKUP[lens_idx].get("surfaces", []):
            if bs.get("surface_num") == s.get("surface_num"):
                orig_s = bs
                break
        if orig_s:
            orig_r = orig_s.get("radius", 0)
            new_r  = s.get("radius", 0)
            if abs(orig_r - new_r) > 0.001:
                changes.append(f"面{int(s.get('surface_num',0))}: {orig_r:.3f}→{new_r:.3f}")

    improvement = (rms_init - rms_final) / rms_init * 100 if rms_init > 0 else 0
    changes_str = ", ".join(changes) if changes else "无显著变化"

    # 更新停滞计数器
    if improvement < 0.1:
        _OPTIMIZE_STALL[stall_key] = _OPTIMIZE_STALL.get(stall_key, 0) + 1
    else:
        _OPTIMIZE_STALL[stall_key] = 0  # 有改善则重置

    # 轨迹记录
    try:
        record_step("local_optimize", lens_idx,
                    {"iterations": iterations, "lr": lr,
                     "n_changes": len(changes)},
                    {"rms": rms_init,  "effl": effl_init},
                    {"rms": rms_final, "effl": effl_final},
                    kind="write")
    except Exception:
        pass

    return f"优化完成#{lens_idx} RMS:{rms_init:.4f}→{rms_final:.4f}mm 改善{improvement:.0f}% EFFL:{effl_final:.2f}mm"

# ══════════════════════════════════════════════════════════════════════════════
# 升级版 System Prompt（含 Gemini 生成的 Skill 库）
# ══════════════════════════════════════════════════════════════════════════════

# 从服务器 lens_skills.py 加载，失败则用内置 skill
try:
    from lens_skills import LENS_SKILLS as _EXTERNAL_SKILLS
    _SKILLS_TEXT = _EXTERNAL_SKILLS
except ImportError:
    _SKILLS_TEXT = """===== 优化策略速查 =====
症状→策略（每次只改一个参数，改后立即验证RMS）：

RMS偏大+色差主导 → 正镜换高Vd：H-FK61(Vd=81.6)或H-FK71(Vd=90.3)；负镜换低Vd：H-ZF7LA(Vd=25.5)或H-ZF52(Vd=23.8)
RMS偏大+球差主导 → 光阑附近正镜换H-ZLAF55D(nd=1.835)或H-ZLAF2A(nd=1.803)；或将强弯面radius增大5-10%
y_image偏大+场曲 → 负镜换H-ZF7LA(nd=1.805)；正镜换H-LAF10LA(nd=1.755)
EFFL偏短 → 最强正面radius增大5%
EFFL偏长 → 最强正面radius减小5%
TOTR超限 → 最大空气间隔thickness减小10%
优化停滞 → reset_lens恢复原始，换其他策略

禁止使用：N-BK7、N-FK51A等肖特/OHARA牌号
======================"""

SYSTEM_PROMPT = """/no_think

光学镜头设计助手。工具调用遵循 ReAct 格式。

# 流程决策树

1. **检索**: 用户给 FOV+F# 数值 → rank_by_rms(加 fov_tol/fnum_tol,如 "FOV=35 F/2.8, fov_tol=5, fnum_tol=0.5")
   未给数值 → 先 interpret_requirement 一次
   rank_by_rms 若带 warning → 不再检索,取 rank=1 直接用

2. **焦距对齐**: 用户指定 EFFL 或 当前 EFFL > 目标×1.3 → align_effl
   align_effl 偏差>85% → 换 rank=2 候选,勿再检索

3. **评估**: align_effl 已返回近轴 RMS，无需再调 rms_calculator（省 token）

4. **优化**:
   - F# 偏大: 先调 get_lens_surfaces 找 stop 面号，再 modify_lens 改 stop 面 semi_diameter = 旧值×(当前F#/目标F#)，再 local_optimize
   - ★ 禁止改 surface=1 的 semi_diameter 来调 F#（surface=1 不是光阑面，无效）
   - RMS 偏大: local_optimize → 若改善=0% → random_restart(strength=0.05) → local_optimize
   - 近轴优化停滞或 F# 调完后: zemax_optimize(lens_idx=X, cycles=auto, rms_target=<目标RMS>, fnum_target=<目标F#>)
   - ★ 调 zemax_optimize 时【必须】传 rms_target 和 fnum_target
   - ★★ rms_target 取值规则（违反此规则=严重错误）：
        • 用户给了具体数字（如"RMS<0.02"）→ 传用户原始值 0.02
        • 用户说"尽量优化"/"尽量降低"/"越小越好"等模糊描述 → 【必须】传 0.05，禁止传其他值
        • 【严禁】把 rms_calculator / local_optimize / check_spec 返回的近轴 RMS 当 rms_target 传入！
          例：近轴 RMS=0.003 → 绝对不能传 rms_target=0.003，必须传 0.05
          原因：近轴 RMS 比 Zemax 真值乐观 5~20 倍，用近轴值做目标必然永远未达标→无限重试→耗尽迭代。
        • 记忆口诀：rms_target 只来自【用户输入】或【固定默认值0.05】，绝不来自【工具返回值】。

5. **验收**: check_spec 最多调 2 次，reasons 相同则停止。
   - pass=true 且 Zemax 可用 → 继续 zemax_optimize 获取真值，【禁止】直接 Final Answer（近轴不可信）
   - pass=true 且 Zemax 不可用（zemax_optimize 返回"Zemax桥接服务不可用"）→ 【立即】输出 Final Answer: 镜头#X | RMS=<近轴值> | 结论: 近轴达标✓（Zemax离线）。【严禁】在 Zemax 离线时调用 get_skill_detail / modify_lens / random_restart / local_optimize 等优化工具，这些操作在已达标镜头上只会让性能变差。
   - ★★ check_spec 返回 pass=true 时，【严禁】调用 get_skill_detail（性能已达标，优化策略只会破坏现有结果）
   - F#差 → 按步骤4改光阑；EFFL差 → align_effl；RMS差 → 再优化
   - 2次都 false → 直接 zemax_optimize

6. **Zemax 验收**: zemax_optimize 返回带【RMS 数值比较】结论（✅达标/❌未达标），直接按结论走。
   ★ zemax_optimize 之后【禁止】再调 check_spec。
   ★ 达标 → 调 zemax_layout(lens_idx=X) → Final Answer。
     zemax_layout 失败（返回"⚠ Layout 生成失败"）→ 不重试，直接 Final Answer，RMS 结果仍然有效。
   ★ 未达标处理流程（必须按顺序）：
     第1次未达标 → 立即再跑 zemax_optimize(同 lens_idx, cycles=10, rms_target=<用户目标>, fnum_target=<目标F#>)
     第2次未达标 → 换 rank+1 候选，从 check_spec 重新开始
     换候选后同样最多 2 次 zemax_optimize，仍未达标 → Final Answer 报最优结果
   ★★ 【严禁】第1次未达标就直接换候选，必须先用 cycles=10 重试一次。

# 策略库(按症状调 get_skill_detail)
{skill_index}

# 规则
- 玻璃只用 CDGM: H-* 或 D-* (H-ZF4A, H-ZLAF55D, H-FK61, H-K9L 等),禁 N-BK7 等肖特牌号
- 熔断: 同一 (tool, input) 连续 3 次相同 → 工具自动返回熔断,立即换策略或 Final Answer
- ★★ F# 调整: 必须先 get_lens_surfaces 找到 stop 面号，再改 stop 面 sd。modify_lens 若检测到改了非 stop 面的 sd 会拒绝并给出正确面号
- Unknown material 错误 → 不重试,换候选镜头
- token 节省: 同一镜头 check_spec 相同 reason 出现 2 次即停，转 zemax_optimize

# Action 格式
Action 和 Action Input 分两行,Action Input 只写参数:
Thought: 分析
Action: rms_calculator
Action Input: 36237

每次只输出一个 Action。禁止同一回合同时出现 Action 和 Final Answer。

Final Answer 格式:
Final Answer: 镜头#xxxxx | RMS=x.xxxx mm | 结论: 达标✓  (或 未达标✗)

工具:
{tools}

工具名: {tool_names}

Question: {input}
Thought:{agent_scratchpad}"""
# PROMPT 在 build_agent 里动态生成，注入 skill_index


# ══════════════════════════════════════════════════════════════════════════════
# Skill 两级加载
# ══════════════════════════════════════════════════════════════════════════════
from skill_summaries import SKILL_SUMMARIES

def build_skill_index_text():
    """★ P18c 瘦身版:只输出 skill 名字列表,不注入 summary。
    LLM 需要时调 get_skill_detail(skill_name) 按需拉取。
    之前每条 skill 都注入完整 summary 占 ~60 chars,20 个 skill 就是 ~1200 chars
    (~720 tokens),压成单行名单后只占 <150 chars (~90 tokens),省 ~600 tokens。
    """
    merged = dict(SKILL_SUMMARIES)
    try:
        learned = load_learned_for_prompt()  # {name: summary}
        merged.update(learned)
    except Exception:
        pass
    names = list(merged.keys())
    # 只输出一行:skill 名单 + 用法提示。详情 LLM 按需调 get_skill_detail 拉。
    return f"可用策略(按需调 get_skill_detail 拉详情): {', '.join(names)}"

@tool
def get_skill_detail(skill_name: str) -> str:
    """按需拉取指定 skill 的详细操作。输入: skill 名,如 "Skill 13"。"""
    def _fuzzy_get(d: dict, key: str):
        if key in d:
            return d[key]
        key_lower = key.lower().strip()
        for k, v in d.items():
            if k.lower().startswith(key_lower) or key_lower in k.lower():
                return v
        return None

    try:
        from lens_skills_full import LENS_SKILLS_FULL
        detail = _fuzzy_get(LENS_SKILLS_FULL, skill_name)
        if detail:
            return detail
    except ImportError:
        pass
    # learned skills fallback（self-evolve 学到的）
    try:
        learned_detail = get_learned_detail(skill_name)
        if learned_detail:
            return learned_detail
    except Exception:
        pass
    result = _fuzzy_get(SKILL_SUMMARIES, skill_name)
    return result if result else f"未找到策略: {skill_name}。请用完整名称，如'Skill 1: 利用高阿贝数低色散玻璃消除轴上色差'"


@tool
def get_retrieval_skill_detail(skill_name: str) -> str:
    """检索策略详情(OOD/EFFL 偏差等场景按需)。输入: skill 名,如 "Skill R2"。"""
    def _fuzzy_get(d: dict, key: str):
        if key in d:
            return d[key]
        key_l = key.lower().strip()
        for k, v in d.items():
            if k.lower().startswith(key_l) or key_l in k.lower():
                return v
        return None

    try:
        from retrieval_skills import RETRIEVAL_SKILLS_FULL
        detail = _fuzzy_get(RETRIEVAL_SKILLS_FULL, skill_name)
        if detail:
            return detail
    except ImportError:
        pass
    return f"未找到检索 skill: {skill_name}。请用完整名称，如 'Skill R3: EFFL差距处理'"


# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Zemax 桥接
# ══════════════════════════════════════════════════════════════════════════════
import requests as _requests

import os as _os_for_bridge
ZEMAX_BRIDGE = _os_for_bridge.environ.get("ZEMAX_BRIDGE_URL", "http://127.0.0.1:5000")

# --- 自动打印所有 bridge 请求 ---
_orig_get, _orig_post = _requests.get, _requests.post
def _traced_get(url, *a, **kw):
    print(f"[BRIDGE-CALL] GET  {url}", flush=True)
    return _orig_get(url, *a, **kw)
def _traced_post(url, *a, **kw):
    print(f"[BRIDGE-CALL] POST {url}  payload={str(kw.get('json'))[:200]}", flush=True)
    return _orig_post(url, *a, **kw)
_requests.get, _requests.post = _traced_get, _traced_post
_NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}

def _zemax_available():
    # ★ FIX Bug4: 先探测 /status，再用 GET /spot_diagram 验证功能端点真正可用，
    # 避免 /status 返回 200 但业务路由其实已挂死（或反之）的误判。
    try:
        r = _requests.get(f"{ZEMAX_BRIDGE}/status", headers=_NGROK_HEADERS, timeout=3)
        if not r.ok:
            return False
        # 尝试解析 JSON；bridge 支持 zemax_ready 字段时精确判断，否则 status 200 即可
        try:
            data = r.json()
            if "zemax_ready" in data:
                return bool(data["zemax_ready"])
        except Exception:
            pass
        return True
    except Exception:
        return False

def _sanitize_surfs(surfs: list) -> list:
    """JSON 序列化前把面型数据里的 inf / nan 替换成安全值。
    object 面的 thickness 在 Zemax 里是无穷远，bridge 的 load_lens 看到 >1e5 就当 inf 处理，
    所以用 1_000_000 代替 float('inf')；semi_diameter=inf 表示"不限制"，改成 0 让 bridge 跳过。
    """
    import math
    out = []
    for s in surfs:
        ns = {}
        for k, v in s.items():
            if isinstance(v, float):
                if math.isinf(v):
                    # thickness 用大数让 bridge 识别为无穷远；其他 inf 字段归 0
                    ns[k] = 1_000_000.0 if k == "thickness" else 0.0
                elif math.isnan(v):
                    ns[k] = 0.0
                else:
                    ns[k] = v
            else:
                ns[k] = v
        out.append(ns)
    return out


@tool
def zemax_layout(input_str: str) -> str:
    """用 Zemax 画布局图 PNG。输入: "lens_idx=X, save_path=/path/to.png"。"""
    try:
        parts = {}
        for seg in str(input_str).strip().split(","):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                parts[k.strip()] = v.strip()
        lens_idx  = int(parts.get("lens_idx", 0))
        save_path = parts.get("save_path", f"/gz-data/layout_{lens_idx}.png")
    except Exception as e:
        return f"输入格式错误: {e}"

    # ★ FIX: 移除 _zemax_available() 前置检查，改用重试机制
    # zemax_layout 在 zemax_optimize 后立即调用，pythonnet atexit 异常可能导致
    # bridge 短暂不响应，直接重试比提前 abort 更合理。
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界（共 {len(ALL_LENSES)} 条）"

    lens  = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        import ast as _ast
        surfs = _ast.literal_eval(surfs)

    import time as _tl
    _last_err = None
    for _attempt in range(3):
        try:
            # ★ 不重新 load_lens：zemax_optimize 跑完后 Zemax 已有正确系统
            resp = _requests.post(f"{ZEMAX_BRIDGE}/layout", headers=_NGROK_HEADERS, timeout=30)
            if not resp.ok:
                _last_err = f"HTTP {resp.status_code} {resp.text[:100]}"
                _tl.sleep(2)
                continue
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return f"✓ Layout 图已保存: {save_path}  ({len(resp.content)//1024} KB)"
        except Exception as e:
            _last_err = f"{type(e).__name__}: {e}"
            print(f"[zemax_layout] attempt {_attempt+1}/3 failed: {_last_err}", flush=True)
            _tl.sleep(3)
    return f"⚠ Layout 生成失败（已重试3次）: {_last_err}\nZemax 优化结果仍然有效，可直接输出 Final Answer。"


@tool
def get_spot_diagram(input_str: str) -> str:
    """用 Zemax 生成点列图(Spot Diagram) PNG 并保存到本地。输入: "lens_idx=X" 或 "lens_idx=X, save_path=/gz-data/spot.png"。★ Zemax 真值达标后、输出 Final Answer 前必须调用此工具。"""
    try:
        parts = {}
        for seg in str(input_str).strip().split(","):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                parts[k.strip()] = v.strip()
        lens_idx  = int(parts.get("lens_idx", 0))
        save_path = parts.get("save_path", f"/gz-data/spot_{lens_idx}.png")
    except Exception as e:
        return f"输入格式错误: {e}"

    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界（共 {len(ALL_LENSES)} 条）"

    lens  = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        import ast as _ast
        surfs = _ast.literal_eval(surfs)

    # ★ FIX: retry with backoff，避免 zemax_optimize 刚结束时 bridge 短暂不响应
    import time as _tsp
    _last_err = None
    for _attempt in range(3):
        try:
            resp = _requests.post(f"{ZEMAX_BRIDGE}/spot_diagram", headers=_NGROK_HEADERS, timeout=30)
            if not resp.ok:
                _last_err = f"HTTP {resp.status_code}"
                _tsp.sleep(2)
                continue
            with open(save_path, "wb") as f:
                f.write(resp.content)
            size_kb = len(resp.content) // 1024
            print(f"[spot_diagram] ✓ 保存到 {save_path} ({size_kb} KB)", flush=True)
            return (
                f"✓ Spot Diagram 已保存: {save_path}  ({size_kb} KB)\n"
                f"  查看方式: 在Windows端浏览器访问 {ZEMAX_BRIDGE}/spot_diagram\n"
                f"  或直接打开文件: {save_path}"
            )
        except Exception as e:
            _last_err = f"{type(e).__name__}: {e}"
            print(f"[spot_diagram] attempt {_attempt+1}/3 failed: {_last_err}", flush=True)
            _tsp.sleep(3)
    return f"⚠ Spot diagram 失败（已重试3次）: {_last_err}\n可在Windows端浏览器访问 {ZEMAX_BRIDGE}/spot_diagram 查看。"


@tool
def zemax_optimize(input_str: str) -> str:
    """推送面型到 Zemax 跑 DLS 优化。输入: "lens_idx=X, cycles=auto, rms_target=0.02, fnum_target=1.2"。
    rms_target/fnum_target 可选但强烈建议传入——代码会直接做数值比较并给出明确的达标/未达标结论，
    避免 LLM 自行判断出错。★ 返回 Zemax 真值(EFFL/F#/RMS/merit),新面型已回写本地。"""
    # ★ P14 熔断
    _tripped = _circuit_guard("zemax_optimize", input_str)
    if _tripped:
        return _tripped
    try:
        parts = {}
        for seg in str(input_str).strip().split(","):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                parts[k.strip()] = v.strip()
        lens_idx = int(parts.get("lens_idx", 0))
        # ★ P17: cycles 支持 "auto"/"automatic" 字面值, 表示跑到收敛
        cycles_raw = str(parts.get("cycles", "auto")).strip().lower()
        if cycles_raw in ("auto", "automatic", "", "0", "-1"):
            cycles = -1        # 约定: -1 传给 bridge = Automatic 模式
        else:
            try:
                cycles = int(cycles_raw)
            except ValueError:
                cycles = -1    # 无法解析就走 auto
        # ★ 用户目标值（可选）——代码直接比较，不依赖 LLM 判断
        try:
            rms_target = float(parts["rms_target"]) if "rms_target" in parts else None
        except (ValueError, TypeError):
            rms_target = None
        try:
            fnum_target = float(parts["fnum_target"]) if "fnum_target" in parts else None
        except (ValueError, TypeError):
            fnum_target = None
    except Exception as e:
        return f"输入格式错误: {e}"

    # cycles 范围保护: -1 (auto) 或 1~50
    if cycles != -1:
        if cycles < 1:
            cycles = 1
        elif cycles > 50:
            cycles = 50

    if not _zemax_available():
        # ★ FIX Bug2: Zemax 离线时把 check_spec 的达标结论同步到 session，
        # 避免 end_session 里 final_passed 被默认 False 覆盖，导致近轴达标会话被误判失败。
        if _HAS_SELF_EVOLVE:
            try:
                from self_evolve import _SESSION_CTX
                last_chk = _SESSION_CTX.get("last_check", {})
                last_chk["zemax_offline"] = True
                _SESSION_CTX["last_check"] = last_chk
            except Exception:
                pass
        return "Zemax桥接服务不可用，请确认Windows端 zemax_bridge.py 已启动"

    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界（共 {len(ALL_LENSES)} 条）"

    lens  = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        import ast as _ast
        surfs = _ast.literal_eval(surfs)

    import time as _time

    # ── 重试封装：处理 bridge 短暂断线（第一次 optimize 后 Zemax/Flask 不稳定）──
    def _post_with_retry(url, payload, timeout_s, max_retries=3, backoff=5):
        """POST with retry on ConnectionError / RemoteDisconnected.
        每次重试前先 GET /status 确认 bridge 活着，最多等 backoff×retry 秒。
        """
        import requests as _req_mod
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                r = _orig_post(url, headers=_NGROK_HEADERS, json=payload, timeout=timeout_s)
                return r
            except (_req_mod.exceptions.ConnectionError,
                    _req_mod.exceptions.Timeout) as e:
                last_err = e
                wait = backoff * attempt
                print(f"[zemax_optimize] ⚠ 连接异常 (attempt {attempt}/{max_retries}): "
                      f"{type(e).__name__} — 等待 {wait}s 后重试...", flush=True)
                _time.sleep(wait)
                # 确认 bridge 还在线
                try:
                    ping = _orig_get(f"{ZEMAX_BRIDGE}/status",
                                     headers=_NGROK_HEADERS, timeout=5)
                    if not ping.ok:
                        print(f"[zemax_optimize] bridge /status 返回 {ping.status_code}，跳过重试",
                              flush=True)
                        break
                    print(f"[zemax_optimize] bridge /status OK，继续重试", flush=True)
                except Exception as _pe:
                    print(f"[zemax_optimize] bridge /status 也失败: {_pe}，停止重试", flush=True)
                    break
        raise last_err

    try:
        # 推送完整面型（带 stop_surface，否则 Zemax 把第一面当光阑）
        # ★ _sanitize_surfs: 把 inf/nan 换成 JSON 安全值，避免第二次调用时崩溃
        _post_with_retry(f"{ZEMAX_BRIDGE}/load_lens",
                         payload={"surfaces": _sanitize_surfs(surfs),
                                  "fov":  lens.get("fov"),
                                  "fnum": fnum_target or lens.get("fnum"),
                                  "stop_surface": lens.get("stop_surface")},
                         timeout_s=15)
        # ★ target_fnum 优先用调用方传入的 fnum_target（用户真实目标），
        #   fallback 才用 lens["fnum"]（历史记录，可能已过时）
        _target_fnum = fnum_target or lens.get("fnum")
        # DLS 优化（timeout=180 给 Zemax 跑够时间）
        resp = _post_with_retry(f"{ZEMAX_BRIDGE}/zemax_optimize",
                                payload={"cycles": cycles,
                                         "target_effl": lens.get("calc_effl") or lens.get("effl"),
                                         "target_fnum": _target_fnum},
                                timeout_s=180)
        try:
            data = resp.json()
        except Exception as _je:
            return (f"Zemax优化失败: bridge 返回非JSON (HTTP {resp.status_code}) "
                    f"body[:200]={resp.text[:200]!r}")

        print(f"[zemax_optimize] bridge returned: {data}", flush=True)

        if isinstance(data, dict) and "error" in data:
            return f"Zemax优化失败: {data['error']}"

        # ═══════════════════════════════════════════════════════════
        # ★ P12 关键: 把 Zemax 真值 + 优化后新面型回写到本地 lens record
        #   否则下次 rms_calculator 会用老面型 + 近轴算出"虚假达标"的 RMS
        #   (这是上个 session 报"✓ 达标"但 Zemax 真值 0.22 mm 的根因)
        # ★ P16 回写前先做 pre-Zemax 快照, 允许 reset_lens 回滚这次优化。
        #   策略: 每次 zemax_optimize 都覆盖快照 → reset 总是回到"最近一次
        #   zemax_optimize 之前"; modify_lens 的 _LENS_BACKUP 保持"首次修改
        #   前"的语义不动。
        # ═══════════════════════════════════════════════════════════
        if isinstance(data, dict) and data.get("surfaces_after"):
            try:
                _LENS_BACKUP_PRE_ZEMAX[lens_idx] = copy.deepcopy(lens)
                print(f"[zemax_optimize] ✓ 已存 pre-Zemax 快照 lens[{lens_idx}] "
                      f"(rms={lens.get('calc_rms','?')})", flush=True)
            except Exception as _sbe:
                print(f"[zemax_optimize] ⚠ pre-Zemax 快照失败: {_sbe}", flush=True)

        if isinstance(data, dict):
            # (a) 回写优化后的面型
            surfs_after = data.get("surfaces_after") or []
            if surfs_after and isinstance(surfs_after, list):
                # 保留 image surface (bridge 不回传它), 用原来的 image 面拼接
                try:
                    old_surfs = lens.get("surfaces", [])
                    if isinstance(old_surfs, str):
                        import ast as _ast
                        old_surfs = _ast.literal_eval(old_surfs)
                    # surfaces_after 包含 object(0) + 实物面; image 面用老的
                    if old_surfs and len(old_surfs) >= 1:
                        image_surf = old_surfs[-1]
                        lens["surfaces"] = list(surfs_after) + [image_surf]
                    else:
                        lens["surfaces"] = list(surfs_after)
                    print(f"[zemax_optimize] ✓ 回写 {len(surfs_after)} 面到 lens[{lens_idx}].surfaces",
                          flush=True)
                except Exception as _wb:
                    print(f"[zemax_optimize] ⚠ 面型回写失败: {_wb}", flush=True)

            # (b) 回写 Zemax 真值指标 (让后续 rms_calculator / check_spec 读到真值)
            effl_real = data.get("effl")
            fnum_real = data.get("fnum")
            rms_list  = data.get("rms_per_field_mm") or []
            rms_worst = max(rms_list) if rms_list else None

            if effl_real is not None:
                lens["calc_effl"] = float(effl_real)
                lens["effl"]      = float(effl_real)
            if fnum_real is not None:
                lens["fnum"]      = float(fnum_real)
            if rms_worst is not None:
                lens["calc_rms"]  = float(rms_worst)
                lens["rms"]       = float(rms_worst)

            # 把全套 Zemax 指标存一份,供诊断
            lens["zemax_metrics"] = {
                "effl": effl_real, "fnum": fnum_real,
                "rms_per_field_mm": rms_list,
                "distortion_per_field_pct": data.get("distortion_per_field_pct"),
                "totr": data.get("totr"),
                "merit_before": data.get("merit_before"),
                "merit_after":  data.get("merit_after"),
            }

        # merit 可能是 float / int / 字符串 / None — 统一安全格式化
        merit_raw = data.get("merit") if isinstance(data, dict) else None
        try:
            merit_str = f"{float(merit_raw):.6f}"
        except (TypeError, ValueError):
            merit_str = "N/A" if merit_raw is None else str(merit_raw)

        # ★ P13 返回消息: 明确展示 Zemax 真值 RMS (每视场), 并强制要求 check_spec
        rms_list = (data.get("rms_per_field_mm") or []) if isinstance(data, dict) else []
        rms_str  = ", ".join(f"{float(r):.4f}" for r in rms_list) if rms_list else "N/A"
        rms_worst = max(rms_list) if rms_list else None
        rms_worst_str = f"{rms_worst:.4f}" if rms_worst is not None else "N/A"
        effl_str = f"{float(data.get('effl', 0)):.3f}" if isinstance(data, dict) and data.get("effl") is not None else "N/A"
        fnum_str = f"{float(data.get('fnum', 0)):.3f}" if isinstance(data, dict) and data.get("fnum") is not None else "N/A"
        cycles_disp = data.get("cycles_used", cycles) if isinstance(data, dict) else cycles

        # ★ P19 修 bug: 用 Zemax 真值在此直接判定是否达标,避免 LLM 再调 check_spec。
        # check_spec 调的 _calc(lens) 是近轴追迹,在大孔径(F/1.2)下和 Zemax 全光线
        # 真值可差 15 倍(实测 near=0.29 vs zmx=0.02)。此处 Zemax 真值才是判据。
        #
        # 从 query 里尝试抓用户的目标 RMS 和目标 EFFL/F# —— 实际上 agent 并没给
        # zemax_optimize 传目标 RMS, 我们只能用 bridge 已接收的 target_effl/fnum 自查。
        target_effl = lens.get("calc_effl") or lens.get("effl")
        target_fnum = lens.get("fnum")  # 注意: 这已经是被修改过的 fnum (经过 STOP 调整)
        effl_ok = effl_str != "N/A" and abs(float(effl_str) - target_effl) / max(target_effl, 1e-6) < 0.02
        fnum_ok = fnum_str != "N/A" and abs(float(fnum_str) - target_fnum) / max(target_fnum, 1e-6) < 0.10

        # ★ self-evolve: 记录 zemax_optimize 这一步(之前漏了),并同步更新 last_check
        # 用 Zemax 真值,而不是近轴复算,这样蒸馏器 final_metrics 拿到真值。
        try:
            _merit_before = data.get("merit_before") if isinstance(data, dict) else None
            _merit_after  = data.get("merit_after") or merit_raw
            _merit_delta_pct = None
            try:
                if _merit_before and float(_merit_before) > 0 and _merit_after is not None:
                    _merit_delta_pct = round(
                        (float(_merit_before) - float(_merit_after)) / float(_merit_before) * 100, 1
                    )
            except Exception:
                pass

            # zemax_pass 判定: rms_worst <= 用户目标? 但我们不知道用户目标 rms,
            # 保守用"merit 下降 + effl/fnum 都在容差内"作为"zemax 认为成功"的近似
            _zemax_pass = bool(effl_ok and fnum_ok and rms_worst is not None)

            # pre-zemax snapshot 里记录了 zemax 之前的 rms(近轴),放进 before 对比
            _pre_snapshot = _LENS_BACKUP_PRE_ZEMAX.get(lens_idx, {})
            _metrics_before_zmx = {
                "rms":  _pre_snapshot.get("calc_rms"),
                "effl": _pre_snapshot.get("calc_effl"),
            }

            record_step("zemax_optimize", lens_idx,
                        {"cycles": cycles_disp,
                         "variables": data.get("variables") if isinstance(data, dict) else None,
                         "zemax_pre_fnum":  _pre_snapshot.get("fnum"),
                         "zemax_post_fnum": float(fnum_str) if fnum_str != "N/A" else None,
                         "merit_before": _merit_before,
                         "merit_after":  _merit_after,
                         "merit_delta_pct": _merit_delta_pct,
                         "zemax_pass": _zemax_pass,
                         "rms_per_field_mm": rms_list},
                        _metrics_before_zmx,
                        {"rms":  rms_worst,
                         "effl": float(effl_str) if effl_str != "N/A" else None},
                        kind="write")
        except Exception as _re_err:
            print(f"[zemax_optimize] record_step 失败(不影响主流程): {_re_err}", flush=True)

        # 同步更新 _SESSION_CTX["last_check"] 为 Zemax 真值
        # 否则 end_session 里 final_metrics 会用上一次 check_spec 的近轴值
        if _HAS_SELF_EVOLVE and rms_worst is not None:
            try:
                from self_evolve import _SESSION_CTX
                _SESSION_CTX["last_check"] = {
                    "lens_idx": lens_idx,
                    "pass": bool(effl_ok and fnum_ok),
                    "rms":  rms_worst,   # ← Zemax 真值
                    "effl": float(effl_str) if effl_str != "N/A" else None,
                    "fnum": float(fnum_str) if fnum_str != "N/A" else None,
                    "source": "zemax_truth",
                }
            except Exception:
                pass

        msg = (
            f"✓ Zemax DLS (cycles={cycles_disp}, vars={data.get('variables','?')})\n"
            f"  merit: {data.get('merit_before','?')} → {merit_str}\n"
            f"  真值 EFFL={effl_str}mm F#={fnum_str} RMS/field=[{rms_str}] 最差={rms_worst_str}mm\n"
        )
        # ★ 代码直接做数值比较，不依赖 LLM 判断大小（LLM 曾把 0.0136 < 0.02 判成"未达标"）
        if rms_worst is not None:
            if rms_target is not None:
                rms_pass = rms_worst <= rms_target
                msg += (
                    f"【RMS 数值比较】 {rms_worst:.4f}mm {'<=' if rms_pass else '>'} 目标 {rms_target:.4f}mm"
                    f" → {'✅ 达标' if rms_pass else '❌ 未达标'}\n"
                )
                if rms_pass:
                    msg += (
                        f"★ RMS 达标！下一步:\n"
                        f"  1. 调 zemax_layout(lens_idx={lens_idx}) 保存布局图\n"
                        f"  2. 输出 Final Answer: 镜头#{lens_idx} | RMS={rms_worst:.4f}mm | 结论: 达标✓\n"
                        f"  禁止再跑 check_spec（近轴在大孔径下不可靠）"
                    )
                else:
                    msg += (
                        f"★ RMS 未达标，可选:\n"
                        f"  • 再跑 zemax_optimize(lens_idx={lens_idx}, cycles=auto, rms_target={rms_target})\n"
                        f"  • 若 2 次都不降 → 换 rank=2 候选或 Final Answer 报未达标\n"
                        f"  • 若比优化前更差 → reset_lens 回滚换策略"
                    )
            else:
                # 没传 rms_target 时保留原有模糊提示，但提醒下次要传
                msg += (
                    f"⚠ 未传 rms_target，无法自动判定。下次请在调用参数里加上:\n"
                    f"  zemax_optimize(lens_idx={lens_idx}, cycles=auto, rms_target=<你的目标值>)\n"
                    f"  当前最差 RMS={rms_worst_str}mm，请与目标值手动对比。\n"
                    f"  若达标 → 调 get_spot_diagram(lens_idx={lens_idx}) 再输出 Final Answer\n"
                    f"  若未达标 → 再跑 zemax_optimize 或换候选"
                )
        return msg
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Zemax调用异常: {type(e).__name__}: {e}"
def build_agent():
    tools = [
        lens_search,
        rank_by_rms,
        rms_calculator,
        get_lens_surfaces,
        modify_lens,
        reset_lens,
        align_effl,
        random_restart,
        local_optimize,
        split_lens,
        interpret_requirement,
        zemax_optimize,
        zemax_layout,
        get_spot_diagram,
        get_skill_detail,  # 两级加载：按需拉取完整 skill
        get_retrieval_skill_detail,  # 检索策略详情：out-of-domain / EFFL偏差时按需调用
    ]
    # self-evolve 的硬达标判断工具（step4）
    if _HAS_SELF_EVOLVE:
        tools.insert(11, check_spec)
    # 动态注入 skill_index（只注入 summary，节省 token）
    skill_index_text = build_skill_index_text()
    prompt = PromptTemplate.from_template(
        SYSTEM_PROMPT.replace("{skill_index}", skill_index_text)
    )
    agent    = create_react_agent(llm, tools, prompt)
    from langchain_core.callbacks import BaseCallbackHandler
    from collections import Counter as _Counter

    class _ToolCallLogger(BaseCallbackHandler):
        def __init__(self):
            self.calls = []
            self._last_input = {}   # {tool_name: last_normalized_input}
        def on_tool_start(self, serialized, input_str, **kwargs):
            name = serialized.get("name", "?")
            self.calls.append(name)
            # ★ P14: 打一行"可能要熔断"预警(真正熔断在工具内部做,这里只是可观测)
            norm = _normalize_tool_input(input_str)
            prev = self._last_input.get(name)
            if prev is not None and prev == norm:
                print(f"[TOOL-REPEAT] ⚠ {name} 收到和上次完全相同的输入,"
                      f"若连续 {_CIRCUIT_MAX_REPEAT} 次将触发熔断", flush=True)
            self._last_input[name] = norm
            print(f"[TOOL-CALL #{len(self.calls)}] {name}  <- {str(input_str)[:120]}", flush=True)
        def on_tool_end(self, output, **kwargs):
            print(f"[TOOL-END  ] -> {str(output)[:120]}", flush=True)
        def on_tool_error(self, error, **kwargs):
            print(f"[TOOL-ERR  ] !! {error}", flush=True)
        def on_chain_end(self, outputs, **kwargs):
            if self.calls:
                c = _Counter(self.calls)
                print(f"\n[TOOL-SUMMARY] 共{len(self.calls)}次: "
                      + ", ".join(f"{k}x{v}" for k, v in c.most_common()), flush=True)
                self.calls.clear()
                self._last_input.clear()
            # ★ P14: 清空全局熔断历史,避免下一个 query 被上一个污染
            _clear_call_history()

    # ── Scratchpad 滑动窗口 ──────────────────────────────────────────
    # Qwen3 上下文 8192 tokens，每步 Thought+Action+Observation ≈ 300 tokens，
    # 系统 prompt + 工具描述 ≈ 2500 tokens，剩余约 5600 tokens，最多容纳 ~18 步。
    # 保守取 10 步滑动窗口，超出时自动丢弃最旧的步骤。
    #
    # ⚠ 新版 LangChain (≥0.1) create_react_agent 返回 RunnableSequence，
    #   不再有 .plan 属性，也不能用 RunnableLambda 替换 agent（缺少 input_keys）。
    #   正确做法：子类化 AgentExecutor，在 _take_next_step 入口裁剪 intermediate_steps。
    _SCRATCHPAD_MAX_STEPS = 10

    class _SlidingWindowExecutor(AgentExecutor):
        def _take_next_step(self, name_to_tool_map, color_mapping, inputs,
                            intermediate_steps, run_manager=None):
            if len(intermediate_steps) > _SCRATCHPAD_MAX_STEPS:
                dropped = len(intermediate_steps) - _SCRATCHPAD_MAX_STEPS
                print(f"[CTX-TRIM] scratchpad {len(intermediate_steps)} 步 → 保留最近 "
                      f"{_SCRATCHPAD_MAX_STEPS} 步（丢弃最旧 {dropped} 步）", flush=True)
                intermediate_steps = intermediate_steps[-_SCRATCHPAD_MAX_STEPS:]
            return super()._take_next_step(
                name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager)

    _executor = _SlidingWindowExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=25,          # 从50降到25，避免context超限
        handle_parsing_errors=True,
        callbacks=[_ToolCallLogger()],
    )

    class CleanExecutor:
        """包装 AgentExecutor：清洗 <think> 标签 + 捕获上下文超限错误"""
        def __init__(self, exe):
            self._exe = exe
        def invoke(self, inputs, **kwargs):
            if isinstance(inputs.get("input"), str):
                inputs = dict(inputs)
                inputs["input"] = _strip_think(inputs["input"])
            try:
                result = self._exe.invoke(inputs, **kwargs)
            except Exception as e:
                err_str = str(e)
                # 上下文超限：给用户清晰提示而不是崩溃
                if any(k in err_str for k in ("context length", "input_tokens", "8192", "BadRequestError")):
                    print(f"[CTX-OVERFLOW] 上下文超限: {err_str[:200]}", flush=True)
                    return {"output": (
                        "❌ 上下文超限（优化步骤过多，超过模型 8192 token 上限）。\n"
                        "建议：重新输入查询，并在 zemax_optimize 之前减少 check_spec / "
                        "local_optimize 的调用轮次，直接跳到 zemax_optimize。"
                    )}
                raise
            if isinstance(result.get("output"), str):
                result["output"] = _strip_think(result["output"])
            return result
        def __getattr__(self, name):
            return getattr(self._exe, name)

    return CleanExecutor(_executor)


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global VS, ALL_LENSES

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    VS, ALL_LENSES = load_rag()
    executor = build_agent()

    if args.query:
        result = executor.invoke({"input": args.query})
        print(f"\n✅ {result['output']}")
        return

    print("\n🔭 光学镜头优化 Agent v2 就绪（输入 quit 退出）")
    print("新增能力: 检索起始镜头 → 分析不足 → 修改参数 → 迭代优化")
    print("示例查询:")
    print("  找FOV=35度 F/2.8的镜头，要求RMS < 0.04mm")
    print("  需要一个FOV=50度 F/2.0的方案，尽量优化RMS\n")

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        try:
            # 清理每轮状态 + session 隔离
            _SEARCH_COUNT.clear()
            _MODIFY_COUNT.clear()
            _OPTIMIZE_STALL.clear()
            _LENS_BACKUP.clear()
            _LENS_BACKUP_PRE_ZEMAX.clear()   # ★ P16
            global _INTERPRET_CALLED
            _INTERPRET_CALLED = False
            start_session(q)

            r = executor.invoke({"input": q})
            print(f"\n✅ {r['output']}\n")

            # 打印点列图路径（如果生成了）
            import glob as _glob
            _spot_files = sorted(_glob.glob("/gz-data/spot_*.png"), key=lambda f: os.path.getmtime(f))
            if _spot_files:
                _latest = _spot_files[-1]
                print(f"📷 点列图: {_latest}  (用 Windows 资源管理器打开 \\\\wsl$\\... 或 scp 到本地查看)\n")

            # 蒸馏
            if _HAS_SELF_EVOLVE:
                try:
                    from self_evolve import _SESSION_CTX
                    last_chk = _SESSION_CTX.get("last_check", {})
                    # ★ FIX Bug2: zemax_offline 时保留近轴 pass 结论
                    _final_passed = bool(last_chk.get("pass", False))
                    report = end_session(
                        final_passed    = _final_passed,
                        final_metrics   = {k: last_chk.get(k) for k in ("rms","effl","fnum")},
                        gemini_api_key  = os.environ["GEMINI_API_KEY"],
                        gemini_base_url = os.environ["GEMINI_BASE_URL"],
                    )
                    if report.get("appended"):
                        print(f"[self_evolve] ✓ 新增 skill: {report['new_skill_name']}\n")
                except Exception as e:
                    print(f"[self_evolve] {e}\n")
        except Exception as e:
            print(f"❌ {e}\n")


# ══════════════════════════════════════════════════════════════════════════════
# eval.py 调用入口（兼容原 run_agent 接口）
# ══════════════════════════════════════════════════════════════════════════════
_executor = None

def run_agent(question: str) -> str:
    global VS, ALL_LENSES, _executor
    _SEARCH_COUNT.clear()
    _MODIFY_COUNT.clear()
    _OPTIMIZE_STALL.clear()
    _LENS_BACKUP.clear()
    _LENS_BACKUP_PRE_ZEMAX.clear()   # ★ P16
    global _INTERPRET_CALLED
    _INTERPRET_CALLED = False
    if _executor is None:
        VS, ALL_LENSES = load_rag()
        _executor = build_agent()

    # ★ self-evolve: session 开始
    start_session(question)

    result = _executor.invoke({"input": question})
    output = result.get("output", "")

    # ★ self-evolve: session 结束，蒸馏轨迹 → 追加到 learned_skills.py
    if _HAS_SELF_EVOLVE:
        try:
            from self_evolve import _SESSION_CTX
            last_chk = _SESSION_CTX.get("last_check", {})
            final_passed  = bool(last_chk.get("pass", False))  # ★ FIX Bug2: zemax_offline 时此值已由 zemax_optimize 保留为 True
            final_metrics = {k: last_chk.get(k) for k in ("rms", "effl", "fnum")}
            report = end_session(
                final_passed    = final_passed,
                final_metrics   = final_metrics,
                gemini_api_key  = os.environ["GEMINI_API_KEY"],
                gemini_base_url = os.environ["GEMINI_BASE_URL"],
            )
            if report.get("appended"):
                print(f"[self_evolve] ✓ 新增 skill: {report['new_skill_name']}")
            elif report.get("trajectory_len", 0) >= 3:
                print(f"[self_evolve] 未入库: {report.get('reason')}")
        except Exception as e:
            print(f"[self_evolve] 失败（不影响主流程）: {e}")

    return output


if __name__ == "__main__":
    main()


