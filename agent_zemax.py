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
_SEARCH_COUNT: dict = {}  # lens_search调用计数，防死循环
_INTERPRET_CALLED: bool = False  # interpret_requirement只能调一次
_OPTIMIZE_STALL: dict = {}
_OPTIMIZE_TOTAL: dict = {}
_MODIFY_COUNT: dict = {}  # modify_lens调用计数，防死循环
_OPTIMIZE_MAX = 6
# local_optimize 停滞计数器 {lens_idx: consecutive_no_improve_count}
_OPTIMIZE_STALL: dict = {}


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
    """
    【语义检索】根据自然语言需求从镜头库中检索最相似的候选镜头。
    输入: 需求描述字符串，例如 'FOV=35度 F/2.8 RMS尽量小'。
    输出: top-5 候选镜头，含编号(lens_idx)、FOV、Fno、EFFL、近轴RMS。
    """
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
    """
    【近轴追迹】查询指定镜头当前的光学性能（含修改后的最新结果）。
    输入: lens_idx，镜头编号整数，例如 36237。
    输出: EFFL / TOTR / 近轴RMS / 像面残差 / merit function。
    """
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
    """
    【RMS排序】检索候选镜头并按近轴RMS升序排列，自动过滤FOV/Fnum范围。
    输入格式（两种均可）:
      1. 自然语言: "FOV=35度 F/2.8 RMS尽量小"
      2. 带过滤: "FOV=35度 F/2.8, fov_tol=5, fnum_tol=0.5"
         fov_tol: FOV容忍范围（度），默认10
         fnum_tol: Fnum容忍范围，默认0.5
    输出: 过滤后按RMS升序的top-10候选，若无匹配则返回最近邻并提示差距。
    """
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
            max_tokens=200,
        )
        resp_text = gemini_resp.choices[0].message.content.strip()
        resp_text = resp_text.replace("```json", "").replace("```", "").strip()
        print(f"[OOD-DEBUG] Gemini 选择: {resp_text[:200]}")
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
            f"⚠ 数据库中无FOV≈{fov_target}° F/{fnum_target}的精确匹配。【禁止继续检索，必须直接用上方返回的镜头作为起点开始优化】\n"
            f"返回镜头EFFL: {effl_hint_str}（缩放比例需在 0.25~4.0 之间，否则 align_effl 会拒绝）\n"
            f"\n{RETRIEVAL_SKILL_SUMMARY}\n"
            f"下一步【必须按顺序执行】：\n"
            f"1. 【必须】先调用align_effl将EFFL缩放到目标值\n"
            f"2. FOV偏大可接受，偏小需优先用modify_lens调整边缘视场\n"
            f"3. 调用local_optimize优化RMS\n"
            f"4. 若F数偏小(光圈偏大)→收缩光圈后再优化"
        )
    return json.dumps(out, ensure_ascii=False)


@tool
def get_lens_surfaces(lens_idx: str) -> str:
    """
    【面型详情】获取指定镜头的完整面型数据（曲率半径、厚度、材料、半径）。
    输入: lens_idx，镜头编号整数，例如 36237。
    输出: 每个光学面的参数表。
    """
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
            "r": round(float(r), 3) if abs(float(r)) < 1e7 else "inf",
            "t": round(float(s.get("thickness", 0)), 3),
            "sd": round(float(sd), 3) if sd else 0,
            "m": s.get("material", "AIR"),
        }
        all_surfs.append(entry)

    # 自动识别光阑面（r≈∞ + AIR + sd 最小）
    stop_candidates = [s for s in all_surfs
                       if s["r"] == "inf" and s["m"] == "AIR"]
    stop_surf = None
    if stop_candidates:
        stop_surf = min(stop_candidates, key=lambda x: x["sd"])["s"]

    return json.dumps({
        "id": lens_idx, "fov": lens.get("fov"), "fnum": lens.get("fnum"),
        "n_surf": len(surfs),
        "stop_surface": stop_surf,  # 光阑面编号（Skill 13 直接用此字段）
        "surfaces": all_surfs,
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# 新增 Tools
# ══════════════════════════════════════════════════════════════════════════════

@tool
def modify_lens(input_str: str) -> str:
    """
    【参数修改】修改指定镜头某个面的参数，用于优化调整。
    输入格式: "lens_idx=<编号>, surface=<面号>, param=<参数名>, value=<新值>"
      - param 可选: radius（曲率半径）、thickness（厚度）、material（材料）、semi_diameter（半径）
      - value: 数值类型（material 传字符串）
    示例: "lens_idx=55097, surface=3, param=radius, value=-48.5"
    示例: "lens_idx=55097, surface=2, param=material, value=N-FK51A"
    注意: 修改后请立即调用 rms_calculator 查看新的 RMS，判断是否改善。
    """
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
        try:
            new_val = float(value_raw)
        except ValueError:
            return f"参数 {param} 需要数值，收到: {value_raw!r}"

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
                    metrics_before=None,
                    metrics_after={"rms":  _r_after.get("rms"),
                                   "effl": _r_after.get("effl")})
    except Exception:
        pass

    return (
        f"✓ 镜头#{lens_idx} 面{surface_id} {param} 修改成功\n"
        f"  {old_val} → {new_val}\n"
        f"请调用 rms_calculator({lens_idx}) 查看修改后的性能。"
    )


@tool
def reset_lens(lens_idx: str) -> str:
    """
    【重置镜头】将镜头恢复到数据库中的原始参数（撤销所有修改）。
    输入: lens_idx，镜头编号整数，例如 36237。
    用途: 当某条优化路径效果不好时，回到起点换另一种修改策略。
    """
    try:
        if isinstance(lens_idx, str) and "=" in lens_idx:
            lens_idx = lens_idx.split("=")[-1].strip()
        lens_idx = int(str(lens_idx).strip())
    except (ValueError, TypeError):
        return f"lens_idx 必须是整数，收到: {lens_idx!r}"

    if lens_idx not in _LENS_BACKUP:
        return f"镜头#{lens_idx} 没有修改记录，无需重置。"

    ALL_LENSES[lens_idx] = copy.deepcopy(_LENS_BACKUP[lens_idx])
    del _LENS_BACKUP[lens_idx]
    # 清空该镜头的修改计数
    for k in list(_MODIFY_COUNT.keys()):
        if k.startswith(f"{lens_idx}_"):
            del _MODIFY_COUNT[k]
    return f"✓ 镜头#{lens_idx} 已恢复到原始参数。"


@tool
def interpret_requirement(description: str) -> str:
    """
    【需求翻译】将用户的自然语言描述转换为光学参数，适合不懂专业术语的用户。
    输入: 用户的自然语言描述，例如"我想拍星空""安防摄像头""手机摄像头"
    输出: 推荐的 FOV、F数、RMS目标值，以及解释说明。
    使用场景: 当用户没有给出具体 FOV/F数/RMS 参数时，先调用此工具。
    """
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

    return ("📋 需求解析结果:\n" +
            "  场景: " + note + "\n" +
            "  推荐 FOV  = " + str(fov_mid) + " 度  （范围 " + str(fov_min) + "~" + str(fov_max) + " 度）\n" +
            "  推荐 F数  = F/" + str(fnum_mid) + "  （范围 F/" + str(fnum_min) + "~F/" + str(fnum_max) + "）\n" +
            "  RMS 目标  = " + str(rms_target) + " mm\n" +
            "  建议查询  : FOV=" + str(fov_mid) + "度 F/" + str(fnum_mid) + " RMS<" + str(rms_target) + "mm\n" +
            "接下来将用以上参数进行镜头检索和优化。")


@tool
def align_effl(input_str: str) -> str:
    """
    【EFFL预对齐】在优化前将镜头焦距缩放到目标值，避免优化过程中焦距漂移。
    输入格式: "lens_idx=<编号>, target_effl=<目标焦距mm>"
    示例: "lens_idx=36237, target_effl=34.0"
    原理: 对所有面的 radius 和 thickness 等比缩放，RMS不变但EFFL对齐目标。
    注意: 缩放后请调用 rms_calculator 确认 EFFL 和 RMS。
    """
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
        warn = ""
        if scale_pct > 150:
            warn = f"  ⚠ 放缩比例较大({scale:.3f})，建议验证后再优化。\n"

        # 轨迹记录
        try:
            record_step("align_effl", lens_idx,
                        {"target_effl": target_effl, "scale": round(scale, 4)},
                        {"rms": r.get("rms"),  "effl": current_effl},
                        {"rms": r2.get("rms"), "effl": r2.get("effl")})
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
    """【随机扰动重启】对镜头参数施加随机扰动后重新优化，用于跳出局部极小值。输入: "lens_idx=<编号>, strength=<扰动强度0.01-0.1>" """
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
                    None,
                    {"rms": r2.get("rms") if r2.get("valid") else None,
                     "effl": r2.get("effl") if r2.get("valid") else None})
    except Exception:
        pass

    return f"扰动完成#{lens_idx} 扰动后RMS:{rms_str} 继续local_optimize或若变差则reset_lens"



@tool
def split_lens(input_str: str) -> str:
    """
    【拆分镜片 Split_Lens】将过厚或光焦度过强的单片镜拆成两片，降低每片的偏折角，
    从而减小球差和高级像差。拆分后自动插入一个薄空气间隔。
    输入格式: "lens_idx=<编号>, surface=<面号>, ratio=<前片光焦度占比0.3-0.7>"
      - surface: 要拆分的玻璃面的面号（从 get_lens_surfaces 获取）
      - ratio: 前片承担的光焦度比例，默认0.5（均分），0.3表示前片弱后片强
    示例: "lens_idx=36237, surface=6, ratio=0.5"
    注意: 拆分后面数增加2，原面号之后的面号全部后移，请重新调用 get_lens_surfaces 查看新面型。
    """
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
                    None,
                    {"rms":  r_check.get("rms")  if r_check.get("valid") else None,
                     "effl": r_check.get("effl") if r_check.get("valid") else None})
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
    """
    【局部优化】对指定镜头的所有可优化面自动做梯度下降，最小化近轴RMS。
    LLM不需要猜浮点数，直接调用此工具让系统自动找最优radius组合。
    输入格式: "lens_idx=<编号>, iterations=<迭代次数>, lr=<学习率>"
      - iterations: 迭代轮数，默认30，建议范围10-100
      - lr: 学习率，默认0.5，太大会震荡，太小收敛慢
    示例: "lens_idx=36237, iterations=50, lr=0.5"
    示例（快速）: "lens_idx=36237"  （使用默认参数）
    输出: 优化前后RMS对比、EFFL变化、每面radius调整量。
    """
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
                    {"rms": rms_final, "effl": effl_final})
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

【全局规则 - 优先阅读】
- 用户给出 FOV/F# 数值需求时，**必须用 rank_by_rms，禁止用 lens_search**
  （lens_search 是语义检索，不按数值距离排序；rank_by_rms 才按数值距离找最接近的）
- 遇到 "Unknown material" 错误不要重复改材料，直接换候选镜头


你是光学镜头设计专家（CDGM玻璃库）。流程：①lens_search检索→②rms_calculator评估→③未达标则modify_lens优化，最多6次。
优化步骤（严格按顺序执行）：
Step0 若用户已明确给出FOV和F数（如"FOV=150度 F/1.2"），【跳过interpret_requirement】，直接进Step0b；若用户未给出FOV/F数等参数→先调用interpret_requirement翻译需求
Step0b rank_by_rms时加fov_tol/fnum_tol过滤，如: FOV=75度 F/2.4, fov_tol=5, fnum_tol=0.5
★ Out-of-domain规则：若rank_by_rms返回warning，【立即停止所有检索】，直接取返回结果中rank=1的镜头，调用rms_calculator评估，然后align_effl对齐焦距，再local_optimize优化。禁止再次调用lens_search或rank_by_rms。若返回镜头FOV < 目标FOV×0.5，Final Answer中必须注明"数据库无精确匹配，以最近邻结果代替"。禁止因RMS达标就直接结束，必须先执行align_effl + local_optimize再输出Final Answer。
Step1【必须】rms_calculator评估初始RMS
Step2 以下两种情况【必须】调用align_effl：①用户明确说"EFFL=xx mm"或"焦距=xx mm"；②rms_calculator返回的EFFL超过目标值的130%，或用户未给目标但EFFL>80mm。target_effl取用户指定值，或取当前EFFL×0.6（启发式）。align_effl失败(偏差>85%)则换rank=2镜头重试，勿再次检索。
Step3 若某面thickness>8mm或RMS主要由球差主导→先split_lens拆分该面，再local_optimize
Step4 若仍未达标：modify_lens换材料（色差→H-FK61/H-FK71正镜+H-ZF7LA负镜；球差→H-ZLAF55D；场曲→负镜H-ZF7LA）
优化策略索引（识别症状后调用get_skill_detail获取详细操作）：
{skill_index}
Step5 rms_calculator验证；若local_optimize返回"改善0.0%"→立即停止重复调用，改用random_restart(strength=0.05)扰动后再跑一次local_optimize；若仍0.0%改善→直接输出Final Answer
Step6 【关键】完成近轴优化后必须调 check_spec 做硬达标判定，格式：
      check_spec(lens_idx=X, target_effl=<用户目标mm>, target_fnum=<用户目标F数>, rms_pass=1.0)
      - 返回 pass=true → 直接调 zemax_optimize 进精优，然后 zemax_layout 出图，输出 Final Answer
      - 返回 pass=false → 按 reasons 提示继续（EFFL差→align_effl；RMS差→local_optimize；F#差→换候选镜头）
      - 同一镜头 check_spec 连续失败 3 次 → 直接输出 Final Answer
      【禁止】不调 check_spec 就直接输出 Final Answer；不经 check_spec 放行就 zemax_optimize
只用CDGM牌号(H-/D-)，禁止N-BK7等肖特牌号。

Step7 【F# 偏大时改光阑扩张入瞳，禁止换候选】
当 check_spec 报 "F#=X vs 目标Y 偏大" 且 infeasible 为空时：
  1) 调 get_skill_detail('Skill 13') 查"扩大入瞳降低 Fnum"（这是 F# 优化主路径）
  2) 按 Skill 13 操作：
     a) get_lens_surfaces 拿到所有面
     b) 识别光阑面（radius 近无穷 + material=AIR + semi_diameter 局部最小）
     c) modify_lens(surface=<光阑面>, param=semi_diameter, value=<原值×当前Fnum/目标Fnum>)
     d) local_optimize 重优化（扩光阑会引入球差，必须重优化）
     e) check_spec 验证 Fnum 是否降到目标
  3) 若 RMS 爆炸（>1mm），调 get_skill_detail('Skill 2') 查换高折射率玻璃方案，再 modify_lens(param=material)
  4) 同一候选最多执行 Step7 两轮；两轮仍未达标再换下一候选
  5) 所有候选都试完仍未达标 → Final Answer 明确写"未达标✗"（禁止写"以最近邻代替"）
若 infeasible 非空（FOV 差距>50%）→ 跳过结构修改，直接换候选或 Final Answer 报未达标
【严禁】F# 偏大时只改 surface=1 的 material/radius——那对 Fnum 无影响！必须改光阑面的 semi_diameter。

工具列表：
{tools}

工具名: {tool_names}

 Action格式必须严格如下，Action和Action Input分两行，Action Input只写参数值不写函数名：
Thought: 分析
Action: rms_calculator
Action Input: 36237

【严禁】每次只能输出一个Action，禁止在同一次输出中同时写多个Action。
【严禁】禁止在同一次输出中同时出现Action和Final Answer，二者只能选其一。
错误示例（禁止）：Action: rms_calculator(lens_idx="36237")

 达到目标或完成判断后，必须立即输出 Final Answer，禁止继续调用工具：
Thought: 已达标，结束
Final Answer: 镜头#xxxxx | RMS=x.xxxx mm | 结论: 达标✓

Question: {input}
Thought:{agent_scratchpad}"""
# PROMPT 在 build_agent 里动态生成，注入 skill_index


# ══════════════════════════════════════════════════════════════════════════════
# Skill 两级加载
# ══════════════════════════════════════════════════════════════════════════════
from skill_summaries import SKILL_SUMMARIES

def build_skill_index_text():
    # 合并手工 skill + 自进化学到的 skill
    merged = dict(SKILL_SUMMARIES)
    try:
        learned = load_learned_for_prompt()  # {name: summary}
        merged.update(learned)
    except Exception:
        pass
    lines = ["===== 优化策略索引（识别症状后调用get_skill_detail获取详情）====="]
    for name, summary in merged.items():
        lines.append(f"・{name}：{summary}")
    lines.append("如需详细操作步骤，调用: get_skill_detail(skill_name=\'策略名\')")
    lines.append("======================")
    return "\n".join(lines)

@tool
def get_skill_detail(skill_name: str) -> str:
    """获取某个优化策略的完整详细内容。输入策略名称，如\'利用高阿贝数低色散玻璃消除轴上色差\'。"""
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
    """
    【检索策略详情】获取某条镜头检索 skill 的完整操作指引。
    适用场景: rank_by_rms 返回 warning（out-of-domain 或 EFFL偏差>60%）时，
              Agent 根据 summary 索引判断需要哪条规则，再调用此工具获取详情。
    输入: skill 名称，如 'Skill R3: EFFL差距处理' 或只写 'R3'
    """
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
_NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}

def _zemax_available():
    try:
        r = _requests.get(f"{ZEMAX_BRIDGE}/status", headers=_NGROK_HEADERS, timeout=3)
        return r.ok
    except Exception:
        return False

@tool
def zemax_layout(input_str: str) -> str:
    """
    【Zemax Layout图】将当前镜头面型推送到 Zemax 并生成 2D 布局图，保存为 PNG。
    输入格式: "lens_idx=<编号>, save_path=<保存路径>"
    示例: "lens_idx=36237, save_path=/gz-data/layout_36237.png"
    save_path 可省略，默认 /gz-data/layout_<lens_idx>.png
    """
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

    if not _zemax_available():
        return "Zemax桥接服务不可用，请确认Windows端 zemax_bridge.py 已启动"

    lens  = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        import ast as _ast
        surfs = _ast.literal_eval(surfs)

    try:
        # 推送完整面型
        _requests.post(f"{ZEMAX_BRIDGE}/load_lens", headers=_NGROK_HEADERS,
                       json={"surfaces": surfs,
                             "fov":  lens.get("fov"),
                             "fnum": lens.get("fnum")},
                       timeout=10)
        # 生成 layout 图
        resp = _requests.post(f"{ZEMAX_BRIDGE}/layout", headers=_NGROK_HEADERS, timeout=30)
        if not resp.ok:
            return f"Layout 生成失败: {resp.text}"
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return f"✓ Layout 图已保存: {save_path}  ({len(resp.content)//1024} KB)"
    except Exception as e:
        return f"Zemax调用异常: {e}"


@tool
def zemax_optimize(input_str: str) -> str:
    """
    【Zemax精确优化】将当前面型推送到 Zemax 做 DLS 局部优化，比近轴追迹更精确。
    适合：近轴优化停滞、需要验证真实光线追迹结果时。
    输入格式: "lens_idx=<编号>, cycles=<优化轮数>"
      - cycles: 默认5，范围1-20
    示例: "lens_idx=36237, cycles=5"
    输出: Zemax优化后的merit function值。优化完毕建议调用 zemax_layout 出图。
    """
    try:
        parts = {}
        for seg in str(input_str).strip().split(","):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                parts[k.strip()] = v.strip()
        lens_idx = int(parts.get("lens_idx", 0))
        cycles   = int(parts.get("cycles", 5))
    except Exception as e:
        return f"输入格式错误: {e}"

    if not _zemax_available():
        return "Zemax桥接服务不可用，请确认Windows端 zemax_bridge.py 已启动"

    lens  = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        import ast as _ast
        surfs = _ast.literal_eval(surfs)

    try:
        # 推送完整面型
        _requests.post(f"{ZEMAX_BRIDGE}/load_lens", headers=_NGROK_HEADERS,
                       json={"surfaces": surfs,
                             "fov":  lens.get("fov"),
                             "fnum": lens.get("fnum")},
                       timeout=10)
        # DLS 优化
        resp = _requests.post(f"{ZEMAX_BRIDGE}/zemax_optimize", headers=_NGROK_HEADERS,
                              json={"cycles": cycles}, timeout=120)
        data = resp.json()
        if "error" in data:
            return f"Zemax优化失败: {data['error']}"
        merit = data.get("merit", "N/A")
        return (f"✓ Zemax DLS优化完成 merit={merit:.6f} (cycles={cycles})\n"
                f"建议调用 rms_calculator 验证近轴RMS，或调用 zemax_layout 生成布局图。")
    except Exception as e:
        return f"Zemax调用异常: {e}"
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
    _executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=50,
        handle_parsing_errors=True,
    )

    class CleanExecutor:
        """包装 AgentExecutor，自动清洗 <think> 标签"""
        def __init__(self, exe):
            self._exe = exe
        def invoke(self, inputs, **kwargs):
            if isinstance(inputs.get("input"), str):
                inputs = dict(inputs)
                inputs["input"] = _strip_think(inputs["input"])
            result = self._exe.invoke(inputs, **kwargs)
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
            global _INTERPRET_CALLED
            _INTERPRET_CALLED = False
            start_session(q)

            r = executor.invoke({"input": q})
            print(f"\n✅ {r['output']}\n")

            # 蒸馏
            if _HAS_SELF_EVOLVE:
                try:
                    from self_evolve import _SESSION_CTX
                    last_chk = _SESSION_CTX.get("last_check", {})
                    report = end_session(
                        final_passed    = bool(last_chk.get("pass", False)),
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
            final_passed  = bool(last_chk.get("pass", False))
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


