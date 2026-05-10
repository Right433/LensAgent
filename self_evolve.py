# -*- coding: utf-8 -*-
"""
self_evolve.py  ── Voyager 风格的光学 Agent 终身学习模块
==========================================================
对标 Voyager (Wang et al. 2023) 三大机制，移植到光学设计领域：

  ① Automatic Curriculum（自动课程）
       propose_curriculum() — 根据已解决/失败规格，建议下一个更难的设计目标，
       让 Agent 能力边界持续向外扩张。

  ② Skill Library with Executable Code（可执行代码库）
       每条 skill 新增 skill_code 字段：直接可 exec() 的 Python 函数，
       调用 agent tools 完成该类操作。技能可组合（compositional）。

  ③ Embedding-based Skill Retrieval（嵌入检索）
       retrieve_relevant_skills(task_desc, k=5) — 用任务描述 embedding 检索 top-k，
       仅将相关 skill 注入 prompt，避免 context 被无关内容撑爆。
       对标 Voyager："querying the library with embedding of task plans"

  ④ Self-Verification Gate（自验证门）
       _verify_skill_effectiveness() — 入库前校验 skill 声称的改善真实可信。
       对标 Voyager："self-verification checks task completion before adding to library"

原有功能（保留+修复）：
  ① check_spec   — 硬达标判断（修复 rms_pass 覆盖 bug）
  ② record_step  — 轨迹记录（补充 fnum/surface/param 字段）
  ③ distill_session + append_skill — 蒸馏 + 三层去重

集成到 agent_zemax.py：
  from self_evolve import (check_spec, record_step, start_session, end_session,
                            retrieve_relevant_skills, propose_curriculum)
  - build_agent() tools 加 check_spec
  - modify_lens/align_effl/etc. return 前调 record_step
  - run_agent 开头 start_session，结尾 end_session
  - build_skill_index_text() 改用 retrieve_relevant_skills(question, k=5)
"""

import json, os, time, re, math
from pathlib import Path
from langchain_core.tools import tool

# ─────────────────────────── 配置 ────────────────────────────
LEARNED_PATH        = "/gz-data/learned_skills.py"
SKILL_EMBED_CACHE   = "/gz-data/skill_embeddings.pkl"
MIN_TRAJ_LEN        = 4
MIN_RMS_GAIN        = 0.05
SIM_THRESHOLD       = 0.72
EFFL_TOL_PCT        = 2.0
FNUM_TOL_PCT        = 2.0
DEFAULT_RMS_PASS_MM = 1.0

# ─────────────────────────── 模块状态 ────────────────────────
_TRAJECTORY:  list = []
_SESSION_CTX: dict = {}


# =============================================================
#   Session 生命周期 hooks
# =============================================================
def start_session(question: str, target_spec: dict | None = None) -> dict | None:
    """
    session 开始。自动检索现有 skill（Voyager：先用已有技能，没有再建新的）。
    返回 recommended_skill（若找到可用的）或 None。
    agent_zemax.py 应检查返回值并优先让 agent 执行 recommended_skill["skill_code"]。
    """
    _TRAJECTORY.clear()
    _SESSION_CTX.clear()
    _SESSION_CTX.update({
        "question":          question,
        "target_spec":       target_spec or {},
        "start_ts":          time.time(),
        "skill_reused":      False,   # 标记：本 session 是否成功复用了现有 skill
        "skill_reused_name": None,
    })

    # ★ Voyager 核心：先检索现有 skill，有匹配的直接推荐给 agent 执行
    # find_applicable_skill 定义在下方（Retrieval 区块），此处延迟调用
    skill = find_applicable_skill(question)
    if skill:
        _SESSION_CTX["recommended_skill"] = skill
        print(f"[self_evolve] ✓ 找到可复用 skill: {skill['name']} "
              f"(score={skill.get('_retrieval_score', '?')})", flush=True)

        # ★ 解析 skill_code 里的 skip/prefer 玻璃规则，存入 SESSION_CTX
        # 供 rank_by_rms 直接重排候选，真正让 skill 生效
        import re as _re2
        _code = skill.get("skill_code", "")
        _skip_m   = _re2.search(r'skip_if_front_glass\s*=\s*(\[[^\]]*\])', _code)
        _prefer_m = _re2.search(r'prefer_front_glass\s*=\s*(\[[^\]]*\])', _code)
        if _skip_m:
            try:
                _SESSION_CTX["skip_if_front_glass"] = json.loads(
                    _skip_m.group(1).replace("'", '"'))
                print(f"[self_evolve] skill skip_if_front_glass: "
                      f"{_SESSION_CTX['skip_if_front_glass']}", flush=True)
            except Exception:
                pass
        if _prefer_m:
            try:
                _SESSION_CTX["prefer_front_glass"] = json.loads(
                    _prefer_m.group(1).replace("'", '"'))
                print(f"[self_evolve] skill prefer_front_glass: "
                      f"{_SESSION_CTX['prefer_front_glass']}", flush=True)
            except Exception:
                pass
    else:
        _SESSION_CTX["recommended_skill"] = None
        print("[self_evolve] 无匹配 skill，将走完整优化流程", flush=True)

    return skill


def record_step(tool_name: str,
                lens_idx:  int,
                action:    dict,
                metrics_before: dict | None = None,
                metrics_after:  dict | None = None,
                note: str  = "",
                kind: str  = "write") -> None:
    delta_rms = None
    if (metrics_before and metrics_after
            and metrics_before.get("rms") is not None
            and metrics_after.get("rms") is not None):
        delta_rms = metrics_before["rms"] - metrics_after["rms"]
    _TRAJECTORY.append({
        "t": len(_TRAJECTORY) + 1, "tool": tool_name, "kind": kind,
        "lens_idx": lens_idx, "action": action,
        "before": metrics_before, "after": metrics_after,
        "delta_rms": delta_rms, "note": note,
    })


# =============================================================
#   ① Voyager: Automatic Curriculum（自动课程）
# =============================================================
_CURRICULUM_PROMPT = """你是光学设计专家，提出下一个训练任务使 Agent 能力持续扩张。

已成功解决的规格（Agent 已掌握）:
{solved}

曾失败的规格（Agent 尚未攻克）:
{failed}

当前 skill 库摘要:
{skill_summaries}

规则：
1. 提出比"已解决"稍难、比"已失败"稍简单的规格，沿能力边界向外扩张。
2. 若失败列表非空，优先从失败案例选最接近成功的方向改进。
3. 规格须光学合理（F#×EFFL >= 20mm，FOV <= 120°）。
4. 严格返回 JSON（不加 markdown fence）:
{{
  "fov":  <半角度 度 浮点>,
  "fnum": <F/# 浮点>,
  "effl": <有效焦距 mm 浮点>,
  "rms_target": <目标 RMS mm 浮点>,
  "rationale": "<30字理由>"
}}
"""


def propose_curriculum(gemini_api_key: str = "",
                       gemini_base_url: str = "",
                       gemini_model:    str = "") -> dict | None:
    """
    Voyager Automatic Curriculum：读历史成功/失败规格，让 Gemini 提出下一训练任务。
    返回 {fov, fnum, effl, rms_target, rationale} 或 None。
    """
    history_path = Path(LEARNED_PATH).parent / "curriculum_history.json"
    solved, failed = [], []
    if history_path.exists():
        try:
            h = json.loads(history_path.read_text())
            solved = h.get("solved", [])
            failed = h.get("failed",  [])
        except Exception:
            pass

    learned = _load_learned()
    summaries = "\n".join(f"- {k}: {v.get('summary','')}"
                          for k, v in list(learned.items())[:20]) or "（暂无）"
    prompt = _CURRICULUM_PROMPT.format(
        solved=json.dumps(solved[-10:], ensure_ascii=False),
        failed=json.dumps(failed[-10:],  ensure_ascii=False),
        skill_summaries=summaries,
    )
    try:
        from openai import OpenAI
        _key   = gemini_api_key  or os.environ.get("GEMINI_API_KEY", "")
        _url   = gemini_base_url or os.environ.get("GEMINI_BASE_URL", "")
        _model = (gemini_model   or os.environ.get("GEMINI_MODEL_SELECT",
                                                    "gemini-2.5-flash-preview-05-20"))
        cli  = OpenAI(api_key=_key, base_url=_url)
        resp = cli.chat.completions.create(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        task = json.loads(m.group(0))
        print(f"[curriculum] 建议下一任务: {task}", flush=True)
        return task
    except Exception as e:
        print(f"[curriculum] 失败: {e}", flush=True)
        return None


def record_curriculum_result(spec: dict, passed: bool) -> None:
    """把本次任务结果写入课程历史，供下次 propose_curriculum 使用。"""
    history_path = Path(LEARNED_PATH).parent / "curriculum_history.json"
    h = {"solved": [], "failed": []}
    if history_path.exists():
        try:
            h = json.loads(history_path.read_text())
        except Exception:
            pass
    key = "solved" if passed else "failed"
    h.setdefault(key, []).append({**spec, "ts": time.strftime("%Y-%m-%d %H:%M")})
    h["solved"] = h["solved"][-50:]
    h["failed"]  = h["failed"][-50:]
    history_path.write_text(json.dumps(h, ensure_ascii=False, indent=2))


# =============================================================
#   ③ Voyager: Embedding-based Skill Retrieval（嵌入检索）
# =============================================================
_EMBEDDER_SINGLETON = None   # Fix #7: 避免每次 retrieve 都重新加载 HuggingFace 模型

def _get_embedder():
    global _EMBEDDER_SINGLETON
    if _EMBEDDER_SINGLETON is not None:
        return _EMBEDDER_SINGLETON
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        EMB_MODEL = os.environ.get(
            "EMB_MODEL",
            "/root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese"
            "/snapshots/183bb99aa7af74355fb58d16edf8c13ae7c5433e",
        )
        _EMBEDDER_SINGLETON = HuggingFaceEmbeddings(
            model_name=EMB_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 32},
        )
        return _EMBEDDER_SINGLETON
    except Exception as e:
        print(f"[skill_retrieval] embedder 加载失败: {e}", flush=True)
        return None


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a)) + 1e-9
    nb  = math.sqrt(sum(x * x for x in b)) + 1e-9
    return dot / (na * nb)


def _load_skill_embeddings() -> dict:
    p = Path(SKILL_EMBED_CACHE)
    if not p.exists():
        return {}
    try:
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _save_skill_embeddings(cache: dict) -> None:
    try:
        import pickle
        with open(SKILL_EMBED_CACHE, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"[skill_retrieval] embedding 缓存保存失败: {e}", flush=True)


def retrieve_relevant_skills(task_desc: str, k: int = 5) -> list:
    """
    Voyager Skill Retrieval：
    用任务描述 embedding 检索 top-k 相关 skill（含 skill_code）。

    用法（agent_zemax.py build_skill_index_text 里）：
        skills = retrieve_relevant_skills(question, k=5)
        prompt_block = "\\n\\n".join(
            f"【{s['name']}】\\n{s['full']}" for s in skills
        )
    """
    learned = _load_learned()
    if not learned:
        return []

    embedder = _get_embedder()
    if embedder is None:
        return _keyword_fallback_retrieve(task_desc, learned, k)

    cache = _load_skill_embeddings()
    dirty = False
    for name in learned:
        if name not in cache:
            try:
                text = (learned[name].get("summary", "")
                        + " " + " ".join(learned[name].get("triggers", [])))
                cache[name] = embedder.embed_query(text)
                dirty = True
            except Exception:
                cache[name] = []
    if dirty:
        _save_skill_embeddings(cache)

    try:
        q_vec = embedder.embed_query(task_desc)
    except Exception:
        return _keyword_fallback_retrieve(task_desc, learned, k)

    scored = []
    for name in learned:
        vec = cache.get(name, [])
        if not vec:
            continue
        scored.append((_cosine(q_vec, vec), name))
    scored.sort(reverse=True)

    result = []
    for sim, name in scored[:k]:
        sk = dict(learned[name])
        sk["name"] = name
        sk["_retrieval_score"] = round(sim, 4)
        result.append(sk)

    print(f"[skill_retrieval] top-{k} for '{task_desc[:40]}': "
          f"{[(s['name'][:30], s['_retrieval_score']) for s in result]}", flush=True)
    return result


def _keyword_fallback_retrieve(task_desc: str, learned: dict, k: int) -> list:
    words = set(re.split(r'\W+', task_desc.lower()))
    scored = []
    for name, sk in learned.items():
        text = (sk.get("summary", "") + " " + " ".join(sk.get("triggers", []))).lower()
        score = sum(1 for w in words if w and w in text)
        scored.append((score, name))
    scored.sort(reverse=True)
    result = []
    for _, name in scored[:k]:
        sk = dict(learned[name])
        sk["name"] = name
        result.append(sk)
    return result


def find_applicable_skill(task_desc: str, min_score: float = 0.72) -> dict | None:
    """
    Voyager 核心：先查现有 skill 库，找到置信度 >= min_score 的才推荐。
    返回最相关的 skill dict（含 skill_code/full/name）或 None。

    min_score=0.75 是经验阈值（从 0.65 提高）：
      - 过高（>0.8）：几乎永远找不到匹配，退化为总是创建新 skill
      - 过低（<0.65）：把不相关的 skill 强行套用，反而误导 agent
      - 0.75：避免 L-1025(超广角FOV) 错误匹配 ood_fnum 题
    """
    candidates = retrieve_relevant_skills(task_desc, k=1)
    if not candidates:
        return None
    top = candidates[0]
    score = top.get("_retrieval_score", 0.0)
    if score < min_score:
        print(f"[self_evolve] 最相关 skill '{top['name']}' 得分 {score:.3f} < {min_score}，"
              f"不复用，走新优化流程", flush=True)
        return None
    return top


def try_execute_skill(skill: dict, lens_idx: int, tools: dict | None = None, **kwargs) -> dict:
    """
    Voyager Iterative Prompting：尝试执行 skill_code。
    返回 {ok: bool, result: any, error: str}。

    tools: agent 工具函数字典，如 {"rank_by_rms": fn, "align_effl": fn, ...}。
           若不传，skill_run 无法调用任何 agent 工具，通常会返回空 steps。
    """
    code = skill.get("skill_code", "").strip()
    if not code:
        return {"ok": False, "result": None, "error": "skill_code 为空"}
    try:
        # Fix #10: 限制可用 builtins，防止 Gemini 生成的代码执行任意系统命令。
        # 保留数学/类型转换等无害内建，禁止 open/exec/eval/compile/__import__ 等。
        _safe_builtins = {
            k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
            for k in (
                "abs", "all", "any", "bool", "dict", "enumerate", "filter",
                "float", "int", "isinstance", "len", "list", "map", "max",
                "min", "print", "range", "round", "set", "sorted", "str",
                "sum", "tuple", "type", "zip", "None", "True", "False",
            )
            if (isinstance(__builtins__, dict) and k in __builtins__)
               or (not isinstance(__builtins__, dict) and hasattr(__builtins__, k))
        }
        # ★ 将 agent 工具函数注入 exec 命名空间，让 skill_run 能真正调用它们
        ns = {"__builtins__": _safe_builtins, "lens_idx": lens_idx, **kwargs}
        if tools:
            ns.update(tools)
        exec(compile(code, "<skill>", "exec"), ns)
        fn = ns.get("skill_run")
        if fn is None:
            return {"ok": False, "result": None, "error": "skill_code 中未定义 skill_run 函数"}
        # 只传 skill_run 函数签名里声明的参数
        _varnames = set(fn.__code__.co_varnames[:fn.__code__.co_argcount])
        _call_kwargs = {k: v for k, v in {**kwargs, **(tools or {})}.items()
                        if k in _varnames and k != "lens_idx"}
        result = fn(lens_idx, **_call_kwargs)
        return {"ok": True, "result": result, "error": ""}
    except Exception as e:
        import traceback
        return {"ok": False, "result": None, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def mark_skill_reused(skill_name: str, succeeded: bool) -> None:
    """
    标记本 session 是否成功复用了现有 skill。
    succeeded=True  → 不需要蒸馏新 skill（已有的就够用）
    succeeded=False → 现有 skill 失效，走完整优化流程并考虑蒸馏更新版
    """
    _SESSION_CTX["skill_reused"]      = succeeded
    _SESSION_CTX["skill_reused_name"] = skill_name
    if succeeded:
        print(f"[self_evolve] ✓ skill '{skill_name}' 复用成功，本 session 不再蒸馏新 skill",
              flush=True)
    else:
        print(f"[self_evolve] ✗ skill '{skill_name}' 复用失败，走完整优化+蒸馏流程",
              flush=True)


# =============================================================
#   ④ Voyager: Self-Verification Gate（自验证门）
# =============================================================
def _verify_skill_effectiveness(skill: dict, trajectory: list) -> tuple:
    """
    Voyager Self-Verification：入库前校验 skill 是否真实可信。
    返回 (ok: bool, reason: str)
    """
    # 1. skill_code 必须存在（Voyager 核心：技能即可执行代码）
    code = skill.get("skill_code", "").strip()
    if not code:
        return False, "skill_code 为空，不符合 Voyager 可执行代码要求"

    # 2. summary 声称 RMS 改善时，轨迹必须有正向改善证据
    # 接受 delta_rms > 0 或 zemax_pass=True（Zemax 真值达标也算改善）
    summary = skill.get("summary", "")
    _rms_claim = re.search(r'RMS.{0,10}([\d.]+)', summary)
    if _rms_claim:
        actual_gains = [s["delta_rms"] for s in trajectory
                        if s.get("delta_rms") is not None and s["delta_rms"] > 0]
        zemax_passed = any(s.get("action", {}).get("zemax_pass") is True
                           for s in trajectory)
        if not actual_gains and not zemax_passed:
            return False, "summary 声称 RMS 改善，但轨迹中无正向 delta_rms 且无 Zemax 达标记录"

    # 3. triggers 必须有 ≥2 个具体词（防止泛化无法触发）
    triggers = skill.get("triggers", [])
    generic  = {"优化", "失败", "改善", "调整", "修改", "问题", "需要", "进行", "操作"}
    specific = [t for t in triggers if t not in generic and len(t) > 2]
    if len(specific) < 2:
        return False, f"triggers 过于泛化（具体词 < 2 个）: {triggers}"

    # 4. full 字段必须含【动作】段
    if "【动作】" not in skill.get("full", ""):
        return False, "full 字段缺少【动作】段"

    return True, "verification passed"


# =============================================================
#   Step3: 蒸馏 & 追加
# =============================================================
def _should_distill(final_passed: bool) -> bool:
    zemax_ran    = any(s["tool"] == "zemax_optimize" for s in _TRAJECTORY)
    zemax_passed = any(s["tool"] == "zemax_optimize"
                       and (s.get("action") or {}).get("zemax_pass") is True
                       for s in _TRAJECTORY)
    # ★ FIX: Zemax 离线时 record_step 会记录 zemax_offline=True，
    # 此时虽然没有 Zemax 真值，但近轴计算达标也是有价值的路径。
    zemax_offline = any(s["tool"] == "zemax_optimize"
                        and (s.get("action") or {}).get("zemax_offline") is True
                        for s in _TRAJECTORY)
    rare = {"split_lens", "random_restart"}

    # ★ FIX: Zemax 达标的会话直接入库，不受 MIN_TRAJ_LEN 限制。
    # 近轴 RMS 通常比 Zemax 真值低，导致 delta_rms 为负，但只要 final_passed
    # 且 zemax_ran，就说明这是一次有价值的成功路径。
    if final_passed and zemax_ran:
        return True

    # ★ FIX: Zemax 离线但近轴达标的会话也允许蒸馏（轨迹 ≥2 步即可）
    # 这解决了 self-evolving 训练时 Zemax bridge 不在线导致 skills=0 的问题
    if final_passed and zemax_offline and len(_TRAJECTORY) >= 2:
        return True

    # 其余情况仍要求轨迹足够长
    if len(_TRAJECTORY) < MIN_TRAJ_LEN:
        return False

    # ★ FIX: 罕见工具必须同时满足 final_passed 或轨迹 ≥ 6 步
    if rare & {s["tool"] for s in _TRAJECTORY}:
        if final_passed or len(_TRAJECTORY) >= 6:
            return True
    # Fix #6: 必须有实际正向 RMS 改善才值得入库，
    # 否则"调错面号被系统拒绝"这种失败操作也会触发蒸馏。
    if any(s["tool"] == "modify_lens"
           and s["action"].get("param") == "semi_diameter"
           and (s.get("delta_rms") or 0) > 0
           for s in _TRAJECTORY):
        return True
    ood_start = any(s["tool"] == "rank_by_rms"
                    and (s.get("action") or {}).get("is_ood") is True
                    for s in _TRAJECTORY)
    if ood_start and zemax_passed:
        return True
    if (not final_passed) and any(s["tool"] == "modify_lens"
                                   and s["action"].get("param") == "material"
                                   for s in _TRAJECTORY):
        return True
    if (not final_passed) and any(s["tool"] == "zemax_optimize"
                                   and (s.get("action") or {}).get("zemax_pass") is False
                                   and (s.get("action") or {}).get("effl_mismatch_pct", 0) > 30
                                   for s in _TRAJECTORY):
        return True
    return False


# ─── _should_distill 前置守卫：现有 skill 成功复用则跳过蒸馏 ──────
def _skill_reuse_succeeded() -> bool:
    """
    Voyager 核心逻辑：
    若本 session 已成功复用现有 skill（mark_skill_reused(succeeded=True) 被调用过），
    则不再蒸馏新 skill —— 现有库已经够用，蒸馏冗余 skill 会稀释库的质量。

    ★ 例外：若复用的 skill 与当前任务 OOD 类型不匹配（如 FOV-OOD skill 被
      fnum-OOD 题命中），仍允许蒸馏，为新类型建立专属 skill。
    """
    if not _SESSION_CTX.get("skill_reused", False):
        return False

    # ★ 检查 OOD 类型是否匹配
    reused_name = _SESSION_CTX.get("skill_reused_name", "") or ""
    spec        = _SESSION_CTX.get("target_spec", {})
    cur_fov     = float(spec.get("fov",  90) or 90)
    cur_fnum    = float(spec.get("fnum", 2.8) or 2.8)

    # skill 是 FOV-OOD 类型（超广角/超大视场系列）
    skill_is_fov_ood = any(kw in reused_name for kw in
                           ["超广角", "超大视场", "110°", "130°", "145°",
                            "大视场", "fov_out_of_range"])

    # 当前任务是 fnum-OOD 类型：FOV 普通但 F# 极端
    task_is_fnum_ood = (cur_fov < 90.0 and cur_fnum < 1.5)

    if skill_is_fov_ood and task_is_fnum_ood:
        print(f"[self_evolve] ⚠ OOD类型不匹配："
              f"skill='{reused_name[:25]}' 为FOV-OOD，"
              f"当前任务 FOV={cur_fov}° F/{cur_fnum} 为fnum-OOD"
              f" → 允许蒸馏专属 skill", flush=True)
        return False

    return True


def _compact_trajectory() -> list:
    """★ FIX: write 类型补充 fnum/surface/param，供 Gemini 判断 fnum 变化和操作特征。"""
    out = []
    for s in _TRAJECTORY:
        kind  = s.get("kind", "write")
        b     = s.get("before") or {}
        a     = s.get("after")  or {}
        entry = {"step": s["t"], "tool": s["tool"], "kind": kind}
        if kind == "write":
            entry["action"] = s.get("action")
            entry["rms"]    = f"{b.get('rms','?')}→{a.get('rms','?')}"
            entry["effl"]   = f"{b.get('effl','?')}→{a.get('effl','?')}"
            if b.get("fnum") is not None or a.get("fnum") is not None:
                entry["fnum"] = f"{b.get('fnum','?')}→{a.get('fnum','?')}"
            if s.get("delta_rms") is not None:
                entry["delta_rms"] = round(s["delta_rms"], 4)
            _act = s.get("action") or {}
            if _act.get("surface") is not None:
                entry["surface"] = _act["surface"]
            if _act.get("param"):
                entry["param"] = _act["param"]
        elif kind == "decide":
            entry["decision"] = s.get("action")
        else:
            entry["observed"] = s.get("action")
        if s.get("note"):
            entry["note"] = s["note"]
        out.append(entry)
    return out


# ② Voyager Skill Library: DISTILL_PROMPT（含 skill_code 字段）
DISTILL_PROMPT = """你是光学设计专家，判断本次 Agent session 是否值得入 skill 库，
并生成可直接执行的 Python skill_code（Voyager 风格：技能即可执行代码）。

用户原始需求: {question}
目标规格: {target_spec}
最终达标: {final_passed}
最终指标: {final_metrics}

本次轨迹（按 kind 分类）:
  - kind="decide": decision 字段带决策结果（is_ood/pass/reasons）
  - kind="write":  有 rms/effl/fnum before→after，以及 surface（面号）和 param（参数名）
  - kind="read":   只有 observed 观察值

{trajectory}

【光学物理参照系——常识，不单独入库】：
像差因果: 球差大→正镜曲率强; 场曲→Petzval和大; 轴上色差→Vd不足; 倍率色差→光阑偏移
玻璃: 消色差正镜Vd>60+负镜Vd<30; 场曲负镜nd>正镜nd且差>0.15
结构: 玻璃厚>=max(0.8mm,SD×8%); 空气>0.3mm; stop_SD=EFFL/(2×F#); FOV>60°需>=6片

【已有 skill 列表】：
{existing_skills}

判断规则（优先级从高到低）：

0. 必须入库（满足任一直接入库）：
   - zemax_optimize 出现，merit 改善率>50%
     （仅 merit_before>0 时计算；merit_before=0 或缺失则跳过此条）
   - modify_lens param=semi_diameter → 必须入库
   - is_ood=true 或 OOD 起点 → 必须入库
   - 换候选镜头且 final_passed=True → 必须入库

1. 基线操作不入库（第0条全不满足时）：
   工具集合⊆{{rank_by_rms,check_spec,zemax_optimize}} 且 merit改善率≤50% 且无特殊操作 → 返回{{}}

2. 去重：参数数值集合完全不相交视为不重复；否则综合相似度>=0.72才重复。

3. 参数极端值即新规律：align_effl scale>2x或<0.5x；sd改动new/old>2x；delta_rms/rms_before>80%

4. OOD/跨域是高价值信号。

5. fnum 变化来自轨迹 fnum 字段（格式"before→after"）。字段缺失时不推断。
   merit_before=0 时跳过改善率判断。

若入库，严格返回 JSON（不加 markdown fence）:
{{
  "name": "Skill L-NNNN: 标题",
  "summary": "<=30字 症状→动作（含关键参数范围）",
  "triggers": ["具体症状+数值范围1", "具体工具名2", "具体参数阈值3"],
  "full": "【症状】具体数值特征\\n【原因】...\\n【动作】\\n  1. 带阈值\\n  2. ...\\n【预期改善】带数值\\n【风险】...",
  "skill_code": "def skill_run(lens_idx, target_effl=None, target_fnum=None):\\n    # Voyager 风格可执行代码\\n    # 按轨迹操作序列，调用 agent tools\\n    # 可组合其他 skill: from learned_skills import LEARNED_SKILLS\\n    steps = []\\n    # 1. 具体操作（含参数阈值注释）\\n    return steps"
}}

skill_code 要求：
- 独立 Python 函数 skill_run(lens_idx, ...)
- 调用 agent tools（modify_lens/align_effl/local_optimize/zemax_optimize 等）
- 注释说明每步的参数阈值
- 可组合：from learned_skills import LEARNED_SKILLS
- 不入库时返回空对象: {{}}
- ★ 若轨迹中出现"换候选后才达标"（即前几个候选场曲/停滞失败，换到 rank=N 后成功），
  skill_code 必须：
  1. 仍然调用 rank_by_rms 获取最新候选（保持"最近原则"）
  2. 记录失败候选的物理特征（如前组玻璃牌号、结构类型）作为跳过条件，
     而不是写死 lens_idx——因为同样的物理缺陷在任何类似镜头上都会重现
  3. 记录成功候选的关键特征，优先选具有相似结构特征的候选
  示例格式：
    # 已知场曲失败特征：前组 H-LAK52 + 大比例缩放 → Petzval 未校正
    skip_if_front_glass = ["H-LAK52"]   # 跳过有此特征的 rank1/2
    # 成功候选特征：前组含 D-LAF50，场曲可被校正
    prefer_front_glass = ["D-LAF50", "H-ZLAF55D"]
    # 从 rank_by_rms 结果中选第一个不含 skip 特征的候选
- ★★ 必须记录本次 session 达到的最佳 RMS 值（best_rms_achieved），
  以及达到该 RMS 所用的 zemax_optimize 轮数（zemax_rounds）：
    best_rms_achieved = <本次最终RMS mm>   # 已验证可达，下次直接以此为目标
    zemax_rounds = <本次zemax_optimize调用次数>  # 下次复用时运行同等轮数
  下次 agent 看到此 skill 后，应以 best_rms_achieved 为 rms_target 调用
  zemax_optimize，而不是用宽松的初始目标，从而直接获得更低的 RMS。

triggers 必须具体含数值范围（不能用"优化失败"等泛化词）。
summary 必须含数值范围。name 前缀 "Skill L-" + 4位数字。
"""


def distill_session(final_passed: bool,
                    final_metrics: dict,
                    gemini_api_key: str,
                    gemini_base_url: str,
                    gemini_model: str = "",   # Fix #8: 空串表示"运行时从 env 读取"
                    ) -> dict | None:
    # Fix #8: 默认参数若在 def 行求值，env 变量在 import 后修改不生效。
    # 改为在函数体内惰性读取，每次调用都能拿到最新值。
    if not gemini_model:
        gemini_model = (
            os.environ.get("GEMINI_MODEL_DISTILL")
            or os.environ.get("GEMINI_MODEL_SELECT", "gemini-2.5-flash-preview-05-20")
        )
    import sys
    print(f"[self_evolve] distill_session ENTER (model={gemini_model})", file=sys.stderr)
    print(f"[self_evolve] traj_len={len(_TRAJECTORY)} final_passed={final_passed}",
          file=sys.stderr, flush=True)

    # ★ Voyager 核心：先用现有 skill，有复用成功的就不蒸馏新的
    if _skill_reuse_succeeded():
        reused = _SESSION_CTX.get("skill_reused_name", "unknown")
        print(f"[self_evolve] skill_reused=True ('{reused}')，跳过蒸馏",
              file=sys.stderr, flush=True)
        _SESSION_CTX["_distill_skip_reason"] = f"现有 skill '{reused}' 复用成功，无需新建"
        return None

    _should = _should_distill(final_passed)
    print(f"[self_evolve] _should_distill={_should}", file=sys.stderr, flush=True)
    if not _should:
        gain      = sum(s["delta_rms"] for s in _TRAJECTORY if s.get("delta_rms") is not None)
        zemax_ran = any(s["tool"] == "zemax_optimize" for s in _TRAJECTORY)
        # ★ FIX: 写入具体原因，供 end_session 区分两种失败
        skip_msg = (f"_should_distill=False: "
                    f"gain={gain:.4f} zemax_ran={zemax_ran} traj_len={len(_TRAJECTORY)}")
        print(f"[self_evolve] skip: {skip_msg}", file=sys.stderr, flush=True)
        _SESSION_CTX["_distill_skip_reason"] = skip_msg
        return None

    _existing_lines = []
    try:
        for k, v in _load_learned().items():
            _existing_lines.append(f"- {k}: {v.get('summary', '')}")
    except Exception:
        pass
    try:
        from skill_summaries import SKILL_SUMMARIES
        for k, v in SKILL_SUMMARIES.items():
            _existing_lines.append(f"- {k}: {v}")
    except Exception:
        pass

    print(f"[self_evolve] sending {len(_TRAJECTORY)} steps", file=sys.stderr, flush=True)
    prompt = DISTILL_PROMPT.format(
        question        = _SESSION_CTX.get("question", ""),
        target_spec     = json.dumps(_SESSION_CTX.get("target_spec", {}), ensure_ascii=False),
        final_passed    = final_passed,
        final_metrics   = json.dumps(final_metrics, ensure_ascii=False),
        trajectory      = json.dumps(_compact_trajectory(), ensure_ascii=False, indent=2),
        existing_skills = "\n".join(_existing_lines) or "（无）",
    )

    try:
        from openai import OpenAI
        print("[self_evolve]   -> calling Gemini ...", file=sys.stderr, flush=True)
        _key  = (gemini_api_key
                 or os.environ.get("GEMINI_API_KEY",
                                   "sk-uwMXbGBi2LKb9EnmGIOQT1QOISpA8jgazzvXwVLq5o5h79WZ"))
        _url  = (gemini_base_url
                 or os.environ.get("GEMINI_BASE_URL", "https://us.novaiapi.com/v1"))
        cli  = OpenAI(api_key=_key, base_url=_url)
        resp = cli.chat.completions.create(
            model=gemini_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
        )
        _msg     = resp.choices[0].message
        _content = _msg.content or ""
        if not _content.strip():
            for _attr in ("reasoning_content", "reasoning", "thinking"):
                _v = getattr(_msg, _attr, None)
                if _v and str(_v).strip():
                    _content = str(_v)
                    print(f"[self_evolve]   (fallback to {_attr})", file=sys.stderr, flush=True)
                    break
        text = _content.strip()
        print(f"[self_evolve]   <- Gemini ({len(text)} chars)", file=sys.stderr, flush=True)
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
        _m = re.search(r"\{[\s\S]*\}", text)
        if not _m:
            _SESSION_CTX["_distill_skip_reason"] = "Gemini返回中找不到JSON对象"
            return None
        raw = _m.group(0)
        # ★ FIX: Gemini 有时在 skill_code 字符串里生成非法 \转义（如 \e \p 等）
        # 先尝试直接解析，失败则修复非法转义后重试
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # 把非法的单反斜杠替换为双反斜杠（仅修非法转义，合法 \n \t \\ 等保留）
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
            try:
                obj = json.loads(fixed)
            except json.JSONDecodeError as _e2:
                _SESSION_CTX["_distill_skip_reason"] = f"JSON修复后仍失败: {_e2}"
                return None
        if not obj or "name" not in obj or "full" not in obj:
            keys = list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
            _SESSION_CTX["_distill_skip_reason"] = f"Gemini JSON缺少必要字段: {keys}"
            return None
        return obj
    except Exception as e:
        import traceback
        print(f"[self_evolve] distill FAILED: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        _SESSION_CTX["_distill_skip_reason"] = f"Gemini调用异常: {type(e).__name__}: {e}"
        return None


# =============================================================
#   写入 learned_skills.py
# =============================================================
def _load_learned() -> dict:
    p = Path(LEARNED_PATH)
    if not p.exists():
        return {}
    try:
        ns: dict = {}
        exec(p.read_text(encoding="utf-8"), ns)
        return ns.get("LEARNED_SKILLS", {})
    except Exception as e:
        print(f"[self_evolve] load failed: {e}")
        return {}


def _save_learned(d: dict) -> None:
    p = Path(LEARNED_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Auto-appended by self_evolve.distill_session.\n"
        f"# Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Total skills: {len(d)}\n\n"
    )
    body = "LEARNED_SKILLS = " + json.dumps(d, ensure_ascii=False, indent=2) + "\n"
    # Fix #11: 原子写入——先写临时文件再 rename，防止进程中途崩溃损坏技能库。
    tmp = p.with_suffix(".tmp")
    tmp.write_text(header + body, encoding="utf-8")
    tmp.replace(p)


# ── 三层相似度融合（★ FIX: 替代原单层 bigram_jaccard）──────────
def _bigram_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ga = {a[i:i+2] for i in range(len(a)-1)}
    gb = {b[i:i+2] for i in range(len(b)-1)}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _trigram_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ga = {a[i:i+3] for i in range(len(a)-2)}
    gb = {b[i:i+3] for i in range(len(b)-2)}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _keyword_overlap(a: str, b: str) -> float:
    tok_a = set(re.split(r'[\s，,。、\-_/]+', a.strip()))
    tok_b = set(re.split(r'[\s，,。、\-_/]+', b.strip()))
    tok_a.discard(""); tok_b.discard("")
    if not tok_a or not tok_b:
        return 0.0
    return len(tok_a & tok_b) / len(tok_a | tok_b)


def _summary_sim(a: str, b: str) -> float:
    """bigram(0.35) + trigram(0.35) + keyword(0.30) 加权融合。"""
    return (0.35 * _bigram_jaccard(a, b)
            + 0.35 * _trigram_jaccard(a, b)
            + 0.30 * _keyword_overlap(a, b))


def append_skill(new_skill: dict) -> tuple:
    name    = new_skill.get("name", "").strip()
    summary = new_skill.get("summary", "").strip()
    if not name:
        return False, "name 为空"

    # ④ Self-Verification Gate（Voyager：入库前先过验证门）
    ok_v, reason_v = _verify_skill_effectiveness(new_skill, _TRAJECTORY)
    if not ok_v:
        return False, f"self-verification 未通过: {reason_v}"

    learned = _load_learned()

    if name in learned:
        return False, f"重名: {name}"

    # 三层相似度去重 + 参数数值范围二次校验
    for existing_name, existing in learned.items():
        sim = _summary_sim(summary, existing.get("summary", ""))
        if sim >= SIM_THRESHOLD:
            nums_new = set(re.findall(r'[\d.]+', summary))
            nums_old = set(re.findall(r'[\d.]+', existing.get("summary", "")))
            # 数值集合完全不相交 → 参数范围不同，放行
            if nums_new and nums_old and not (nums_new & nums_old):
                continue
            return False, f"与 {existing_name} 相似度 {sim:.2f} 过高"

    if not re.match(r"Skill L-\d{4}:", name):
        existing_ids = [int(m.group(1)) for k in learned
                        for m in [re.match(r"Skill L-(\d{4})", k)] if m]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        title   = name.split(":", 1)[-1].strip() or summary[:30]
        name    = f"Skill L-{next_id:04d}: {title}"
        new_skill["name"] = name

    learned[name] = new_skill
    _save_learned(learned)

    # ③ 新 skill 入库后清空 embedding 缓存，下次检索时重建
    try:
        Path(SKILL_EMBED_CACHE).unlink(missing_ok=True)
    except Exception:
        pass

    return True, "appended"


# =============================================================
#   Skill 原地更新：把本次实测 RMS 回写到已有 skill 的 skill_code
# =============================================================
def update_skill_best_rms(skill_name: str, achieved_rms: float) -> tuple:
    """
    每次 skill 复用成功后调用。
    把本次 Zemax 实测的最佳 RMS 写入该 skill 的 skill_code，
    让库随使用次数越来越准，下次 zemax_optimize 直接以此为更紧目标。

    策略：
    - 若 skill_code 已有 best_rms_achieved，且新值更好（更低） → 覆盖
    - 若 skill_code 尚无该字段 → 追加
    - 若新值 >= 已有值（变差了） → 跳过，保留历史最优

    返回 (ok: bool, reason: str)
    """
    import sys as _sys
    if not skill_name or achieved_rms is None:
        return False, "参数不完整"
    # ★ 拒绝 RMS=0 哨兵值（追迹失败），防止覆盖真实历史最优
    if achieved_rms <= 1e-6:
        return False, f"RMS={achieved_rms} 为追迹失败哨兵值，拒绝回写"
    # ★ 拒绝极端小值：RMS < 0.001mm 时停止覆盖
    # 防止偶发极好结果把 best_rms 压到 0.0001 量级，导致后续题
    # 用 rms_target=0.0001 优化，所有候选必然失败。
    # 0.001mm 已是衍射极限量级，低于此不应作为"可复用目标"。
    _RMS_FLOOR = 0.010
    if achieved_rms < _RMS_FLOOR:
        return False, (f"RMS={achieved_rms:.6f} < 下限 {_RMS_FLOOR}，"
                       f"属偶发极优结果，不覆盖 skill 目标（避免下次 rms_target 过严）")

    learned = _load_learned()
    if skill_name not in learned:
        # 模糊匹配
        skill_name_lower = skill_name.lower().strip()
        for k in learned:
            if k.lower().startswith(skill_name_lower) or skill_name_lower in k.lower():
                skill_name = k
                break
        else:
            return False, f"skill '{skill_name}' 不在库中"

    skill  = learned[skill_name]
    code   = skill.get("skill_code", "")

    # ── 1. 解析现有 best_rms_achieved ──────────────────────────
    existing_m = re.search(r'(best_rms_achieved\s*=\s*)([0-9]+\.?[0-9]*)', code)
    if existing_m:
        existing_rms = float(existing_m.group(2))
        if achieved_rms >= existing_rms:
            return False, (f"新 RMS {achieved_rms:.6f} >= 已有最优 {existing_rms:.6f}，"
                           f"保留历史最优，不更新")
        # 新值更好：原地替换数值
        new_code = code[:existing_m.start(2)] + f"{achieved_rms:.6f}" + code[existing_m.end(2):]
        action = f"覆盖: {existing_rms:.6f} → {achieved_rms:.6f}"
    else:
        # skill_code 里尚无该字段，追加到函数体首行（def 下一行）
        insert_comment = (
            f"\n    # ★ 经 Zemax 实测验证可达的最佳 RMS，下次直接覆盖宽松目标\n"
            f"    best_rms_achieved = {achieved_rms:.6f}   # 已验证可达\n"
            f"    zemax_rounds = 2               # 达标所用轮数\n"
        )
        # 找 def skill_run(...): 后第一个换行处插入
        def_m = re.search(r'(def\s+skill_run\s*\([^)]*\)\s*:)', code)
        if def_m:
            insert_pos = def_m.end()
            new_code   = code[:insert_pos] + insert_comment + code[insert_pos:]
        else:
            # 找不到函数头，直接在末尾追加变量
            new_code = code.rstrip() + f"\nbest_rms_achieved = {achieved_rms:.6f}\nzemax_rounds = 2\n"
        action = f"新增: best_rms_achieved={achieved_rms:.6f}"

    skill["skill_code"] = new_code

    # ── 2. 同步更新 summary 里的 RMS 数值（可选，便于检索时看到最新值）──
    old_summary = skill.get("summary", "")
    new_summary = re.sub(
        r'RMS[可达到至为]*[\s:：]*[\d.]+\s*mm',
        f'RMS可达{achieved_rms:.4f}mm',
        old_summary,
    )
    if new_summary != old_summary:
        skill["summary"] = new_summary

    learned[skill_name] = skill
    _save_learned(learned)

    # 清空 embedding 缓存（summary 变了需重建）
    try:
        Path(SKILL_EMBED_CACHE).unlink(missing_ok=True)
    except Exception:
        pass

    print(f"[self_evolve] ✓ update_skill_best_rms '{skill_name}': {action}",
          file=_sys.stderr, flush=True)
    return True, action


def end_session(final_passed: bool,
                final_metrics: dict,
                gemini_api_key: str,
                gemini_base_url: str,
                gemini_model: str = os.environ.get(
                    "GEMINI_MODEL_DISTILL",
                    os.environ.get("GEMINI_MODEL_SELECT", "gemini-2.5-flash-preview-05-20")
                )) -> dict:
    report = {
        "trajectory_len": len(_TRAJECTORY),
        "final_passed":   final_passed,
        "distilled":      False,
        "appended":       False,
        "new_skill_name": None,
        "reason":         "",
    }

    import sys as _sys2
    if not gemini_api_key:
        print("[self_evolve] ⚠ gemini_api_key 为空", file=_sys2.stderr)
    if not gemini_base_url:
        print("[self_evolve] ⚠ gemini_base_url 为空", file=_sys2.stderr)

    new_skill = distill_session(final_passed, final_metrics,
                                gemini_api_key, gemini_base_url, gemini_model)
    if new_skill is None:
        # ★ FIX: 区分"不值得蒸馏"和"Gemini返回空"两种失败
        report["reason"] = _SESSION_CTX.get("_distill_skip_reason",
                                             "Gemini returned empty or parse failed")
        import sys as _sys
        print(f"[self_evolve] 未入库: {report['reason']}", file=_sys.stderr, flush=True)
        _TRAJECTORY.clear()
        return report

    report["distilled"] = True
    ok, msg = append_skill(new_skill)
    report["appended"]       = ok
    report["new_skill_name"] = new_skill.get("name") if ok else None
    report["reason"]         = msg

    # ① 更新课程历史
    try:
        record_curriculum_result(_SESSION_CTX.get("target_spec", {}), final_passed)
    except Exception:
        pass

    import sys as _sys
    if ok:
        print(f"[self_evolve] ✓ 新增 skill: {new_skill.get('name')}", file=_sys.stderr, flush=True)
    else:
        print(f"[self_evolve] 未入库: {msg}", file=_sys.stderr, flush=True)

    _TRAJECTORY.clear()
    return report


# =============================================================
#   Prompt 侧接口
# =============================================================
def load_learned_for_prompt() -> dict:
    """全量加载（兼容旧接口）。推荐改用 retrieve_relevant_skills(question, k=5)。"""
    return {name: sk.get("summary", "") for name, sk in _load_learned().items()}


def get_learned_detail(name: str) -> str | None:
    learned = _load_learned()
    if name in learned:
        return learned[name].get("full", "")
    key_l = name.lower().strip()
    for k, v in learned.items():
        if k.lower().startswith(key_l) or key_l in k.lower():
            return v.get("full", "")
    return None
