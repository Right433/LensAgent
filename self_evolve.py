import os
# -*- coding: utf-8 -*-
"""
self_evolve.py
==============
补齐 agent_zemax.py 的两个缺口：
  ① step4 硬达标判断（check_spec）
  ② step3 self-evolve：把本次 session 的 trajectory 蒸馏成新 skill，
     追加到 /gz-data/learned_skills.py，下次 run_agent 会加载进 prompt。

集成到 agent_zemax.py 的四处（见 integration_patch.md）：
  1. import: from self_evolve import check_spec, record_step, start_session, end_session
  2. 在 build_agent() 的 tools=[...] 里把 check_spec 加进去
  3. 在 modify_lens / align_effl / split_lens / local_optimize / random_restart 的
     return 前各调一次 record_step(...)
  4. run_agent(question) 开头调 start_session，结尾调 end_session
"""

import json, os, time, re
from pathlib import Path
from langchain_core.tools import tool

# ─────────────────────────── 配置 ────────────────────────────
LEARNED_PATH   = "/gz-data/learned_skills.py"
MIN_TRAJ_LEN   = 2           # 轨迹太短不蒸馏
MIN_RMS_GAIN   = 0.05        # 成功 session 累计 RMS 改善阈值（mm）
SIM_THRESHOLD  = 0.80        # 与已有 skill 相似度 >= 此值视为重复，跳过
EFFL_TOL_PCT   = 2.0         # 达标判定：EFFL 相对偏差 < 2%
FNUM_TOL_PCT   = 2.0         # 达标判定：F# 相对偏差 < 2%
DEFAULT_RMS_PASS_MM = 1.0    # 近轴 RMS 放行阈值 = 1000 μm

# ─────────────────────────── 模块状态 ────────────────────────
_TRAJECTORY: list = []       # 本次 session 的动作轨迹
_SESSION_CTX: dict = {}      # 本次 session 的上下文


# =============================================================
#                  Session 生命周期 hooks
# =============================================================
def start_session(question: str, target_spec: dict | None = None) -> None:
    """run_agent(question) 开头调用。清空上一次的状态。"""
    _TRAJECTORY.clear()
    _SESSION_CTX.clear()
    _SESSION_CTX.update({
        "question": question,
        "target_spec": target_spec or {},
        "start_ts": time.time(),
    })


def record_step(tool_name: str,
                lens_idx: int,
                action: dict,
                metrics_before: dict | None = None,
                metrics_after: dict | None = None,
                note: str = "",
                kind: str = "write") -> None:
    """
    记录一步轨迹。
    metrics_before/after 里关键字段：rms, effl, totr（单位 mm）。

    kind:
      "write"  — 写入型（修改镜头面型），需要 before/after
      "read"   — 读取型（rms_calculator / get_lens_surfaces），只填 action.observed
      "decide" — 决策型（rank_by_rms / check_spec），记录 input→pick/pass/fail
    """
    delta_rms = None
    if metrics_before and metrics_after \
       and metrics_before.get("rms") is not None \
       and metrics_after.get("rms") is not None:
        delta_rms = metrics_before["rms"] - metrics_after["rms"]

    _TRAJECTORY.append({
        "t": len(_TRAJECTORY) + 1,
        "tool": tool_name,
        "kind": kind,
        "lens_idx": lens_idx,
        "action": action,
        "before": metrics_before,
        "after": metrics_after,
        "delta_rms": delta_rms,
        "note": note,
    })


# =============================================================
#                   Step4: 硬达标判断工具
# =============================================================
@tool
def check_spec(input_str: str) -> str:
    """
    【达标判断】统一判定镜头是否满足用户规格。是 step4 的硬门，决定能否进 Zemax 精优。
    输入格式: "lens_idx=<id>, target_effl=<mm>, target_fnum=<F>, rms_pass=<mm>"
      - rms_pass 可省略，默认 1.0 mm（=1000 μm）
      - target_effl / target_fnum 为用户需求，相对偏差须 < 2%
    示例: "lens_idx=36237, target_effl=34.0, target_fnum=2.8"
    返回 JSON: {"pass": bool, "metrics": {...}, "reasons": [...], "next": "..."}
      - pass=True  → 下一步调 zemax_optimize
      - pass=False → 看 reasons 决定调 modify_lens/align_effl/换其他镜头
    """
    try:
        parts = {}
        for seg in str(input_str).strip().split(","):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                parts[k.strip()] = v.strip()
        lens_idx    = int(parts["lens_idx"])
        target_effl = float(parts["target_effl"]) if parts.get("target_effl") else None
        target_fnum = float(parts["target_fnum"]) if parts.get("target_fnum") else None
        rms_pass    = DEFAULT_RMS_PASS_MM  # 固定1mm，忽略query传入
    except Exception as e:
        return json.dumps({"pass": False, "error": f"输入格式错误: {e}"}, ensure_ascii=False)

    # 运行时从主 agent 模块取数据（避免循环导入）
    try:
        from agent_zemax import ALL_LENSES, _calc
    except ImportError:
        return json.dumps({"pass": False, "error": "agent_zemax 未加载"}, ensure_ascii=False)

    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return json.dumps({"pass": False, "error": f"lens_idx {lens_idx} 越界"}, ensure_ascii=False)

    lens = ALL_LENSES[lens_idx]
    r = _calc(lens)
    if not r.get("valid"):
        return json.dumps({
            "pass": False,
            "reasons": [f"追迹失败: {r.get('msg')}"],
            "next": "reset_lens",
        }, ensure_ascii=False)

    rms      = float(r.get("rms", 1e9))
    effl_cur = float(r.get("effl", 0.0))
    fnum_cur = float(lens.get("fnum") or 0.0)

    reasons = []
    if rms > rms_pass:
        reasons.append(f"RMS={rms:.4f}mm 超过放行阈值{rms_pass}mm")
    if target_effl is not None and target_effl > 0:
        dev = abs(effl_cur - target_effl) / abs(target_effl) * 100.0
        if dev > EFFL_TOL_PCT:
            reasons.append(f"EFFL={effl_cur:.2f}mm vs 目标{target_effl}mm 偏差{dev:.1f}%>2%")
    # Fnum 单向判定：光圈偏大（fnum < target）可通过收缩光阑解决，视为 pass
    # 只惩罚光圈偏小（fnum > target），且大于 tol 才判未达标
    if target_fnum is not None and target_fnum > 0:
        if fnum_cur > target_fnum * (1 + FNUM_TOL_PCT / 100.0):
            dev = (fnum_cur - target_fnum) / abs(target_fnum) * 100.0
            reasons.append(f"F#={fnum_cur} vs 目标{target_fnum} 偏大{dev:.1f}%>2% (光圈偏小,无法通过优化扩张)")

    passed = len(reasons) == 0

    # ─── 不可达检测：F# 偏差>30% 或 FOV 偏差>50% 标 infeasible ───
    import re as _re
    infeasible = []
    for _reason in reasons:
        _m = _re.search(r'偏差([\d.]+)%', _reason)
        if not _m:
            continue
        _dev = float(_m.group(1))
        if 'F#' in _reason and _dev > 30:
            infeasible.append(f'F#差距{_dev:.0f}%超出local_optimize能力,需换候选')
        elif 'FOV' in _reason and _dev > 50:
            infeasible.append(f'FOV差距{_dev:.0f}%需换结构,无法通过优化修复')

    # 基于失败原因给 agent 下一步提示
    if passed:
        nxt = "zemax_optimize"
    elif any("EFFL" in x for x in reasons):
        nxt = "align_effl"
    elif any("RMS" in x for x in reasons):
        nxt = "local_optimize"
    else:
        nxt = "continue_optimize"

    # 存进 session ctx，供蒸馏时用
    _SESSION_CTX["last_check"] = {
        "lens_idx": lens_idx, "pass": passed,
        "rms": rms, "effl": effl_cur, "fnum": fnum_cur,
        "reasons": reasons,
        "source": "paraxial_check_spec",
    }
    # 同时把 target 记下来（蒸馏时需要）
    if target_effl:  _SESSION_CTX.setdefault("target_spec", {})["effl"] = target_effl
    if target_fnum:  _SESSION_CTX.setdefault("target_spec", {})["fnum"] = target_fnum
    _SESSION_CTX.setdefault("target_spec", {})["rms_pass"] = rms_pass

    # infeasible 时覆盖 next，强制 agent 换候选或 Final Answer
    if infeasible:
        nxt = "try_next_candidate_or_final_answer"

    # ★ self-evolve: 记录 check_spec 决策
    try:
        record_step("check_spec", lens_idx,
                    {"target_effl": target_effl, "target_fnum": target_fnum,
                     "pass": passed,
                     "rms_paraxial": round(rms, 4),
                     "effl_cur": round(effl_cur, 2),
                     "fnum_cur": fnum_cur,
                     "reasons": reasons,
                     "next": nxt,
                     "note": "近轴追迹(F/1.2 大孔径下与真值可差 >10x)"},
                    kind="decide")
    except Exception:
        pass

    return json.dumps({
        "pass": passed,
        "metrics": {"rms": round(rms, 4), "effl": round(effl_cur, 2), "fnum": fnum_cur},
        "reasons": reasons,
        "infeasible": infeasible,
        "next": nxt,
    }, ensure_ascii=False)


# =============================================================
#              Step3 self-evolve: 蒸馏 & 追加
# =============================================================
def _should_distill(final_passed: bool) -> bool:
    """启发式决定本次 session 是否值得蒸馏。避免 skill 库被噪声稀释。"""
    if len(_TRAJECTORY) < MIN_TRAJ_LEN:
        return False

    zemax_ran    = any(s["tool"] == "zemax_optimize" for s in _TRAJECTORY)
    zemax_passed = any(
        s["tool"] == "zemax_optimize"
        and (s.get("action") or {}).get("zemax_pass") is True
        for s in _TRAJECTORY
    )
    rare = {"split_lens", "random_restart"}

    # ── A：成功 session 跑了 Zemax（最常见有价值路径）────────────────────────
    if final_passed and zemax_ran:
        return True

    # ── B：用到了罕见工具 ──────────────────────────────────────────────────────
    if rare & {s["tool"] for s in _TRAJECTORY}:
        return True

    # ── C：光阑 SD 被修改（F# 调整，无论成败）────────────────────────────────
    sd_modified = any(
        s["tool"] == "modify_lens" and s["action"].get("param") == "semi_diameter"
        for s in _TRAJECTORY
    )
    if sd_modified:
        return True

    # ── D：OOD 起点 + Zemax 真值达标 ─────────────────────────────────────────
    ood_start = any(
        s["tool"] == "rank_by_rms" and (s.get("action") or {}).get("is_ood") is True
        for s in _TRAJECTORY
    )
    if ood_start and zemax_passed:
        return True

    # ── E：失败 + 材料被修改（负向经验：哪种策略无效）────────────────────────
    materials_changed = any(
        s["tool"] == "modify_lens" and s["action"].get("param") == "material"
        for s in _TRAJECTORY
    )
    if (not final_passed) and materials_changed:
        return True

    # ── F：★ EFFL 严重不匹配导致失败（新增）────────────────────────────────
    # 捕获「RAG 镜头 EFFL 与目标偏差 >30%，优化失败」→ 教 agent 先 align_effl
    effl_mismatch_fail = any(
        s["tool"] == "zemax_optimize"
        and (s.get("action") or {}).get("zemax_pass") is False
        and (s.get("action") or {}).get("effl_mismatch_pct", 0) > 30
        for s in _TRAJECTORY
    )
    if (not final_passed) and effl_mismatch_fail:
        return True

    return False


def _compact_trajectory() -> list:
    """
    给 Gemini 的精简轨迹。按 kind 分别渲染:
      - write:  完整 before→after + delta_rms
      - decide: 决策意图 + 关键观察 (is_ood / pass / selected_id)
      - read:   只保留最重要的 observation (如 rms_calculator 的 rms)
    """
    out = []
    for s in _TRAJECTORY:
        kind = s.get("kind", "write")
        b = s.get("before") or {}
        a = s.get("after") or {}
        entry = {"step": s["t"], "tool": s["tool"], "kind": kind}

        if kind == "write":
            entry["action"] = s.get("action")
            entry["rms"]    = f"{b.get('rms','?')}→{a.get('rms','?')}"
            entry["effl"]   = f"{b.get('effl','?')}→{a.get('effl','?')}"
            if s.get("delta_rms") is not None:
                entry["delta_rms"] = round(s["delta_rms"], 4)
        elif kind == "decide":
            # 决策型:action 里就带关键观察(picked_id / is_ood / pass / reasons)
            entry["decision"] = s.get("action")
        else:  # read
            # 读取型:只保留 action 里最小信息量
            entry["observed"] = s.get("action")

        note = s.get("note")
        if note:
            entry["note"] = note
        out.append(entry)
    return out


DISTILL_PROMPT = """你是光学设计专家，判断本次 Agent session 是否值得入 skill 库。

用户原始需求: {question}
目标规格: {target_spec}
最终达标: {final_passed}
最终指标: {final_metrics}

本次轨迹（时间顺序，按 kind 分类）:
  - kind="decide": 决策型步骤(rank_by_rms 选候选 / check_spec 判达标)。action 字段内带决策结果
  - kind="write":  写入型步骤(modify_lens / align_effl / zemax_optimize 等)。有 before/after rms 对比
  - kind="read":   读取型步骤(rms_calculator / get_lens_surfaces)。只有 observed 观察值

{trajectory}

【光学物理参照系——判断 skill 是否真正新颖时的对照基准】
以下是已知的光学设计常识，Gemini 归纳时应以此为参照，
只有超出以下常识范围的路径才算"真正新颖"值得入库：

像差因果（常识，不单独入库）：
  • 球差大 → 正镜曲率强或孔径大；换高nd正镜或拆分强弯面
  • 场曲大 → Petzval和过大；负镜换高nd(>1.78)，正镜换低nd(<1.55)
  • 轴上色差 → 正镜Vd不足；换H-FK61(Vd=70)/H-FK71(Vd=84)消色差
  • 倍率色差 → 光阑偏离主组；Vd差需>20才能有效消色差
  这些是教科书级常识，若本次 session 只做了以上操作，无需入库。

玻璃选型（常识，不单独入库）：
  • 消色差：正镜高Vd(>60) + 负镜低Vd(<30)，Vd差>40最佳
  • 场曲：负镜nd要>正镜nd，nd差>0.15效果显著
  • CDGM替换：非H-*/D-*牌号按nd最近邻换，ΔEFFL<5%可不对齐

结构约束（常识，不单独入库）：
  • 玻璃最小厚度=max(0.8mm, SD×8%)；空气间隔>0.3mm
  • stop_SD = EFFL/(2×F#)；偏差>10%需手动调
  • FOV>60°需≥6片；F#<2.0需≥5片

值得入库的情况（超出以上常识的新发现）：
  ★ 特定OOD参数组合（如FOV=58°+F/1.8）的具体成功路径
  ★ 某种玻璃组合在特定FOV/F#范围内的实测收敛规律
  ★ 非常规的操作顺序（如先random_restart再换候选）取得成功
  ★ 极端参数下（EFFL缩放>2×，SD扩大>2×）的修复策略
  ★ 物理修复（auto_fix_physics）发现并修复了影响收敛的结构问题

【已有 skill 列表（手写 + 已学到）】：
{existing_skills}

判断规则（**严格执行**，优先级从高到低）：

0. **必须入库的情况（优先级最高，满足任一条直接入库，不受第1条限制）**：
   - zemax_optimize 出现且 merit_delta/merit_before > 0.5（merit 改善 >50%）→ **必须入库**
   - 轨迹中有 modify_lens 且 param=semi_diameter（扩/缩光阑调 F#）→ **必须入库**
   - rank_by_rms 的 decision 里 is_ood=true，或者 top20 Fnum 与 target_fnum 不完全匹配（OOD 起点）→ **必须入库**
   - 换了候选镜头（轨迹中出现多个不同 lens_idx）且最终 final_passed=True → **必须入库**

1. **基线操作不入库**（仅当第0条一项都不满足时才执行此条）：
   轨迹工具集合是 {{rank_by_rms, check_spec, zemax_optimize}} 的子集，
   且 zemax_optimize 的 merit_delta/merit_before ≤ 0.5，
   且无光阑修改、无换候选、无 OOD → 返回 {{}}

2. **去重（工具组合+参数范围都要比）**：
   - 如果 tool 序列和已有 skill 相同，**还要比较参数范围**
   - 已有 skill 覆盖的参数范围才算去重
   - 例：Skill 13 "扩光阑降 F#" 只覆盖常规 sd 调整（<2x），若本次 sd 改动 >2x 则**不算重复**
   - 例：已有 skill 覆盖 align_effl scale 0.5~2.0，若本次 scale >2.0 或 <0.5 则**不算重复**

3. **参数极端值即新规律**（即使工具组合看似平常）：
   - `align_effl` 的 `scale > 2.0` 或 `< 0.5` → **非常规 EFFL 缩放**，应入库
   - `modify_lens` 改 `semi_diameter` 时 `new/old > 2.0` → **非常规光阑扩张**
   - `local_optimize` 单步 `delta_rms / rms_before > 0.8`（改善 >80%）→ **稀有的大幅改善**
   - 这些情况 summary 必须**写出具体参数范围**，例如"OOD 场景 EFFL 3x 缩放后可优化"而非笼统"扩光阑降 F#"

4. **OOD/跨域泛化是高价值信号**：
   - 当 target_spec 远离数据库分布（如极端 FOV、F#、EFFL 组合），且 final_passed=True
   - 轨迹里 rank_by_rms 的 decision 字段若 is_ood=true,说明起点就是 OOD
   - 归纳"如何在数据库没覆盖的规格上硬搞出解"，这种 skill 比常规技巧价值更高

5. **zemax_optimize 的真值优化路径** (kind=write, tool=zemax_optimize):
   - 若 action.zemax_pass=True 且 merit_delta/merit_before > 0.5 (merit 改善 >50%),
     说明 DLS 对该候选结构的收敛能力强,值得记录"什么起点 → 什么结果"的经验
   - 特别是 F# 被修改(zemax_pre_fnum ≠ zemax_post_fnum)后仍然真值达标的,
     说明 OOD 扩光阑后 Zemax 也能兜住

6. **真正新的动作组合才入库**：基于上述规则判断，不重复也非常规才入库

若入库，严格返回 JSON（不加 markdown fence）:
{{
  "name": "Skill L-NNNN: 标题",
  "summary": "≤30字 症状→动作（含关键参数范围）",
  "triggers": ["关键词1", "关键词2", "关键词3"],
  "full": "【症状】具体数值特征...
【原因】...
【动作】
  1. 【带具体参数阈值的公式】
  2. ...
【预期改善】带数值...
【风险】..."
}}

summary 必须含**数值范围**，不能只说"扩光阑"而要说"扩光阑 sd >2x"。
name 前缀用 "Skill L-" + 4 位数字。
"""


def distill_session(final_passed: bool,
                    final_metrics: dict,
                    gemini_api_key: str,
                    gemini_base_url: str,
                    gemini_model: str = os.environ.get("GEMINI_MODEL_DISTILL", os.environ.get("GEMINI_MODEL_SELECT", "gemini-3-flash-preview"))) -> dict | None:
    """让 Gemini 归纳，返回新 skill dict 或 None。"""
    import sys
    print(f"[self_evolve] distill_session ENTER (model={gemini_model})", file=sys.stderr)
    print(f"[self_evolve] trajectory_len={len(_TRAJECTORY)} final_passed={final_passed}", file=sys.stderr, flush=True)
    _should = _should_distill(final_passed)
    print(f"[self_evolve] _should_distill={_should}", file=sys.stderr, flush=True)
    if not _should:
        # 打印具体原因
        import sys as _sys3
        gain = sum(s["delta_rms"] for s in _TRAJECTORY if s.get("delta_rms") is not None)
        zemax_ran = any(s["tool"] == "zemax_optimize" for s in _TRAJECTORY)
        print(f"[self_evolve] skip reason: gain={gain:.4f} zemax_ran={zemax_ran} traj_len={len(_TRAJECTORY)}", file=_sys3.stderr, flush=True)
        return None

    # 加载已有 skill 列表（learned + 手写）供 Gemini 去重
    _existing_lines = []
    try:
        _learned = _load_learned()
        for k, v in _learned.items():
            _existing_lines.append(f"- {k}: {v.get('summary', '')}")
    except Exception:
        pass
    try:
        from skill_summaries import SKILL_SUMMARIES
        for k, v in SKILL_SUMMARIES.items():
            _existing_lines.append(f"- {k}: {v}")
    except Exception:
        pass
    _existing_str = "\n".join(_existing_lines) if _existing_lines else "（无）"

    import sys as _sys
    print(f"[self_evolve] sending {len(_TRAJECTORY)} trajectory steps", file=_sys.stderr, flush=True)

    prompt = DISTILL_PROMPT.format(
        question       = _SESSION_CTX.get("question", ""),
        target_spec    = json.dumps(_SESSION_CTX.get("target_spec", {}), ensure_ascii=False),
        final_passed   = final_passed,
        final_metrics  = json.dumps(final_metrics, ensure_ascii=False),
        trajectory     = json.dumps(_compact_trajectory(), ensure_ascii=False, indent=2),
        existing_skills = _existing_str,
    )

    try:
        import sys as _sys
        from openai import OpenAI
        print("[self_evolve]   -> calling Gemini ...", file=_sys.stderr, flush=True)
        # ★ FIX: fallback 到 rank_by_rms 使用的 novaiapi key（当 .env 未设置时）
        _api_key  = gemini_api_key or os.environ.get("GEMINI_API_KEY", "sk-uwMXbGBi2LKb9EnmGIOQT1QOISpA8jgazzvXwVLq5o5h79WZ")
        _base_url = gemini_base_url or os.environ.get("GEMINI_BASE_URL", "https://us.novaiapi.com/v1")
        cli = OpenAI(api_key=_api_key, base_url=_base_url)
        resp = cli.chat.completions.create(
            model=gemini_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
        )
        _msg = resp.choices[0].message
        _content = _msg.content or ""
        # Pro 在 thinking 模式下 content 可能为空，fallback 到 reasoning_content
        if not _content.strip():
            for _attr in ("reasoning_content", "reasoning", "thinking"):
                _v = getattr(_msg, _attr, None)
                if _v and str(_v).strip():
                    _content = str(_v)
                    print(f"[self_evolve]   (fallback to {_attr})", file=_sys.stderr, flush=True)
                    break
        text = _content.strip()
        print(f"[self_evolve]   <- Gemini returned ({len(text)} chars)", file=_sys.stderr, flush=True)
        print(f"[self_evolve]   raw[:200]: {text[:200]!r}", file=_sys.stderr, flush=True)
        # 先剥 markdown 代码块
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
        # Gemini 有时在 JSON 前输出推理文字，用正则提取第一个完整 {...} 块
        _m = re.search(r"\{[\s\S]*\}", text)
        if not _m:
            print("[self_evolve]   -> 返回文本中找不到 JSON 对象", file=_sys.stderr, flush=True)
            return None
        text = _m.group(0)
        obj = json.loads(text)
        if not obj or "name" not in obj or "full" not in obj:
            print(f"[self_evolve]   -> empty/missing keys, keys={list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__}", file=_sys.stderr, flush=True)
            return None
        return obj
    except Exception as e:
        import sys, traceback
        print(f"[self_evolve] distill FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


# =============================================================
#                    写入 learned_skills.py
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
        print(f"[self_evolve] load learned_skills.py failed: {e}")
        return {}


def _save_learned(d: dict) -> None:
    p = Path(LEARNED_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Auto-appended by self_evolve.distill_session.\n"
        "# Do not hand-edit individual entries (will be regenerated).\n"
        f"# Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Total skills: {len(d)}\n\n"
    )
    body = "LEARNED_SKILLS = " + json.dumps(d, ensure_ascii=False, indent=2) + "\n"
    p.write_text(header + body, encoding="utf-8")


def _bigram_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ga = {a[i:i+2] for i in range(len(a)-1)}
    gb = {b[i:i+2] for i in range(len(b)-1)}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def append_skill(new_skill: dict) -> tuple[bool, str]:
    """(appended?, reason)"""
    name = new_skill.get("name", "").strip()
    summary = new_skill.get("summary", "").strip()
    if not name:
        return False, "name 为空"

    learned = _load_learned()

    # 1. 精确重名
    if name in learned:
        return False, f"重名: {name}"

    # 2. summary 相似度去重
    for existing_name, existing in learned.items():
        sim = _bigram_jaccard(summary, existing.get("summary", ""))
        if sim >= SIM_THRESHOLD:
            return False, f"与 {existing_name} 相似度 {sim:.2f} 过高"

    # 3. 自动分配 L-NNNN（如果 Gemini 给的不唯一）
    if not re.match(r"Skill L-\d{4}:", name):
        existing_ids = [int(m.group(1)) for k in learned
                        for m in [re.match(r"Skill L-(\d{4})", k)] if m]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        title = name.split(":", 1)[-1].strip() or summary[:30]
        name = f"Skill L-{next_id:04d}: {title}"
        new_skill["name"] = name

    learned[name] = new_skill
    _save_learned(learned)
    return True, "appended"


def end_session(final_passed: bool,
                final_metrics: dict,
                gemini_api_key: str,
                gemini_base_url: str,
                gemini_model: str = os.environ.get("GEMINI_MODEL_DISTILL", os.environ.get("GEMINI_MODEL_SELECT", "gemini-3-flash-preview"))) -> dict:
    """run_agent 收尾时调用。返回这次 session 的蒸馏摘要。"""
    report = {
        "trajectory_len": len(_TRAJECTORY),
        "final_passed":   final_passed,
        "distilled":      False,
        "appended":       False,
        "new_skill_name": None,
        "reason":         "",
    }
    # ★ FIX: gemini_api_key/base_url 缺失时给出明确提示，不再静默吞错
    if not gemini_api_key:
        import sys as _sys2
        print("[self_evolve] ⚠ gemini_api_key 为空，使用 novaiapi 内置 key", file=_sys2.stderr)
    if not gemini_base_url:
        import sys as _sys2
        print("[self_evolve] ⚠ gemini_base_url 为空，使用 novaiapi 内置 base_url", file=_sys2.stderr)
    new_skill = distill_session(
        final_passed, final_metrics,
        gemini_api_key, gemini_base_url, gemini_model,
    )
    if new_skill is None:
        report["reason"] = "not worth distilling or Gemini returned empty"
        import sys as _sys
        print(f"[self_evolve] 未入库: {report['reason']}", file=_sys.stderr, flush=True)
        _TRAJECTORY.clear()
        return report

    # 成功入库时也打日志

    report["distilled"] = True
    ok, msg = append_skill(new_skill)
    report["appended"]       = ok
    report["new_skill_name"] = new_skill.get("name") if ok else None
    report["reason"]         = msg
    import sys as _sys
    if ok:
        print(f"[self_evolve] ✓ 新增 skill: {new_skill.get('name')}", file=_sys.stderr, flush=True)
    else:
        print(f"[self_evolve] 未入库: {msg}", file=_sys.stderr, flush=True)
    # 清 trajectory 防止二次蒸馏污染
    _TRAJECTORY.clear()
    return report


# =============================================================
#      让 prompt 里能读到 learned skills（build_skill_index 用）
# =============================================================
def load_learned_for_prompt() -> dict:
    """返回 {name: summary}，build_skill_index_text 里合并到 SKILL_SUMMARIES。"""
    learned = _load_learned()
    return {name: sk.get("summary", "") for name, sk in learned.items()}


def get_learned_detail(name: str) -> str | None:
    """get_skill_detail fallback：找不到手工 skill 时查 learned。"""
    learned = _load_learned()
    if name in learned:
        return learned[name].get("full", "")
    # 模糊匹配
    key_l = name.lower().strip()
    for k, v in learned.items():
        if k.lower().startswith(key_l) or key_l in k.lower():
            return v.get("full", "")
    return None
