import os
# -*- coding: utf-8 -*-
"""
self_evolve.py
==============
补齐 agent_zemax.py 的两个缺口：
   step4 硬达标判断（check_spec）
   step3 self-evolve：把本次 session 的 trajectory 蒸馏成新 skill，
     追加到 /gz-data/learned_skills.py，下次 run_agent 会加载进 prompt。

集成到 agent_zemax.py 的四处：
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
                metrics_before: dict | None,
                metrics_after: dict | None,
                note: str = "") -> None:
    """
    在每个修改类 tool 返回前调用。
    metrics_before/after 里关键字段：rms, effl, totr（单位 mm）。
    """
    delta_rms = None
    if metrics_before and metrics_after \
       and metrics_before.get("rms") is not None \
       and metrics_after.get("rms") is not None:
        delta_rms = metrics_before["rms"] - metrics_after["rms"]

    _TRAJECTORY.append({
        "t": len(_TRAJECTORY) + 1,
        "tool": tool_name,
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
    }
    # 同时把 target 记下来（蒸馏时需要）
    if target_effl:  _SESSION_CTX.setdefault("target_spec", {})["effl"] = target_effl
    if target_fnum:  _SESSION_CTX.setdefault("target_spec", {})["fnum"] = target_fnum
    _SESSION_CTX.setdefault("target_spec", {})["rms_pass"] = rms_pass

    # infeasible 时覆盖 next，强制 agent 换候选或 Final Answer
    if infeasible:
        nxt = "try_next_candidate_or_final_answer"

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

    # 情况 A：成功 session 且累计 RMS 改善够大
    if final_passed:
        gain = sum(s["delta_rms"] for s in _TRAJECTORY if s.get("delta_rms") is not None)
        if gain > MIN_RMS_GAIN:
            return True

    # 情况 B：用到了罕见工具（split_lens / random_restart），信号价值高
    rare = {"split_lens", "random_restart"}
    if rare & {s["tool"] for s in _TRAJECTORY}:
        return True
    # 情况 B2: 成功 session 中改了光阑面 semi_diameter（Fnum 优化成功，高价值信号）
    sd_modified = any(
        s["tool"] == "modify_lens" and s["action"].get("param") == "semi_diameter"
        for s in _TRAJECTORY
    )
    if final_passed and sd_modified:
        return True

    # 情况 C：有过达标失败，且轨迹包含换材料动作（对"哪种材料在哪种症状下无效"有信息量）
    materials_changed = any(
        s["tool"] == "modify_lens" and s["action"].get("param") == "material"
        for s in _TRAJECTORY
    )
    if (not final_passed) and materials_changed:
        return True

    return False


def _compact_trajectory() -> list:
    """给 Gemini 的精简轨迹。"""
    out = []
    for s in _TRAJECTORY:
        b = s.get("before") or {}
        a = s.get("after") or {}
        out.append({
            "step":   s["t"],
            "tool":   s["tool"],
            "action": s["action"],
            "rms":    f"{b.get('rms','?')}→{a.get('rms','?')}",
            "effl":   f"{b.get('effl','?')}→{a.get('effl','?')}",
            "delta_rms": (round(s["delta_rms"], 4) if s.get("delta_rms") is not None else None),
        })
    return out


DISTILL_PROMPT = """你是光学设计专家，判断本次 Agent session 是否值得入 skill 库。

用户原始需求: {question}
目标规格: {target_spec}
最终达标: {final_passed}
最终指标: {final_metrics}

本次轨迹（时间顺序）:
{trajectory}

【已有 skill 列表（手写 + 已学到）】：
{existing_skills}

判断规则（**严格执行**）：

1. **基线操作不入库**：仅靠 align_effl + local_optimize + check_spec 且所有参数在常规范围内 → 返回 {{}}

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
   - 归纳"如何在数据库没覆盖的规格上硬搞出解"，这种 skill 比常规技巧价值更高

5. **真正新的动作组合才入库**：基于上述规则判断，不重复也非常规才入库

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
                    gemini_model: str = os.environ.get("GEMINI_MODEL_DISTILL", "gemini-3.1-pro-preview")) -> dict | None:
    """让 Gemini 归纳，返回新 skill dict 或 None。"""
    import sys
    print(f"[self_evolve] distill_session ENTER (model={gemini_model})", file=sys.stderr)
    if not _should_distill(final_passed):
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
        cli = OpenAI(api_key=gemini_api_key, base_url=gemini_base_url)
        resp = cli.chat.completions.create(
            model=gemini_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
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
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
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
                gemini_model: str = os.environ.get("GEMINI_MODEL_DISTILL", "gemini-3.1-pro-preview")) -> dict:
    """run_agent 收尾时调用。返回这次 session 的蒸馏摘要。"""
    report = {
        "trajectory_len": len(_TRAJECTORY),
        "final_passed":   final_passed,
        "distilled":      False,
        "appended":       False,
        "new_skill_name": None,
        "reason":         "",
    }
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
