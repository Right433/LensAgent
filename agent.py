

import os, sys, ast, pickle, argparse
from pathlib import Path

import torch
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

sys.path.insert(0, "/gz-data")
sys.path.insert(0, str(Path(__file__).parent))

from lens_optimizer import optimize_lens, format_optimize_result

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════
FAISS_DIR = "/gz-data/faiss_index"
EMB_MODEL = "shibing624/text2vec-base-chinese"

MODEL_CONFIGS = {
    "gemini": {
        "base_url":    "https://api.gptoai.top/v1",
        "api_key":     "sk-cjgfHqA58fZNqN8wKb7cffBAY7GyhTaJqzg3zkn4vm0orDV8",
        "model":       "gemini-2.5-pro-preview",
        "max_tokens":  4096,
        "temperature": 0,
        # Gemini 用 ReAct 文本格式
        "agent_type":  "react",
    },
    "qwen3": {
        "base_url":    "http://localhost:8000/v1",
        "api_key":     "EMPTY",
        # served-model-name 必须与 vllm serve --served-model-name 一致
        "model":       "qwen3-14b",
        "max_tokens":  4096,
        "temperature": 0,
        # Qwen3 用 OpenAI tools 格式（结构化 JSON tool call）
        "agent_type":  "openai_tools",
        # ⚠ 关闭 thinking 模式：避免 <think>...</think> 污染输出
        "extra_body":  {"chat_template_kwargs": {"enable_thinking": False}},
    },
}

# ── 全局状态 ──────────────────────────────────────────────────────────────────
VS:         FAISS = None
ALL_LENSES: list  = []
_device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": _device},
    encode_kwargs={"batch_size": 64},
)


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


# ══════════════════════════════════════════════════════════════════════════════
# RAG 加载
# ══════════════════════════════════════════════════════════════════════════════
def load_rag():
    faiss_path = Path(FAISS_DIR) / "index.faiss"
    pkl_path   = Path(FAISS_DIR) / "lenses.pkl"
    if not faiss_path.exists():
        raise FileNotFoundError(f"向量库不存在: {FAISS_DIR}")
    vs = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    lenses = []
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            lenses = pickle.load(f)
        for i, l in enumerate(lenses):
            l["lens_idx"] = i
        print(f"加载镜头数据: {len(lenses)} 条")
    return vs, lenses


# ══════════════════════════════════════════════════════════════════════════════
# 数值约束解析
# ══════════════════════════════════════════════════════════════════════════════
import re

def _parse_constraints(text: str) -> dict:
    def _first(patterns, s):
        for p in patterns:
            m = re.search(p, s)
            if m:
                return float(m.group(1))
        return None

    fov = _first([
        r'FOV\s*[=≈]\s*([\d.]+)',
        r'视场角\s*([\d.]+)',
        r'全视场\s*([\d.]+)',
        r'视场\s*([\d.]+)\s*[度°]',
        r'([\d.]+)\s*[度°]\s*视场',
    ], text)

    fnum = _first([
        r'F\s*/\s*([\d.]+)',
        r'F数\s*[=≈]?\s*([\d.]+)',
        r'Fno\s*[=≈]\s*([\d.]+)',
        r'光圈\s*F\s*/?\s*([\d.]+)',
        r'F\s*([\d.]+)',
    ], text)

    aper = _first([
        r'入瞳[径直径]*\s*[=≈]?\s*([\d.]+)\s*mm',
        r'口径\s*[=≈]?\s*([\d.]+)\s*mm',
        r'Aper\s*[=≈]\s*([\d.]+)',
        r'通光口径\s*([\d.]+)',
        r'入瞳\s*([\d.]+)',
    ], text)

    return {"fov": fov, "fnum": fnum, "aper": aper}


# ══════════════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════════════

@tool
def lens_search(query: str) -> str:
    """
    【语义检索】从镜头库中语义检索最相似的 top-5 候选镜头。
    输入: 自然语言需求，例如 'FOV=35度 F/2.8 RMS尽量小'。
    输出: top-5 候选，含 lens_idx / FOV / F# / Aper / EFFL / RMS。
    注意: lens_idx 可传给 rms_calculator 或 optimize_lens_tool 使用。
    """
    results = VS.similarity_search(query, k=5)
    if not results:
        return "未检索到相关镜头"
    lines = []
    for i, doc in enumerate(results):
        m    = doc.metadata
        rms  = m.get("calc_rms")
        effl = m.get("calc_effl")
        lines.append(
            f"[{i+1}] 镜头#{m.get('lens_idx')} | "
            f"FOV={m.get('fov')}° F/{m.get('fnum')} 入瞳径={m.get('aper')}mm | "
            f"EFFL={f'{effl:.2f}mm' if effl else '未知'} | "
            f"近轴RMS={f'{rms:.4f}mm' if rms else '未知'} | "
            f"来源={Path(str(m.get('source',''))).name}"
        )
    return "\n".join(lines)


@tool
def rank_by_rms(query: str) -> str:
    """
    【RMS排序】检索候选镜头并按近轴RMS升序排列，找到光学性能最优方案。
    输入: 需求描述字符串。
    输出: top-10 候选，按 RMS 从小到大排列。
    """
    results = VS.similarity_search(query, k=20)
    candidates = [(doc.metadata.get("calc_rms"), doc.metadata)
                  for doc in results if doc.metadata.get("calc_rms") is not None]
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        return "未找到有效候选"
    lines = ["按近轴RMS升序："]
    for rank, (rms, m) in enumerate(candidates[:10], 1):
        effl = m.get("calc_effl")
        lines.append(
            f"[{rank}] 镜头#{m.get('lens_idx')} | RMS={rms:.4f}mm | "
            f"FOV={m.get('fov')}° F/{m.get('fnum')} | "
            f"EFFL={f'{effl:.2f}mm' if effl else '未知'}"
        )
    return "\n".join(lines)


@tool
def rms_calculator(lens_idx: int) -> str:
    """
    【近轴追迹】精确计算指定镜头的光学性能（EFFL/TOTR/RMS）。
    输入: lens_idx 整数。
    输出: 详细近轴追迹结果。
    """
    try:
        lens_idx = int(lens_idx)
    except (ValueError, TypeError):
        return f"lens_idx 必须是整数，收到: {lens_idx!r}"
    if not ALL_LENSES:
        return "镜头数据未加载"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界（范围 0~{len(ALL_LENSES)-1}）"

    lens = ALL_LENSES[lens_idx]
    rms  = lens.get("calc_rms")
    effl = lens.get("calc_effl")
    totr = lens.get("calc_totr")
    yimg = lens.get("calc_yimg")

    if rms is None:
        r = _calc(lens)
        if not r.get("valid"):
            return f"追迹失败: {r.get('msg')}"
        effl, totr, rms, yimg = r["effl"], r["totr"], r["rms"], r["y_image"]

    return (
        f"镜头#{lens_idx} | FOV={lens.get('fov')}° F/{lens.get('fnum')} Aper={lens.get('aper')}mm\n"
        f"  EFFL    = {effl:.4f} mm\n"
        f"  TOTR    = {totr:.4f} mm\n"
        f"  近轴RMS = {rms:.6f} mm\n"
        f"  像面残差= {yimg:.4f} mm\n"
        f"  Zemax   = {lens.get('fit')}"
    )


@tool
def optimize_lens_tool(lens_idx: int,
                        target_fov:  float = None,
                        target_fnum: float = None,
                        target_aper: float = None) -> str:
    """
    【镜头优化】以检索到的镜头为基础，通过等比缩放+近轴数值优化，
    使其满足 out-of-domain 目标参数（FOV / F# / 入瞳径）。

    适用场景: 数据库中没有精确匹配的镜头，但有相近的方案可以作为起点。

    输入:
      lens_idx   — 来自 lens_search / rank_by_rms 的镜头编号
      target_fov  — 目标视场角（度），可为 None
      target_fnum — 目标 F/#，可为 None
      target_aper — 目标入瞳径（mm），可为 None

    优化流程:
      Step 1: 等比焦距缩放 → 满足宏观 Fno × Aper = EFFL 关系
      Step 2: Nelder-Mead 微调曲率半径 → 压低近轴 RMS

    输出: 优化前/后参数对比，RMS 改善量，达标检查。
    """
    try:
        lens_idx = int(lens_idx)
    except (ValueError, TypeError):
        return "lens_idx 必须是整数"
    if not ALL_LENSES:
        return "镜头数据未加载"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界"

    original = ALL_LENSES[lens_idx]
    print(f"  [optimizer] 优化镜头#{lens_idx}: "
          f"FOV={target_fov}° F/{target_fnum} Aper={target_aper}mm")

    optimized = optimize_lens(
        original,
        target_fov=target_fov,
        target_fnum=target_fnum,
        target_aper=target_aper,
        do_refine=True,
        refine_iters=80,
    )

    return format_optimize_result(original, optimized,
                                   target_fov, target_fnum, target_aper)


# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════

# ── Gemini 用：ReAct 文本格式 ─────────────────────────────────────────────────
REACT_PROMPT_STR = """你是专业光学镜头设计助手，帮助工程师设计和优化光学镜头方案。

可用工具：
{tools}

工具名列表: {tool_names}

【工作流程 — 必须按以下逻辑执行】

情形A — 数据库中有精确匹配（in-domain）:
  1. lens_search 检索候选
  2. rank_by_rms 找 RMS 最优
  3. rms_calculator 精确验证
  4. Final Answer

情形B — 数据库中无精确匹配（out-of-domain，检索结果 FOV/Fno 与目标偏差 > 10%）:
  1. lens_search 检索最近邻
  2. rank_by_rms 找最近邻中 RMS 最优的作为起点
  3. optimize_lens_tool 以该镜头为基础做参数优化（必须传入 target_fov/target_fnum/target_aper）
  4. Final Answer（引用优化后的 RMS 和达标检查结果）

判断依据：若 lens_search 返回的镜头 FOV 与目标偏差 > 5°，或 F# 偏差 > 0.5，视为 out-of-domain。

【输出格式】严格按以下格式，不得省略任何字段：

Question: {input}
Thought: 分析需求，提取 FOV / F# / 入瞳径，判断情形A还是B
Action: 工具名
Action Input: 参数
Observation: 工具返回
... (循环)
Thought: 已有足够信息
Final Answer: 推荐方案（含 lens_idx / FOV / F# / Aper / 近轴RMS / EFFL / 是否经过优化）

---示例（out-of-domain）---
Question: 帮我设计FOV=45度、F/3.5、入瞳径15mm的镜头
Thought: 提取参数：FOV=45°, F/3.5, Aper=15mm。先语义检索。
Action: lens_search
Action Input: FOV=45度 F/3.5 入瞳径15mm
Observation: [1] 镜头#22 | FOV=35° F/2.8 ...  ← 偏差较大，判断为 out-of-domain
Thought: 检索结果 FOV=35° 与目标 45° 偏差 10°，属于 out-of-domain，需要优化
Action: rank_by_rms
Action Input: FOV=45度 F/3.5
Observation: [1] 镜头#22 RMS=0.002mm ...
Thought: 以镜头#22 为起点做优化
Action: optimize_lens_tool
Action Input: {{"lens_idx": 22, "target_fov": 45.0, "target_fnum": 3.5, "target_aper": 15.0}}
Observation: 优化结果 ... RMS after=0.0008mm ... F/# ✅ Aper ✅
Thought: 优化完成，达标
Final Answer: 基于镜头#22 优化的方案，FOV=45°，F/3.5，入瞳径15mm，近轴RMS=0.0008mm（经近轴缩放+Nelder-Mead优化）
---

Question: {input}
Thought:{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(REACT_PROMPT_STR)

# ── Qwen3 用：OpenAI tools 格式（system + human + agent_scratchpad）────────────
# 不需要 {tools}/{tool_names}，工具 schema 由 LangChain 自动注入到 API 的 tools 字段
QWEN3_SYSTEM = """你是专业光学镜头设计助手，帮助工程师设计和优化光学镜头方案。

【工作流程 — 必须按以下逻辑执行】

情形A — 数据库中有精确匹配（in-domain）:
  1. 调用 lens_search 检索候选
  2. 调用 rank_by_rms 找 RMS 最优
  3. 调用 rms_calculator 精确验证
  4. 给出最终推荐

情形B — 数据库中无精确匹配（out-of-domain，检索结果 FOV/Fno 与目标偏差 > 10%）:
  1. 调用 lens_search 检索最近邻
  2. 调用 rank_by_rms 找 RMS 最优的作为起点
  3. 调用 optimize_lens_tool 做参数优化（必须传入 target_fov/target_fnum/target_aper）
  4. 给出最终推荐（引用优化后的 RMS 和达标检查结果）

判断依据：若 lens_search 返回的镜头 FOV 与目标偏差 > 5°，或 F# 偏差 > 0.5，视为 out-of-domain。

最终回答必须包含：lens_idx / FOV / F# / Aper / 近轴RMS / EFFL / 是否经过优化。"""

QWEN3_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QWEN3_SYSTEM),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# ══════════════════════════════════════════════════════════════════════════════
# LLM & Agent 构建
# ══════════════════════════════════════════════════════════════════════════════
def build_llm(model_name: str = "gemini") -> ChatOpenAI:
    cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["gemini"])

    kwargs = dict(
        model=cfg["model"],
        openai_api_key=cfg["api_key"],
        openai_api_base=cfg["base_url"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )
    # Qwen3 专属：关闭 thinking 模式，避免 <think> 块干扰工具调用解析
    if "extra_body" in cfg:
        kwargs["extra_body"] = cfg["extra_body"]

    return ChatOpenAI(**kwargs)


def build_agent(model_name: str = "gemini") -> AgentExecutor:
    llm    = build_llm(model_name)
    tools  = [lens_search, rank_by_rms, rms_calculator, optimize_lens_tool]
    cfg    = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["gemini"])

    if cfg["agent_type"] == "openai_tools":
        # Qwen3：结构化 tool call，不用 ReAct 文本解析
        from langchain.agents import create_openai_tools_agent
        agent = create_openai_tools_agent(llm, tools, QWEN3_PROMPT)
    else:
        # Gemini / 其他：ReAct 文本格式
        agent = create_react_agent(llm, tools, REACT_PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=12,
        handle_parsing_errors=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global VS, ALL_LENSES

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini",
                        choices=["gemini", "qwen3"],
                        help="使用的 LLM 模型")
    parser.add_argument("--query", default=None,
                        help="单次查询（不传则进入交互模式）")
    args = parser.parse_args()

    VS, ALL_LENSES = load_rag()
    executor  = build_agent(args.model)
    model_tag = f"[{args.model.upper()}]"

    print(f"\n{model_tag} 光学镜头优化 Agent 就绪")
    print("工具: lens_search / rank_by_rms / rms_calculator / optimize_lens_tool")

    if args.query:
        result = executor.invoke({"input": args.query})
        print(f"\n✅ {result['output']}")
        return

    print("示例: 帮我设计FOV=45度 F/3.5 入瞳径15mm的镜头")
    print("输入 quit 退出\n")

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        try:
            r = executor.invoke({"input": q})
            print(f"\n✅ {r['output']}\n")
        except Exception as e:
            print(f"❌ {e}\n")


if __name__ == "__main__":
    main()
