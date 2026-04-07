"""
agent.py  —  第二步：ReAct Agent baseline
依赖 build_rag.py 已经跑完（FAISS_DIR 里有 index.faiss + lenses.pkl）

用法:
    python agent.py                 # 交互模式
    python agent.py --query "找RMS最小的镜头"   # 单次查询
"""

import os, sys, ast, pickle, argparse
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

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════
BASE_URL  = "https://api.gptoai.top/v1"
API_KEY   = "sk-cjgfHqA58fZNqN8wKb7cffBAY7GyhTaJqzg3zkn4vm0orDV8"
LLM_MODEL = "gemini-2.5-pro-preview-05-06"
EMB_MODEL = "shibing624/text2vec-base-chinese"
FAISS_DIR = "/gz-data/faiss_index"

# ══════════════════════════════════════════════════════════════════════════════
# 模型初始化
# ══════════════════════════════════════════════════════════════════════════════
_device = "cuda" if torch.cuda.is_available() else "cpu"

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0,
    max_tokens=2048,
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": _device},
    encode_kwargs={"batch_size": 64},
)

# ══════════════════════════════════════════════════════════════════════════════
# 加载 RAG 向量库 + 镜头数据
# ══════════════════════════════════════════════════════════════════════════════
def load_rag():
    faiss_path = Path(FAISS_DIR) / "index.faiss"
    pkl_path   = Path(FAISS_DIR) / "lenses.pkl"
    if not faiss_path.exists():
        raise FileNotFoundError(f"向量库不存在，请先运行 build_rag.py：{FAISS_DIR}")

    print(f"加载向量库: {FAISS_DIR}")
    vs = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

    lenses = []
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            lenses = pickle.load(f)
        print(f"加载镜头数据: {len(lenses)} 条")
    else:
        print("⚠ lenses.pkl 不存在，rms_calculator 将不可用")

    return vs, lenses

# ══════════════════════════════════════════════════════════════════════════════
# optical_calculator（用于实时追迹）
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
# 全局状态（由 main() 初始化）
# ══════════════════════════════════════════════════════════════════════════════
VS: FAISS       = None
ALL_LENSES: list = []

# ══════════════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════════════

@tool
def lens_search(query: str) -> str:
    """
    【语义检索】根据自然语言需求从镜头库中检索最相似的候选镜头。
    输入: 需求描述字符串，例如 'FOV=35度 F/2.8 RMS尽量小' 或 '大视场小光圈镜头'。
    输出: top-5 候选镜头，含编号(lens_idx)、FOV、Fno、EFFL、近轴RMS、玻璃组合。
    注意: lens_idx 是全局唯一编号，后续可传给 rms_calculator 做精确计算。
    """
    results = VS.similarity_search(query, k=5)
    if not results:
        return "未检索到相关镜头"
    lines = []
    for i, doc in enumerate(results):
        m = doc.metadata
        rms   = m.get("calc_rms")
        effl  = m.get("calc_effl")
        rms_s = f"{rms:.4f}mm"   if rms  is not None else "未知"
        efl_s = f"{effl:.2f}mm"  if effl is not None else "未知"
        lines.append(
            f"[{i+1}] 镜头#{m.get('lens_idx')} | "
            f"FOV={m.get('fov')}° F/{m.get('fnum')} 入瞳径={m.get('aper')}mm | "
            f"EFFL={efl_s} | 近轴RMS={rms_s} | "
            f"玻璃={m.get('surfaces','')[:40]}... | "   # 截断防止过长
            f"来源={Path(str(m.get('source',''))).name}"
        )
    return "\n".join(lines)


@tool
def rms_calculator(lens_idx: int) -> str:
    """
    【近轴追迹】查询或重新计算指定镜头的光学性能。
    输入: lens_idx 整数（从 lens_search 结果中获取）。
    输出: EFFL / TOTR / 近轴RMS spot size / 像面残差 / Zemax merit function 对比。
    用途: 对 lens_search 返回的候选镜头做精确验证，确认其光学性能是否符合要求。
    """
    try:
        lens_idx = int(lens_idx)   # LLM 有时传字符串，强制转 int
    except (ValueError, TypeError):
        return f"lens_idx 必须是整数，收到: {lens_idx!r}"
    if not ALL_LENSES:
        return "镜头数据未加载（lenses.pkl 缺失）"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界（共 {len(ALL_LENSES)} 条，范围 0~{len(ALL_LENSES)-1}）"

    lens      = ALL_LENSES[lens_idx]
    calc_rms  = lens.get("calc_rms")
    calc_effl = lens.get("calc_effl")
    calc_totr = lens.get("calc_totr")
    calc_yimg = lens.get("calc_yimg")
    fit_val   = lens.get("fit")

    if calc_rms is None:
        r = _calc(lens)
        if not r.get("valid"):
            return f"追迹失败: {r.get('msg')}"
        calc_effl = r["effl"]
        calc_totr = r["totr"]
        calc_rms  = r["rms"]
        calc_yimg = r["y_image"]

    return (
        f"镜头#{lens_idx} | "
        f"FOV={lens.get('fov')}° F/{lens.get('fnum')} 入瞳径={lens.get('aper')}mm\n"
        f"  EFFL       = {calc_effl:.4f} mm\n"
        f"  TOTR       = {calc_totr:.4f} mm\n"
        f"  近轴RMS    = {calc_rms:.6f} mm\n"
        f"  像面残差 y = {calc_yimg:.4f} mm  （理想=0）\n"
        f"  Zemax fit  = {fit_val}  （merit function 参考值）\n"
        f"  来源: {Path(str(lens.get('source',''))).name}"
    )


@tool
def rank_by_rms(query: str) -> str:
    """
    【RMS排序】检索候选镜头并按近轴RMS从小到大排序，快速找到光学性能最优方案。
    输入: 需求描述字符串，与 lens_search 相同。
    输出: 按近轴RMS升序排列的 top-10 候选镜头列表。
    """
    results = VS.similarity_search(query, k=20)
    candidates = []
    for doc in results:
        m   = doc.metadata
        rms = m.get("calc_rms")
        if rms is not None:
            candidates.append((rms, m))

    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:10]

    if not candidates:
        return "未找到有效候选"

    lines = ["按近轴RMS升序排列（越小越好）："]
    for rank, (rms, m) in enumerate(candidates, 1):
        effl = m.get("calc_effl")
        lines.append(
            f"[{rank}] 镜头#{m.get('lens_idx')} | "
            f"近轴RMS={rms:.4f}mm | "
            f"FOV={m.get('fov')}° F/{m.get('fnum')} | "
            f"EFFL={effl:.2f}mm | "
            f"来源={Path(str(m.get('source',''))).name}"
        )
    return "\n".join(lines)


@tool
def get_lens_surfaces(lens_idx: int) -> str:
    """
    【面型详情】获取指定镜头的完整面型数据（曲率半径、厚度、材料、半径）。
    输入: lens_idx 整数。
    输出: 每个光学面的详细参数表，可用于深入分析镜头结构。
    """
    try:
        lens_idx = int(lens_idx)
    except (ValueError, TypeError):
        return f"lens_idx 必须是整数，收到: {lens_idx!r}"
    if not ALL_LENSES:
        return "镜头数据未加载"
    if lens_idx < 0 or lens_idx >= len(ALL_LENSES):
        return f"lens_idx {lens_idx} 越界"

    lens = ALL_LENSES[lens_idx]
    surfs = lens.get("surfaces", [])
    if isinstance(surfs, str):
        import ast
        surfs = ast.literal_eval(surfs)

    lines = [
        f"镜头#{lens_idx} 面型数据 | FOV={lens.get('fov')}° F/{lens.get('fnum')}",
        f"{'面号':>4} {'曲率半径(mm)':>14} {'厚度(mm)':>10} {'材料':>12} {'半径(mm)':>10}",
        "-" * 56,
    ]
    for s in surfs:
        r   = s.get("radius")
        r_s = f"{r:.4f}" if r is not None else "∞(平面)"
        lines.append(
            f"{int(s.get('surface_num',0)):>4} "
            f"{r_s:>14} "
            f"{s.get('thickness',0):>10.4f} "
            f"{s.get('material','AIR'):>12} "
            f"{s.get('semi_diameter',0):>10.4f}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ReAct Prompt
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """你是一位专业的光学镜头设计助手，帮助工程师从大型镜头数据库中检索和评估镜头方案。

你有以下工具可以使用：
{tools}

工具名列表: {tool_names}

工作流程建议：
1. 先用 lens_search 做语义检索，获取候选镜头编号
2. 用 rank_by_rms 对候选按RMS性能排序，找到最优方案
3. 用 rms_calculator 对最优候选做精确验证
4. 如需查看面型结构，用 get_lens_surfaces

严格按以下格式输出（不得省略任何字段）：
Question: {input}
Thought: （分析用户需求，决定下一步行动）
Action: （工具名，必须是工具名列表中的一个）
Action Input: （工具的输入参数）
Observation: （工具返回结果）
...（可循环多次 Thought/Action/Observation）
Thought: 我已经得到足够信息，可以给出最终答案
Final Answer: （中文回答，包含推荐镜头编号、关键参数、推荐理由）

开始！
Question: {input}
Thought:{agent_scratchpad}"""

PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)

# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════
def build_agent():
    tools    = [lens_search, rank_by_rms, rms_calculator, get_lens_surfaces]
    agent    = create_react_agent(llm, tools, PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,          # 打印完整 Thought/Action/Observation 链
        max_iterations=10,
        handle_parsing_errors=True,
    )
    return executor

def main():
    global VS, ALL_LENSES

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="单次查询（不进入交互模式）")
    args = parser.parse_args()

    VS, ALL_LENSES = load_rag()
    executor = build_agent()

    if args.query:
        # 单次查询模式
        result = executor.invoke({"input": args.query})
        print(f"\n✅ {result['output']}")
        return

    # 交互模式
    print("\n🔭 光学镜头 ReAct Agent 就绪（输入 quit 退出）")
    print("工具: lens_search / rank_by_rms / rms_calculator / get_lens_surfaces")
    print("示例查询:")
    print("  找FOV=35度 F/2.8 近轴RMS最小的镜头")
    print("  有没有用H-ZF4A玻璃的方案")
    print("  比较镜头#8和#18的面型结构\n")

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
