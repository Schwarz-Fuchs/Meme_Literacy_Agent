"""
LangGraph Demo

Pros:
1.能跑，至少在我的机器上接 deepseek API 能跑
Cons:
1.目前只是基础运行框架，用LLM分析用户的语境情绪，提取关键信息或许是可行的，但LLM/VLM能否实现用户需求与返回meme图的对齐不好搞，换言之，LLM/VLM懂用户，但懂meme吗？
（用RAG 基于know your meme 做外挂知识库？ VLM in the loop?）
2.用户意图可能有多义性，可能合适的meme很多，但用户有自己的偏好(不同对话环境可能偏好还不同)，大概需要加上 human in the loop
3.只使用了单一 agent 来直接输出 image_search_queries 模版，分析用户对话的情景情绪，但没有agent协作
（参考deepseek Chain of Thought 功能？）
4.只使用了duckduckgo 单一免费搜索引擎，或许可以增加专门搜表情包的引擎
5.没有数据库交互，全是往内存和json文件里放，为了贴近工业，可以考虑整一个来避免相似输入也要调用LLM API
6.没有设置验证指标. 用LLM/VLM自己评分又会陷入它是否懂meme的悖论，需要人类用户评判。
(参考一下这篇文章？ TransMeme: A Multi-Agent Framework for Cross-Cultural Meme Transcreation)
"""
from __future__ import annotations
import json
import os
import re
import operator
from typing import Any

from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict

load_dotenv()
class MemeIntent(BaseModel):
    """
    (Pydantic 模型)：定义了一个表情包意图的标准结构（包含情境、情绪、立场、视觉隐喻、模板假设、搜索关键词和置信度）
    """
    situation: str = Field(description="事件发生了什么")
    emotion: str = Field(description="主要情绪，如尴尬、幸灾乐祸、无奈")
    stance: str = Field(description="说话者希望呈现的态度或反转")
    visual_metaphors: list[str] = Field(description="可被画面表达的动作/关系", min_length=2, max_length=4)
    template_hypotheses: list[str] = Field(description="经典 meme 模板名或视觉原型", min_length=2, max_length=4)
    search_queries: list[str] = Field(description="英文检索式；每条必须包含 meme", min_length=2, max_length=4)
    confidence: float = Field(ge=0, le=1, description="意图判断的自信度")


class MemeState(TypedDict, total=False):
    """
    (TypedDict)：LangGraph 的全局状态载体, 在各个节点间传递用户输入、对话历史、意图、候选表情包和 trace
    """
    user_text: str  #用户输入
    conversation: list[dict[str, str]] #对话历史(上下文)
    context_user_turns: int  #传给LLM 的上下文轮次限制
    intent: dict[str, Any] #用户意图 由 understand_meme 生成，存的是 MemeIntent 模型的字典化数据
    candidates: list[dict[str, Any]] #原始候选表情包 由 search_memes 生成，存 DDGS 搜回来的原始图片列表（URL、标题、来源等）
    ranked: list[dict[str, Any]] #重排后的表情包 由 rank_candidates 生成，经过打分排好序的表情包列表
    answer: str #最终回复文本 由 reflect 生成，用来给用户的文字反馈或追问。
    trace: Annotated[list[str], operator.add]  #运行日志
    needs_clarification: bool #是否需要追问 由 reflect 判断。如果置信度太低或搜不到图，设为 True，系统会转去向用户发起追问

# 这里可能需要进一步 Prompt Engineering
SYSTEM_PROMPT = """你是 Meme Literacy Agent，不是普通搜索助手。
你的工作是解释用户话语中的“笑点机制”，再把它映射到互联网已存在的 meme 模板。

先识别：事件、关系/权力结构、情绪、说话者姿态（自嘲/吐槽/旁观/庆祝）以及画面中的反转。
再提出 2-4 个经典 meme 模板或稳定视觉原型，并生成适合英文图片索引的检索式。
不要捏造冷门模板；若不确定，使用明确的视觉描述。检索式必须带 meme。

严禁无脑滥用 This Is Fine 或 Cheems，必须根据当前情境的独特性定制模版。
注意：互联网 meme 库极其庞大。请根据用户输入的具体动作（如‘哈气’、‘发疯’、‘物理逃跑’等微观行为）寻找最具画面契合度的梗，避免每次都使用万能通用梗。

只返回一个 JSON object，不要使用 Markdown，不要返回 `analysis`、`meme_templates`、
`image_search_queries` 等嵌套字段。必须使用以下**完全相同的顶层字段**：
{
  "situation": "老板临时要求员工加班",
  "emotion": "无奈而烦躁",
  "stance": "用自嘲表达不敢当面反抗的压抑",
  "visual_metaphors": ["被困在燃烧办公室仍强装镇定", "内心愤怒、现实顺从的反差"],
  "template_hypotheses": ["This Is Fine", "Buff Doge vs Cheems"],
  "search_queries": ["This Is Fine meme overtime office", "Buff Doge vs Cheems meme boss overtime"],
  "confidence": 0.82
}"""

def _fallback_intent(text: str) -> MemeIntent:
    """
    当程序没有配置 OPENAI_API_KEY，或者大模型调用失败时，默认返回该intent
    看到 this is fine 🔥 ，SpongeBob writing essay ，Panik Kalm Panik  可能就是something went wrong 了
    """
    emotion = "awkwardness / helplessness"
    templates = ["This Is Fine", "Disaster Girl", "side eye reaction"]
    if any(word in text.lower() for word in ("deadline", "论文", "作业", "考试", "ddl")):
        emotion, templates = "procrastination panic", ["This Is Fine", "SpongeBob writing essay", "Panik Kalm Panik"]
    elif any(word in text.lower() for word in ("老板", "领导", "群", "收到")):
        emotion, templates = "social awkwardness after accidental visibility", ["Hide the Pain Harold", "awkward monkey puppet", "Disaster Girl"]
    queries = [f"{template} meme" for template in templates]
    return MemeIntent(
        situation=text,
        emotion=emotion,
        stance="self-deprecating reaction to an uncomfortable social moment",
        visual_metaphors=["one person exposed in a silent crowd", "forced smile while everything goes wrong"],
        template_hypotheses=templates,
        search_queries=queries,
        confidence=0.35,
    )


def _strings(value: Any) -> list[str]:
    """
    LLM返回结果清洗，转字符串列表
    """
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _parse_deepseek_intent(content: str, user_text: str) -> MemeIntent:
    """
    大肥鱼可能会自作聪明地把结果嵌套在 analysis 或 meme_templates 字段里
    将非标准的嵌套结构，强行归一化 MemeIntent 结构
    """
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON 根节点必须是 object")

    # Preferred contract: the exact MemeIntent flat schema.
    if {"situation", "emotion", "stance"}.issubset(payload):
        return MemeIntent.model_validate(payload)

    # Compatibility for the nested shape DeepSeek returned in the error report.
    analysis = payload.get("analysis", {})
    analysis = analysis if isinstance(analysis, dict) else {}
    templates = payload.get("meme_templates", [])
    templates = templates if isinstance(templates, list) else []
    template_names = [item["name"] for item in templates if isinstance(item, dict) and isinstance(item.get("name"), str)]
    template_visuals = [item["visual"] for item in templates if isinstance(item, dict) and isinstance(item.get("visual"), str)]
    template_queries = [query for item in templates if isinstance(item, dict) for query in _strings(item.get("search_queries"))]
    queries = _strings(payload.get("image_search_queries")) + template_queries
    queries = list(dict.fromkeys(query for query in queries if "meme" in query.lower()))[:4]
    visuals = _strings(analysis.get("visual_reversal")) + template_visuals
    visuals = list(dict.fromkeys(visuals))[:4]

    normalized = {
        "situation": analysis.get("event") or user_text,
        "emotion": analysis.get("emotion") or "mixed reaction",
        "stance": analysis.get("speaker_stance") or "reaction",
        "visual_metaphors": (visuals + ["reaction to a social situation", "contrast between expectation and reality"])[:4],
        "template_hypotheses": (template_names + ["reaction meme", "This Is Fine"])[:4],
        "search_queries": (queries + ["reaction meme template", "This Is Fine meme"])[:4],
        "confidence": 0.70,
    }
    return MemeIntent.model_validate(normalized)


def _select_context(conversation: list[dict[str, str]], user_turn_limit: int) -> list[dict[str, str]]:
    """控制只保留最近的 N 个用户发言 上传LLM API
    user_turn_limit=0时，只使用当前的对话内容搜索meme
    """
    if user_turn_limit <= 0:
        return []
    user_indices = [index for index, turn in enumerate(conversation) if turn.get("role") == "user"]
    if len(user_indices) <= user_turn_limit:
        return conversation
    return conversation[user_indices[-user_turn_limit]:]


def understand_meme(state: MemeState) -> dict[str, Any]:
    """
    LLM 扮演 SYSTEM_PROMPT 中定义的 Meme Literacy Agent, 分析用户的文本，输出符合 MemeIntent 规范的 JSON (目前已使用Deepseek 测试)
    """
    if not os.getenv("OPENAI_API_KEY"):
        intent = _fallback_intent(state["user_text"])
        return {"intent": intent.model_dump(), "trace": ["理解：未配置 API key，使用可解释的规则回退（非 LLM）。"]}

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    context = _select_context(
        state.get("conversation", []), state.get("context_user_turns", 3)
    )
    for turn in context:
        message_type = HumanMessage if turn.get("role") == "user" else AIMessage
        messages.append(message_type(content=turn["content"]))
    messages.append(HumanMessage(content=state["user_text"]))

    # Do not use `with_structured_output` here. Its Pydantic parser runs before
    # this node can recover from an OpenAI-compatible provider's alternate JSON.

    json_llm = llm.bind(response_format={"type": "json_object"})

    try:
        response = json_llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else ""
        intent = _parse_deepseek_intent(content, state["user_text"])

    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        intent = _fallback_intent(state["user_text"])
        return {
            "intent": intent.model_dump(),
            "trace": [f"理解：LLM JSON 未通过协议校验，已安全回退（{type(exc).__name__}）。"],
        }
    return {"intent": intent.model_dump(), "trace": ["理解：LLM JSON 已归一化并通过 MemeIntent 校验。"]}


def search_memes(state: MemeState) -> dict[str, Any]:
    """
    从上一步生成的意图里提取 search_queries，利用 ddgs（DuckDuckGo 搜索）去网络上抓取相关的图片 URL、标题和来源
    这里用 ddgs 纯是懒和 demo 测试用，实际的工作流可以尝试多引擎来源搜索
    """
    candidates: list[dict[str, Any]] = []
    failures: list[str] = []
    for query in state["intent"]["search_queries"]:
        try:
            with DDGS() as ddgs:
                for result in ddgs.images(query, max_results=4, safesearch="moderate"):
                    candidates.append({
                        "url": result.get("image", ""),
                        "thumbnail": result.get("thumbnail", ""),
                        "title": result.get("title", ""),
                        "source": result.get("source", ""),
                        "query": query,
                    })
        except Exception as exc:  # search providers can rate-limit a class demo
            failures.append(f"{query}（{type(exc).__name__}）")
    note = f"检索：用 {len(state['intent']['search_queries'])} 条 meme 查询收集到 {len(candidates)} 个候选。"
    if failures:
        note += f" {len(failures)} 条查询暂不可用，已继续尝试其余查询。"
    return {"candidates": candidates, "trace": [note]}


def _tokens(value: str) -> set[str]:
    """
    正则，拿出所有字母数字，用于重合度计算
    """
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def rank_candidates(state: MemeState) -> dict[str, Any]:
    """
    匹配度打分器，对比图片的标题、查询词与意图中的关键词（如情绪、视觉隐喻、模板名）重合度，给候选表情包打分并降序排列。
    demo 只靠文本（标题、查询词）做关键词重合度匹配 (overlap/wanted)，可以考虑用 CLIP方法做图像语义对齐，或者接入另一个多模态API来评价找到图片和intent 是否对齐
    """
    intent = state["intent"]
    wanted = _tokens(" ".join(intent["template_hypotheses"] + intent["visual_metaphors"] + intent["emotion"].split()))
    unique: dict[str, dict[str, Any]] = {}
    for item in state.get("candidates", []):
        if not item["url"] or item["url"] in unique:
            continue
        observed = _tokens(f"{item['title']} {item['query']}")
        overlap = len(wanted & observed)
        item["score"] = round(overlap / max(len(wanted), 1) + (0.10 if "meme" in item["query"].lower() else 0), 3)
        item["why"] = f"标题/查询与意图词重合 {overlap} 个；来自查询：{item['query']}"
        unique[item["url"]] = item
    ranked = sorted(unique.values(), key=lambda item: item["score"], reverse=True)[:6]
    return {"ranked": ranked, "trace": ["重排：当前为透明的元数据基线；可在此节点接入 SigLIP 做真正图文对齐。"]}


def reflect(state: MemeState) -> dict[str, Any]:
    """
    检查候选结果是否足够。如果候选太少或置信度太低，决定触发追问（needs_clarification=True）；否则准备输出推荐结果
    """
    intent = state["intent"]
    needs_clarification = len(state.get("ranked", [])) < 2 or intent["confidence"] < 0.25
    if needs_clarification:
        answer = "我还缺少一个决定笑点的线索：你想表达自嘲、吐槽别人，还是单纯尴尬？"
    else:
        answer = (
            f"我把这句话理解为「{intent['emotion']}」：{intent['stance']}。"
            f"优先尝试 { '、'.join(intent['template_hypotheses'][:3]) }；下面是按匹配度排过序的候选。"
        )
    return {"answer": answer, "needs_clarification": needs_clarification, "trace": ["反思：检查候选数量与语义置信度，决定展示或追问。"]}

def build_graph():
    """
    Graph 逻辑 understand->search->rank->reflect
    """
    graph = StateGraph(MemeState)

    graph.add_node("understand", understand_meme)
    graph.add_node("search", search_memes)
    graph.add_node("rank", rank_candidates)
    graph.add_node("reflect", reflect)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "search")
    graph.add_edge("search", "rank")
    graph.add_edge("rank", "reflect")
    graph.add_edge("reflect", END)
    return graph.compile()

def run(
    text: str,
    conversation: list[dict[str, str]] | None = None,
    context_user_turns: int = 3,
) -> MemeState:
    return build_graph().invoke({
        "user_text": text,
        "conversation": conversation or [],
        "context_user_turns": context_user_turns,
        "trace": [],
    })

if __name__ == "__main__":
    # JSON format test
    result = run("今天又要加班，我想对领导哈气")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "ranked"},
            ensure_ascii=False,
            indent=2,
        )
    )
