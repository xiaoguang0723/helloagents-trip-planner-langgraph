"""三个子智能体：LLM + 文本 TOOL_CALL 协议 + MCP（与 HelloAgents SimpleAgent 行为同类）。"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from ..models.schemas import TripRequest
from ..services.mcp_amap import pick_tool


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts).strip()
    return str(content).strip()


TOOL_CALL_BRACKET_RE = re.compile(r"\[TOOL_CALL:\s*([^:\]]+)\s*:\s*([^\]]+)\]", re.IGNORECASE)
TOOL_CALL_PLAIN_RE = re.compile(
    r"^TOOL_CALL:\s*([^:]+):\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _fix_common_tool_call_typos(text: str) -> str:
    """纠正常见 TOOL_CALL 笔误。"""
    text = re.sub(
        r"(\[TOOL_CALL:\s*maps_text_search)\s*keywords\s*=",
        r"\1:keywords=",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\[TOOL_CALL:\s*maps_text_search)\s*&\s*keywords\s*=",
        r"\1:keywords=",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    text = _fix_common_tool_call_typos(text)
    m = TOOL_CALL_BRACKET_RE.search(text)
    if not m:
        m = TOOL_CALL_PLAIN_RE.search(text.strip())
    if not m:
        return None
    raw_name, raw_args = m.group(1).strip(), m.group(2).strip()
    args: Dict[str, str] = {}
    for part in re.split(r"[,，&]", raw_args):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        args[k.strip()] = v.strip()
    return raw_name, args


def _resolve_tool(tools: list[BaseTool], requested: str) -> Optional[BaseTool]:
    req = requested.strip().lower().removeprefix("amap_")
    for t in tools:
        if t.name.lower() == req:
            return t
    for t in tools:
        tl, rl = t.name.lower(), req.lower()
        if rl in tl or tl in rl:
            return t
    try:
        return pick_tool(tools, requested, requested.replace("amap_", ""))
    except ValueError:
        return None


def _tool_subset_attraction(all_tools: list[BaseTool]) -> list[BaseTool]:
    return [pick_tool(all_tools, "maps_text_search", "text_search")]


def _tool_subset_weather(all_tools: list[BaseTool]) -> list[BaseTool]:
    return [pick_tool(all_tools, "maps_weather", "weather")]


def _tool_subset_hotel(all_tools: list[BaseTool]) -> list[BaseTool]:
    return [pick_tool(all_tools, "maps_text_search", "text_search")]


ATTRACTION_SYSTEM = """你是景点搜索专家。

**你必须使用工具查询真实景点，禁止编造。**

**工具调用格式（整行输出，勿改格式）：**
`[TOOL_CALL:工具名:keywords=检索词,city=城市名,citylimit=true]`

说明：
- `工具名` 必须使用下方列出的确切名称之一。
- keywords 结合用户偏好、美食/文化等需求自拟（可含多个简短词）。
- city 与用户目的地一致。
- citylimit 固定为 true。

收到工具结果后，若仍不足可再次输出同一格式的 TOOL_CALL；信息已足够时，用中文简要列出景点名称与要点（须基于工具返回）。"""

WEATHER_SYSTEM = """你是天气查询专家。

**你必须使用工具查询真实天气，禁止编造。**

**工具调用格式：**
`[TOOL_CALL:工具名:city=城市名]`

`工具名` 必须使用下方列出的确切名称之一。

收到工具返回后，用中文简要总结天气与气温要点（须基于工具返回）。"""

HOTEL_SYSTEM = """你是酒店推荐专家。

**你必须使用工具搜索真实酒店，禁止编造。**

**工具调用格式（keywords 与 city 缺一不可）：**
`[TOOL_CALL:工具名:keywords=住宿类型或酒店,city=城市名,citylimit=true]`

- **city=** 必须单独写出目的地城市（与用户行程城市一致），禁止把城市名只写在 keywords 里而不写 city。
- keywords 写住宿类型即可（如：经济型酒店、商务酒店）。
- `工具名` 必须使用下方列出的确切名称之一。

收到工具返回后，用中文简要归纳酒店名称与区域（须基于工具返回）。"""


def _system_with_tool_names(base: str, tools: list[BaseTool]) -> str:
    names = "、".join(t.name for t in tools)
    return f"{base}\n\n**当前可用工具名称：** {names}"


def _build_attraction_user_message(req: TripRequest) -> str:
    prefs = "、".join(req.preferences) if req.preferences else "无"
    extra = req.free_text_input.strip() if req.free_text_input else "无"
    return (
        f"请为用户搜索「{req.city}」的合适景点。\n"
        f"- 偏好标签：{prefs}\n"
        f"- 额外说明：{extra}\n"
        f"请先输出 TOOL_CALL 调用搜索工具。"
    )


def _build_weather_user_message(req: TripRequest) -> str:
    return f"请查询「{req.city}」的天气。请先输出 TOOL_CALL。"


def _build_hotel_user_message(req: TripRequest) -> str:
    extra = req.free_text_input.strip() if req.free_text_input else "无"
    return (
        f"请在「{req.city}」搜索符合需求的酒店。\n"
        f"- 住宿偏好：{req.accommodation}\n"
        f"- 其他说明：{extra}\n"
        f"请先输出 TOOL_CALL。"
    )


MAX_TEXT_PROTOCOL_TURNS = 6


async def run_text_protocol_specialist(
    model: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    user_text: str,
    fallback: Callable[[], Awaitable[str]],
    label: str,
) -> str:
    if not tools:
        return await fallback()

    system = _system_with_tool_names(system_prompt, tools)
    messages: List[Any] = [
        SystemMessage(content=system),
        HumanMessage(content=user_text),
    ]
    transcript: List[str] = []

    try:
        for turn in range(MAX_TEXT_PROTOCOL_TURNS):
            ai = await model.ainvoke(messages)
            content = _text_from_content(ai.content).strip().strip("`").strip()
            transcript.append(f"[子智能体 第{turn + 1}轮]\n{content}")

            parsed = _parse_tool_call(content)
            if not parsed:
                if turn == 0:
                    messages.append(AIMessage(content=content))
                    messages.append(
                        HumanMessage(
                            content=(
                                "你尚未输出工具调用。请**单独一行**输出符合系统说明的 "
                                "`[TOOL_CALL:工具名:参数...]`，不要省略。"
                            )
                        )
                    )
                    continue
                break

            raw_name, args = parsed
            tool = _resolve_tool(tools, raw_name)
            if tool is None:
                messages.append(AIMessage(content=content))
                messages.append(
                    HumanMessage(
                        content=(
                            f"系统：未找到工具「{raw_name}」。"
                            f"请仅使用以下名称之一重试 TOOL_CALL：{', '.join(t.name for t in tools)}"
                        )
                    )
                )
                continue

            if "citylimit" in args:
                v = args["citylimit"].lower()
                if v in ("true", "1", "yes"):
                    args["citylimit"] = "true"
                else:
                    args["citylimit"] = "false"

            try:
                raw_out = await tool.ainvoke(args)
            except Exception as e:
                raw_out = f"工具执行失败: {e!s}"

            obs = f"[{tool.name} 返回]\n{_text_from_content(str(raw_out))}"
            transcript.append(obs)

            messages.append(AIMessage(content=content))
            messages.append(
                HumanMessage(
                    content=(
                        f"{obs}\n\n"
                        f"请根据上述结果：若需补充检索可再输出一条 TOOL_CALL；"
                        f"否则直接给出中文简要总结（勿再调用工具）。"
                    )
                )
            )

        body = "\n\n".join(transcript)
        if len(body.strip()) < 40:
            print(f"   ⚠️ {label} 子智能体轨迹过短，改用直连 MCP 兜底")
            return await fallback()
        return f"[{label} 子智能体]\n{body}"
    except Exception as e:
        print(f"   ⚠️ {label} 子智能体异常: {e!s}，改用直连 MCP 兜底")
        return await fallback()


async def attraction_agent_context(
    model: BaseChatModel,
    all_tools: list[BaseTool],
    req: TripRequest,
    fallback: Callable[[], Awaitable[str]],
) -> str:
    tools = _tool_subset_attraction(all_tools)
    return await run_text_protocol_specialist(
        model,
        tools,
        ATTRACTION_SYSTEM,
        _build_attraction_user_message(req),
        fallback,
        "景点",
    )


async def weather_agent_context(
    model: BaseChatModel,
    all_tools: list[BaseTool],
    req: TripRequest,
    fallback: Callable[[], Awaitable[str]],
) -> str:
    tools = _tool_subset_weather(all_tools)
    return await run_text_protocol_specialist(
        model,
        tools,
        WEATHER_SYSTEM,
        _build_weather_user_message(req),
        fallback,
        "天气",
    )


async def hotel_agent_context(
    model: BaseChatModel,
    all_tools: list[BaseTool],
    req: TripRequest,
    fallback: Callable[[], Awaitable[str]],
) -> str:
    tools = _tool_subset_hotel(all_tools)
    return await run_text_protocol_specialist(
        model,
        tools,
        HOTEL_SYSTEM,
        _build_hotel_user_message(req),
        fallback,
        "酒店",
    )
