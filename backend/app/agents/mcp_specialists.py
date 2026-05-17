"""LangGraph 节点：直连高德 MCP 工具（与原先「先采集再汇总」一致，不依赖 LLM tool-calling）。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from ..models.schemas import TripRequest
from ..services.mcp_amap import pick_tool


def _as_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


async def mcp_attraction_context(tools: list[BaseTool], request: TripRequest) -> str:
    kw = request.preferences[0] if request.preferences else "景点"
    try:
        tool = pick_tool(tools, "maps_text_search", "text_search")
        raw = await tool.ainvoke(
            {
                "keywords": f"{kw} 景点",
                "city": request.city,
                "citylimit": "true",
            }
        )
        return f"[maps_text_search 景点]\n{_as_str(raw)}"
    except Exception as e:
        return f"[maps_text_search 景点 失败] {e!s}"


async def mcp_weather_context(tools: list[BaseTool], request: TripRequest) -> str:
    try:
        tool = pick_tool(tools, "maps_weather", "weather")
        raw = await tool.ainvoke({"city": request.city})
        return f"[maps_weather]\n{_as_str(raw)}"
    except Exception as e:
        return f"[maps_weather 失败] {e!s}"


async def mcp_hotel_context(tools: list[BaseTool], request: TripRequest) -> str:
    try:
        tool = pick_tool(tools, "maps_text_search", "text_search")
        raw = await tool.ainvoke(
            {
                "keywords": f"{request.accommodation} 酒店",
                "city": request.city,
                "citylimit": "true",
            }
        )
        return f"[maps_text_search 住宿]\n{_as_str(raw)}"
    except Exception as e:
        return f"[maps_text_search 住宿 失败] {e!s}"
