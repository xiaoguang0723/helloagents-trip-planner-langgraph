"""高德地图 MCP：通过 langchain-mcp-adapters 加载为 LangChain tools（进程内缓存）"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config import get_settings

_lock = asyncio.Lock()
_tools_cache: Optional[list[BaseTool]] = None


def amap_mcp_connections() -> dict:
    settings = get_settings()
    if not settings.amap_api_key:
        raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")

    if shutil.which("uvx"):
        cmd, args = "uvx", ["amap-mcp-server"]
    else:
        py_root = Path(sys.executable).resolve().parent
        if sys.platform == "win32":
            bundled = py_root / "Scripts" / "amap-mcp-server.exe"
        else:
            bundled = py_root / "bin" / "amap-mcp-server"
        if bundled.is_file():
            cmd, args = str(bundled), []
        else:
            which_amap = shutil.which("amap-mcp-server")
            if which_amap:
                cmd, args = which_amap, []
            else:
                raise ValueError(
                    "无法启动高德 MCP：未找到 uvx 或 amap-mcp-server。"
                    "请安装 uv 或将 amap-mcp-server 加入 PATH：pip install amap-mcp-server"
                )

    return {
        "amap": {
            "command": cmd,
            "args": args,
            "transport": "stdio",
            "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key},
        }
    }


async def get_amap_langchain_tools() -> list[BaseTool]:
    global _tools_cache
    async with _lock:
        if _tools_cache is None:
            client = MultiServerMCPClient(amap_mcp_connections())
            _tools_cache = await client.get_tools()
        return _tools_cache


def clear_amap_tools_cache() -> None:
    global _tools_cache
    _tools_cache = None


def pick_tool(tools: list[BaseTool], *candidates: str) -> BaseTool:
    """按名称或子串匹配 MCP 工具（不同版本命名可能略有差异）"""
    names = [t.name for t in tools]
    for c in candidates:
        for t in tools:
            if t.name == c or c in t.name:
                return t
    raise ValueError(f"未找到工具 {candidates!r}，当前有: {names}")
