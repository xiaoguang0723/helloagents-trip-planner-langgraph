"""高德地图服务：通过 MCP（langchain-mcp-adapters）调用"""

from typing import Any, Dict, List, Optional

from ..models.schemas import Location, POIInfo, WeatherInfo
from .mcp_amap import get_amap_langchain_tools, pick_tool


class AmapService:
    """高德地图服务封装类（异步，供 FastAPI async 路由 await）"""

    async def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        try:
            tools = await get_amap_langchain_tools()
            tool = pick_tool(tools, "maps_text_search", "text_search")
            result = await tool.ainvoke(
                {
                    "keywords": keywords,
                    "city": city,
                    "citylimit": str(citylimit).lower(),
                }
            )
            print(f"POI搜索结果: {str(result)[:200]}...")
            return []
        except Exception as e:
            print(f"❌ POI搜索失败: {str(e)}")
            return []

    async def get_weather(self, city: str) -> List[WeatherInfo]:
        try:
            tools = await get_amap_langchain_tools()
            tool = pick_tool(tools, "maps_weather", "weather")
            result = await tool.ainvoke({"city": city})
            print(f"天气查询结果: {str(result)[:200]}...")
            return []
        except Exception as e:
            print(f"❌ 天气查询失败: {str(e)}")
            return []

    async def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking",
    ) -> Dict[str, Any]:
        try:
            tool_map = {
                "walking": ("maps_direction_walking_by_address", "walking"),
                "driving": ("maps_direction_driving_by_address", "driving"),
                "transit": ("maps_direction_transit_integrated_by_address", "transit"),
            }
            tool_key, _ = tool_map.get(route_type, tool_map["walking"])
            tools = await get_amap_langchain_tools()
            tool = pick_tool(tools, tool_key)

            arguments: Dict[str, Any] = {
                "origin_address": origin_address,
                "destination_address": destination_address,
            }
            if route_type == "transit":
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city
            else:
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city

            result = await tool.ainvoke(arguments)
            print(f"路线规划结果: {str(result)[:200]}...")
            return {}
        except Exception as e:
            print(f"❌ 路线规划失败: {str(e)}")
            return {}

    async def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        try:
            tools = await get_amap_langchain_tools()
            tool = pick_tool(tools, "maps_geo", "geo")
            arguments: Dict[str, Any] = {"address": address}
            if city:
                arguments["city"] = city
            result = await tool.ainvoke(arguments)
            print(f"地理编码结果: {str(result)[:200]}...")
            return None
        except Exception as e:
            print(f"❌ 地理编码失败: {str(e)}")
            return None

    async def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        try:
            tools = await get_amap_langchain_tools()
            tool = pick_tool(tools, "maps_search_detail", "search_detail")
            result = await tool.ainvoke({"id": poi_id})
            print(f"POI详情结果: {str(result)[:200]}...")

            import json
            import re

            json_match = re.search(r"\{.*\}", str(result), re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"raw": result}
        except Exception as e:
            print(f"❌ 获取POI详情失败: {str(e)}")
            return {}


_amap_service: Optional[AmapService] = None


def get_amap_service() -> AmapService:
    global _amap_service
    if _amap_service is None:
        _amap_service = AmapService()
    return _amap_service
