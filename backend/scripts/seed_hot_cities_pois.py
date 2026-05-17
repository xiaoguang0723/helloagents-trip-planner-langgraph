"""预置热门城市景点到 Elasticsearch 向量库。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Tuple

import sys

CURRENT = Path(__file__).resolve()
BACKEND_ROOT = CURRENT.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.mcp_amap import get_amap_langchain_tools, pick_tool  # noqa: E402
from app.services.vector_store_service import (  # noqa: E402
    POIDocument,
    get_vector_store_service,
    parse_pois_from_amap_payload,
)

HOT_CITIES = [
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "成都",
    "西安",
    "重庆",
    "苏州",
    "南京",
]

CITY_KEYWORDS = ["热门景点", "历史文化景点", "地标建筑", "博物馆", "自然风景"]


def _dedup_pois(pois: List[POIDocument]) -> List[POIDocument]:
    seen: set[Tuple[str, str, str]] = set()
    out: List[POIDocument] = []
    for poi in pois:
        key = (
            poi.get("city", ""),
            poi.get("name", ""),
            poi.get("address", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(poi)
    return out


async def _collect_city_pois(city: str) -> List[POIDocument]:
    tools = await get_amap_langchain_tools()
    tool = pick_tool(tools, "maps_text_search", "text_search")
    city_pois: List[POIDocument] = []

    for keyword in CITY_KEYWORDS:
        payload = await tool.ainvoke(
            {
                "keywords": f"{city} {keyword}",
                "city": city,
                "citylimit": "true",
            }
        )
        extracted = parse_pois_from_amap_payload(payload, city=city, source="seed_hot_city")
        city_pois.extend(extracted)

    return _dedup_pois(city_pois)


async def main() -> None:
    service = get_vector_store_service()
    if not service.is_index_ready():
        raise RuntimeError("Elasticsearch 索引初始化失败，请确认 ES 地址与鉴权配置")

    total = 0
    detail: Dict[str, int] = {}
    for city in HOT_CITIES:
        pois = await _collect_city_pois(city)
        count = service.upsert_pois(pois)
        detail[city] = count
        total += count
        print(f"[seed] {city}: 采集 {len(pois)} 条，写入 {count} 条")

    print("[seed] 热门城市景点导入完成")
    print(f"[seed] 总写入条数: {total}")
    for city, count in detail.items():
        print(f"  - {city}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
