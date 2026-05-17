"""旅行规划：LangGraph 多智能体顺序执行（景点/天气/酒店 子 Agent + ReAct 工具调用 → 规划 LLM）。"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, TypedDict

try:
    import json5  # type: ignore
except ImportError:
    json5 = None  # type: ignore

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ..config import settings
from .mcp_specialists import mcp_attraction_context, mcp_hotel_context, mcp_weather_context
from .specialist_llm_agents import (
    attraction_agent_context,
    hotel_agent_context,
    weather_agent_context,
)
from ..models.schemas import (
    Attraction,
    DayPlan,
    Hotel,
    Location,
    Meal,
    TripPlan,
    TripRequest,
    WeatherInfo,
)
from ..services.llm_service import get_chat_model
from ..services.mcp_amap import get_amap_langchain_tools
from ..services.vector_store_service import (
    get_vector_store_service,
    parse_pois_from_amap_payload,
)


def _loads_llm_json(raw: str) -> Any:
    """
    解析 LLM 输出的 JSON：先做常见修复，再 json.loads；失败时尝试 json5（更宽松）。
    """
    s = raw.strip()
    if s.startswith("\ufeff"):
        s = s[1:]

    for old, new in (
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2018", "'"),
        ("\u2019", "'"),
    ):
        s = s.replace(old, new)

    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bNaN\b", "null", s, flags=re.IGNORECASE)
    s = re.sub(r"\bInfinity\b", "null", s, flags=re.IGNORECASE)

    s = re.sub(r"/\*[\s\S]*?\*/", "", s)

    prev: Optional[str] = None
    while prev != s:
        prev = s
        s = re.sub(r",\s*([}\]])", r"\1", s)

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        if json5 is not None:
            try:
                return json5.loads(s)
            except Exception as e2:
                raise ValueError(f"JSON解析失败(stdlib): {e}; json5: {e2}") from e2
        raise


_DAY_PLAN_ALLOWED = {
    "date",
    "day_index",
    "description",
    "transportation",
    "accommodation",
    "hotel",
    "attractions",
    "meals",
}

_MEAL_TYPE_CN = {
    "早餐": "breakfast",
    "午饭": "lunch",
    "午餐": "lunch",
    "晚餐": "dinner",
    "晚饭": "dinner",
    "加餐": "snack",
}


def _coerce_visit_duration_minutes(val: Any) -> int:
    """把「2小时」「1.5小时」「90分钟」等转为整数分钟。"""
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, float):
        return int(val)
    if not isinstance(val, str):
        return 120
    s = val.strip().replace(" ", "")
    if not s:
        return 120
    try:
        return int(float(s))
    except ValueError:
        pass
    import re

    if "分钟" in s:
        m = re.search(r"([\d.]+)\s*分钟", s)
        if m:
            return int(float(m.group(1)))
    if "小时" in s:
        m = re.search(r"([\d.]+)\s*小时", s)
        if m:
            return int(float(m.group(1)) * 60)
    m = re.search(r"([\d.]+)\s*h", s, re.I)
    if m:
        return int(float(m.group(1)) * 60)
    return 120


def _normalize_llm_trip_dict(data: dict) -> None:
    """去掉模型多写的字段、补 day_index、餐饮 type 英文化，降低 Pydantic 校验失败。"""
    days = data.get("days")
    if not isinstance(days, list):
        return
    for i, day in enumerate(days):
        if not isinstance(day, dict):
            continue
        for k in list(day.keys()):
            if k not in _DAY_PLAN_ALLOWED:
                day.pop(k, None)
        day.setdefault("day_index", i)
        meals = day.get("meals")
        if isinstance(meals, list):
            for m in meals:
                if not isinstance(m, dict):
                    continue
                mt = m.get("type")
                if isinstance(mt, str) and mt in _MEAL_TYPE_CN:
                    m["type"] = _MEAL_TYPE_CN[mt]
        atts = day.get("attractions")
        if isinstance(atts, list):
            for a in atts:
                if isinstance(a, dict) and "visit_duration" in a:
                    a["visit_duration"] = _coerce_visit_duration_minutes(a["visit_duration"])


def _message_content_to_text(content: Any) -> str:
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


PLANNER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息,生成详细的旅行计划。

请严格按照以下JSON格式返回旅行计划:
```json
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397128, "latitude": 39.916527},
        "price_range": "300-500元",
        "rating": "4.5",
        "distance": "距离景点2公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "详细地址",
          "location": {"longitude": 116.397128, "latitude": 39.916527},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50},
        {"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "总体建议",
  "budget": {
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }
}
```

**重要提示:**
1. weather_info数组必须包含每一天的天气信息
2. 温度必须是纯数字(不要带°C等单位)
3. 每天安排2-3个景点
4. 考虑景点之间的距离和游览时间
5. 每天必须包含早中晚三餐
6. 提供实用的旅行建议
7. **必须包含预算信息**:
   - 景点门票价格(ticket_price)
   - 餐饮预估费用(estimated_cost)
   - 酒店预估费用(estimated_cost)
   - 预算汇总(budget)包含各项总费用

**JSON输出硬性要求(违反将导致系统无法解析):**
1. 只输出合法JSON：键与字符串必须用英文双引号，禁止单引号键
2. 禁止使用 Placeholder、占位、TBD、未完句、单独一行的 `...`；所有字符串必须写完整并正确闭合
3. 不要使用 // 或 /* */ 注释；不要尾随逗号(虽然系统会尝试修复)
4. 不要使用 Python 的 True/False/None，必须使用 true/false/null
5. days[].accommodation 必须是**字符串**(如与用户请求一致的住宿偏好)；酒店详情必须放在 days[].hotel 对象中，不要把酒店对象赋给 accommodation
"""


_SKIP_POI_NAMES = frozenset(
    {
        "keywords",
        "cities",
        "suggestion",
        "pois",
        "forecasts",
        "text",
        "type",
    }
)


def _unique_poi_names_from_mcp_text(text: str, limit: int = 40) -> List[str]:
    """从高德 MCP 返回的文本/JSON 串中提取 POI name（去重保序）。"""
    if not text:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for m in re.finditer(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        n = m.group(1)
        if len(n) < 2 or len(n) > 120:
            continue
        ln = n.lower()
        if ln in _SKIP_POI_NAMES:
            continue
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= limit:
            break
    return out


def _build_attraction_vector_query(request: TripRequest) -> str:
    pref = "、".join(request.preferences) if request.preferences else "热门景点"
    extra = (request.free_text_input or "").strip()
    return f"{request.city} {pref} {extra}".strip()


def _format_vector_attraction_context(request: TripRequest, hits: List[Dict[str, Any]]) -> str:
    lines = [f"[vector_search 景点] city={request.city} hits={len(hits)}"]
    for idx, hit in enumerate(hits[:20], start=1):
        score = float(hit.get("score") or 0.0)
        name = str(hit.get("name") or "")
        addr = str(hit.get("address") or request.city)
        category = str(hit.get("category") or "景点")
        lines.append(f'{idx}. {name} | 地址: {addr} | 类别: {category} | score={score:.4f}')
    return "\n".join(lines)


def _pick_hotel_from_mcp_text(hotel_ctx: str, city: str) -> Optional[Hotel]:
    names = _unique_poi_names_from_mcp_text(hotel_ctx, 25)
    for n in names:
        if any(k in n for k in ("酒店", "宾馆", "民宿", "客栈", "公寓", "旅馆")):
            return Hotel(
                name=n,
                address=f"{city}（地图检索）",
                type="酒店",
            )
    if names:
        return Hotel(name=names[0], address=f"{city}（地图检索）", type="酒店")
    return None


def _weather_rows_from_mcp_text(weather_ctx: str, max_rows: int = 14) -> List[WeatherInfo]:
    """从天气 MCP 文本中尽量抽出 forecast 行（字段不齐则留空）。"""
    if not weather_ctx:
        return []
    dates = re.findall(r'"date"\s*:\s*"(\d{4}-\d{2}-\d{2})"', weather_ctx)
    if not dates:
        return []
    day_w = re.findall(r'"dayweather"\s*:\s*"([^"]*)"', weather_ctx)
    night_w = re.findall(r'"nightweather"\s*:\s*"([^"]*)"', weather_ctx)
    day_t = re.findall(r'"daytemp"\s*:\s*"?([^",}\s]+)', weather_ctx)
    night_t = re.findall(r'"nighttemp"\s*:\s*"?([^",}\s]+)', weather_ctx)
    rows: List[WeatherInfo] = []
    n = min(len(dates), max_rows)
    for i in range(n):
        d = dates[i]
        dw = day_w[i] if i < len(day_w) else ""
        nw = night_w[i] if i < len(night_w) else ""
        dt: Any = day_t[i] if i < len(day_t) else 0
        nt: Any = night_t[i] if i < len(night_t) else 0
        try:
            dt = int(float(str(dt).replace("℃", "").strip()))
        except ValueError:
            dt = 0
        try:
            nt = int(float(str(nt).replace("℃", "").strip()))
        except ValueError:
            nt = 0
        rows.append(
            WeatherInfo(
                date=d,
                day_weather=dw,
                night_weather=nw,
                day_temp=dt,
                night_temp=nt,
            )
        )
    return rows


class TripState(TypedDict, total=False):
    request: Dict[str, Any]
    attraction_context: str
    weather_context: str
    hotel_context: str
    planner_raw: str


class LangGraphTripPlanner:
    """LangGraph：三子智能体（LLM+ReAct+MCP）顺序采集 + 规划 LLM 汇总。"""

    def __init__(self) -> None:
        print("🔄 开始初始化 LangGraph 旅行规划系统...")
        _ = get_chat_model()
        print("✅ LangGraph 旅行规划器就绪（首次规划时加载 MCP 工具）")

    def plan_trip(self, request: TripRequest) -> TripPlan:
        return asyncio.run(self._plan_trip_async(request))

    async def _plan_trip_async(self, request: TripRequest) -> TripPlan:
        print(f"\n{'='*60}")
        print(f"🚀 开始 LangGraph 协作规划旅行...")
        print(f"目的地: {request.city}")
        print(f"日期: {request.start_date} 至 {request.end_date}")
        print(f"天数: {request.travel_days}天")
        print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
        print(f"{'='*60}\n")

        tools = await get_amap_langchain_tools()
        model = get_chat_model()

        async def node_attraction(state: TripState) -> dict:
            req = TripRequest(**state["request"])
            print("📍 步骤1: 景点子智能体（LLM 决策 + MCP 工具）...")

            if settings.vector_search_enabled and settings.elasticsearch_url:
                try:
                    vector_service = get_vector_store_service()
                    is_ready = await asyncio.to_thread(vector_service.is_index_ready)
                    if is_ready:
                        query = _build_attraction_vector_query(req)
                        hits = await asyncio.to_thread(
                            vector_service.search_similar_pois,
                            req.city,
                            query,
                            settings.elasticsearch_knn_top_k,
                        )
                        top_score = float(hits[0].get("score", 0.0)) if hits else 0.0
                        enough = (
                            len(hits) >= settings.vector_recall_min_items
                            and top_score >= settings.vector_recall_score_threshold
                        )
                        if enough:
                            print(
                                f"[vector_hit] city={req.city} hits={len(hits)} top_score={top_score:.4f}"
                            )
                            return {"attraction_context": _format_vector_attraction_context(req, hits)}
                        print(
                            f"[vector_miss] city={req.city} hits={len(hits)} top_score={top_score:.4f}"
                        )
                    else:
                        print(f"[vector_miss] city={req.city} reason=index_not_ready")
                except Exception as e:
                    print(f"[vector_miss] city={req.city} reason={e!s}")

            async def _fb_attr():
                return await mcp_attraction_context(tools, req)

            print(f"[api_fallback] city={req.city}")
            ctx = await attraction_agent_context(model, tools, req, _fb_attr)
            if settings.vector_search_enabled and settings.elasticsearch_url:
                try:
                    vector_service = get_vector_store_service()
                    parsed = parse_pois_from_amap_payload(ctx, city=req.city, source="trip_runtime")
                    upserted = await asyncio.to_thread(vector_service.upsert_pois, parsed)
                    print(f"[es_upsert_count] city={req.city} count={upserted}")
                except Exception as e:
                    print(f"[es_upsert_count] city={req.city} count=0 reason={e!s}")
            preview = ctx[:200] + "..." if len(ctx) > 200 else ctx
            print(f"景点侧上下文: {preview}\n")
            return {"attraction_context": ctx}

        async def node_weather(state: TripState) -> dict:
            req = TripRequest(**state["request"])
            print("🌤️  步骤2: 天气子智能体（LLM 决策 + MCP 工具）...")

            async def _fb_w():
                return await mcp_weather_context(tools, req)

            ctx = await weather_agent_context(model, tools, req, _fb_w)
            preview = ctx[:200] + "..." if len(ctx) > 200 else ctx
            print(f"天气侧上下文: {preview}\n")
            return {"weather_context": ctx}

        async def node_hotel(state: TripState) -> dict:
            req = TripRequest(**state["request"])
            print("🏨 步骤3: 酒店子智能体（LLM 决策 + MCP 工具）...")

            async def _fb_h():
                return await mcp_hotel_context(tools, req)

            ctx = await hotel_agent_context(model, tools, req, _fb_h)
            preview = ctx[:200] + "..." if len(ctx) > 200 else ctx
            print(f"酒店侧上下文: {preview}\n")
            return {"hotel_context": ctx}

        async def node_planner(state: TripState) -> dict:
            req = TripRequest(**state["request"])
            print("📋 步骤4: 生成行程计划...")
            query = self._build_planner_query(
                req,
                state.get("attraction_context", ""),
                state.get("weather_context", ""),
                state.get("hotel_context", ""),
            )
            msg = await model.ainvoke(
                [
                    SystemMessage(content=PLANNER_AGENT_PROMPT),
                    HumanMessage(content=query),
                ]
            )
            raw = _message_content_to_text(msg.content)
            preview = raw[:300] + "..." if len(raw) > 300 else raw
            print(f"行程规划结果: {preview}\n")
            return {"planner_raw": raw}

        g = StateGraph(TripState)
        g.add_node("attraction", node_attraction)
        g.add_node("weather", node_weather)
        g.add_node("hotel", node_hotel)
        g.add_node("planner", node_planner)
        g.add_edge(START, "attraction")
        g.add_edge("attraction", "weather")
        g.add_edge("weather", "hotel")
        g.add_edge("hotel", "planner")
        g.add_edge("planner", END)
        app = g.compile()

        out = await app.ainvoke({"request": request.model_dump()})
        raw = out.get("planner_raw", "")
        trip_plan = self._parse_response(raw, request, graph_state=out)

        if self._looks_like_fallback_plan(trip_plan, request):
            print("🔁 规划 JSON 未通过校验或解析失败，尝试让模型修正输出（最多 1 次）...\n")
            fix_query = (
                "你上一次输出的旅行计划 JSON 无法被严格解析，或 days[].accommodation 误写成了对象。\n"
                "请**只输出一个**完整、可 json.loads 的 JSON 对象，且 days[].accommodation 必须是字符串，酒店放在 days[].hotel。\n"
                "不要 markdown 代码块，不要注释，不要尾随逗号。\n\n"
                "待修正的原文（截断）：\n"
                + raw[:6000]
            )
            msg2 = await model.ainvoke(
                [
                    SystemMessage(content=PLANNER_AGENT_PROMPT),
                    HumanMessage(content=fix_query),
                ]
            )
            raw2 = _message_content_to_text(msg2.content)
            trip_plan = self._parse_response(raw2, request, graph_state=out)

        if self._looks_like_fallback_plan(trip_plan, request):
            mc = self._create_mcp_grounded_plan(
                request,
                out.get("attraction_context", ""),
                out.get("weather_context", ""),
                out.get("hotel_context", ""),
            )
            if mc is not None:
                print("📌 规划 JSON 仍不可用，已改用 **MCP 检索上下文** 拼装降级行程（非占位假数据）\n")
                trip_plan = mc

        print(f"{'='*60}")
        print(f"✅ 旅行计划生成完成!")
        print(f"{'='*60}\n")

        return trip_plan

    def _looks_like_fallback_plan(self, plan: TripPlan, request: TripRequest) -> bool:
        """占位行程：解析失败后的通用假景点名（如「金华景点1」）。"""
        if not plan.days:
            return True
        a0 = plan.days[0].attractions[0].name if plan.days[0].attractions else ""
        return a0 == f"{request.city}景点1"

    def _create_mcp_grounded_plan(
        self,
        request: TripRequest,
        attraction_ctx: str,
        weather_ctx: str,
        hotel_ctx: str,
    ) -> Optional[TripPlan]:
        """
        规划 JSON 失败时：从子智能体已拿到的 MCP 文本里抽 POI/天气/酒店，拼装合法 TripPlan。
        若上下文里几乎抽不到名称，返回 None（继续保留占位行程）。
        """
        from datetime import datetime, timedelta

        attr_names = _unique_poi_names_from_mcp_text(attraction_ctx, 48)
        hotel = _pick_hotel_from_mcp_text(hotel_ctx, request.city)
        if not attr_names and hotel is None:
            merged = _unique_poi_names_from_mcp_text(
                attraction_ctx + "\n" + hotel_ctx + "\n" + weather_ctx, 48
            )
            attr_names = [n for n in merged if not any(k in n for k in ("酒店", "宾馆"))][:20] or merged[:20]
        if not attr_names and hotel is None:
            return None

        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        pool = list(attr_names)
        if not pool:
            pool = [f"{request.city}热门地点（详见 MCP 酒店/检索摘要）"]

        idx = 0
        days_out: List[DayPlan] = []
        for i in range(request.travel_days):
            cur = start_date + timedelta(days=i)
            day_names: List[str] = []
            for _ in range(2):
                if idx < len(pool):
                    day_names.append(pool[idx])
                    idx += 1
            if not day_names:
                day_names = [pool[0]]
            attractions = [
                Attraction(
                    name=nm,
                    address=request.city,
                    location=Location(
                        longitude=116.4 + i * 0.01 + j * 0.005,
                        latitude=39.9 + i * 0.01 + j * 0.005,
                    ),
                    visit_duration=120,
                    description="名称来自高德地图 MCP 检索结果（降级拼装）",
                    category="景点",
                )
                for j, nm in enumerate(day_names)
            ]
            days_out.append(
                DayPlan(
                    date=cur.strftime("%Y-%m-%d"),
                    day_index=i,
                    description=f"第{i + 1}天：基于地图检索的参考行程（规划 JSON 未生成）",
                    transportation=request.transportation,
                    accommodation=request.accommodation,
                    hotel=hotel,
                    attractions=attractions,
                    meals=[
                        Meal(type="breakfast", name=f"第{i + 1}天早餐", description="当地早餐"),
                        Meal(type="lunch", name=f"第{i + 1}天午餐", description="当地午餐"),
                        Meal(type="dinner", name=f"第{i + 1}天晚餐", description="当地晚餐"),
                    ],
                )
            )

        weather_rows = _weather_rows_from_mcp_text(weather_ctx, max_rows=request.travel_days + 7)
        if not weather_rows:
            weather_rows = []

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days_out,
            weather_info=weather_rows,
            overall_suggestions=(
                "规划模型输出的 JSON 未能通过校验；本行程根据**已执行的高德 MCP 检索结果**自动拼装，"
                "景点与酒店名称来自地图数据摘要，坐标为省内参考占位，出行前请再次核实。"
            ),
        )

    def _build_planner_query(
        self,
        request: TripRequest,
        attractions: str,
        weather: str,
        hotels: str = "",
    ) -> str:
        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

**景点子智能体检索摘要与工具结果:**
{attractions}

**天气子智能体检索摘要与工具结果:**
{weather}

**酒店子智能体检索摘要与工具结果:**
{hotels}

**要求:**
1. 每天安排2-3个景点
2. 每天必须包含早中晚三餐
3. 每天推荐一个具体的酒店(从酒店信息中选择)
3. 考虑景点之间的距离和交通方式
4. 返回完整的JSON格式数据
5. 景点的经纬度坐标要真实准确
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"

        return query

    def _parse_response(
        self,
        response: str,
        request: TripRequest,
        graph_state: Optional[Dict[str, Any]] = None,
    ) -> TripPlan:
        last_json_sample: Optional[str] = None
        try:
            if isinstance(response, dict):
                return TripPlan(**response)
            if not isinstance(response, str):
                response = str(response)

            response = re.sub(
                r"<redacted_thinking>[\s\S]*?</redacted_thinking>", "", response, flags=re.IGNORECASE
            )
            response = re.sub(
                r"</redacted_thinking>[\s\S]*?</redacted_thinking>", "", response, flags=re.IGNORECASE
            )

            json_str: str | None = None
            fenced_json_blocks = re.findall(
                r"```json\s*([\s\S]*?)\s*```",
                response,
                flags=re.IGNORECASE,
            )
            if fenced_json_blocks:
                json_str = fenced_json_blocks[-1].strip()

            if not json_str:
                fenced_blocks = re.findall(
                    r"```\s*([\s\S]*?)\s*```",
                    response,
                    flags=re.IGNORECASE,
                )
                if fenced_blocks:
                    for block in reversed(fenced_blocks):
                        if '"city"' in block and '"days"' in block:
                            json_str = block.strip()
                            break
                    if not json_str:
                        json_str = fenced_blocks[-1].strip()

            if not json_str:
                start_idx = response.find("{")
                if start_idx == -1:
                    raise ValueError("响应中未找到JSON数据")

                depth = 0
                in_string = False
                escape = False
                end_idx = None

                for i in range(start_idx, len(response)):
                    ch = response[i]

                    if in_string:
                        if escape:
                            escape = False
                            continue
                        if ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_string = False
                        continue

                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break

                if end_idx is None:
                    raise ValueError("JSON花括号未能正确匹配")

                json_str = response[start_idx:end_idx].strip()

            if not json_str:
                raise ValueError("响应中未提取到JSON字符串")

            last_json_sample = json_str[:300]

            data = _loads_llm_json(json_str)
            _normalize_llm_trip_dict(data)

            days = data.get("days")
            if isinstance(days, list):
                for i, day in enumerate(days):
                    if not isinstance(day, dict):
                        continue

                    accommodation = day.get("accommodation")
                    if isinstance(accommodation, dict):
                        if not isinstance(day.get("hotel"), dict):
                            day["hotel"] = accommodation

                        normalized = (
                            accommodation.get("type")
                            or accommodation.get("name")
                            or "酒店"
                        )
                        day["accommodation"] = str(normalized)

                    for j, att in enumerate(day.get("attractions") or []):
                        if not isinstance(att, dict):
                            continue
                        loc = att.get("location")
                        if not isinstance(loc, dict) or "longitude" not in loc or "latitude" not in loc:
                            att["location"] = {
                                "longitude": 116.4 + i * 0.02 + j * 0.01,
                                "latitude": 39.9 + i * 0.02 + j * 0.01,
                            }

            if not (data.get("overall_suggestions") and str(data["overall_suggestions"]).strip()):
                data["overall_suggestions"] = (
                    f"{request.city}行程建议：结合天气与体力安排游览顺序，热门景点尽量提前预约。"
                )

            if "weather_info" not in data or data["weather_info"] is None:
                data["weather_info"] = []

            return TripPlan(**data)

        except Exception as e:
            print(f"⚠️  解析响应失败: {str(e)}")
            try:
                if last_json_sample:
                    print(f"   提取到的JSON片段(前300字符): {last_json_sample}...")
            except Exception:
                pass
            print("   将尝试用 MCP 检索上下文拼装降级行程；若仍不可行则使用占位行程")
            if graph_state:
                mc = self._create_mcp_grounded_plan(
                    request,
                    str(graph_state.get("attraction_context", "")),
                    str(graph_state.get("weather_context", "")),
                    str(graph_state.get("hotel_context", "")),
                )
                if mc is not None:
                    print("   ✅ 已用 MCP 上下文生成降级行程")
                    return mc
            print("   使用占位行程（无可用 MCP 摘要）")
            return self._create_fallback_plan(request)

    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        from datetime import datetime, timedelta

        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")

        days: List[DayPlan] = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)

            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}",
                        address=f"{request.city}市",
                        location=Location(
                            longitude=116.4 + i * 0.01 + j * 0.005,
                            latitude=39.9 + i * 0.01 + j * 0.005,
                        ),
                        visit_duration=120,
                        description=f"这是{request.city}的著名景点",
                        category="景点",
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐"),
                ],
            )
            days.append(day_plan)

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程,建议提前查看各景点的开放时间。",
        )


_planner: Optional[LangGraphTripPlanner] = None


def get_trip_planner_agent() -> LangGraphTripPlanner:
    global _planner
    if _planner is None:
        _planner = LangGraphTripPlanner()
    return _planner
