"""多智能体行程样例自检（成都 1 天）。"""
from dotenv import load_dotenv

load_dotenv()

from app.agents import trip_planner_agent as tpa
from app.agents.trip_planner_agent import get_trip_planner_agent
from app.models.schemas import TripRequest

tpa._planner = None

req = TripRequest(
    city="成都",
    start_date="2026-06-01",
    end_date="2026-06-01",
    travel_days=1,
    transportation="地铁",
    accommodation="经济型酒店",
    preferences=["美食", "文化"],
    free_text_input="想体验火锅和茶馆",
)
plan = get_trip_planner_agent().plan_trip(req)
fb = (
    plan.days[0].attractions[0].name == f"{req.city}景点1"
    if plan.days and plan.days[0].attractions
    else True
)
print("---RESULT---")
print("city", plan.city, "days", len(plan.days), "fallback_placeholder", fb)
if plan.days and plan.days[0].attractions:
    for a in plan.days[0].attractions[:5]:
        print("  attraction:", a.name)
print("weather_entries", len(plan.weather_info), "has_budget", plan.budget is not None)
