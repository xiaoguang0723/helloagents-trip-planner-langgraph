"""同城连续两次请求实测：统计向量命中率与 API 回落次数。"""

from __future__ import annotations

import io
import json
import os
import re
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict

import sys

CURRENT = Path(__file__).resolve()
BACKEND_ROOT = CURRENT.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# 必须在导入 app 之前设置，确保 Settings 读取正确
os.environ.setdefault("VECTOR_SEARCH_ENABLED", "true")
os.environ.setdefault("ELASTICSEARCH_URL", "http://localhost:9200")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from app.agents import trip_planner_agent as tpa  # noqa: E402
from app.agents.trip_planner_agent import get_trip_planner_agent  # noqa: E402
from app.models.schemas import TripRequest  # noqa: E402


def _metrics_from_log(text: str) -> Dict[str, int]:
    return {
        "vector_hit": len(re.findall(r"\[vector_hit\]", text)),
        "vector_miss": len(re.findall(r"\[vector_miss\]", text)),
        "api_fallback": len(re.findall(r"\[api_fallback\]", text)),
        "es_upsert_count_logs": len(re.findall(r"\[es_upsert_count\]", text)),
    }


def _run_once(request: TripRequest) -> Dict[str, object]:
    buffer = io.StringIO()
    started = time.time()
    with redirect_stdout(buffer):
        plan = get_trip_planner_agent().plan_trip(request)
    elapsed = round(time.time() - started, 2)
    log_text = buffer.getvalue()
    metrics = _metrics_from_log(log_text)
    denom = metrics["vector_hit"] + metrics["vector_miss"]
    hit_rate = round(metrics["vector_hit"] / denom, 4) if denom > 0 else 0.0
    return {
        "elapsed_sec": elapsed,
        "hit_rate": hit_rate,
        "api_fallback_count": metrics["api_fallback"],
        "vector_hit_count": metrics["vector_hit"],
        "vector_miss_count": metrics["vector_miss"],
        "es_upsert_count_logs": metrics["es_upsert_count_logs"],
        "plan_days": len(plan.days),
        "first_day_attractions": [a.name for a in (plan.days[0].attractions if plan.days else [])[:3]],
        "raw_log": log_text,
    }


def main() -> None:
    # 重置单例，确保配置重新生效
    tpa._planner = None

    req = TripRequest(
        city="成都",
        start_date="2026-06-01",
        end_date="2026-06-01",
        travel_days=1,
        transportation="地铁",
        accommodation="经济型酒店",
        preferences=["美食", "文化"],
        free_text_input="希望多安排热门景点，少走回头路",
    )

    first = _run_once(req)
    second = _run_once(req)

    summary = {
        "city": req.city,
        "run_1": {k: v for k, v in first.items() if k != "raw_log"},
        "run_2": {k: v for k, v in second.items() if k != "raw_log"},
        "compare": {
            "hit_rate_delta": round(second["hit_rate"] - first["hit_rate"], 4),
            "api_fallback_delta": int(second["api_fallback_count"]) - int(first["api_fallback_count"]),
            "vector_hit_delta": int(second["vector_hit_count"]) - int(first["vector_hit_count"]),
            "vector_miss_delta": int(second["vector_miss_count"]) - int(first["vector_miss_count"]),
        },
    }

    out_dir = BACKEND_ROOT / "test_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "same_city_twice_report.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n报告已保存: {out_file}")


if __name__ == "__main__":
    main()
