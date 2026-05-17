# 参考：直连 Agent 落盘（不经 HTTP）

在 `backend` 目录下执行，将结果写入项目根目录 `test_output/`：

```python
# 保存为 backend/run_smoke_to_file.py 或由 Agent 内联执行
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from app.models.schemas import TripRequest
from app.agents.trip_planner_agent import get_trip_planner_agent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test_output"
OUT.mkdir(exist_ok=True)

req = TripRequest(
    city="金华",
    start_date="2026-06-15",
    end_date="2026-06-16",
    travel_days=2,
    transportation="自驾",
    accommodation="舒适型酒店",
    preferences=["自然风光", "古村落"],
    free_text_input="想去横店或周边古镇",
)
plan = get_trip_planner_agent().plan_trip(req)
payload = {"request": req.model_dump(), "plan": plan.model_dump(mode="json")}
path = OUT / "jinhua_sample.json"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(path)
```

PDF：前端页面内导出依赖浏览器；自动化 PDF 非本仓库默认能力。
