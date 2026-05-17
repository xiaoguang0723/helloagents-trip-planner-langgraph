---
name: trip-planner-smoke-test
description: >-
  Runs end-to-end trip planning smoke tests for the helloagents-trip-planner project (FastAPI, LangGraph, MCP), validates agent output sanity, and saves artifacts under test_output/. Use after code changes, when the user asks to try a city scenario, simulate user flow, verify agent output, or export/save results locally. Also when the user mentions smoke test, 用例, 试一个城市, or saving trip output to disk/PDF.
---

# 旅行规划 Agent 手工/半自动验收

## 目标

在**修改后端/agent 相关代码后**，用**接近真实用户**的方式跑一条「某城市 + 若干天」的规划请求，检查返回是否合理，并把**完整输出落盘**到可配置目录（默认 `test_output/`）。PDF 为可选项（见下文）。

## 默认约定

| 项 | 约定 |
|----|------|
| 后端目录 | `backend/`（工作目录设为此处再执行命令） |
| 落盘目录 | 项目根下 `test_output/`（若用户指定其他路径，以用户为准） |
| 主接口 | `POST http://127.0.0.1:8000/api/trip/plan`（端口以 `.env` 的 `PORT` 为准） |
| 直连 Agent（不经 HTTP） | `python -c` 调 `get_trip_planner_agent().plan_trip(TripRequest(...))`（需已 `load_dotenv()`） |

## 执行流程（Agent 按序做）

1. **确认环境**  
   - 需要：可用的 LLM Key、高德 `AMAP_API_KEY`（或等价配置），否则规划会失败或降级。  
   - **不要**在日志/落盘文件里写入完整 Key。

2. **启动后端（若未运行）**  
   - `cd backend` → `python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000`（或与 `.env` 一致）。  
   - 单测脚本方式可跳过 HTTP，但一次完整「用户路径」仍建议至少验证一次 HTTP。

3. **构造请求体**（与 `TripRequest` 一致）  
   - 必填：`city`, `start_date`, `end_date`, `travel_days`, `transportation`, `accommodation`  
   - 可选：`preferences`（数组）, `free_text_input`（字符串）

4. **发送请求**  
   - 推荐：`Invoke-RestMethod` / `curl` POST JSON；或运行 `backend` 内短脚本调用 `plan_trip`。

5. **快速判断「是否正常」（启发式，非严格）**  
   - HTTP 200 且 `success: true`。  
   - `data.city` 与请求城市一致或合理（模型偶发写别名字段时需结合 `data.days`）。  
   - `days` 长度与 `travel_days` 一致。  
   - 若首景点名为 `{city}景点1` 且 `overall_suggestions` 像占位文案，可能为**解析失败后的占位/降级**，应在结果里**标注**「疑似占位或 MCP 拼装降级」，便于人工核对。

6. **落盘**  
   - 文件名建议：`test_output/<城市拼音或汉字>_<YYYYMMDD_HHMMSS>.json` 或 `.txt`。  
   - 内容：JSON 时保存 `request` + 完整 `plan`（或整份 API 响应），UTF-8。  
   - 若用户指定 `test_output` 子目录或其他根路径，创建目录后写入。

7. **PDF（可选）**  
   - **本仓库前端**已依赖 `jspdf` 与 `html2canvas`，**面向浏览器内导出**；自动化流水线里**不默认**生成 PDF，除非用户明确要求且环境具备无头浏览器/现成脚本。  
   - 可行替代：**将同一份 JSON 交给用户**，用浏览器打开前端结果页手动导出；或系统「打印为 PDF」打开保存的 `.json`/`.md`。  
   - 若用户坚持命令行一键 PDF，可在 `backend` 增加独立脚本（如 `reportlab`/`weasyprint`）——**属新功能，不在本 Skill 默认步骤**。

## 最小 HTTP 示例（Windows PowerShell）

请求体保存为 `body.json` 后：

```powershell
$body = Get-Content -Raw -Encoding UTF8 body.json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/trip/plan" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType "application/json; charset=utf-8" -TimeoutSec 600
```

（单次规划可能需数分钟，超时建议 ≥ 300～600 秒。）

## Agent 行为约束

- 优先**实际执行**命令并读取结果，再汇报；不要只给「用户可自行运行」的说明就结束（除非环境无法联网/缺 Key）。  
- 落盘路径使用**正斜杠**或用户确认的绝对路径，避免仅写反斜杠导致跨工具问题。  
- 改动代码与本次验收无关时，不要扩大范围。

## 额外资源

- 更长的示例请求、与 `smoke_multi_agent.py` 对齐的脚本片段，可写在同目录 `reference.md`（若存在则按需阅读）。
