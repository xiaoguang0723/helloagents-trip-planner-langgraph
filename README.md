# HelloAgents 智能旅行助手---0514编辑更新

基于 **LangGraph** 构建的智能旅行规划助手，后端通过 **FastAPI** 提供 API，并通过 **MCP** 接入高德地图能力；同时支持使用 **Elasticsearch** 作为向量数据库进行 POI 向量检索（可选），提升同城多次规划的命中率与稳定性。

## ✨ 功能特点

- 🤖 **LangGraph 协作式规划**: 多子智能体顺序采集（景点/天气/酒店）+ 规划模型汇总，输出可解析的行程 JSON
- 🗺️ **高德地图集成（MCP）**: 通过 MCP 接入高德地图工具，支持 POI 搜索、天气查询、路线规划
- 🧠 **自动工具调用（ReAct）**: 子智能体可按需调用 MCP 工具获取实时数据，再交给规划模型整合
- 🔎 **向量检索增强（可选）**: 后端使用 Elasticsearch 作为向量数据库，支持语义 KNN + BM25 + RRF 融合召回
- 🎨 **现代化前端**: Vue 3 + TypeScript + Vite，配合 Ant Design Vue

## 🏗️ 技术栈

### 后端
- **框架**: LangGraph
- **API**: FastAPI
- **MCP**: `amap-mcp-server`（高德地图工具进程，由 `uvx`/PATH 启动）
- **LLM**: 基于 `langchain-openai` 的 Chat 模型接入（兼容 OpenAI/DeepSeek 等 OpenAI-API 形态）
- **向量库（可选）**: Elasticsearch（dense_vector + KNN）
- **Embedding（可选）**: `sentence-transformers`（默认 `BAAI/bge-m3`）

### 前端
- **框架**: Vue 3 + TypeScript
- **构建工具**: Vite
- **UI组件库**: Ant Design Vue
- **地图服务**: 高德地图 JavaScript API
- **HTTP客户端**: Axios

## 📁 项目结构

```
helloagents-trip-planner/
├── backend/                    # 后端服务
│   ├── app/
│   │   ├── agents/            # Agent实现
│   │   │   └── trip_planner_agent.py
│   │   ├── api/               # FastAPI路由
│   │   │   ├── main.py
│   │   │   └── routes/
│   │   │       ├── trip.py
│   │   │       ├── poi.py
│   │   │       └── map.py
│   │   ├── services/          # 服务层
│   │   │   ├── llm_service.py
│   │   │   ├── mcp_amap.py
│   │   │   ├── embedding_service.py
│   │   │   └── vector_store_service.py
│   │   ├── models/            # 数据模型
│   │   │   └── schemas.py
│   │   └── config.py          # 配置管理
│   ├── scripts/               # 脚本（可选：向量库预热/压测）
│   │   ├── seed_hot_cities_pois.py
│   │   └── benchmark_same_city_twice.py
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
├── frontend/                   # 前端应用
│   ├── src/
│   │   ├── components/        # Vue组件
│   │   ├── services/          # API服务
│   │   ├── types/             # TypeScript类型
│   │   └── views/             # 页面视图
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 🚀 快速开始

### 前提条件

- Python 3.10+
- Node.js 16+
- 高德地图API密钥 (Web服务API和Web端(JS API))
- LLM API密钥 (OpenAI/DeepSeek等)
- （可选）Elasticsearch 8.x（用于向量检索增强）

### 后端安装

1. 进入后端目录
```bash
cd backend
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件,填入你的API密钥
```

5. 启动后端服务
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### （可选）启用 Elasticsearch 向量检索

该项目支持把高德 MCP 返回的 POI（景点等）写入 Elasticsearch，并在后续同城规划中优先走 **向量召回**（语义 KNN，必要时融合 BM25 + RRF），以减少对实时 API 的依赖。

1. 启动 Elasticsearch（示例：单机开发）

```bash
docker run --name es-trip -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.14.0
```

2. 在 `backend/.env` 中开启并配置（参考 `backend/.env.example`）

- `VECTOR_SEARCH_ENABLED=true`
- `ELASTICSEARCH_URL=http://localhost:9200`
- `ELASTICSEARCH_INDEX=trip_poi`
- `EMBEDDING_MODEL_ID=BAAI/bge-m3`
- `EMBEDDING_DIMENSION=1024`

3. （推荐）预置热门城市景点到向量库（一次性预热）

```bash
cd backend
python scripts/seed_hot_cities_pois.py
```

4. （可选）验证“同城第二次请求”命中率提升（会输出报告到 `backend/test_output/`）

```bash
cd backend
python scripts/benchmark_same_city_twice.py
```

### 前端安装

1. 进入前端目录
```bash
cd frontend
```

2. 安装依赖
```bash
npm install
```

3. 配置环境变量
```bash
# 创建.env文件, 填入高德地图Web API Key 和 Web端JS API Key
cp .env.example .env
```

4. 启动开发服务器
```bash
npm run dev
```

5. 打开浏览器访问 `http://localhost:5173`

## 📝 使用指南

1. 在首页填写旅行信息:
   - 目的地城市
   - 旅行日期和天数
   - 交通方式偏好
   - 住宿偏好
   - 旅行风格标签

2. 点击"生成旅行计划"按钮

3. 系统将:
   - 调用HelloAgents Agent生成初步计划
   - Agent自动调用高德地图MCP工具搜索景点
   - Agent获取天气信息和路线规划
   - 整合所有信息生成完整行程

4. 查看结果:
   - 每日详细行程
   - 景点信息与地图标记
   - 交通路线规划
   - 天气预报
   - 餐饮推荐

## 🔧 核心实现

### LangGraph 旅行规划器（当前实现）

后端核心在 `backend/app/agents/trip_planner_agent.py`：使用 `StateGraph` 串联 4 个节点：

- `attraction`：景点子智能体（优先向量库召回；不足则 MCP 回落；回落结果可回写 ES）
- `weather`：天气子智能体（MCP）
- `hotel`：酒店子智能体（MCP）
- `planner`：规划模型汇总，产出最终行程 JSON（并带解析修复/降级拼装）

### MCP工具调用

子智能体可自动调用以下高德地图 MCP 工具（以实际可用工具为准）:
- `maps_text_search`: 搜索景点POI
- `maps_weather`: 查询天气
- `maps_direction_walking_by_address`: 步行路线规划
- `maps_direction_driving_by_address`: 驾车路线规划
- `maps_direction_transit_integrated_by_address`: 公共交通路线规划

## 📄 API文档

启动后端服务后,访问 `http://localhost:8000/docs` 查看完整的API文档。

主要端点:
- `POST /api/trip/plan` - 生成旅行计划
- `GET /api/map/poi` - 搜索POI
- `GET /api/map/weather` - 查询天气
- `POST /api/map/route` - 规划路线

## 🔐 配置与安全提示

- **请不要直接使用仓库里的示例 Key**：`backend/.env.example` 与 `frontend/.env.example` 仅用于展示字段，实际部署请替换为你自己的密钥。
- **LLM 环境变量兼容**：后端读取 `LLM_API_KEY/LLM_BASE_URL/LLM_MODEL_ID`，同时也兼容 `OPENAI_API_KEY` 等 OpenAI 生态变量（详见 `backend/app/config.py`）。

## 🤝 贡献指南

欢迎提交Pull Request或Issue!

## 📜 开源协议

CC BY-NC-SA 4.0

## 🙏 致谢

- [HelloAgents](https://github.com/datawhalechina/Hello-Agents) - 智能体教程
- [HelloAgents框架](https://github.com/jjyaoao/HelloAgents) - 智能体框架
- [高德地图开放平台](https://lbs.amap.com/) - 地图服务
- [amap-mcp-server](https://github.com/sugarforever/amap-mcp-server) - 高德地图MCP服务器

---

**HelloAgents智能旅行助手** - 让旅行计划变得简单而智能 🌈

