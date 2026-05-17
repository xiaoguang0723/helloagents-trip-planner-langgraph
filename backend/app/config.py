"""配置管理模块"""

import os
import sys
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载环境变量
# 首先尝试加载当前目录的.env
load_dotenv()

# 然后尝试加载HelloAgents的.env(如果存在)
helloagents_env = Path(__file__).parent.parent.parent.parent / "HelloAgents" / ".env"
if helloagents_env.exists():
    load_dotenv(helloagents_env, override=False)  # 不覆盖已有的环境变量


class Settings(BaseSettings):
    """应用配置"""

    # 应用基本配置
    app_name: str = "HelloAgents智能旅行助手"
    app_version: str = "1.0.0"
    debug: bool = False

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS配置 - 使用字符串,在代码中分割
    cors_origins: str = (
        "http://localhost:5173,http://localhost:5174,http://localhost:3000,"
        "http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:3000"
    )

    # 高德地图API配置
    amap_api_key: str = ""

    # Unsplash API配置
    unsplash_access_key: str = ""
    unsplash_secret_key: str = ""

    # LLM配置 (从环境变量读取，与 LangChain ChatOpenAI 兼容)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4"

    # 日志配置
    log_level: str = "INFO"

    # 向量检索配置
    vector_search_enabled: bool = False
    vector_recall_min_items: int = 8
    vector_recall_score_threshold: float = 0.45
    vector_enable_bm25_branch: bool = True
    vector_rrf_k: int = 60
    vector_bm25_top_k: int = 20

    # Elasticsearch 配置
    elasticsearch_url: str = ""
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""
    elasticsearch_api_key: str = ""
    elasticsearch_index: str = "trip_poi"
    elasticsearch_verify_certs: bool = False
    elasticsearch_request_timeout: int = 30
    elasticsearch_knn_top_k: int = 20
    elasticsearch_num_candidates: int = 100

    # Embedding 配置（BGE-M3）
    embedding_model_id: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    embedding_normalize: bool = True
    embedding_max_length: int = 8192
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # 忽略额外的环境变量

    def get_cors_origins_list(self) -> List[str]:
        """获取CORS origins列表"""
        return [origin.strip() for origin in self.cors_origins.split(',')]


# 创建全局配置实例
settings = Settings()

# 同样为了避免 Windows 控制台（GBK）对 Emoji 输出报 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


# 验证必要的配置
def validate_config():
    """验证配置是否完整"""
    errors = []
    warnings = []

    if not settings.amap_api_key:
        errors.append("AMAP_API_KEY未配置")

    # ChatOpenAI 使用 LLM_API_KEY 或 OPENAI_API_KEY
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        warnings.append("LLM_API_KEY或OPENAI_API_KEY未配置,LLM功能可能无法使用")

    if settings.vector_search_enabled and not settings.elasticsearch_url:
        errors.append("已启用向量检索，但 ELASTICSEARCH_URL 未配置")

    if settings.embedding_dimension <= 0:
        errors.append("EMBEDDING_DIMENSION 必须大于0")

    if settings.vector_recall_min_items <= 0:
        errors.append("VECTOR_RECALL_MIN_ITEMS 必须大于0")

    if settings.vector_search_enabled and not settings.embedding_model_id:
        errors.append("已启用向量检索，但 EMBEDDING_MODEL_ID 未配置")

    if settings.vector_rrf_k <= 0:
        errors.append("VECTOR_RRF_K 必须大于0")

    if settings.vector_bm25_top_k <= 0:
        errors.append("VECTOR_BM25_TOP_K 必须大于0")

    if errors:
        error_msg = "配置错误:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    if warnings:
        print("\n⚠️  配置警告:")
        for w in warnings:
            print(f"  - {w}")

    return True


# 打印配置信息(用于调试)
def print_config():
    """打印当前配置(隐藏敏感信息)"""
    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"高德地图API Key: {'已配置' if settings.amap_api_key else '未配置'}")

    # 检查LLM配置
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL") or settings.openai_base_url
    llm_model = os.getenv("LLM_MODEL_ID") or settings.openai_model

    print(f"LLM API Key: {'已配置' if llm_api_key else '未配置'}")
    print(f"LLM Base URL: {llm_base_url}")
    print(f"LLM Model: {llm_model}")
    print(f"向量检索: {'启用' if settings.vector_search_enabled else '禁用'}")
    print(f"关键词分支(BM25): {'启用' if settings.vector_enable_bm25_branch else '禁用'}")
    print(
        f"Elasticsearch: {settings.elasticsearch_url if settings.elasticsearch_url else '未配置'}"
    )
    print(f"Embedding Model: {settings.embedding_model_id}")
    print(f"日志级别: {settings.log_level}")

