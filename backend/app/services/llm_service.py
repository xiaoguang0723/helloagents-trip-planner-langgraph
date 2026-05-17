"""LLM 服务：LangChain ChatOpenAI（供 LangGraph / 直连调用）"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI

from ..config import get_settings

_chat_model: Optional[ChatOpenAI] = None


def get_chat_model() -> ChatOpenAI:
    """
    获取 ChatOpenAI 实例（单例）。

    环境变量：LLM_API_KEY / OPENAI_API_KEY，LLM_BASE_URL，LLM_MODEL_ID。
    """
    global _chat_model

    if _chat_model is None:
        settings = get_settings()
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or settings.openai_api_key
        )
        base_url = os.getenv("LLM_BASE_URL") or settings.openai_base_url
        model = os.getenv("LLM_MODEL_ID") or settings.openai_model

        _chat_model = ChatOpenAI(
            api_key=api_key or None,
            base_url=base_url or None,
            model=model,
            temperature=0.2,
        )

        print("✅ LLM 服务初始化成功 (LangChain ChatOpenAI)")
        print(f"   Base URL: {base_url}")
        print(f"   Model: {model}")

    return _chat_model


def reset_chat_model() -> None:
    global _chat_model
    _chat_model = None


def reset_llm() -> None:
    """兼容旧名称"""
    reset_chat_model()
