"""Embedding 服务：使用 BGE-M3 生成向量。"""

from __future__ import annotations

from threading import Lock
from typing import List

from ..config import settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


class EmbeddingService:
    """文本嵌入服务（单例复用模型实例）。"""

    def __init__(self) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers 未安装，请先安装 requirements.txt 中的向量依赖"
            )
        self.model = SentenceTransformer(settings.embedding_model_id, device=settings.embedding_device)

    def _build_input(self, text: str, is_query: bool) -> str:
        prefix = settings.embedding_query_prefix if is_query else settings.embedding_passage_prefix
        clean = (text or "").strip()
        return f"{prefix}{clean}" if prefix else clean

    def embed_query(self, text: str) -> List[float]:
        """生成查询向量。"""
        value = self._build_input(text, is_query=True)
        vec = self.model.encode(
            [value],
            normalize_embeddings=settings.embedding_normalize,
            convert_to_numpy=True,
        )[0]
        return vec.astype("float32").tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档向量。"""
        if not texts:
            return []
        values = [self._build_input(t, is_query=False) for t in texts]
        vectors = self.model.encode(
            values,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=settings.embedding_normalize,
            convert_to_numpy=True,
        )
        return [v.astype("float32").tolist() for v in vectors]


_embedding_service: EmbeddingService | None = None
_embedding_lock = Lock()


def get_embedding_service() -> EmbeddingService:
    """获取 embedding 服务单例。"""
    global _embedding_service
    if _embedding_service is None:
        with _embedding_lock:
            if _embedding_service is None:
                _embedding_service = EmbeddingService()
    return _embedding_service
