"""Elasticsearch 向量存储服务。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, TypedDict

from ..config import settings
from .embedding_service import get_embedding_service

try:
    from elasticsearch import Elasticsearch
except ImportError:  # pragma: no cover
    Elasticsearch = None  # type: ignore


class POIDocument(TypedDict, total=False):
    poi_id: str
    city: str
    name: str
    address: str
    category: str
    source: str
    source_time: str
    location: Dict[str, float]
    raw: str


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _parse_location(location: Any) -> Dict[str, float] | None:
    if isinstance(location, str):
        raw = location.strip()
        if "," in raw:
            lng, lat = raw.split(",", 1)
            try:
                return {"longitude": float(lng), "latitude": float(lat)}
            except ValueError:
                return None
    if isinstance(location, dict):
        lon = location.get("longitude") or location.get("lng")
        lat = location.get("latitude") or location.get("lat")
        try:
            if lon is not None and lat is not None:
                return {"longitude": float(lon), "latitude": float(lat)}
        except ValueError:
            return None
    return None


def _extract_json_like_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    stack = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
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
            continue
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0 and start != -1:
                blocks.append(text[start : i + 1])
                start = -1
    return blocks


def parse_pois_from_amap_payload(payload: Any, city: str, source: str = "mcp_amap") -> List[POIDocument]:
    """从高德返回（dict/json/text）中尽量提取 POI。"""
    items: List[POIDocument] = []
    now = datetime.now(timezone.utc).isoformat()

    def append_one(one: dict) -> None:
        name = _as_str(one.get("name")).strip()
        if not name:
            return
        loc = _parse_location(one.get("location"))
        items.append(
            {
                "poi_id": _as_str(one.get("id")).strip(),
                "city": city,
                "name": name,
                "address": _as_str(one.get("address")).strip() or city,
                "category": _as_str(one.get("type")).strip() or "景点",
                "source": source,
                "source_time": now,
                "location": loc or {},
                "raw": _as_str(one),
            }
        )

    if isinstance(payload, dict):
        pois = payload.get("pois")
        if isinstance(pois, list):
            for one in pois:
                if isinstance(one, dict):
                    append_one(one)
        return items

    if isinstance(payload, list):
        for one in payload:
            if isinstance(one, dict):
                append_one(one)
        return items

    text = _as_str(payload).strip()
    if not text:
        return items

    try:
        decoded = json.loads(text)
        return parse_pois_from_amap_payload(decoded, city=city, source=source)
    except Exception:
        pass

    for block in _extract_json_like_blocks(text):
        try:
            decoded = json.loads(block)
            extracted = parse_pois_from_amap_payload(decoded, city=city, source=source)
            if extracted:
                return extracted
        except Exception:
            continue

    # 最差兜底：仅提取 name
    names = re.findall(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    for name in names[:40]:
        clean_name = name.strip()
        if not clean_name:
            continue
        items.append(
            {
                "poi_id": "",
                "city": city,
                "name": clean_name,
                "address": city,
                "category": "景点",
                "source": source,
                "source_time": now,
                "location": {},
                "raw": clean_name,
            }
        )
    return items


class VectorStoreService:
    """Elasticsearch 向量检索服务。"""

    def __init__(self) -> None:
        if Elasticsearch is None:
            raise RuntimeError("elasticsearch 未安装，请先安装 requirements.txt 中的向量依赖")

        kwargs: Dict[str, Any] = {
            "hosts": [settings.elasticsearch_url],
            "request_timeout": settings.elasticsearch_request_timeout,
            "verify_certs": settings.elasticsearch_verify_certs,
        }
        if settings.elasticsearch_api_key:
            kwargs["api_key"] = settings.elasticsearch_api_key
        elif settings.elasticsearch_username:
            kwargs["basic_auth"] = (
                settings.elasticsearch_username,
                settings.elasticsearch_password,
            )

        self.client = Elasticsearch(**kwargs)
        self.index_name = settings.elasticsearch_index
        self.embedding_service = get_embedding_service()

    def _ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "poi_id": {"type": "keyword"},
                    "city": {"type": "keyword"},
                    "name": {"type": "text"},
                    "address": {"type": "text"},
                    "category": {"type": "keyword"},
                    "content": {"type": "text"},
                    "source": {"type": "keyword"},
                    "source_time": {"type": "date"},
                    "location": {"type": "geo_point"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": settings.embedding_dimension,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=mapping)

    def is_index_ready(self) -> bool:
        try:
            self._ensure_index()
            return True
        except Exception:
            return False

    def _build_doc_id(self, poi: POIDocument) -> str:
        if poi.get("poi_id"):
            return str(poi["poi_id"])
        raw = f"{poi.get('city','')}|{poi.get('name','')}|{poi.get('address','')}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _build_content(self, poi: POIDocument) -> str:
        parts = [
            poi.get("city", ""),
            poi.get("name", ""),
            poi.get("address", ""),
            poi.get("category", ""),
        ]
        return " ".join([p for p in parts if p]).strip()

    def upsert_pois(self, pois: List[POIDocument]) -> int:
        if not pois:
            return 0
        self._ensure_index()
        docs: List[POIDocument] = []
        contents: List[str] = []
        for poi in pois:
            if not poi.get("name"):
                continue
            content = self._build_content(poi)
            if not content:
                continue
            docs.append(poi)
            contents.append(content)
        if not docs:
            return 0

        vectors = self.embedding_service.embed_documents(contents)
        upserted = 0
        for poi, content, vector in zip(docs, contents, vectors):
            body: Dict[str, Any] = {
                "poi_id": poi.get("poi_id", ""),
                "city": poi.get("city", ""),
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "category": poi.get("category", "景点"),
                "content": content,
                "source": poi.get("source", "unknown"),
                "source_time": poi.get("source_time") or datetime.now(timezone.utc).isoformat(),
                "embedding": vector,
            }
            loc = poi.get("location", {})
            lon = loc.get("longitude")
            lat = loc.get("latitude")
            if lon is not None and lat is not None:
                body["location"] = {"lat": lat, "lon": lon}
            self.client.index(index=self.index_name, id=self._build_doc_id(poi), document=body)
            upserted += 1
        return upserted

    def _normalize_hit(self, hit: Dict[str, Any], city: str, source_type: str) -> Dict[str, Any]:
        src = hit.get("_source", {})
        loc = src.get("location") or {}
        return {
            "id": str(hit.get("_id", "")),
            "score": float(hit.get("_score", 0.0)),
            "source_type": source_type,
            "poi_id": src.get("poi_id", ""),
            "city": src.get("city", city),
            "name": src.get("name", ""),
            "address": src.get("address", city),
            "category": src.get("category", "景点"),
            "location": {
                "longitude": float(loc.get("lon")) if isinstance(loc, dict) and loc.get("lon") is not None else None,
                "latitude": float(loc.get("lat")) if isinstance(loc, dict) and loc.get("lat") is not None else None,
            },
        }

    def _search_semantic_knn(self, city: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        q_vec = self.embedding_service.embed_query(query)
        body = {
            "knn": {
                "field": "embedding",
                "query_vector": q_vec,
                "k": top_k,
                "num_candidates": settings.elasticsearch_num_candidates,
                "filter": [{"term": {"city": city}}],
            }
        }
        resp = self.client.search(index=self.index_name, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [self._normalize_hit(h, city, "semantic") for h in hits]

    def _search_keyword_bm25(self, city: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "filter": [{"term": {"city": city}}],
                    "should": [
                        {"match": {"name": {"query": query, "boost": 3.0}}},
                        {"match": {"category": {"query": query, "boost": 2.0}}},
                        {"match": {"address": {"query": query, "boost": 1.0}}},
                        {"match": {"content": {"query": query, "boost": 2.0}}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        }
        resp = self.client.search(index=self.index_name, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [self._normalize_hit(h, city, "keyword") for h in hits]

    def _rrf_fuse(
        self, semantic_hits: List[Dict[str, Any]], keyword_hits: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        rrf_k = settings.vector_rrf_k
        merged: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(semantic_hits, start=1):
            doc_id = hit.get("id") or f"semantic::{hit.get('name')}::{rank}"
            base = merged.get(doc_id) or {**hit, "score": 0.0, "matched_by": []}
            base["score"] += 1.0 / (rrf_k + rank)
            base["matched_by"].append("semantic")
            merged[doc_id] = base

        for rank, hit in enumerate(keyword_hits, start=1):
            doc_id = hit.get("id") or f"keyword::{hit.get('name')}::{rank}"
            base = merged.get(doc_id) or {**hit, "score": 0.0, "matched_by": []}
            base["score"] += 1.0 / (rrf_k + rank)
            if "keyword" not in base["matched_by"]:
                base["matched_by"].append("keyword")
            merged[doc_id] = base

        ranked = sorted(merged.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return ranked[:top_k]

    def search_similar_pois(self, city: str, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        self._ensure_index()
        if not query.strip():
            return []
        k = top_k or settings.elasticsearch_knn_top_k

        semantic_hits = self._search_semantic_knn(city=city, query=query, top_k=k)
        if not settings.vector_enable_bm25_branch:
            return semantic_hits

        keyword_k = max(k, settings.vector_bm25_top_k)
        keyword_hits = self._search_keyword_bm25(city=city, query=query, top_k=keyword_k)
        return self._rrf_fuse(semantic_hits=semantic_hits, keyword_hits=keyword_hits, top_k=k)


_vector_store_service: VectorStoreService | None = None
_vector_store_lock = Lock()


def get_vector_store_service() -> VectorStoreService:
    """获取向量服务单例。"""
    global _vector_store_service
    if _vector_store_service is None:
        with _vector_store_lock:
            if _vector_store_service is None:
                _vector_store_service = VectorStoreService()
    return _vector_store_service
