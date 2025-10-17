from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from .config import WeaviateConfig

logger = logging.getLogger(__name__)


@dataclass
class WeaviateManager:
    """Simple wrapper around the Weaviate client with graceful fallbacks."""

    config: WeaviateConfig
    fallback_store: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._client = None
        self._connected = False
        self._init_error: Optional[str] = None
        self._ensure_client()

    # ------------------------------------------------------------------
    # Client bootstrap
    # ------------------------------------------------------------------
    def _ensure_client(self) -> None:
        try:
            import weaviate

            auth = None
            if self.config.api_key:
                auth = weaviate.AuthApiKey(api_key=self.config.api_key)

            self._client = weaviate.Client(
                url=self.config.url,
                auth_client_secret=auth,
                timeout_config=(5, 15),
            )
            self._connected = True
        except Exception as exc:  # pragma: no cover - network dependent
            self._client = None
            self._connected = False
            self._init_error = str(exc)
            logger.warning("Falling back to in-memory vector store: %s", exc)

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def ensure_schema(self) -> bool:
        if not self._connected or self._client is None:
            return False

        try:
            schema = self._client.schema.get()
            if any(cls.get("class") == self.config.class_name for cls in schema.get("classes", [])):
                return True

            class_obj = {
                "class": self.config.class_name,
                "vectorizer": "text2vec-openai",
                "properties": [
                    {"name": "text", "dataType": ["text"]},
                    {"name": "source", "dataType": ["text"]},
                    {"name": "chunk_index", "dataType": ["int"]},
                    {"name": "document_id", "dataType": ["int"]},
                ],
            }
            self._client.schema.create_class(class_obj)
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Unable to ensure Weaviate schema: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------
    def upsert_texts(self, payloads: List[Dict[str, Any]]) -> int:
        if not payloads:
            return 0

        stored = 0
        if self._connected and self._client is not None and self.ensure_schema():
            for payload in payloads:
                data_object = {
                    "text": payload.get("text"),
                    "source": payload.get("source"),
                    "chunk_index": payload.get("chunk_index"),
                    "document_id": payload.get("document_id"),
                }
                try:
                    self._client.data_object.create(data_object=data_object, class_name=self.config.class_name)
                    stored += 1
                except Exception as exc:  # pragma: no cover - network dependent
                    logger.warning("Failed to insert into Weaviate: %s", exc)
                    self.fallback_store.append(payload)
        else:
            self.fallback_store.extend(payloads)
            stored = len(payloads)

        return stored

    def query(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not query:
            return []

        if self._connected and self._client is not None:
            try:
                response = (
                    self._client.query.get(self.config.class_name, ["text", "source", "chunk_index", "document_id"])
                    .with_near_text({"concepts": [query]})
                    .with_limit(limit)
                    .do()
                )
                results = response.get("data", {}).get("Get", {}).get(self.config.class_name, [])
                return [
                    {
                        "text": item.get("text"),
                        "source": item.get("source"),
                        "chunk_index": item.get("chunk_index"),
                        "document_id": item.get("document_id"),
                        "score": None,  # Weaviate returns distance but only when requested; keep placeholder
                        "provider": "weaviate",
                    }
                    for item in results
                ]
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Weaviate query failed, switching to fallback: %s", exc)
                self._connected = False

        return self._fallback_query(query, limit)

    # ------------------------------------------------------------------
    # Fallback retrieval
    # ------------------------------------------------------------------
    def _fallback_query(self, query: str, limit: int) -> List[Dict[str, Any]]:
        if not self.fallback_store:
            return []

        scored = []
        for payload in self.fallback_store:
            text = payload.get("text", "")
            score = SequenceMatcher(None, query, text).ratio()
            scored.append((score, payload))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_items = scored[:limit]
        return [
            {
                "text": payload.get("text"),
                "source": payload.get("source"),
                "chunk_index": payload.get("chunk_index"),
                "document_id": payload.get("document_id"),
                "score": round(score, 4),
                "provider": "fallback",
            }
            for score, payload in top_items
        ]

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error
