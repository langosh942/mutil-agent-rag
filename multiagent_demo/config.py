from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(eq=True, frozen=True)
class ModelConfig:
    """Configuration for chat/completion model."""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.2
    max_tokens: int = 512


@dataclass(eq=True, frozen=True)
class WeaviateConfig:
    """Configuration for Weaviate client."""

    url: str
    api_key: Optional[str] = None
    class_name: str = "DocumentChunk"


@dataclass(eq=True, frozen=True)
class DatabaseConfig:
    """Configuration for relational database connection."""

    url: str = "sqlite:///demo.db"
    echo: bool = False


@dataclass(eq=True, frozen=True)
class AppConfig:
    """Aggregate configuration for the application."""

    model: ModelConfig
    weaviate: WeaviateConfig
    database: DatabaseConfig
    top_k: int = field(default=4)
