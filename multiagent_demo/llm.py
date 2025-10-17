from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from .config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResources:
    """Container for the optional chat model and any initialisation error."""

    model: Optional[BaseChatModel]
    error: Optional[str]


def build_chat_model(config: ModelConfig) -> LLMResources:
    """Create a LangChain chat model from the provided configuration.

    Returns a container object holding the model (if creation was successful)
    and a potential error message (if creation failed). The application can use
    the error string to inform users that they are running in fallback mode.
    """

    if not config.api_key:
        message = "Missing API key for OpenAI-compatible endpoint; falling back to rule-based responses."
        logger.warning(message)
        return LLMResources(model=None, error=message)

    try:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            base_url=config.base_url,
        )
        return LLMResources(model=model, error=None)
    except Exception as exc:  # pragma: no cover - network dependent
        error = f"Failed to initialise OpenAI-compatible chat model: {exc}"
        logger.warning(error)
        return LLMResources(model=None, error=error)
