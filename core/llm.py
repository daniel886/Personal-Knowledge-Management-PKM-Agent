"""Multi-provider LLM and embedding factory."""
from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from core.config import settings
from utils.logger import logger


def get_chat_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a LangChain chat model based on provider configuration."""
    provider = (provider or settings.llm_provider).lower()
    temperature = temperature if temperature is not None else settings.llm_temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY is not configured.")
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=model or settings.llm_model,
            temperature=temperature,
            max_tokens=settings.llm_max_tokens,
            streaming=streaming,
            **kwargs,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=model or settings.llm_model or "claude-3-5-sonnet-latest",
            temperature=temperature,
            max_tokens=settings.llm_max_tokens,
            streaming=streaming,
            **kwargs,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model or settings.ollama_model,
            temperature=temperature,
            **kwargs,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


def get_embeddings(provider: str | None = None) -> Embeddings:
    """Get an embedding model instance."""
    provider = (provider or settings.embedding_provider).lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
        )

    if provider == "huggingface":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "huggingface embeddings require sentence-transformers"
            ) from exc

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(f"Unknown embedding provider: {provider}")
