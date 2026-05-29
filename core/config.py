"""Application configuration via Pydantic Settings v2."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------- LLM --------
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    anthropic_api_key: str | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # -------- Embedding --------
    embedding_provider: Literal["openai", "huggingface"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    hf_embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # -------- Storage --------
    database_url: str = "sqlite+aiosqlite:///./data/pkm.db"
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "pkm_knowledge"

    # -------- Obsidian --------
    obsidian_vault_path: str = "./vault"
    obsidian_inbox_folder: str = "PKM/Inbox"
    obsidian_daily_folder: str = "PKM/Daily"
    obsidian_review_folder: str = "PKM/Reviews"
    obsidian_auto_sync: bool = True

    # -------- Scrapers --------
    youtube_api_key: str | None = None
    notion_api_key: str | None = None
    notion_database_id: str | None = None

    email_host: str = "imap.gmail.com"
    email_port: int = 993
    email_user: str | None = None
    email_password: str | None = None
    email_folder: str = "INBOX"

    wechat_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    )

    playwright_headless: bool = True
    playwright_timeout: int = 30000

    # -------- Scheduler --------
    weekly_review_cron: str = "0 20 * * 0"
    monthly_review_cron: str = "0 21 28-31 * *"
    rss_fetch_interval_min: int = 60

    # -------- App --------
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = False
    app_log_level: str = "INFO"
    app_secret_key: str = "change-me"

    # -------- Logging --------
    log_dir: str = "./logs"
    log_level: str = "INFO"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def vault_path(self) -> Path:
        p = Path(self.obsidian_vault_path).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def ensure_dirs(self) -> None:
        for d in (
            self.log_dir,
            self.chroma_persist_dir,
            "./data/uploads",
        ):
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
