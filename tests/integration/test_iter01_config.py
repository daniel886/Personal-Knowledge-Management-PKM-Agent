"""Iter 1 — verify Settings load + ensure_dirs creates dirs and is idempotent."""
from __future__ import annotations

import os
from pathlib import Path


def test_settings_load(monkeypatch, tmp_path):
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    # bypass cached lru_cache by importing module fresh
    from core.config import Settings

    s = Settings()
    s.ensure_dirs()
    assert (tmp_path / "chroma").exists()
    assert (tmp_path / "logs").exists()
    assert s.vault_path.exists()


def test_provider_default():
    from core.config import settings

    assert settings.llm_provider in {"openai", "anthropic", "ollama"}


def test_ensure_dirs_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "l"))
    from core.config import Settings

    s = Settings()
    for _ in range(3):
        s.ensure_dirs()
    assert (tmp_path / "c").exists()
