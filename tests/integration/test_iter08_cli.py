"""Iter 8 — Typer CLI: --help and rss-add and init-db."""
from __future__ import annotations

import os
from typer.testing import CliRunner


runner = CliRunner()


def test_help_lists_commands():
    from main import app
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0, r.output
    for cmd in ["serve", "ingest", "search", "chat", "review", "init-db", "rss-add"]:
        assert cmd in r.output


def test_init_db():
    """init-db should create whatever DB the cached settings point to."""
    from core.config import settings
    from main import app

    # Wipe + clear engine cache
    import models.database as dbmod
    dbmod._engine = None
    dbmod._session_factory = None

    r = runner.invoke(app, ["init-db"])
    assert r.exit_code == 0, r.output

    # Resolve sqlite path from the URL the running settings use
    url = settings.database_url
    assert url.startswith("sqlite+aiosqlite:///")
    db_path = url.replace("sqlite+aiosqlite:///", "", 1)
    from pathlib import Path
    assert Path(db_path).exists(), f"db not created at {db_path}"


def test_rss_add(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from main import app
    r = runner.invoke(app, ["rss-add", "https://example.com/feed.xml"])
    assert r.exit_code == 0, r.output
    sub = tmp_path / "data" / "rss_subscriptions.txt"
    assert sub.exists()
    assert "https://example.com/feed.xml" in sub.read_text(encoding="utf-8")
