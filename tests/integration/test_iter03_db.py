"""Iter 3 — verify async DB init creates tables and CRUD works on Knowledge."""
from __future__ import annotations

import os
from datetime import datetime

import pytest


@pytest.mark.asyncio
async def test_db_init_and_insert(tmp_path, monkeypatch):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/iter3.db"
    monkeypatch.setenv("DATABASE_URL", db_url)

    # Ensure fresh module state with new env
    import importlib
    from models import database as dbmod

    # Override the global engine for this test by manually building one
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    engine = create_async_engine(db_url, echo=False, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(dbmod.Base.metadata.create_all)

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        s.add(dbmod.Knowledge(
            id="k-1",
            title="Test",
            source_type="web",
            source="https://x.com",
            summary="hi",
            tags="a,b",
            obsidian_path="/tmp/x.md",
            extra={"k": "v"},
            chunks_indexed=2,
        ))
        await s.commit()

    async with Session() as s:
        from sqlalchemy import select
        rows = (await s.execute(select(dbmod.Knowledge))).scalars().all()
        assert len(rows) == 1
        assert rows[0].title == "Test"
        assert rows[0].extra == {"k": "v"}


@pytest.mark.asyncio
async def test_db_review_table(tmp_path):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from models.database import Base, ReviewReport

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/r.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        s.add(ReviewReport(
            period="weekly",
            start_at=datetime.utcnow(),
            end_at=datetime.utcnow(),
            summary="abc",
        ))
        await s.commit()
