"""Iter 4 — Obsidian vault end-to-end: write, read, search, daily, backlink."""
from __future__ import annotations

from pathlib import Path

from core.obsidian import ObsidianVault


def test_write_and_read_note(tmp_path: Path):
    v = ObsidianVault(root=tmp_path)
    p = v.write_note(
        title="测试笔记 / 含特殊字符*",
        content="正文 #机器学习 [[相关]]",
        folder="PKM/Inbox",
        frontmatter={"tags": ["机器学习", "测试"]},
    )
    assert p.exists()
    note = v.read_note(p)
    assert note.title.startswith("测试笔记")
    assert "正文" in note.content
    assert "机器学习" in note.tags
    assert "相关" in note.links


def test_search_notes(tmp_path: Path):
    v = ObsidianVault(root=tmp_path)
    v.write_note(title="Alpha", content="alpha beta")
    v.write_note(title="Beta", content="只有 BETA")
    hits = v.search_notes("BETA")
    assert len(hits) >= 2


def test_daily_append(tmp_path: Path):
    v = ObsidianVault(root=tmp_path)
    p1 = v.append_to_daily("第一条")
    p2 = v.append_to_daily("第二条")
    assert p1 == p2
    txt = p1.read_text(encoding="utf-8")
    assert "第一条" in txt and "第二条" in txt


def test_backlink_section(tmp_path: Path):
    v = ObsidianVault(root=tmp_path)
    p = v.write_note(title="Origin", content="body")
    v.add_backlink_section(p, ["Note A", "Note B"])
    txt = p.read_text(encoding="utf-8")
    assert "[[Note A]]" in txt and "[[Note B]]" in txt


def test_overwrite_false_creates_unique(tmp_path: Path):
    v = ObsidianVault(root=tmp_path)
    p1 = v.write_note(title="dup", content="1")
    p2 = v.write_note(title="dup", content="2", overwrite=False)
    assert p1 != p2
    assert p1.exists() and p2.exists()
