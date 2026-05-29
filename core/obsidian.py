"""Obsidian vault read/write & two-way sync helpers."""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml

from core.config import settings
from utils.logger import logger


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write atomically: same-dir temp file + os.replace to avoid partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]\|]+?)(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([\w\u4e00-\u9fa5\-/]+)")


@dataclass
class ObsidianNote:
    path: Path
    title: str
    content: str
    frontmatter: dict
    tags: list[str]
    links: list[str]

    @property
    def relative_path(self) -> str:
        try:
            return str(self.path.relative_to(settings.vault_path))
        except ValueError:
            return str(self.path)


class ObsidianVault:
    """Filesystem-based Obsidian vault adapter."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or settings.vault_path).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in (
            settings.obsidian_inbox_folder,
            settings.obsidian_daily_folder,
            settings.obsidian_review_folder,
        ):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        logger.info(f"Obsidian vault @ {self.root}")

    # ---------------- write ----------------
    def write_note(
        self,
        title: str,
        content: str,
        *,
        folder: str | None = None,
        frontmatter: dict | None = None,
        overwrite: bool = True,
    ) -> Path:
        folder = folder or settings.obsidian_inbox_folder
        safe_title = self._safe_filename(title)
        target_dir = self.root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{safe_title}.md"
        if path.exists() and not overwrite:
            stamp = datetime.now().strftime("%H%M%S")
            path = target_dir / f"{safe_title}-{stamp}.md"

        body = self._render(title=title, body=content, frontmatter=frontmatter or {})
        _atomic_write_text(path, body)
        logger.info(f"Wrote Obsidian note: {path.relative_to(self.root)}")
        return path

    def append_to_daily(self, content: str, *, day: datetime | None = None) -> Path:
        day = day or datetime.now()
        filename = day.strftime("%Y-%m-%d") + ".md"
        path = self.root / settings.obsidian_daily_folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                self._render(
                    title=day.strftime("%Y-%m-%d"),
                    body=f"## {day.strftime('%H:%M')}\n\n{content}\n",
                    frontmatter={"type": "daily", "date": day.strftime("%Y-%m-%d")},
                ),
                encoding="utf-8",
            )
        else:
            with path.open("a", encoding="utf-8") as f:
                f.write(f"\n## {day.strftime('%H:%M')}\n\n{content}\n")
        return path

    # ---------------- read ----------------
    def read_note(self, path: str | Path) -> ObsidianNote:
        p = Path(path) if Path(path).is_absolute() else self.root / path
        raw = p.read_text(encoding="utf-8", errors="ignore")
        fm, body = self._split_frontmatter(raw)
        return ObsidianNote(
            path=p,
            title=fm.get("title") or p.stem,
            content=body,
            frontmatter=fm,
            tags=self._extract_tags(raw, fm),
            links=WIKILINK_RE.findall(body),
        )

    def list_notes(self, folder: str | None = None) -> list[ObsidianNote]:
        base = self.root / folder if folder else self.root
        return [self.read_note(p) for p in base.rglob("*.md")]

    def search_notes(self, keyword: str) -> list[ObsidianNote]:
        out: list[ObsidianNote] = []
        for p in self.root.rglob("*.md"):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if keyword.lower() in txt.lower():
                out.append(self.read_note(p))
        return out

    # ---------------- linking ----------------
    @staticmethod
    def make_wikilink(target: str, alias: str | None = None) -> str:
        return f"[[{target}|{alias}]]" if alias else f"[[{target}]]"

    def add_backlink_section(
        self, note_path: str | Path, related: Iterable[str]
    ) -> None:
        related_list = list(related)
        if not related_list:
            return
        p = Path(note_path) if Path(note_path).is_absolute() else self.root / note_path
        if not p.exists():
            return
        section = "\n\n## 🔗 相关笔记\n\n" + "\n".join(
            f"- {self.make_wikilink(r)}" for r in related_list
        )
        with p.open("a", encoding="utf-8") as f:
            f.write(section)

    # ---------------- helpers ----------------
    @staticmethod
    def _safe_filename(title: str) -> str:
        cleaned = re.sub(r'[\\/:"*?<>|]+', "-", title).strip()
        return cleaned[:120] or f"note-{datetime.now():%Y%m%d-%H%M%S}"

    @staticmethod
    def _render(title: str, body: str, frontmatter: dict) -> str:
        fm = {
            "title": title,
            "created": datetime.now().isoformat(timespec="seconds"),
            **(frontmatter or {}),
        }
        fm_yaml = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip()
        return f"---\n{fm_yaml}\n---\n\n# {title}\n\n{body}\n"

    @staticmethod
    def _split_frontmatter(raw: str) -> tuple[dict, str]:
        m = FRONTMATTER_RE.match(raw)
        if not m:
            return {}, raw
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            fm = {}
        return fm, raw[m.end():]

    @staticmethod
    def _extract_tags(raw: str, fm: dict) -> list[str]:
        tags = set()
        fm_tags = fm.get("tags") or []
        if isinstance(fm_tags, str):
            fm_tags = [t.strip() for t in fm_tags.split(",")]
        tags.update(fm_tags)
        tags.update(TAG_RE.findall(raw))
        return sorted(tags)


_VAULT: ObsidianVault | None = None


def get_vault() -> ObsidianVault:
    global _VAULT
    if _VAULT is None:
        _VAULT = ObsidianVault()
    return _VAULT
