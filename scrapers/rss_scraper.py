"""RSS scraper using feedparser."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import feedparser

from core.config import settings
from scrapers.base import BaseScraper, ScrapedDocument
from utils.logger import logger

SUBSCRIPTIONS_PATH = Path("./data/rss_subscriptions.txt")


class RSSScraper(BaseScraper):
    source_type = "rss"

    def _parse(self, url: str) -> list[ScrapedDocument]:
        feed = feedparser.parse(
            url,
            agent=settings.wechat_user_agent,
        )
        if feed.bozo and not feed.entries:
            logger.warning(f"RSS parse error: {feed.bozo_exception}")
            return []
        items: list[ScrapedDocument] = []
        feed_title = (feed.feed or {}).get("title", url)
        for entry in feed.entries[:30]:
            content = entry.get("summary") or entry.get("description") or ""
            if hasattr(entry, "content") and entry.content:
                content = entry.content[0].get("value", content)
            items.append(
                ScrapedDocument(
                    title=entry.get("title", "(untitled)"),
                    content=content,
                    source=entry.get("link", url),
                    source_type=self.source_type,
                    metadata={
                        "feed_title": feed_title,
                        "published": entry.get("published", ""),
                        "feed_url": url,
                    },
                )
            )
        logger.info(f"RSS '{feed_title}' returned {len(items)} entries")
        return items

    async def scrape(self, target: str, **kwargs: Any) -> ScrapedDocument:
        items = await asyncio.to_thread(self._parse, target)
        if not items:
            raise RuntimeError(f"No entries parsed from {target}")
        primary = items[0]
        primary.metadata["all"] = [i.to_dict() for i in items]
        return primary

    async def fetch_all(self, target: str) -> list[ScrapedDocument]:
        return await asyncio.to_thread(self._parse, target)

    # ---------------- subscription file ----------------
    @classmethod
    def list_subscriptions(cls) -> list[str]:
        if not SUBSCRIPTIONS_PATH.exists():
            return []
        return [
            line.strip()
            for line in SUBSCRIPTIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]

    @classmethod
    def add_subscription(cls, url: str) -> None:
        SUBSCRIPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = cls.list_subscriptions()
        if url in existing:
            return
        with SUBSCRIPTIONS_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{url}\n")
        logger.info(f"RSS subscription added: {url}")

    @classmethod
    async def refresh_subscribed(cls) -> int:
        """Periodic job: pull every subscribed feed and ingest new items."""
        from workflows.ingest_workflow import ingest_document

        scraper = cls()
        ingested = 0
        for url in cls.list_subscriptions():
            try:
                items = await scraper.fetch_all(url)
                for doc in items[:5]:  # limit per cycle
                    await ingest_document(doc)
                    ingested += 1
            except Exception as exc:
                logger.warning(f"RSS refresh failed for {url}: {exc}")
        logger.info(f"RSS refresh: {ingested} items ingested")
        return ingested
