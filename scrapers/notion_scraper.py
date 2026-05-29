"""Notion scraper using notion-client."""
from __future__ import annotations

import asyncio
from typing import Any

from core.config import settings
from scrapers.base import BaseScraper, ScrapedDocument
from utils.logger import logger


class NotionScraper(BaseScraper):
    source_type = "notion"

    def __init__(self) -> None:
        if not settings.notion_api_key:
            logger.warning("NOTION_API_KEY missing — NotionScraper will fail when invoked")
            self._client = None
        else:
            from notion_client import Client

            self._client = Client(auth=settings.notion_api_key)

    def _ensure(self) -> None:
        if self._client is None:
            raise RuntimeError("NOTION_API_KEY not configured")

    def _block_to_text(self, block: dict[str, Any]) -> str:
        btype = block.get("type")
        data = block.get(btype, {})
        rich = data.get("rich_text") or []
        text = "".join(t.get("plain_text", "") for t in rich)
        if btype == "heading_1":
            return f"# {text}"
        if btype == "heading_2":
            return f"## {text}"
        if btype == "heading_3":
            return f"### {text}"
        if btype == "bulleted_list_item":
            return f"- {text}"
        if btype == "numbered_list_item":
            return f"1. {text}"
        if btype == "to_do":
            checked = "x" if data.get("checked") else " "
            return f"- [{checked}] {text}"
        if btype == "code":
            lang = data.get("language", "")
            return f"```{lang}\n{text}\n```"
        if btype == "quote":
            return f"> {text}"
        return text

    def _fetch_page(self, page_id: str) -> ScrapedDocument:
        self._ensure()
        page = self._client.pages.retrieve(page_id=page_id)  # type: ignore[union-attr]
        # title detection
        title = "Notion Page"
        for prop in (page.get("properties") or {}).values():
            if prop.get("type") == "title":
                title = "".join(t.get("plain_text", "") for t in prop["title"]) or title
                break
        # blocks
        blocks: list[str] = []
        cursor: str | None = None
        while True:
            resp = self._client.blocks.children.list(  # type: ignore[union-attr]
                block_id=page_id, start_cursor=cursor
            )
            for blk in resp.get("results", []):
                blocks.append(self._block_to_text(blk))
            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")
        content = "\n\n".join(b for b in blocks if b)
        return ScrapedDocument(
            title=title,
            content=content,
            source=f"notion://{page_id}",
            source_type=self.source_type,
            metadata={"page_id": page_id, "url": page.get("url")},
        )

    async def scrape(self, target: str, **kwargs: Any) -> ScrapedDocument:
        page_id = target.replace("-", "")
        return await asyncio.to_thread(self._fetch_page, page_id)
