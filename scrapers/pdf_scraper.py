"""PDF scraper - works with both local files and remote URLs."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import httpx

from scrapers.base import BaseScraper, ScrapedDocument
from utils.logger import logger
from utils.parsers import parse_pdf
from utils.retry import async_retry


class PDFScraper(BaseScraper):
    source_type = "pdf"

    @async_retry(max_attempts=3, exceptions=(httpx.HTTPError,))
    async def _download(self, url: str) -> Path:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            tmp = Path(tempfile.mkstemp(suffix=".pdf")[1])
            tmp.write_bytes(resp.content)
            logger.info(f"Downloaded PDF -> {tmp}")
            return tmp

    async def scrape(self, target: str, **kwargs: Any) -> ScrapedDocument:
        path: Path
        if target.startswith(("http://", "https://")):
            path = await self._download(target)
        else:
            path = Path(target).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(target)

        text = parse_pdf(path)
        title = kwargs.get("title") or path.stem

        return ScrapedDocument(
            title=title,
            content=text,
            source=str(target),
            source_type=self.source_type,
            metadata={"file": str(path), "chars": len(text)},
        )
