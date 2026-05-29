"""Web scraper - Playwright preferred, httpx fallback."""
from __future__ import annotations

from typing import Any

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from core.config import settings
from scrapers.base import BaseScraper, ScrapedDocument
from utils.logger import logger
from utils.retry import async_retry


class WebScraper(BaseScraper):
    source_type = "web"

    def __init__(self, *, headless: bool | None = None, timeout: int | None = None) -> None:
        self.headless = settings.playwright_headless if headless is None else headless
        self.timeout = timeout or settings.playwright_timeout

    @async_retry(max_attempts=3, exceptions=(httpx.HTTPError,))
    async def _fetch_with_httpx(self, url: str) -> str:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": settings.wechat_user_agent},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    async def _fetch_with_playwright(self, url: str) -> str | None:
        """Render dynamic JS pages via Playwright; gracefully degrade if not installed."""
        try:
            from playwright.async_api import async_playwright
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Playwright not available, falling back: {exc}")
            return None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context(
                    user_agent=settings.wechat_user_agent
                )
                page = await context.new_page()
                await page.goto(url, timeout=self.timeout, wait_until="networkidle")
                html = await page.content()
                await context.close()
                await browser.close()
                return html
        except Exception as exc:
            logger.warning(f"Playwright fetch failed for {url}: {exc}")
            return None

    async def scrape(self, target: str, **kwargs: Any) -> ScrapedDocument:
        logger.info(f"Web scraping {target}")
        html = await self._fetch_with_playwright(target)
        if not html:
            html = await self._fetch_with_httpx(target)

        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string.strip() if soup.title and soup.title.string else target)

        # Strip non-content elements
        for tag in soup(["script", "style", "nav", "footer", "noscript", "header"]):
            tag.decompose()

        article = soup.find("article") or soup.find("main") or soup.body or soup
        content = md(str(article), heading_style="ATX").strip()

        return ScrapedDocument(
            title=title,
            content=content,
            source=target,
            source_type=self.source_type,
            metadata={"length": len(content)},
        )
