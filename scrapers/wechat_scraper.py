"""WeChat 公众号 article scraper. Implementation reuses WebScraper with WeChat-specific
content extraction logic (the article body sits in #js_content)."""
from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup
from markdownify import markdownify as md

from scrapers.base import BaseScraper, ScrapedDocument
from scrapers.web_scraper import WebScraper
from utils.logger import logger


class WeChatScraper(BaseScraper):
    source_type = "wechat"

    def __init__(self) -> None:
        self._web = WebScraper()

    async def scrape(self, target: str, **kwargs: Any) -> ScrapedDocument:
        if "mp.weixin.qq.com" not in target:
            logger.warning(f"{target} is not a WeChat article URL — using generic scraper")
        # Reuse Playwright-aware WebScraper to obtain HTML
        html = await self._web._fetch_with_playwright(target)
        if not html:
            html = await self._web._fetch_with_httpx(target)
        soup = BeautifulSoup(html, "lxml")

        title_tag = soup.find(id="activity-name") or soup.title
        title = (title_tag.get_text(strip=True) if title_tag else target)
        author_tag = soup.find(id="js_name") or soup.find("a", id="js_name")
        author = author_tag.get_text(strip=True) if author_tag else ""
        publish_tag = soup.find(id="publish_time")
        published = publish_tag.get_text(strip=True) if publish_tag else ""

        body_node = soup.find(id="js_content") or soup.body
        for tag in body_node(["script", "style"]):
            tag.decompose()
        content_md = md(str(body_node), heading_style="ATX").strip()

        return ScrapedDocument(
            title=title,
            content=content_md,
            source=target,
            source_type=self.source_type,
            metadata={"author": author, "published": published},
        )
