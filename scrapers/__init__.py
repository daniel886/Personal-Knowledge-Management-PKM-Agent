"""Scraper registry and factory."""
from __future__ import annotations

from scrapers.base import BaseScraper, ScrapedDocument
from scrapers.email_scraper import EmailScraper
from scrapers.notion_scraper import NotionScraper
from scrapers.pdf_scraper import PDFScraper
from scrapers.rss_scraper import RSSScraper
from scrapers.web_scraper import WebScraper
from scrapers.wechat_scraper import WeChatScraper
from scrapers.youtube_scraper import YouTubeScraper

SCRAPERS: dict[str, type[BaseScraper]] = {
    "web": WebScraper,
    "pdf": PDFScraper,
    "youtube": YouTubeScraper,
    "wechat": WeChatScraper,
    "email": EmailScraper,
    "notion": NotionScraper,
    "rss": RSSScraper,
}


def get_scraper(source_type: str) -> BaseScraper:
    cls = SCRAPERS.get(source_type)
    if cls is None:
        raise ValueError(f"No scraper registered for source_type='{source_type}'")
    return cls()


__all__ = [
    "BaseScraper",
    "ScrapedDocument",
    "WebScraper",
    "PDFScraper",
    "YouTubeScraper",
    "WeChatScraper",
    "EmailScraper",
    "NotionScraper",
    "RSSScraper",
    "SCRAPERS",
    "get_scraper",
]
