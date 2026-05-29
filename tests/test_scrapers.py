"""Smoke tests that don't require external API access."""
from __future__ import annotations

import pytest

from scrapers.youtube_scraper import YouTubeScraper


def test_extract_video_id_from_url() -> None:
    assert YouTubeScraper.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert YouTubeScraper.extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=10") == "dQw4w9WgXcQ"
    assert YouTubeScraper.extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid() -> None:
    with pytest.raises(ValueError):
        YouTubeScraper.extract_video_id("https://example.com/no-video")


def test_scrapers_registry() -> None:
    from scrapers import SCRAPERS, get_scraper

    assert {"web", "pdf", "youtube", "wechat", "rss"}.issubset(SCRAPERS.keys())
    assert get_scraper("web").source_type == "web"
    with pytest.raises(ValueError):
        get_scraper("unknown")
