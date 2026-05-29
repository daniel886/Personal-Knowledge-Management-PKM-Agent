"""YouTube scraper using youtube-transcript-api + optional metadata via API key."""
from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from core.config import settings
from scrapers.base import BaseScraper, ScrapedDocument
from utils.logger import logger
from utils.retry import async_retry

YT_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})"
)


class YouTubeScraper(BaseScraper):
    source_type = "youtube"

    @staticmethod
    def extract_video_id(url: str) -> str:
        m = YT_ID_RE.search(url)
        if not m:
            raise ValueError(f"Could not extract video id from {url}")
        return m.group(1)

    async def _fetch_transcript(
        self, video_id: str, languages: list[str]
    ) -> tuple[str, str]:
        def _sync() -> tuple[str, str]:
            try:
                segments = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            except (NoTranscriptFound, TranscriptsDisabled):
                segments = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join(seg["text"].strip() for seg in segments if seg.get("text"))
            return text, "transcript"

        return await asyncio.to_thread(_sync)

    @async_retry(max_attempts=2, exceptions=(httpx.HTTPError,))
    async def _fetch_metadata(self, video_id: str) -> dict[str, Any]:
        if not settings.youtube_api_key:
            return {}
        url = (
            "https://www.googleapis.com/youtube/v3/videos"
            f"?id={video_id}&part=snippet,statistics&key={settings.youtube_api_key}"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        items = data.get("items", [])
        if not items:
            return {}
        snippet = items[0].get("snippet", {})
        stats = items[0].get("statistics", {})
        return {
            "title": snippet.get("title"),
            "channel": snippet.get("channelTitle"),
            "published_at": snippet.get("publishedAt"),
            "description": snippet.get("description", "")[:1000],
            "view_count": stats.get("viewCount"),
        }

    async def scrape(self, target: str, **kwargs: Any) -> ScrapedDocument:
        video_id = self.extract_video_id(target)
        logger.info(f"YouTube scraping video_id={video_id}")
        languages = kwargs.get("languages") or ["zh-Hans", "zh-CN", "en"]
        try:
            transcript, _kind = await self._fetch_transcript(video_id, languages)
        except Exception as exc:
            logger.warning(f"Transcript fetch failed: {exc}")
            transcript = ""
        meta = await self._fetch_metadata(video_id)
        title = meta.get("title") or f"YouTube {video_id}"
        body_parts: list[str] = []
        if meta.get("description"):
            body_parts.append("## 视频描述\n\n" + meta["description"])
        if transcript:
            body_parts.append("## 字幕\n\n" + transcript)
        body = "\n\n".join(body_parts) or "(no transcript available)"
        return ScrapedDocument(
            title=title,
            content=body,
            source=target,
            source_type=self.source_type,
            metadata={"video_id": video_id, **meta},
        )
