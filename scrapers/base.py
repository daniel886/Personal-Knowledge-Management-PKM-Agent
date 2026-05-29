"""Scraper base abstractions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ScrapedDocument:
    """Normalised scraping result."""

    title: str
    content: str
    source: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "source_type": self.source_type,
            "metadata": self.metadata,
            "fetched_at": self.fetched_at.isoformat(),
        }


class BaseScraper(ABC):
    """Abstract base scraper."""

    source_type: str = "base"

    @abstractmethod
    async def scrape(self, target: str, **kwargs: Any) -> ScrapedDocument:
        """Run scraping and return a normalised document."""
        raise NotImplementedError
