"""Email scraper - IMAP based."""
from __future__ import annotations

import asyncio
from email import message_from_bytes
from email.header import decode_header
from typing import Any

from imapclient import IMAPClient

from core.config import settings
from scrapers.base import BaseScraper, ScrapedDocument
from utils.logger import logger


def _decode_header(raw: str | bytes | None) -> str:
    if not raw:
        return ""
    parts = decode_header(raw if isinstance(raw, str) else raw.decode(errors="ignore"))
    return "".join(
        (p.decode(enc or "utf-8", errors="ignore") if isinstance(p, bytes) else p)
        for p, enc in parts
    )


class EmailScraper(BaseScraper):
    source_type = "email"

    def _fetch_recent(self, limit: int = 10) -> list[ScrapedDocument]:
        if not settings.email_user or not settings.email_password:
            raise RuntimeError("EMAIL_USER / EMAIL_PASSWORD not configured")

        out: list[ScrapedDocument] = []
        with IMAPClient(settings.email_host, port=settings.email_port, ssl=True) as server:
            server.login(settings.email_user, settings.email_password)
            server.select_folder(settings.email_folder, readonly=True)
            uids = server.search(["ALL"])[-limit:]
            messages = server.fetch(uids, ["RFC822", "INTERNALDATE"])
            for uid, data in messages.items():
                msg = message_from_bytes(data[b"RFC822"])
                subject = _decode_header(msg.get("Subject"))
                sender = _decode_header(msg.get("From"))
                body = self._extract_body(msg)
                out.append(
                    ScrapedDocument(
                        title=subject or f"Email {uid}",
                        content=body,
                        source=f"imap://{settings.email_user}/{uid}",
                        source_type=self.source_type,
                        metadata={
                            "from": sender,
                            "uid": uid,
                            "date": str(data.get(b"INTERNALDATE", "")),
                        },
                    )
                )
        logger.info(f"Fetched {len(out)} emails")
        return out

    @staticmethod
    def _extract_body(msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    payload = part.get_payload(decode=True) or b""
                    return payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    payload = part.get_payload(decode=True) or b""
                    return payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
            return ""
        payload = msg.get_payload(decode=True) or b""
        return payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")

    async def scrape(self, target: str, **kwargs: Any) -> ScrapedDocument:
        # target is treated as "limit" here, e.g. "5"
        limit = int(target) if target.isdigit() else int(kwargs.get("limit", 1))
        items = await asyncio.to_thread(self._fetch_recent, limit)
        if not items:
            raise RuntimeError("No emails fetched")
        # Return latest as primary; metadata carries others
        primary = items[-1]
        primary.metadata["all"] = [i.to_dict() for i in items]
        return primary

    async def fetch_recent(self, limit: int = 10) -> list[ScrapedDocument]:
        return await asyncio.to_thread(self._fetch_recent, limit)
