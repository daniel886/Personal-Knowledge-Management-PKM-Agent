"""APScheduler-based background scheduler."""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from core.config import settings
from utils.logger import logger


class TaskScheduler:
    """Singleton wrapper around APScheduler with async job support."""

    _instance: "TaskScheduler | None" = None

    def __new__(cls) -> "TaskScheduler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.scheduler = AsyncIOScheduler(timezone="UTC")
        return cls._instance

    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("APScheduler started")

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("APScheduler stopped")

    def add_cron(
        self,
        func: Callable[..., Awaitable[Any]],
        cron_expr: str,
        *,
        job_id: str,
        kwargs: dict | None = None,
    ) -> None:
        trigger = CronTrigger.from_crontab(cron_expr)
        self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            kwargs=kwargs or {},
            replace_existing=True,
            misfire_grace_time=60 * 30,
        )
        logger.info(f"Cron job '{job_id}' registered: {cron_expr}")

    def add_interval(
        self,
        func: Callable[..., Awaitable[Any]],
        *,
        minutes: int,
        job_id: str,
        kwargs: dict | None = None,
    ) -> None:
        self.scheduler.add_job(
            func,
            trigger=IntervalTrigger(minutes=minutes),
            id=job_id,
            kwargs=kwargs or {},
            replace_existing=True,
        )
        logger.info(f"Interval job '{job_id}' every {minutes}m registered")

    def list_jobs(self) -> list[dict]:
        return [
            {
                "id": j.id,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
                "trigger": str(j.trigger),
            }
            for j in self.scheduler.get_jobs()
        ]


def get_scheduler() -> TaskScheduler:
    return TaskScheduler()


async def register_default_jobs() -> None:
    """Register the default cron + interval jobs."""
    from agents.review_agent import run_weekly_review, run_monthly_review
    from scrapers.rss_scraper import RSSScraper

    sched = get_scheduler()
    sched.add_cron(run_weekly_review, settings.weekly_review_cron, job_id="weekly_review")
    sched.add_cron(run_monthly_review, settings.monthly_review_cron, job_id="monthly_review")
    sched.add_interval(
        RSSScraper.refresh_subscribed,
        minutes=settings.rss_fetch_interval_min,
        job_id="rss_pull",
    )
    sched.start()
