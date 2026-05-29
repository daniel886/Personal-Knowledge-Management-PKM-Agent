"""Loguru-based unified logging."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from core.config import settings


def setup_logger() -> None:
    """Configure Loguru sinks (console + rotating file)."""
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=log_format,
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        f"{settings.log_dir}/pkm-{{time:YYYY-MM-DD}}.log",
        level=settings.log_level,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        format=log_format,
        backtrace=True,
        diagnose=True,
        enqueue=True,
        encoding="utf-8",
    )


setup_logger()

__all__ = ["logger", "setup_logger"]
