"""Iter 2 — verify logger writes to file and respects level."""
from __future__ import annotations

import time
from pathlib import Path


def test_logger_writes_file():
    from utils.logger import logger
    from core.config import settings

    msg = f"hello-{time.time_ns()}"
    logger.info(msg)
    # flush async sinks
    import loguru
    loguru.logger.complete()

    log_dir = Path(settings.log_dir)
    found = False
    for p in log_dir.glob("pkm-*.log"):
        if msg in p.read_text(encoding="utf-8", errors="ignore"):
            found = True
            break
    assert found, f"message not found in any log file under {log_dir}"


def test_logger_levels_no_crash():
    from utils.logger import logger

    logger.debug("debug")
    logger.info("info")
    logger.warning("warn")
    logger.error("err")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception("captured")
