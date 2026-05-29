"""Async-friendly retry decorators based on tenacity."""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.logger import logger

T = TypeVar("T")


def async_retry(
    *,
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Async retry decorator with exponential backoff."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential(multiplier=initial_wait, max=max_wait),
                    retry=retry_if_exception_type(exceptions),
                    reraise=True,
                ):
                    with attempt:
                        if attempt.retry_state.attempt_number > 1:
                            logger.warning(
                                f"Retry {attempt.retry_state.attempt_number}/{max_attempts} "
                                f"for {func.__name__}"
                            )
                        return await func(*args, **kwargs)
            except RetryError as e:  # pragma: no cover - safety net
                logger.error(f"All retries exhausted for {func.__name__}: {e}")
                raise

        return wrapper

    return decorator
