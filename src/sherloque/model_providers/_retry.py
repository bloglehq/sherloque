import asyncio
import functools
from random import random

import httpx

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def with_retry(
    max_retries: int = 3,
    delay: float = 0.1,
    max_delay: float = 10.0,
):
    """Retry an async function on transient HTTP errors with exponential backoff + jitter.

    Retries connection/timeout errors and retryable status codes (429, 5xx).
    Non-retryable status codes (e.g. 4xx) are re-raised immediately.
    The wrapped function is expected to call ``response.raise_for_status()`` so
    that HTTP error statuses surface as ``httpx.HTTPStatusError``.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (httpx.TransportError, httpx.HTTPStatusError) as e:
                    if (
                        isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code not in RETRYABLE_STATUS
                    ):
                        raise
                    if attempt == max_retries - 1:
                        raise
                    backoff = min(delay * 2**attempt, max_delay) + random()
                    await asyncio.sleep(backoff)

        return wrapper

    return decorator


__all__ = [
    "with_retry",
]
