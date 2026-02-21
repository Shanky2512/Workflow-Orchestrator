"""
Retry and timeout helpers for async validation.
"""
import asyncio
from typing import Callable, Any

# Constants for async validation
ASYNC_TIMEOUT_SECONDS = 5
ASYNC_MAX_RETRIES = 2


async def retry_with_timeout(
    coro_fn: Callable[[], Any],
    error_message: str,
    retries: int = ASYNC_MAX_RETRIES,
    timeout: int = ASYNC_TIMEOUT_SECONDS
) -> Any:
    """
    Execute async function with retries and timeout.

    Args:
        coro_fn: Async function to execute
        error_message: Error message if all retries fail
        retries: Number of retry attempts
        timeout: Timeout in seconds for each attempt

    Returns:
        Result from coro_fn

    Raises:
        Exception: If all retries fail
    """
    last_exception = None

    for attempt in range(retries):
        try:
            return await asyncio.wait_for(coro_fn(), timeout=timeout)
        except Exception as e:
            last_exception = e
            # Log attempt failure
            if attempt < retries - 1:
                # Brief delay before retry
                await asyncio.sleep(0.1 * (attempt + 1))

    raise Exception(f"{error_message}: {str(last_exception)}")
