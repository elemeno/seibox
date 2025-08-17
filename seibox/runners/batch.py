"""Batch execution utilities with parallelism and retry logic."""

import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: Attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds with jitter
    """
    delay = min(base_delay * (2**attempt), max_delay)
    # Add jitter (±25%)
    jitter = delay * 0.25 * (2 * random.random() - 1)
    return max(0, delay + jitter)


def execute_with_retry(
    func: Callable[[T], R],
    item: T,
    max_retries: int = 3,
    retry_exceptions: tuple = (Exception,),
) -> R:
    """Execute a function with retry logic.

    Args:
        func: Function to execute
        item: Item to process
        max_retries: Maximum number of retries
        retry_exceptions: Tuple of exceptions to retry on

    Returns:
        Result from the function

    Raises:
        Exception: The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(item)
        except retry_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = exponential_backoff(attempt)
                time.sleep(delay)
            else:
                raise

    raise last_exception


def batch_execute(
    func: Callable[[T], R],
    items: list[T],
    max_workers: int = 10,
    rate_limit: float | None = None,
    max_retries: int = 3,
    show_progress: bool = True,
    desc: str = "Processing",
) -> list[R]:
    """Execute a function over items in parallel with rate limiting.

    Args:
        func: Function to execute on each item
        items: List of items to process
        max_workers: Maximum number of parallel workers
        rate_limit: Optional rate limit (calls per second)
        max_retries: Maximum retries per item
        show_progress: Whether to show progress bar
        desc: Description for progress bar

    Returns:
        List of results in the same order as inputs
    """
    if not items:
        return []

    results: dict[int, R] = {}
    last_call_time = 0.0
    min_interval = 1.0 / rate_limit if rate_limit else 0.0

    def rate_limited_func(indexed_item: tuple[int, T]) -> tuple[int, R]:
        """Wrapper to apply rate limiting."""
        nonlocal last_call_time

        idx, item = indexed_item

        # Apply rate limiting
        if min_interval > 0:
            current_time = time.time()
            elapsed = current_time - last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_call_time = time.time()

        result = execute_with_retry(func, item, max_retries=max_retries)
        return idx, result

    # Create indexed items to preserve order
    indexed_items = list(enumerate(items))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(rate_limited_func, indexed_item) for indexed_item in indexed_items
        ]

        # Process completed futures
        progress_bar = tqdm(total=len(items), desc=desc, disable=not show_progress)

        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
                progress_bar.update(1)
            except Exception as e:
                progress_bar.write(f"Error processing item: {e}")
                raise

        progress_bar.close()

    # Return results in original order
    return [results[i] for i in range(len(items))]
