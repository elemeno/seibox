"""Response caching utilities for deterministic evaluation."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import orjson
import xxhash


def compute_cache_key(
    model_name: str, temperature: float, system_hash: str | None, prompt: str
) -> str:
    """Compute a deterministic cache key for a completion request.

    Args:
        model_name: Name of the model
        temperature: Temperature setting
        system_hash: Hash of the system prompt (if any)
        prompt: The user prompt

    Returns:
        Hexadecimal cache key
    """
    system_hash = system_hash or ""
    key_string = f"{model_name}|{temperature}|{system_hash}|{prompt}"
    return xxhash.xxh64_hexdigest(key_string)


def get_cache_dir() -> Path:
    """Get the cache directory path.

    Returns:
        Path to the cache directory
    """
    cache_dir = Path("artifacts/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached(cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve a cached response if it exists.

    Args:
        cache_key: The cache key to look up

    Returns:
        Cached response dict with text, usage, and latency_ms, or None if not found
    """
    cache_file = get_cache_dir() / f"{cache_key}.json"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "rb") as f:
            result = orjson.loads(f.read())
            return dict(result) if isinstance(result, dict) else None
    except (orjson.JSONDecodeError, IOError):
        return None


def set_cached(cache_key: str, response: Dict[str, Any]) -> None:
    """Store a response in the cache.

    Args:
        cache_key: The cache key to store under
        response: Response dict with text, usage, and latency_ms

    Raises:
        IOError: If unable to write to cache
    """
    cache_file = get_cache_dir() / f"{cache_key}.json"

    try:
        with open(cache_file, "wb") as f:
            f.write(orjson.dumps(response, option=orjson.OPT_INDENT_2))
    except IOError as e:
        raise IOError(f"Failed to write cache file {cache_file}: {e}")
