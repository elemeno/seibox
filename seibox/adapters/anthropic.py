"""Anthropic adapter implementation."""

import os
import time
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv


class AnthropicAdapter:
    """Adapter for Anthropic Claude models."""

    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        """Initialize the Anthropic adapter.

        Args:
            model_name: Anthropic model name (e.g., claude-3-haiku-20240307, claude-3-sonnet-20240229)
        """
        self.name = f"anthropic:{model_name}"
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        """Lazily initialize the Anthropic client.

        Returns:
            Anthropic client instance

        Raises:
            ImportError: If anthropic package is not installed
            ValueError: If API key is not set
        """
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")

            load_dotenv()
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Set it in .env file or environment variables."
                )

            self._client = anthropic.Anthropic(api_key=api_key)

        return self._client

    def complete(
        self,
        *,
        system: str | None,
        prompt: str,
        temperature: float,
        max_tokens: int,
        extra: Mapping[str, Any] | None = None,
    ) -> dict:
        """Generate a completion from Anthropic Claude.

        Args:
            system: Optional system prompt
            prompt: User prompt
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            extra: Additional Anthropic-specific parameters

        Returns:
            Dictionary with text, usage, and latency_ms
        """
        client = self._get_client()

        start_time = time.time()

        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add system prompt if provided
        if system:
            kwargs["system"] = system

        # Add any extra parameters
        if extra:
            # Filter out any None values from extra
            extra_filtered = {k: v for k, v in extra.items() if v is not None}
            kwargs.update(extra_filtered)

        response = client.messages.create(**kwargs)

        latency_ms = (time.time() - start_time) * 1000

        # Extract text from response
        text = ""
        if response.content:
            # Response content is a list of content blocks
            for block in response.content:
                if block.type == "text":
                    text += block.text

        # Calculate token usage - use Claude's counts if available, otherwise estimate
        if response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        else:
            # Estimate tokens if Anthropic doesn't provide usage info
            from seibox.utils.tokens import estimate_tokens

            full_input = f"{system}\n{prompt}" if system else prompt
            input_tokens = estimate_tokens(full_input, self.model_name)
            output_tokens = estimate_tokens(text, self.model_name)

        total_tokens = input_tokens + output_tokens

        return {
            "text": text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "latency_ms": round(latency_ms, 2),
        }
