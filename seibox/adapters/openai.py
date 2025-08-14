"""OpenAI adapter implementation."""

import os
import time
from typing import Any, Mapping, Optional

from dotenv import load_dotenv


class OpenAIAdapter:
    """Adapter for OpenAI models."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the OpenAI adapter.

        Args:
            model_name: OpenAI model name (e.g., gpt-4o-mini, gpt-4)
        """
        self.name = f"openai:{model_name}"
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI client.

        Returns:
            OpenAI client instance

        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not set
        """
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment. "
                    "Set it in .env file or environment variables."
                )

            self._client = openai.OpenAI(api_key=api_key)

        return self._client

    def complete(
        self,
        *,
        system: Optional[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """Generate a completion from OpenAI.

        Args:
            system: Optional system prompt
            prompt: User prompt
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            extra: Additional OpenAI-specific parameters

        Returns:
            Dictionary with text, usage, and latency_ms
        """
        client = self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if extra:
            kwargs.update(extra)

        response = client.chat.completions.create(**kwargs)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "text": response.choices[0].message.content or "",
            "usage": {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            "latency_ms": round(latency_ms, 2),
        }
