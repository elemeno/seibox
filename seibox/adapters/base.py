"""Base protocol for model adapters."""

from typing import Any, Mapping, Optional, Protocol


class ModelAdapter(Protocol):
    """Protocol defining the interface for model adapters."""

    name: str

    def complete(
        self,
        *,
        system: Optional[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """Generate a completion from the model.

        Args:
            system: Optional system prompt
            prompt: User prompt
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            extra: Additional provider-specific parameters

        Returns:
            Dictionary with keys:
                - text: Generated text
                - usage: Dict with input_tokens, output_tokens, total_tokens
                - latency_ms: Latency in milliseconds
        """
        ...
