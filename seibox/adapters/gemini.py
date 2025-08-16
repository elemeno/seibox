"""Google Gemini adapter implementation."""

import os
import time
from typing import Any, Mapping, Optional

from dotenv import load_dotenv


class GeminiAdapter:
    """Adapter for Google Gemini models."""

    def __init__(self, model_name: str = "gemini-2.0-flash-001"):
        """Initialize the Gemini adapter.

        Args:
            model_name: Gemini model name (e.g., gemini-2.0-flash-001, gemini-1.5-pro)
        """
        self.name = f"gemini:{model_name}"
        self.model_name = model_name
        self._client = None
        self._use_vertexai = False

    def _get_client(self):
        """Lazily initialize the Gemini client.

        Returns:
            Google GenAI client instance

        Raises:
            ImportError: If google-genai package is not installed
            ValueError: If API key or project configuration is not set
        """
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError("Google GenAI package not installed. Run: pip install google-genai")

            load_dotenv()

            # Check for Vertex AI configuration first
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")

            if use_vertexai and project_id:
                # Use Vertex AI
                self._client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location
                )
                self._use_vertexai = True
            else:
                # Use Gemini Developer API
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GEMINI_API_KEY not found in environment. "
                        "Set it in .env file or environment variables. "
                        "Alternatively, set GOOGLE_CLOUD_PROJECT and GOOGLE_GENAI_USE_VERTEXAI=true for Vertex AI."
                    )

                self._client = genai.Client(api_key=api_key)
                self._use_vertexai = False

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
        """Generate a completion from Gemini.

        Args:
            system: Optional system prompt
            prompt: User prompt
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            extra: Additional Gemini-specific parameters

        Returns:
            Dictionary with text, usage, and latency_ms
        """
        client = self._get_client()

        # Build the content - Gemini uses a single content string with system instruction separate
        content = prompt

        start_time = time.time()

        # Prepare generation config
        try:
            from google.genai import types

            config_params = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }

            # Add system instruction if provided
            if system:
                config_params["system_instruction"] = system

            # Add any extra parameters
            if extra:
                config_params.update(extra)

            config = types.GenerateContentConfig(**config_params)

            # Generate content
            response = client.models.generate_content(
                model=self.model_name,
                contents=content,
                config=config,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract text from response
            text_content = response.text if hasattr(response, 'text') else ""

            # Extract token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_metadata = response.usage_metadata
                input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
                total_tokens = getattr(usage_metadata, 'total_token_count', input_tokens + output_tokens)
            else:
                # Estimate tokens if Gemini doesn't provide usage info
                from seibox.utils.tokens import estimate_tokens

                # Estimate input tokens (prompt + system)
                full_input = f"{system or ''}\n{prompt}".strip()
                input_tokens = estimate_tokens(full_input, "gpt-4")  # Use GPT-4 tokenizer as approximation
                output_tokens = estimate_tokens(text_content, "gpt-4")
                total_tokens = input_tokens + output_tokens

            return {
                "text": text_content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                "latency_ms": round(latency_ms, 2),
            }

        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Gemini API call failed: {str(e)}") from e