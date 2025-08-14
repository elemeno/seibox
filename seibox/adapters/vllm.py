"""vLLM adapter for local model inference via HTTP API."""

import os
import time
import json
from typing import Any, Mapping, Optional
import requests
from dotenv import load_dotenv


class VLLMAdapter:
    """Adapter for local vLLM server via HTTP API."""

    def __init__(self, model_name: str = "default", base_url: str = None):
        """Initialize the vLLM adapter.

        Args:
            model_name: Model name for identification (can be any string)
            base_url: Base URL of the vLLM server (e.g., http://localhost:8000)
        """
        self.name = f"vllm:{model_name}"
        self.model_name = model_name
        
        # Load environment variables
        load_dotenv()
        
        # Use provided base_url or fall back to environment variable or default
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        
        # Ensure base URL doesn't end with slash
        self.base_url = self.base_url.rstrip("/")
        
        # Construct the completions endpoint
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_url = f"{self.base_url}/v1/chat/completions"

    def complete(
        self,
        *,
        system: Optional[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """Generate a completion from vLLM server.

        Args:
            system: Optional system prompt
            prompt: User prompt
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            extra: Additional vLLM-specific parameters

        Returns:
            Dictionary with text, usage, and latency_ms
        """
        from seibox.utils.tokens import estimate_tokens

        start_time = time.time()

        # Prepare the request payload
        # If we have a system prompt, use chat completions format, otherwise use completions
        if system:
            # Use chat completions format
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            
            # Add any extra parameters
            if extra:
                extra_filtered = {k: v for k, v in extra.items() if v is not None}
                payload.update(extra_filtered)
            
            url = self.chat_url
            
        else:
            # Use completions format
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            
            # Add any extra parameters
            if extra:
                extra_filtered = {k: v for k, v in extra.items() if v is not None}
                payload.update(extra_filtered)
            
            url = self.completions_url

        try:
            # Make the HTTP request
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            
            result = response.json()
            latency_ms = (time.time() - start_time) * 1000

            # Extract text and usage from response
            if system:  # Chat completions format
                text = ""
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        text = choice["message"]["content"] or ""
            else:  # Completions format
                text = ""
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    text = choice.get("text", "")

            # Extract usage information if provided by vLLM
            usage = result.get("usage", {})
            
            # Calculate token counts - use vLLM's counts if available, otherwise estimate
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                input_tokens = usage["prompt_tokens"]
                output_tokens = usage["completion_tokens"]
                total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            else:
                # Estimate tokens if vLLM doesn't provide them
                full_prompt = f"{system}\n{prompt}" if system else prompt
                input_tokens = estimate_tokens(full_prompt, self.model_name)
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

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"vLLM server request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from vLLM server: {e}") from e
        except Exception as e:
            raise RuntimeError(f"vLLM adapter error: {e}") from e