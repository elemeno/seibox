"""Token counting utilities with fallback estimation methods."""

import re
from typing import Optional


def estimate_tokens_tiktoken(text: str, model_name: str = "gpt-3.5-turbo") -> Optional[int]:
    """Estimate tokens using tiktoken library.
    
    Args:
        text: Text to count tokens for
        model_name: Model name for tiktoken encoding selection
        
    Returns:
        Token count if tiktoken is available, None otherwise
    """
    try:
        import tiktoken
        
        # Map model names to tiktoken encodings
        encoding_map = {
            # OpenAI models
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base", 
            "gpt-4o": "o200k_base",
            "gpt-4o-mini": "o200k_base",
            "gpt-4.1": "o200k_base",
            "gpt-4.1-mini": "o200k_base",
            "gpt-4.1-nano": "o200k_base",
            "gpt-5": "o200k_base",
            "gpt-5-mini": "o200k_base", 
            "gpt-5-nano": "o200k_base",
            "gpt-3.5-turbo": "cl100k_base",
            # Default encoding
            "default": "cl100k_base"
        }
        
        # Extract base model name (remove provider prefix)
        base_model = model_name
        if ":" in model_name:
            base_model = model_name.split(":", 1)[1]
        
        # Get encoding name
        encoding_name = encoding_map.get(base_model, encoding_map["default"])
        
        # Get encoding and count tokens
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
        
    except ImportError:
        # tiktoken not available
        return None
    except Exception:
        # Any other error with tiktoken
        return None


def estimate_tokens_char_based(text: str, model_name: str = None) -> int:
    """Character-based token estimation fallback.
    
    Args:
        text: Text to estimate tokens for
        model_name: Model name (unused but kept for API consistency)
        
    Returns:
        Estimated token count based on character analysis
    """
    if not text:
        return 0
    
    # More sophisticated character-based estimation
    # Based on empirical analysis of various models
    
    # Remove extra whitespace and normalize
    normalized_text = re.sub(r'\s+', ' ', text.strip())
    
    # Count different types of tokens
    char_count = len(normalized_text)
    word_count = len(normalized_text.split())
    
    # Special token patterns that typically count as single tokens
    punctuation_count = len(re.findall(r'[.,!?;:()\[\]{}"\'-]', normalized_text))
    number_count = len(re.findall(r'\b\d+\b', normalized_text))
    
    # Estimate based on multiple factors
    # - Base estimate: ~4 characters per token (common rule of thumb)
    # - Word-based estimate: ~0.75 tokens per word on average
    # - Account for punctuation and numbers as separate tokens
    
    char_based_estimate = max(1, char_count // 4)
    word_based_estimate = max(1, int(word_count * 0.75))
    
    # Use the average of char and word estimates, plus punctuation/numbers
    base_estimate = (char_based_estimate + word_based_estimate) // 2
    special_tokens = (punctuation_count + number_count) // 2
    
    final_estimate = max(1, base_estimate + special_tokens)
    
    return final_estimate


def estimate_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Estimate token count with fallback methods.
    
    Tries tiktoken first, falls back to character-based estimation.
    
    Args:
        text: Text to count tokens for
        model_name: Model name for encoding selection
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Try tiktoken first
    tiktoken_count = estimate_tokens_tiktoken(text, model_name)
    if tiktoken_count is not None:
        return tiktoken_count
    
    # Fall back to character-based estimation
    return estimate_tokens_char_based(text, model_name)


def estimate_tokens_for_messages(messages: list, model_name: str = "gpt-3.5-turbo") -> int:
    """Estimate tokens for a list of chat messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model_name: Model name for encoding selection
        
    Returns:
        Estimated token count including message formatting overhead
    """
    if not messages:
        return 0
    
    total_tokens = 0
    
    # Account for message formatting overhead
    # Each message has some overhead for role and structure
    message_overhead_per_message = 3  # Approximate tokens per message for formatting
    
    for message in messages:
        content = message.get("content", "")
        role = message.get("role", "")
        
        # Count tokens for content and role
        content_tokens = estimate_tokens(content, model_name)
        role_tokens = estimate_tokens(role, model_name)
        
        total_tokens += content_tokens + role_tokens + message_overhead_per_message
    
    # Add some overhead for the conversation structure
    conversation_overhead = 3
    
    return max(1, total_tokens + conversation_overhead)