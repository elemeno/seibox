"""Cost accounting utilities for token usage and pricing."""

from typing import Any, Dict, Optional

import yaml


def load_cost_table(config_path: str = "configs/models.yaml") -> Dict[str, Dict[str, float]]:
    """Load the cost table from the models configuration.
    
    Args:
        config_path: Path to the models configuration file
        
    Returns:
        Dictionary mapping model names to cost information
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config format is invalid
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Return default cost table if config doesn't exist
        return {
            "openai:gpt-4o-mini": {
                "input_per_1k": 0.005,
                "output_per_1k": 0.015,
            }
        }
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}")
    
    cost_table = {}
    for model in config.get("models", []):
        if "cost" in model:
            cost_table[model["name"]] = model["cost"]
    
    return cost_table


def compute_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cost_table: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Compute the cost for a completion.
    
    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_table: Optional cost table (will load default if not provided)
        
    Returns:
        Dictionary with input_tokens, output_tokens, and usd cost
    """
    if cost_table is None:
        cost_table = load_cost_table()
    
    result = {
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "usd": 0.0,
    }
    
    if model_name in cost_table:
        costs = cost_table[model_name]
        input_cost = (input_tokens / 1000.0) * costs.get("input_per_1k", 0)
        output_cost = (output_tokens / 1000.0) * costs.get("output_per_1k", 0)
        result["usd"] = round(input_cost + output_cost, 6)
    
    return result


def estimate_tokens(text: str) -> int:
    """Simple fallback token estimation.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count (roughly 4 chars per token)
    """
    # Simple heuristic: ~4 characters per token on average
    return max(1, len(text) // 4)