"""Configuration validation with friendly error messages."""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import difflib


class ConfigValidationError(Exception):
    """Custom exception for configuration errors with helpful suggestions."""
    
    def __init__(self, message: str, path: str = "", suggestions: List[str] = None):
        """Initialize config validation error.
        
        Args:
            message: Error message
            path: Path to the problematic key (e.g., "datasets.pii.sampling")
            suggestions: List of valid alternatives
        """
        self.path = path
        self.suggestions = suggestions or []
        
        # Build detailed error message
        full_message = f"\n❌ Configuration Error"
        if path:
            full_message += f" at '{path}'"
        full_message += f":\n   {message}"
        
        if self.suggestions:
            full_message += "\n\n   Did you mean one of these?"
            for suggestion in self.suggestions[:3]:  # Show top 3 suggestions
                full_message += f"\n   • {suggestion}"
        
        super().__init__(full_message)


def find_similar_keys(invalid_key: str, valid_keys: List[str], threshold: float = 0.6) -> List[str]:
    """Find similar keys using fuzzy matching.
    
    Args:
        invalid_key: The key that was not found
        valid_keys: List of valid keys
        threshold: Minimum similarity ratio (0-1)
        
    Returns:
        List of similar keys sorted by similarity
    """
    matches = []
    for valid_key in valid_keys:
        ratio = difflib.SequenceMatcher(None, invalid_key.lower(), valid_key.lower()).ratio()
        if ratio >= threshold:
            matches.append((valid_key, ratio))
    
    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)
    return [key for key, _ in matches]


def validate_config_keys(config: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> None:
    """Recursively validate configuration keys against a schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: Expected schema with valid keys
        path: Current path in the configuration (for error messages)
        
    Raises:
        ConfigValidationError: If invalid keys are found
    """
    for key, value in config.items():
        current_path = f"{path}.{key}" if path else key
        
        if key not in schema:
            # Find similar valid keys
            valid_keys = list(schema.keys())
            suggestions = find_similar_keys(key, valid_keys)
            
            # Create helpful error message
            if suggestions:
                raise ConfigValidationError(
                    f"Unknown configuration key '{key}'",
                    path=current_path,
                    suggestions=[f"{path}.{s}" if path else s for s in suggestions]
                )
            else:
                raise ConfigValidationError(
                    f"Unknown configuration key '{key}'. Valid keys are: {', '.join(valid_keys)}",
                    path=current_path
                )
        
        # Recursively validate nested dictionaries
        if isinstance(value, dict) and isinstance(schema[key], dict):
            validate_config_keys(value, schema[key], current_path)


def validate_required_keys(config: Dict[str, Any], required: List[str], path: str = "") -> None:
    """Check that all required keys are present.
    
    Args:
        config: Configuration dictionary
        required: List of required keys
        path: Current path in the configuration
        
    Raises:
        ConfigValidationError: If required keys are missing
    """
    missing = []
    for key in required:
        if key not in config:
            missing.append(key)
    
    if missing:
        raise ConfigValidationError(
            f"Missing required configuration keys: {', '.join(missing)}",
            path=path
        )


def validate_value_type(value: Any, expected_type: type, key: str, path: str = "") -> None:
    """Validate that a value has the expected type.
    
    Args:
        value: Value to check
        expected_type: Expected type
        key: Key name
        path: Current path
        
    Raises:
        ConfigValidationError: If type is incorrect
    """
    if not isinstance(value, expected_type):
        actual_type = type(value).__name__
        expected_name = expected_type.__name__
        
        raise ConfigValidationError(
            f"Invalid type for '{key}': expected {expected_name}, got {actual_type}",
            path=f"{path}.{key}" if path else key
        )


def validate_eval_config(config: Dict[str, Any]) -> None:
    """Validate evaluation configuration with helpful error messages.
    
    Args:
        config: Configuration dictionary from YAML
        
    Raises:
        ConfigValidationError: If configuration is invalid
    """
    # Define the expected schema
    schema = {
        "run": {
            "temperature": float,
            "max_tokens": int,
            "rate_limit": int,
            "cache": bool,
            "retry": {
                "max_attempts": int,
                "backoff_base": float,
            }
        },
        "datasets": {
            "pii": {
                "sampling": {
                    "n": int,
                },
                "source": str,
            },
            "injection": {
                "sampling": {
                    "n": int,
                },
                "source": str,
            },
            "benign": {
                "sampling": {
                    "n": int,
                },
                "source": str,
            }
        },
        "scoring": {
            "pii": {
                "detect_email": bool,
                "detect_phone": bool,
                "detect_credit_card": bool,
                "detect_uk_ni": bool,
                "detect_address": bool,
            },
            "injection": {
                "detect_system_prompt": bool,
                "detect_conflicts": bool,
                "detect_obfuscation": bool,
            },
            "benign": {
                "detect_refusals": bool,
                "llm_quality_proxy": {
                    "enabled": bool,
                    "temperature": float,
                    "max_tokens": int,
                }
            }
        },
        "mitigations": {
            "available": list,
        },
        "reporting": {
            "html_path": str,
            "dashboard_enabled": bool,
            "include_examples": bool,
        }
    }
    
    # Check required top-level keys
    required_top = ["run", "datasets", "scoring"]
    validate_required_keys(config, required_top)
    
    # Validate against schema
    validate_config_keys(config, schema)
    
    # Type validation for common fields
    if "run" in config:
        run_config = config["run"]
        if "temperature" in run_config:
            validate_value_type(run_config["temperature"], (int, float), "temperature", "run")
            if not 0 <= run_config["temperature"] <= 2:
                raise ConfigValidationError(
                    f"Temperature must be between 0 and 2, got {run_config['temperature']}",
                    path="run.temperature"
                )
        
        if "max_tokens" in run_config:
            validate_value_type(run_config["max_tokens"], int, "max_tokens", "run")
            if run_config["max_tokens"] <= 0:
                raise ConfigValidationError(
                    f"max_tokens must be positive, got {run_config['max_tokens']}",
                    path="run.max_tokens"
                )
    
    # Validate dataset configurations
    if "datasets" in config:
        for dataset_name, dataset_config in config["datasets"].items():
            if isinstance(dataset_config, dict) and "sampling" in dataset_config:
                sampling = dataset_config["sampling"]
                if "n" in sampling:
                    validate_value_type(sampling["n"], int, "n", f"datasets.{dataset_name}.sampling")
                    if sampling["n"] <= 0:
                        raise ConfigValidationError(
                            f"Sample size 'n' must be positive, got {sampling['n']}",
                            path=f"datasets.{dataset_name}.sampling.n"
                        )


def get_config_schema_info() -> str:
    """Get a formatted string describing the expected configuration schema.
    
    Returns:
        Formatted schema information
    """
    return """
Expected configuration structure:

run:
  temperature: <float>      # 0.0 to 2.0
  max_tokens: <int>         # Positive integer
  rate_limit: <int>         # Calls per second
  cache: <bool>             # true/false
  retry:
    max_attempts: <int>     # Positive integer
    backoff_factor: <float> # Multiplier for retries

datasets:
  pii/injection/benign:
    sampling:
      n: <int>              # Number of samples
      seed: <int>           # Random seed
    source: <string>        # Data source

scoring:
  pii:
    strict_mode: <bool>
    entities: <list>
  injection:
    detect_obfuscation: <bool>
    patterns: <list>
  benign:
    refusal_detection: <bool>
    allow_patterns: <list>

mitigations: <list>         # List of mitigation IDs

reporting:
  html_path: <string>       # Output path for HTML report
  dashboard: <bool>         # Enable dashboard
  include_examples: <bool>  # Include examples in report
"""