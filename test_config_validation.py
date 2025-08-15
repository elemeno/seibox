#!/usr/bin/env python3
"""Test configuration validation with friendly errors."""

from seibox.runners.eval_runner import load_config
from seibox.utils.config_validation import ConfigValidationError

# Test 1: Config with typo
print("Test 1: Loading config with typos...")
print("-" * 50)
try:
    config = load_config("test_config_error.yaml")
    print("Config loaded successfully (unexpected!)")
except ConfigValidationError as e:
    print("✅ Caught ConfigValidationError:")
    print(str(e))
except Exception as e:
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")

print("\n" + "=" * 50 + "\n")

# Test 2: Try a different typo
print("Test 2: Testing specific key typo...")
print("-" * 50)

from seibox.utils.config_validation import find_similar_keys, validate_config_keys

# Test fuzzy matching
test_keys = ["temperature", "max_tokens", "rate_limit", "cache", "retry"]
typo = "temprature"

matches = find_similar_keys(typo, test_keys)
print(f"Typo: '{typo}'")
print(f"Similar keys found: {matches}")

print("\n" + "=" * 50 + "\n")

# Test 3: Validate nested config
print("Test 3: Testing nested config validation...")
print("-" * 50)

test_config = {
    "run": {
        "temperature": 0.0,
        "max_token": 512,  # Typo: should be max_tokens
    }
}

schema = {
    "run": {
        "temperature": float,
        "max_tokens": int,
        "rate_limit": int,
    }
}

try:
    validate_config_keys(test_config, schema)
    print("Config validated (unexpected!)")
except ConfigValidationError as e:
    print("✅ Caught nested key error:")
    print(str(e))