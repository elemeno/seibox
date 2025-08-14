"""Prompt hardening utilities for system prompt injection."""

from pathlib import Path
from typing import Optional

import yaml


def load_templates() -> dict:
    """Load prompt hardening templates from YAML file.
    
    Returns:
        Dictionary of template configurations
    """
    template_path = Path(__file__).parent / "templates.yaml"
    
    with open(template_path, "r") as f:
        return yaml.safe_load(f)


def get_system_prompt(template_name: str = "default_v0") -> Optional[str]:
    """Get a system prompt template by name.
    
    Args:
        template_name: Name of the template to retrieve
        
    Returns:
        System prompt string, or None if template not found
    """
    templates = load_templates()
    
    if template_name in templates:
        return templates[template_name].get("system_prompt")
    
    return None