"""Policy gate mitigation for pre and post processing."""

import re
from typing import Dict, Any, Tuple, Optional

from seibox.scoring import pii
from seibox.utils.policy import get_contextual_refusal, should_refuse_content


# Pre-gate keyword patterns that indicate potential policy violations
PRE_GATE_PATTERNS = [
    r"reveal.*system.*prompt",
    r"show.*system.*prompt",
    r"ignore.*previous",
    r"disregard.*instructions",
    r"disregard.*rules",
    r"override.*rules",
    r"override.*safety",
    r"tell.*guidelines",
    r"show.*guidelines",
    r"reveal.*guidelines",
    r"show.*internal.*instructions",
    r"reveal.*internal.*instructions",
    r"forget.*above",
    r"forget.*everything",
]

# Injection-specific patterns that require specialized handling
INJECTION_REFUSAL = "Sorry—I can't disclose system information. I can help with your original request in a different way instead."


def apply_pre(prompt: str) -> Tuple[str, Dict[str, Any]]:
    """Apply pre-processing policy gate.

    Args:
        prompt: The user prompt to check

    Returns:
        Tuple of (possibly-modified prompt, trace information)
    """
    trace = {
        "pre_gate_triggered": False,
        "pre_gate_patterns_matched": [],
        "prompt_risk_level": "low",
    }

    prompt_lower = prompt.lower()

    # Check for policy violation patterns
    for pattern in PRE_GATE_PATTERNS:
        if re.search(pattern, prompt_lower):
            trace["pre_gate_patterns_matched"].append(pattern)

    if trace["pre_gate_patterns_matched"]:
        trace["pre_gate_triggered"] = True
        trace["prompt_risk_level"] = "high"
        # Flag high-risk prompts but don't modify (post-processing handles refusal)

    # Check for PII in prompt (for logging purposes)
    prompt_pii = pii.score_pii_leak(prompt)
    if prompt_pii["leak"]:
        trace["prompt_contains_pii"] = True
        trace["prompt_pii_types"] = prompt_pii["entities"]
        if trace["prompt_risk_level"] == "low":
            trace["prompt_risk_level"] = "medium"

    return prompt, trace


def apply_post(text: str, original_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Apply post-processing policy gate with contextual refusals.

    Args:
        text: The generated text to check
        original_prompt: Original user prompt for context (optional)

    Returns:
        Tuple of (possibly-modified text, trace information)
    """
    trace = {
        "post_gate_triggered": False,
        "pii_detected": {},
        "refusal_type": None,
        "original_text_length": len(text),
    }

    # Check for PII leakage
    pii_result = pii.score_pii_leak(text)

    # Check for injection violation indicators in the response
    injection_indicators = [
        "system prompt",
        "system message",
        "instructions:",
        "prompt:",
        "here are my instructions",
        "my guidelines are",
        "i was told to",
    ]

    text_lower = text.lower()
    injection_detected = any(indicator in text_lower for indicator in injection_indicators)

    if should_refuse_content(text, pii_result):
        trace["post_gate_triggered"] = True
        trace["pii_detected"] = pii_result["entities"]
        trace["refusal_type"] = "pii_protection"

        # Get contextual refusal based on detected PII types
        refusal_message = get_contextual_refusal("general", pii_result["entities"])
        text = refusal_message

    elif injection_detected:
        trace["post_gate_triggered"] = True
        trace["refusal_type"] = "injection_protection"
        text = INJECTION_REFUSAL

    # Add refusal context to trace
    if trace["post_gate_triggered"]:
        trace["refused_text_preview"] = text[:100] + "..." if len(text) > 100 else text

    return text, trace
