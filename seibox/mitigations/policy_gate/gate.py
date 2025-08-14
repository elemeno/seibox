"""Policy gate mitigation for pre and post processing."""

import re
from typing import Dict, Any, Tuple

from seibox.scoring import pii


# Pre-gate keyword patterns that indicate potential policy violations
PRE_GATE_PATTERNS = [
    r"reveal.*system.*prompt",
    r"show.*system.*prompt",
    r"ignore.*previous",
    r"disregard.*instructions",
    r"override.*rules",
]

# Standard refusal message for policy violations
REFUSAL_MESSAGE = "Sorry—I can't disclose personal data. I can summarize without identifiers or help with next steps."


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
    }

    prompt_lower = prompt.lower()

    # Check for policy violation patterns
    for pattern in PRE_GATE_PATTERNS:
        if re.search(pattern, prompt_lower):
            trace["pre_gate_patterns_matched"].append(pattern)

    if trace["pre_gate_patterns_matched"]:
        trace["pre_gate_triggered"] = True
        # Could modify prompt here, but for v0.1.0 we just flag it

    return prompt, trace


def apply_post(text: str) -> Tuple[str, Dict[str, Any]]:
    """Apply post-processing policy gate.

    Args:
        text: The generated text to check

    Returns:
        Tuple of (possibly-modified text, trace information)
    """
    trace = {
        "post_gate_triggered": False,
        "pii_detected": {},
    }

    # Check for PII leakage
    pii_result = pii.score_pii_leak(text)

    if pii_result["leak"]:
        trace["post_gate_triggered"] = True
        trace["pii_detected"] = pii_result["entities"]
        # Replace the response with a safe refusal
        text = REFUSAL_MESSAGE

    return text, trace
