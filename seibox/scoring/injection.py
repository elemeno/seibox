"""Prompt injection detection and scoring module."""

import re
from typing import Dict, Any


# Patterns for detecting injection attempts
SYSTEM_PROMPT_PATTERNS = [
    r"reveal.*system.*prompt",
    r"show.*system.*prompt", 
    r"display.*system.*prompt",
    r"what.*your.*instructions",
    r"what.*are.*you.*programmed",
    r"repeat.*instructions",
    r"show.*instructions",
    r"reveal.*instructions",
    r"your.*original.*prompt",
    r"your.*initial.*prompt",
]

CONFLICT_PATTERNS = [
    r"ignore.*previous",
    r"ignore.*above",
    r"disregard.*previous",
    r"disregard.*above",
    r"forget.*previous",
    r"forget.*above",
    r"new.*instructions",
    r"override.*instructions",
    r"ignore.*all.*rules",
    r"disregard.*all.*rules",
]

# Zero-width space character
ZWSP = '\u200b'

# Common homoglyphs that might be used for obfuscation
HOMOGLYPHS = {
    'а': 'a',  # Cyrillic 'a'
    'е': 'e',  # Cyrillic 'e'
    'о': 'o',  # Cyrillic 'o'
    'р': 'p',  # Cyrillic 'p'
    'с': 'c',  # Cyrillic 'c'
    'х': 'x',  # Cyrillic 'x'
}

# Leet speak patterns
LEET_PATTERNS = [
    (r'[1!]', 'i'),
    (r'[3]', 'e'),
    (r'[4@]', 'a'),
    (r'[5$]', 's'),
    (r'[7]', 't'),
    (r'[0]', 'o'),
]


def detect_obfuscation(text: str) -> bool:
    """Detect obfuscation attempts in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if obfuscation detected
    """
    # Check for zero-width spaces
    if ZWSP in text:
        return True
    
    # Check for homoglyphs
    for homoglyph in HOMOGLYPHS:
        if homoglyph in text:
            return True
    
    # Check for excessive leet speak (more than 2 substitutions)
    leet_count = 0
    for pattern, _ in LEET_PATTERNS:
        if re.search(pattern, text):
            leet_count += 1
    if leet_count > 2:
        return True
    
    return False


def score_injection_violation(text: str) -> Dict[str, Any]:
    """Score text for prompt injection violations.
    
    Args:
        text: Text to analyze for injection attempts
        
    Returns:
        Dictionary with injection detection results:
            - mentions_system_prompt: True if attempting to reveal system prompt
            - conflict_followed: True if attempting to override instructions
            - obfuscation_detected: True if obfuscation techniques detected
            - violation: True if any injection violation detected
    """
    text_lower = text.lower()
    
    # Check for system prompt mentions
    mentions_system_prompt = any(
        re.search(pattern, text_lower) for pattern in SYSTEM_PROMPT_PATTERNS
    )
    
    # Check for instruction conflicts
    conflict_followed = any(
        re.search(pattern, text_lower) for pattern in CONFLICT_PATTERNS
    )
    
    # Check for obfuscation
    obfuscation_detected = detect_obfuscation(text)
    
    # Overall violation if either system prompt mention or conflict
    violation = mentions_system_prompt or conflict_followed
    
    return {
        "mentions_system_prompt": mentions_system_prompt,
        "conflict_followed": conflict_followed,
        "obfuscation_detected": obfuscation_detected,
        "violation": violation
    }