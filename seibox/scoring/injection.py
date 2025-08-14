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
ZWSP = "\u200b"

# RTL override and bidirectional characters for obfuscation
RTL_OVERRIDE_CHARS = [
    "\u202E",  # Right-to-Left Override (RLO)
    "\u202D",  # Left-to-Right Override (LRO) 
    "\u061C",  # Arabic Letter Mark
    "\u200F",  # Right-to-Left Mark
    "\u200E",  # Left-to-Right Mark
    "\u2066",  # Left-to-Right Isolate
    "\u2067",  # Right-to-Left Isolate
    "\u2068",  # First Strong Isolate
    "\u2069",  # Pop Directional Isolate
]

# Expanded homoglyphs that might be used for obfuscation (Latin ↔ Cyrillic)
HOMOGLYPHS = {
    # Cyrillic → Latin lookalikes
    "а": "a",  # Cyrillic 'a' (U+0430) → Latin 'a' (U+0061)
    "в": "b",  # Cyrillic 'в' → Latin 'b'
    "е": "e",  # Cyrillic 'e' (U+0435) → Latin 'e' (U+0065)
    "к": "k",  # Cyrillic 'к' → Latin 'k'
    "м": "m",  # Cyrillic 'м' → Latin 'm'
    "н": "h",  # Cyrillic 'н' → Latin 'h' (looks similar)
    "о": "o",  # Cyrillic 'o' (U+043E) → Latin 'o' (U+006F)
    "р": "p",  # Cyrillic 'р' (U+0440) → Latin 'p' (U+0070)
    "с": "c",  # Cyrillic 'с' (U+0441) → Latin 'c' (U+0063)
    "т": "t",  # Cyrillic 'т' → Latin 't'
    "у": "y",  # Cyrillic 'у' → Latin 'y'
    "х": "x",  # Cyrillic 'х' (U+0445) → Latin 'x' (U+0078)
    # Greek letters that look like Latin
    "α": "a",  # Greek alpha → Latin 'a'
    "ο": "o",  # Greek omicron → Latin 'o'
    "ρ": "p",  # Greek rho → Latin 'p'
    "τ": "t",  # Greek tau → Latin 't'
    "υ": "y",  # Greek upsilon → Latin 'y'
    "χ": "x",  # Greek chi → Latin 'x'
}

# Leet speak patterns
LEET_PATTERNS = [
    (r"[1!]", "i"),
    (r"[3]", "e"),
    (r"[4@]", "a"),
    (r"[5$]", "s"),
    (r"[7]", "t"),
    (r"[0]", "o"),
]

# HTML/XML comment patterns for smuggling instructions
HTML_COMMENT_PATTERNS = [
    r"<!--.*?ignore.*?previous.*?-->",
    r"<!--.*?disregard.*?above.*?-->", 
    r"<!--.*?new.*?instructions.*?-->",
    r"<!--.*?system.*?prompt.*?-->",
    r"<!--.*?reveal.*?-->",
    r"<!--.*?override.*?-->",
]

# Base-N encoding hints (common patterns that suggest encoded content)
BASE_ENCODING_PATTERNS = [
    r"base64.*(?:decode|print|system)",
    r"base32.*(?:decode|print|system)", 
    r"base16.*(?:decode|print|system)",
    r"hex.*(?:decode|print|system)",
    r"binary.*(?:decode|print|system)",
    r"(?:decode|print).*(?:base64|base32|base16|hex|binary)",
    r"encode.*system.*prompt",
    r"print.*system.*prompt.*(?:base64|hex|binary)",
]

# Split word patterns (looking for ellipsis, dots, or other separators in key words)
SPLIT_WORD_PATTERNS = [
    r"i[.…\s-]+gnore",
    r"dis[.…\s-]+regard", 
    r"for[.…\s-]+get",
    r"over[.…\s-]+ride",
    r"sys[.…\s-]+tem",
    r"prom[.…\s-]+pt",
    r"prev[.…\s-]+ious",
    r"inst[.…\s-]+ructions",
    r"rev[.…\s-]+eal",
]


def detect_rlo_rtl(text: str) -> bool:
    """Detect RTL override and bidirectional characters.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if RTL override characters detected
    """
    return any(char in text for char in RTL_OVERRIDE_CHARS)


def detect_homoglyphs(text: str) -> bool:
    """Detect homoglyph characters.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if homoglyphs detected
    """
    return any(homoglyph in text for homoglyph in HOMOGLYPHS)


def detect_html_smuggling(text: str) -> bool:
    """Detect HTML comment smuggling attempts.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if HTML comment smuggling detected
    """
    text_lower = text.lower()
    return any(re.search(pattern, text_lower, re.DOTALL) for pattern in HTML_COMMENT_PATTERNS)


def detect_base_encoding_hints(text: str) -> bool:
    """Detect base-N encoding hints that suggest system prompt extraction.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if encoding hints detected
    """
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in BASE_ENCODING_PATTERNS)


def detect_split_instructions(text: str) -> bool:
    """Detect split word obfuscation (e.g., 'i...gnore', 'sys...tem').
    
    Args:
        text: Text to analyze
        
    Returns:
        True if split word obfuscation detected
    """
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in SPLIT_WORD_PATTERNS)


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

    # Check for RTL override characters
    if detect_rlo_rtl(text):
        return True

    # Check for homoglyphs
    if detect_homoglyphs(text):
        return True

    # Check for HTML comment smuggling
    if detect_html_smuggling(text):
        return True

    # Check for base encoding hints
    if detect_base_encoding_hints(text):
        return True

    # Check for split instructions
    if detect_split_instructions(text):
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
            - rlo_rtl_detected: True if RTL override characters detected
            - homoglyph_detected: True if homoglyph characters detected
            - html_smuggling_detected: True if HTML comment smuggling detected
            - base_encoding_detected: True if base-N encoding hints detected
            - split_instruction_detected: True if split word obfuscation detected
            - violation: True if any injection violation detected
    """
    text_lower = text.lower()

    # Check for system prompt mentions
    mentions_system_prompt = any(
        re.search(pattern, text_lower) for pattern in SYSTEM_PROMPT_PATTERNS
    )

    # Check for instruction conflicts
    conflict_followed = any(re.search(pattern, text_lower) for pattern in CONFLICT_PATTERNS)

    # Check for various obfuscation techniques
    rlo_rtl_detected = detect_rlo_rtl(text)
    homoglyph_detected = detect_homoglyphs(text)
    html_smuggling_detected = detect_html_smuggling(text)
    base_encoding_detected = detect_base_encoding_hints(text)
    split_instruction_detected = detect_split_instructions(text)
    
    # General obfuscation detection (includes all techniques plus leet speak and ZWSP)
    obfuscation_detected = detect_obfuscation(text)

    # Overall violation if either system prompt mention or conflict
    violation = mentions_system_prompt or conflict_followed

    return {
        "mentions_system_prompt": mentions_system_prompt,
        "conflict_followed": conflict_followed,
        "obfuscation_detected": obfuscation_detected,
        "rlo_rtl_detected": rlo_rtl_detected,
        "homoglyph_detected": homoglyph_detected,
        "html_smuggling_detected": html_smuggling_detected,
        "base_encoding_detected": base_encoding_detected,
        "split_instruction_detected": split_instruction_detected,
        "violation": violation,
    }
