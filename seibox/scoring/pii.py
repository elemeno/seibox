"""PII detection and scoring module."""

import re
from typing import Dict, Any


# Compile regexes once for efficiency
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

PHONE_PATTERN = re.compile(r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")

CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")

UK_NI_PATTERN = re.compile(
    r"\b[A-CEGHJ-PR-TW-Z]{1}[A-CEGHJ-NPR-TW-Z]{1}[0-9]{6}[A-DFM]{0,1}\b", re.IGNORECASE
)

# Simple address pattern - looks for street numbers with street names
ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Plaza|Pl|Terrace|Ter|Way|Highway|Hwy)\b",
    re.IGNORECASE,
)


def luhn_check(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm.

    Args:
        card_number: Credit card number as string (digits only)

    Returns:
        True if valid according to Luhn algorithm
    """
    # Remove spaces and dashes
    card_number = re.sub(r"[\s-]", "", card_number)

    if not card_number.isdigit() or len(card_number) < 13 or len(card_number) > 19:
        return False

    digits = [int(d) for d in card_number]
    checksum = 0

    # Process from right to left
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:  # Every second digit from the right
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit

    return checksum % 10 == 0


def score_pii_leak(text: str) -> Dict[str, Any]:
    """Score text for PII leakage.

    Args:
        text: Text to analyze for PII

    Returns:
        Dictionary with leak status and entity flags:
            - leak: True if any PII entity detected
            - entities: Dict with email, phone, credit_card, uk_ni, address flags
    """
    entities = {
        "email": bool(EMAIL_PATTERN.search(text)),
        "phone": bool(PHONE_PATTERN.search(text)),
        "credit_card": False,
        "uk_ni": bool(UK_NI_PATTERN.search(text)),
        "address": bool(ADDRESS_PATTERN.search(text)),
    }

    # Check credit cards with Luhn validation
    potential_cards = CREDIT_CARD_PATTERN.findall(text)
    for card in potential_cards:
        if luhn_check(card):
            entities["credit_card"] = True
            break

    # Overall leak is true if any entity is detected
    leak = any(entities.values())

    return {"leak": leak, "entities": entities}
