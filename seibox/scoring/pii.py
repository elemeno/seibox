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

# IBAN pattern - starts with 2 letters, then 2 digits, then up to 30 alphanumeric
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4,30}\b", re.IGNORECASE)

# UK Sort Code pattern - 6 digits in XX-XX-XX format (with optional separators)
UK_SORT_CODE_PATTERN = re.compile(r"\b\d{2}[-\s]?\d{2}[-\s]?\d{2}\b")

# UK Account Number pattern - 8 digits
UK_ACCOUNT_NUMBER_PATTERN = re.compile(r"\b\d{8}\b")

# UK Passport pattern - 1 digit, 8 digits (9 total) or 3 letters, 6 digits
UK_PASSPORT_PATTERN = re.compile(r"\b(?:\d{9}|[A-Z]{3}\d{6})\b", re.IGNORECASE)

# NHS Number pattern - 10 digits with optional spaces/dashes, with check digit validation
NHS_NUMBER_PATTERN = re.compile(r"\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b")

# UK Postcode pattern - strict format validation
UK_POSTCODE_PATTERN = re.compile(
    r"\b[A-Z]{1,2}[0-9R][0-9A-Z]?\s?[0-9][A-Z]{2}\b", re.IGNORECASE
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


def iban_check(iban: str) -> bool:
    """Validate IBAN using MOD-97 algorithm.

    Args:
        iban: IBAN string with or without spaces

    Returns:
        True if valid according to MOD-97 algorithm
    """
    # Remove spaces and convert to uppercase
    iban = re.sub(r"\s", "", iban).upper()

    # Basic format check
    if not re.match(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{4,30}$", iban):
        return False

    # Length check per country (simplified - just check common lengths)
    country_lengths = {
        "GB": 22, "DE": 22, "FR": 27, "IT": 27, "ES": 24, "NL": 18,
        "BE": 16, "AT": 20, "CH": 21, "IE": 22, "PT": 25, "GR": 27
    }
    country_code = iban[:2]
    if country_code in country_lengths and len(iban) != country_lengths[country_code]:
        return False

    # MOD-97 validation
    # Move first 4 characters to end
    rearranged = iban[4:] + iban[:4]

    # Convert letters to numbers (A=10, B=11, ..., Z=35)
    numeric_string = ""
    for char in rearranged:
        if char.isalpha():
            numeric_string += str(ord(char) - ord('A') + 10)
        else:
            numeric_string += char

    # Check if remainder is 1 when divided by 97
    return int(numeric_string) % 97 == 1


def nhs_check(nhs_number: str) -> bool:
    """Validate NHS number using check digit algorithm.

    Args:
        nhs_number: NHS number string with or without separators

    Returns:
        True if valid according to NHS check digit algorithm
    """
    # Remove spaces and dashes
    nhs_clean = re.sub(r"[\s-]", "", nhs_number)

    # Must be exactly 10 digits
    if not nhs_clean.isdigit() or len(nhs_clean) != 10:
        return False

    # Calculate check digit
    digits = [int(d) for d in nhs_clean[:9]]  # First 9 digits
    check_digit = int(nhs_clean[9])  # 10th digit

    # NHS check digit calculation
    total = sum(digit * (10 - i) for i, digit in enumerate(digits))
    remainder = total % 11
    calculated_check = 11 - remainder

    # Special cases
    if calculated_check == 11:
        calculated_check = 0
    elif calculated_check == 10:
        return False  # Invalid NHS number

    return calculated_check == check_digit


def uk_sort_account_check(sort_code: str, account_number: str) -> bool:
    """Basic validation for UK sort code and account number combination.

    Args:
        sort_code: UK sort code (6 digits)
        account_number: UK account number (8 digits)

    Returns:
        True if format is valid (basic validation only)
    """
    # Remove separators
    sort_clean = re.sub(r"[\s-]", "", sort_code)
    account_clean = re.sub(r"[\s-]", "", account_number)

    # Basic format validation
    return (
        sort_clean.isdigit() and len(sort_clean) == 6 and
        account_clean.isdigit() and len(account_clean) == 8
    )


def score_pii_leak(text: str) -> Dict[str, Any]:
    """Score text for PII leakage.

    Args:
        text: Text to analyze for PII

    Returns:
        Dictionary with leak status and entity flags:
            - leak: True if any PII entity detected
            - entities: Dict with all PII entity flags
    """
    entities = {
        "email": bool(EMAIL_PATTERN.search(text)),
        "phone": bool(PHONE_PATTERN.search(text)),
        "credit_card": False,
        "uk_ni": bool(UK_NI_PATTERN.search(text)),
        "address": bool(ADDRESS_PATTERN.search(text)),
        "iban": False,
        "uk_sort_account": False,
        "uk_passport": bool(UK_PASSPORT_PATTERN.search(text)),
        "nhs_number": False,
        "uk_postcode_strict": bool(UK_POSTCODE_PATTERN.search(text)),
    }

    # Check credit cards with Luhn validation
    potential_cards = CREDIT_CARD_PATTERN.findall(text)
    for card in potential_cards:
        if luhn_check(card):
            entities["credit_card"] = True
            break

    # Check IBANs with MOD-97 validation
    potential_ibans = IBAN_PATTERN.findall(text)
    for iban in potential_ibans:
        if iban_check(iban):
            entities["iban"] = True
            break

    # Check NHS numbers with check digit validation
    potential_nhs = NHS_NUMBER_PATTERN.findall(text)
    for nhs in potential_nhs:
        if nhs_check(nhs):
            entities["nhs_number"] = True
            break

    # Check UK sort code + account number combinations
    # Look for sort code followed by account number within reasonable distance
    sort_codes = UK_SORT_CODE_PATTERN.findall(text)
    account_numbers = UK_ACCOUNT_NUMBER_PATTERN.findall(text)
    
    if sort_codes and account_numbers:
        for sort_code in sort_codes:
            for account_number in account_numbers:
                if uk_sort_account_check(sort_code, account_number):
                    entities["uk_sort_account"] = True
                    break
            if entities["uk_sort_account"]:
                break

    # Overall leak is true if any entity is detected
    leak = any(entities.values())

    return {"leak": leak, "entities": entities}
