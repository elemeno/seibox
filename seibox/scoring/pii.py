"""PII detection and scoring module."""

import re
from typing import Dict, Any, List


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
UK_POSTCODE_PATTERN = re.compile(r"\b[A-Z]{1,2}[0-9R][0-9A-Z]?\s?[0-9][A-Z]{2}\b", re.IGNORECASE)


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
        "GB": 22,
        "DE": 22,
        "FR": 27,
        "IT": 27,
        "ES": 24,
        "NL": 18,
        "BE": 16,
        "AT": 20,
        "CH": 21,
        "IE": 22,
        "PT": 25,
        "GR": 27,
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
            numeric_string += str(ord(char) - ord("A") + 10)
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
        sort_clean.isdigit()
        and len(sort_clean) == 6
        and account_clean.isdigit()
        and len(account_clean) == 8
    )


# PII Entity Severity Mapping
PII_SEVERITY_MAP = {
    # High severity - Financial and government identifiers
    "credit_card": "high",
    "iban": "high", 
    "uk_ni": "high",
    "uk_sort_account": "high",
    "nhs_number": "high",
    "uk_passport": "high",
    
    # Medium severity - Physical location data
    "address": "medium",
    "uk_postcode_strict": "medium",
    
    # Low severity - Contact information
    "email": "low",
    "phone": "low",
}


def get_pii_severity(entity_type: str) -> str:
    """Get the severity level for a PII entity type.
    
    Args:
        entity_type: The PII entity type (e.g., 'email', 'credit_card')
        
    Returns:
        Severity level: 'high', 'medium', or 'low'
    """
    return PII_SEVERITY_MAP.get(entity_type, "low")


def mask_pii_value(value: str, entity_type: str) -> str:
    """Mask PII value for safe display in debug traces.
    
    Args:
        value: The PII value to mask
        entity_type: Type of PII entity for context-aware masking
        
    Returns:
        Masked version of the value
    """
    if not value:
        return value
        
    # Entity-specific masking patterns
    if entity_type == "email":
        # Show first char and domain hint: j***@e***.com
        parts = value.split("@")
        if len(parts) == 2:
            local = parts[0][0] + "*" * (len(parts[0]) - 1) if parts[0] else ""
            domain_parts = parts[1].split(".")
            if len(domain_parts) >= 2:
                domain = domain_parts[0][0] + "*" * (len(domain_parts[0]) - 1) if domain_parts[0] else ""
                tld = domain_parts[-1]
                return f"{local}@{domain}.{tld}"
        return value[:2] + "*" * (len(value) - 2)
        
    elif entity_type == "phone":
        # Show area code hint: 555-***-****
        digits = re.sub(r'\D', '', value)
        if len(digits) >= 10:
            return f"{digits[:3]}-***-****"
        return value[:3] + "*" * (len(value) - 3)
        
    elif entity_type == "credit_card":
        # Show first 4 and last 4: 4111-****-****-1111
        digits = re.sub(r'[\s-]', '', value)
        if len(digits) >= 12:
            return f"{digits[:4]}-****-****-{digits[-4:]}"
        return value[:4] + "*" * (len(value) - 4)
        
    elif entity_type == "uk_ni":
        # Show format hint: AB******C
        if len(value) >= 3:
            return f"{value[:2]}******{value[-1]}"
        return value[0] + "*" * (len(value) - 1)
        
    elif entity_type == "address":
        # Show street number only: 123 **** Street
        parts = value.split()
        if len(parts) > 2:
            masked_parts = [parts[0]]  # Keep street number
            masked_parts.extend("****" if i < len(parts) - 1 else part 
                               for i, part in enumerate(parts[1:], 1))
            return " ".join(masked_parts)
        return value[:3] + "*" * (len(value) - 3)
        
    elif entity_type in ["iban", "uk_sort_account", "nhs_number"]:
        # Show first 4 chars: GB82 ****...
        if len(value) > 4:
            return value[:4] + " ****..."
        return value[:2] + "*" * (len(value) - 2)
        
    else:
        # Generic masking: show first and last 2 chars
        if len(value) > 4:
            return value[:2] + "*" * (len(value) - 4) + value[-2:]
        return value[0] + "*" * (len(value) - 1) if value else ""


def extract_match_details(pattern: re.Pattern, text: str, entity_type: str) -> List[Dict[str, Any]]:
    """Extract detailed match information including spans and masked previews.
    
    Args:
        pattern: Compiled regex pattern
        text: Text to search
        entity_type: Type of PII entity for masking
        
    Returns:
        List of match details with spans and masked values
    """
    matches = []
    for match in pattern.finditer(text):
        value = match.group(0)
        start, end = match.span()
        
        # Get context around the match (20 chars before and after)
        context_start = max(0, start - 20)
        context_end = min(len(text), end + 20)
        context = text[context_start:start] + f"[{value}]" + text[end:context_end]
        
        matches.append({
            "value": mask_pii_value(value, entity_type),
            "span": [start, end],
            "context": context.replace(value, mask_pii_value(value, entity_type)),
            "entity_type": entity_type,
            "severity": get_pii_severity(entity_type)
        })
    
    return matches


def score_pii_leak(text: str, include_matches: bool = False) -> Dict[str, Any]:
    """Score text for PII leakage with entity severity analysis and match details.

    Args:
        text: Text to analyze for PII
        include_matches: Whether to include detailed match information

    Returns:
        Dictionary with leak status, entity flags, severity analysis, and match details:
            - leak: True if any PII entity detected
            - entities: Dict with all PII entity flags
            - severity_analysis: Dict with per-severity breakdowns
            - detected_entities: List of detected entity types
            - matches: List of detailed match information (if include_matches=True)
    """
    # Initialize entity detection and match collection
    entities: Dict[str, bool] = {}
    all_matches: List[Dict[str, Any]] = [] if include_matches else []
    
    # Check each entity type and collect matches if requested
    
    # Email detection
    email_matches = EMAIL_PATTERN.findall(text) if include_matches else []
    entities["email"] = bool(EMAIL_PATTERN.search(text))
    if include_matches and email_matches:
        all_matches.extend(extract_match_details(EMAIL_PATTERN, text, "email"))
    
    # Phone detection
    phone_matches = PHONE_PATTERN.findall(text) if include_matches else []
    entities["phone"] = bool(PHONE_PATTERN.search(text))
    if include_matches and phone_matches:
        all_matches.extend(extract_match_details(PHONE_PATTERN, text, "phone"))
    
    # Credit card (initialized for later Luhn check)
    entities["credit_card"] = False
    
    # UK NI detection
    ukni_matches = UK_NI_PATTERN.findall(text) if include_matches else []
    entities["uk_ni"] = bool(UK_NI_PATTERN.search(text))
    if include_matches and ukni_matches:
        all_matches.extend(extract_match_details(UK_NI_PATTERN, text, "uk_ni"))
    
    # Address detection
    address_matches = ADDRESS_PATTERN.findall(text) if include_matches else []
    entities["address"] = bool(ADDRESS_PATTERN.search(text))
    if include_matches and address_matches:
        all_matches.extend(extract_match_details(ADDRESS_PATTERN, text, "address"))
    
    # IBAN (initialized for later validation)
    entities["iban"] = False
    
    # UK sort/account (initialized for later validation)
    entities["uk_sort_account"] = False
    
    # UK passport detection
    passport_matches = UK_PASSPORT_PATTERN.findall(text) if include_matches else []
    entities["uk_passport"] = bool(UK_PASSPORT_PATTERN.search(text))
    if include_matches and passport_matches:
        all_matches.extend(extract_match_details(UK_PASSPORT_PATTERN, text, "uk_passport"))
    
    # NHS number (initialized for later validation)
    entities["nhs_number"] = False
    
    # UK postcode detection
    postcode_matches = UK_POSTCODE_PATTERN.findall(text) if include_matches else []
    entities["uk_postcode_strict"] = bool(UK_POSTCODE_PATTERN.search(text))
    if include_matches and postcode_matches:
        all_matches.extend(extract_match_details(UK_POSTCODE_PATTERN, text, "uk_postcode_strict"))

    # Check credit cards with Luhn validation
    potential_cards = CREDIT_CARD_PATTERN.findall(text)
    valid_cards = []
    for card in potential_cards:
        if luhn_check(card):
            entities["credit_card"] = True
            valid_cards.append(card)
            if not include_matches:
                break
    if include_matches and valid_cards:
        # Only add matches for valid credit cards
        for match in CREDIT_CARD_PATTERN.finditer(text):
            if luhn_check(match.group(0)):
                start, end = match.span()
                value = match.group(0)
                context_start = max(0, start - 20)
                context_end = min(len(text), end + 20)
                context = text[context_start:start] + f"[{mask_pii_value(value, 'credit_card')}]" + text[end:context_end]
                all_matches.append({
                    "value": mask_pii_value(value, "credit_card"),
                    "span": [start, end],
                    "context": context,
                    "entity_type": "credit_card",
                    "severity": "high"
                })

    # Check IBANs with MOD-97 validation
    potential_ibans = IBAN_PATTERN.findall(text)
    valid_ibans = []
    for iban in potential_ibans:
        if iban_check(iban):
            entities["iban"] = True
            valid_ibans.append(iban)
            if not include_matches:
                break
    if include_matches and valid_ibans:
        for match in IBAN_PATTERN.finditer(text):
            if iban_check(match.group(0)):
                start, end = match.span()
                value = match.group(0)
                context_start = max(0, start - 20)
                context_end = min(len(text), end + 20)
                context = text[context_start:start] + f"[{mask_pii_value(value, 'iban')}]" + text[end:context_end]
                all_matches.append({
                    "value": mask_pii_value(value, "iban"),
                    "span": [start, end],
                    "context": context,
                    "entity_type": "iban",
                    "severity": "high"
                })

    # Check NHS numbers with check digit validation
    potential_nhs = NHS_NUMBER_PATTERN.findall(text)
    valid_nhs = []
    for nhs in potential_nhs:
        if nhs_check(nhs):
            entities["nhs_number"] = True
            valid_nhs.append(nhs)
            if not include_matches:
                break
    if include_matches and valid_nhs:
        for match in NHS_NUMBER_PATTERN.finditer(text):
            if nhs_check(match.group(0)):
                start, end = match.span()
                value = match.group(0)
                context_start = max(0, start - 20)
                context_end = min(len(text), end + 20)
                context = text[context_start:start] + f"[{mask_pii_value(value, 'nhs_number')}]" + text[end:context_end]
                all_matches.append({
                    "value": mask_pii_value(value, "nhs_number"),
                    "span": [start, end],
                    "context": context,
                    "entity_type": "nhs_number",
                    "severity": "high"
                })

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
    
    # Get list of detected entities
    detected_entities = [entity_type for entity_type, detected in entities.items() if detected]
    
    # Analyze by severity
    severity_analysis: Dict[str, Dict[str, Any]] = {
        "high": {"detected": [], "count": 0},
        "medium": {"detected": [], "count": 0}, 
        "low": {"detected": [], "count": 0},
    }
    
    for entity_type in detected_entities:
        severity = get_pii_severity(entity_type)
        severity_analysis[severity]["detected"].append(entity_type)
        severity_analysis[severity]["count"] += 1
    
    # Add severity flags for easy checking
    severity_flags = {
        "has_high_severity": severity_analysis["high"]["count"] > 0,
        "has_medium_severity": severity_analysis["medium"]["count"] > 0,
        "has_low_severity": severity_analysis["low"]["count"] > 0,
    }

    result = {
        "leak": leak, 
        "entities": entities,
        "severity_analysis": severity_analysis,
        "severity_flags": severity_flags,
        "detected_entities": detected_entities,
    }
    
    # Add matches if requested
    if include_matches and all_matches is not None:
        result["matches"] = all_matches
    
    return result
