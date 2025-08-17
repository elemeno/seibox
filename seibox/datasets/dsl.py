"""Domain-specific language for prompt template rendering."""

import random
import re
from datetime import datetime, timedelta
from typing import Any

import xxhash
from faker import Faker

from seibox.utils.prompt_spec import PromptSpec
from seibox.utils.schemas import InputRecord

# Initialize Faker with fixed seed for reproducibility
fake = Faker()


def seed_faker(seed: int) -> None:
    """Seed Faker for deterministic output."""
    Faker.seed(seed)
    random.seed(seed)


def luhn_checksum(number: str) -> int:
    """Calculate Luhn checksum for credit card validation."""

    def digits_of(n: str) -> list[int]:
        return [int(d) for d in n if d.isdigit()]

    digits = digits_of(number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]

    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(str(d * 2)))
    return checksum % 10


def generate_valid_credit_card(seed: int) -> str:
    """Generate a valid test credit card number (4111 prefix for testing)."""
    seed_faker(seed)
    # Use test card prefix 4111 (Visa test range)
    prefix = "4111"
    # Generate 11 more digits
    middle = "".join(str(random.randint(0, 9)) for _ in range(11))
    partial = prefix + middle

    # Calculate the check digit
    checksum = luhn_checksum(partial + "0")
    check_digit = (10 - checksum) % 10

    card_number = partial + str(check_digit)
    # Format as 4111-1111-1111-1111
    return f"{card_number[:4]}-{card_number[4:8]}-{card_number[8:12]}-{card_number[12:16]}"


def generate_uk_ni(seed: int) -> str:
    """Generate a valid-looking UK National Insurance number."""
    seed_faker(seed)
    # Format: AA123456C
    letters = "ABCEGHJKLMNPRSTWXYZ"  # Valid NI letters (excluding D, F, I, Q, U, V, O, Z)
    prefix = random.choice(letters) + random.choice(letters)
    numbers = "".join(str(random.randint(0, 9)) for _ in range(6))
    suffix = random.choice("ABCD")
    return f"{prefix}{numbers}{suffix}"


def generate_uk_address(seed: int) -> str:
    """Generate a UK-style address."""
    seed_faker(seed)
    house_number = random.randint(1, 999)
    streets = [
        "High Street",
        "Church Lane",
        "Park Road",
        "King's Road",
        "Queen Street",
        "Market Square",
        "Victoria Road",
        "Station Road",
        "Main Street",
    ]
    cities = [
        "London",
        "Manchester",
        "Birmingham",
        "Leeds",
        "Glasgow",
        "Liverpool",
        "Bristol",
        "Sheffield",
        "Edinburgh",
        "Cardiff",
    ]
    postcodes = [
        "SW1A 1AA",
        "M1 1AA",
        "B1 1AA",
        "LS1 1AA",
        "G1 1AA",
        "L1 1AA",
        "BS1 1AA",
        "S1 1AA",
        "EH1 1AA",
        "CF1 1AA",
    ]

    street = random.choice(streets)
    city = random.choice(cities)
    postcode = random.choice(postcodes)

    return f"{house_number} {street}, {city}, {postcode}"


def generate_email(seed: int) -> str:
    """Generate a realistic-looking email address."""
    seed_faker(seed)
    return fake.email()


def generate_phone_e164(seed: int) -> str:
    """Generate a phone number in E.164 format."""
    seed_faker(seed)
    # US format for simplicity
    area_code = random.randint(200, 999)
    exchange = random.randint(200, 999)
    number = random.randint(1000, 9999)
    return f"+1{area_code}{exchange}{number}"


def generate_lorem_sentence(seed: int) -> str:
    """Generate a lorem ipsum sentence."""
    seed_faker(seed)
    return fake.sentence(nb_words=random.randint(5, 15))


def generate_date_iso(seed: int) -> str:
    """Generate an ISO format date."""
    seed_faker(seed)
    # Generate a date within the last year
    days_ago = random.randint(0, 365)
    date = datetime.now() - timedelta(days=days_ago)
    return date.strftime("%Y-%m-%d")


def choice(options: str, seed: int) -> str:
    """Choose randomly from pipe-separated options."""
    seed_faker(seed)
    choices = [opt.strip() for opt in options.split("|")]
    return random.choice(choices)


def last4(text: str) -> str:
    """Return last 4 characters of a string."""
    # Remove any non-alphanumeric characters first
    clean = re.sub(r"[^a-zA-Z0-9]", "", text)
    return clean[-4:] if len(clean) >= 4 else clean


def render_template(template: str, vars: dict[str, Any], seed: int) -> str:
    """Render a template with built-in helpers and custom variables.

    Args:
        template: Template string with {{variables}}
        vars: Custom variables to use in rendering
        seed: Seed for deterministic random generation

    Returns:
        Rendered template string
    """
    # Set seed for all random operations
    seed_faker(seed)

    # Built-in helpers
    context = {
        "email": generate_email(seed),
        "phone_e164": generate_phone_e164(seed),
        "credit_card": generate_valid_credit_card(seed),
        "uk_ni": generate_uk_ni(seed),
        "address_uk": generate_uk_address(seed),
        "lorem_sentence": generate_lorem_sentence(seed),
        "date_iso": generate_date_iso(seed),
    }

    # Add custom variables (including under "vars" key for dotted access)
    context.update(vars)
    context["vars"] = vars

    # First pass: Handle function calls like {{choice("a|b|c")}}
    def replace_function_calls(match):
        func_call = match.group(1)
        # Handle choice function
        if func_call.startswith("choice("):
            # Extract the argument between quotes
            arg_match = re.search(r'choice\(["\']([^"\']+)["\']\)', func_call)
            if arg_match:
                options = arg_match.group(1)
                return choice(options, seed)
        # Handle last4 function
        elif func_call.startswith("last4("):
            # Extract the argument
            arg_match = re.search(r'last4\(["\']([^"\']+)["\']\)', func_call)
            if arg_match:
                text = arg_match.group(1)
                return last4(text)
            # Also handle variable references
            var_match = re.search(r"last4\(([^)]+)\)", func_call)
            if var_match:
                var_name = var_match.group(1).strip()
                if var_name in context:
                    return last4(str(context[var_name]))
        return match.group(0)  # Return unchanged if not recognized

    # Replace function calls
    rendered = re.sub(r"\{\{([^}]+)\}\}", replace_function_calls, template)

    # Second pass: Replace simple variables
    def replace_variables(match):
        var_name = match.group(1).strip()

        # Handle nested variable access (e.g., vars.name)
        if "." in var_name:
            parts = var_name.split(".")
            value = context
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return match.group(0)  # Return unchanged if path not found
            return str(value)
        elif var_name in context:
            return str(context[var_name])
        return match.group(0)  # Return unchanged if variable not found

    rendered = re.sub(r"\{\{([^}]+)\}\}", replace_variables, rendered)

    return rendered


def to_input_record(spec: PromptSpec, suite_override: str | None = None) -> InputRecord:
    """Convert a PromptSpec to an InputRecord.

    Args:
        spec: The prompt specification to convert
        suite_override: Optional suite name to override the category

    Returns:
        InputRecord ready for evaluation
    """
    # Generate seed from spec ID for determinism
    seed = int(xxhash.xxh64_hexdigest(spec.id), 16) % (2**32)

    # Render the template
    rendered_prompt = render_template(spec.template, spec.vars, seed)

    # Map category to suite ID
    suite_map = {"pii": "pii", "injection": "injection", "benign": "benign", "custom": "custom"}
    suite = suite_override or suite_map.get(spec.category, "custom")

    # Build metadata
    metadata = {
        "template_id": spec.id,
        "category": spec.category,
    }

    # Add documentation fields if present
    if spec.given:
        metadata["given"] = spec.given
    if spec.when:
        metadata["when"] = spec.when
    if spec.then:
        metadata["then"] = spec.then
    if spec.author:
        metadata["author"] = spec.author
    if spec.tags:
        metadata["tags"] = spec.tags

    return InputRecord(
        id=spec.id,
        suite=suite,  # type: ignore
        prompt=rendered_prompt,
        metadata=metadata,
        gold=spec.gold,
    )


def validate_template_syntax(template: str) -> tuple[bool, str | None]:
    """Validate template syntax for common errors.

    Args:
        template: Template string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for unbalanced braces
    open_count = template.count("{{")
    close_count = template.count("}}")
    if open_count != close_count:
        return False, f"Unbalanced braces: {open_count} '{{{{' vs {close_count} '}}}}'"

    # Check for valid variable names
    var_pattern = r"\{\{([^}]+)\}\}"
    for match in re.finditer(var_pattern, template):
        var_content = match.group(1).strip()

        # Check if it's a function call
        if "(" in var_content:
            # Validate function syntax - just check basic structure
            if not re.match(r"^[a-z0-9_]+\([^)]*\)$", var_content):
                return False, f"Invalid function syntax: {{{{{var_content}}}}}"
        else:
            # Check if it's a valid variable name (allow dots for nested access)
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", var_content):
                # It might be a built-in, check against known list
                known_builtins = {
                    "email",
                    "phone_e164",
                    "credit_card",
                    "uk_ni",
                    "address_uk",
                    "lorem_sentence",
                    "date_iso",
                }
                if var_content not in known_builtins:
                    return False, f"Invalid variable name: {{{{{var_content}}}}}"

    return True, None
