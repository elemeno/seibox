"""Unit tests for prompt DSL and template rendering."""

import pytest
import re
from seibox.utils.prompt_spec import PromptSpec, PromptSpecValidationResult
from seibox.datasets.dsl import (
    render_template,
    to_input_record,
    validate_template_syntax,
    luhn_checksum,
    generate_valid_credit_card,
    generate_uk_ni,
    generate_uk_address,
    generate_email,
    generate_phone_e164,
    choice,
    last4,
)


class TestDeterminism:
    """Test that all DSL functions produce deterministic output."""

    def test_render_template_deterministic(self):
        """Same seed should produce same output."""
        template = "Email: {{email}}, Phone: {{phone_e164}}"
        seed = 12345

        # Render multiple times with same seed
        result1 = render_template(template, {}, seed)
        result2 = render_template(template, {}, seed)
        result3 = render_template(template, {}, seed)

        assert result1 == result2 == result3

    def test_different_seeds_different_output(self):
        """Different seeds should produce different output."""
        template = "{{email}}"

        result1 = render_template(template, {}, 1)
        result2 = render_template(template, {}, 2)

        assert result1 != result2

    def test_choice_deterministic(self):
        """Choice function should be deterministic."""
        template = '{{choice("apple|banana|orange")}}'
        seed = 42

        result1 = render_template(template, {}, seed)
        result2 = render_template(template, {}, seed)

        assert result1 == result2
        assert result1 in ["apple", "banana", "orange"]


class TestBuiltinHelpers:
    """Test built-in helper functions."""

    def test_email_format(self):
        """Generated emails should be valid format."""
        email = generate_email(123)
        assert "@" in email
        assert "." in email.split("@")[1]

    def test_phone_e164_format(self):
        """Phone numbers should be E.164 format."""
        phone = generate_phone_e164(456)
        assert phone.startswith("+1")
        assert len(phone) == 12  # +1 followed by 10 digits
        assert phone[2:].isdigit()

    def test_credit_card_luhn_valid(self):
        """Generated credit cards should pass Luhn check."""
        for seed in [1, 42, 999]:
            card = generate_valid_credit_card(seed)
            # Remove hyphens for validation
            card_digits = card.replace("-", "")
            assert len(card_digits) == 16
            assert luhn_checksum(card_digits) == 0
            assert card.startswith("4111")  # Test prefix

    def test_uk_ni_format(self):
        """UK NI numbers should match expected format."""
        ni = generate_uk_ni(789)
        # Format: AA123456C
        assert len(ni) == 9
        assert ni[:2].isalpha()
        assert ni[2:8].isdigit()
        assert ni[8].isalpha()

    def test_uk_address_format(self):
        """UK addresses should have expected components."""
        address = generate_uk_address(321)
        # Should contain number, street, city, postcode
        assert re.search(r"\d+", address)  # Has house number
        assert "," in address  # Has comma separators
        # Check for UK postcode pattern
        assert re.search(r"[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}", address)

    def test_last4_function(self):
        """Last4 should return last 4 characters."""
        assert last4("1234567890") == "7890"
        assert last4("abc") == "abc"
        assert last4("test-123-456") == "3456"  # Strips non-alphanumeric


class TestTemplateRendering:
    """Test template rendering functionality."""

    def test_simple_variable_substitution(self):
        """Basic variable substitution should work."""
        template = "Hello {{name}}"
        vars = {"name": "Alice"}
        result = render_template(template, vars, 1)
        assert result == "Hello Alice"

    def test_multiple_variables(self):
        """Multiple variables should be substituted."""
        template = "{{greeting}} {{name}}, you are {{age}} years old"
        vars = {"greeting": "Hi", "name": "Bob", "age": 25}
        result = render_template(template, vars, 1)
        assert result == "Hi Bob, you are 25 years old"

    def test_builtin_helpers(self):
        """Built-in helpers should be accessible."""
        template = "Contact: {{email}}"
        result = render_template(template, {}, 42)
        assert "@" in result
        assert "Contact:" in result

    def test_function_calls(self):
        """Function calls in templates should work."""
        template = 'Pick: {{choice("red|blue|green")}}'
        result = render_template(template, {}, 123)
        assert result.startswith("Pick: ")
        assert result.split(": ")[1] in ["red", "blue", "green"]

    def test_last4_with_variable(self):
        """Last4 function should work with variables."""
        template = "Card ending in {{last4(credit_card)}}"
        result = render_template(template, {}, 456)
        # Should contain last 4 digits of a credit card
        assert "Card ending in" in result
        last_digits = result.split("in ")[1]
        assert len(last_digits) == 4
        assert last_digits.isdigit()


class TestTemplateSyntaxValidation:
    """Test template syntax validation."""

    def test_valid_templates(self):
        """Valid templates should pass validation."""
        valid_templates = [
            "Simple text",
            "Hello {{name}}",
            "{{email}} and {{phone_e164}}",
            '{{choice("a|b|c")}}',
            '{{last4("test")}}',
        ]

        for template in valid_templates:
            is_valid, error = validate_template_syntax(template)
            assert is_valid, f"Template '{template}' should be valid: {error}"

    def test_invalid_templates(self):
        """Invalid templates should fail validation."""
        invalid_templates = [
            ("{{unclosed", "Unbalanced braces"),
            ("{{123invalid}}", "Invalid variable name"),
            ("{{bad function()}}", "Invalid function syntax"),
        ]

        for template, expected_error in invalid_templates:
            is_valid, error = validate_template_syntax(template)
            assert not is_valid
            assert error is not None


class TestPromptSpec:
    """Test PromptSpec model validation."""

    def test_valid_spec(self):
        """Valid spec should be created successfully."""
        spec = PromptSpec(
            id="test_001",
            category="pii",
            template="Test {{email}}",
            gold={"should_block": True},
            vars={"name": "Test"},
        )
        assert spec.id == "test_001"
        assert spec.category == "pii"

    def test_invalid_category(self):
        """Invalid category should raise error."""
        with pytest.raises(ValueError):
            PromptSpec(
                id="test",
                category="invalid",  # Not in allowed categories
                template="Test",
            )

    def test_empty_id(self):
        """Empty ID should raise error."""
        with pytest.raises(ValueError):
            PromptSpec(
                id="",
                category="benign",
                template="Test",
            )

    def test_empty_template(self):
        """Empty template should raise error."""
        with pytest.raises(ValueError):
            PromptSpec(
                id="test",
                category="benign",
                template="",
            )

    def test_optional_fields(self):
        """Optional fields should work correctly."""
        spec = PromptSpec(
            id="test",
            category="benign",
            template="Test",
            given="Given context",
            when="When action",
            then="Then expected",
            author="Test Author",
            tags=["test", "example"],
        )
        assert spec.given == "Given context"
        assert spec.when == "When action"
        assert spec.then == "Then expected"
        assert spec.author == "Test Author"
        assert spec.tags == ["test", "example"]


class TestInputRecordConversion:
    """Test conversion from PromptSpec to InputRecord."""

    def test_basic_conversion(self):
        """Basic spec should convert to InputRecord."""
        spec = PromptSpec(
            id="test_001",
            category="pii",
            template="Test {{email}}",
            gold={"should_block": True},
        )

        record = to_input_record(spec)

        assert record.id == "test_001"
        assert record.suite == "pii"
        assert "@" in record.prompt  # Should have rendered email
        assert record.gold == {"should_block": True}
        assert record.metadata["template_id"] == "test_001"
        assert record.metadata["category"] == "pii"

    def test_suite_override(self):
        """Suite override should work."""
        spec = PromptSpec(
            id="test",
            category="pii",
            template="Test",
        )

        record = to_input_record(spec, suite_override="benign")
        assert record.suite == "benign"

    def test_metadata_inclusion(self):
        """Documentation fields should be included in metadata."""
        spec = PromptSpec(
            id="test",
            category="benign",
            template="Test",
            given="Given text",
            when="When text",
            then="Then text",
            author="Author",
            tags=["tag1", "tag2"],
        )

        record = to_input_record(spec)

        assert record.metadata["given"] == "Given text"
        assert record.metadata["when"] == "When text"
        assert record.metadata["then"] == "Then text"
        assert record.metadata["author"] == "Author"
        assert record.metadata["tags"] == ["tag1", "tag2"]


class TestHelperFunctions:
    """Test individual helper functions."""

    def test_luhn_checksum(self):
        """Luhn checksum should work correctly."""
        # Valid credit card numbers
        assert luhn_checksum("4111111111111111") == 0
        assert luhn_checksum("5500000000000004") == 0

        # Invalid numbers
        assert luhn_checksum("4111111111111112") != 0
        assert luhn_checksum("1234567890123456") != 0

    def test_choice_function(self):
        """Choice function should select from options."""
        options = "apple|banana|orange"
        result = choice(options, 42)
        assert result in ["apple", "banana", "orange"]

    def test_last4_edge_cases(self):
        """Last4 should handle edge cases."""
        assert last4("") == ""
        assert last4("123") == "123"
        assert last4("a-b-c-d") == "abcd"
