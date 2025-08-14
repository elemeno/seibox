"""Policy management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PolicyManager:
    """Manages loading and accessing policy configurations."""

    def __init__(self, policies_dir: Optional[Path] = None):
        """Initialize policy manager.

        Args:
            policies_dir: Directory containing policy files. Defaults to repo policies/ dir.
        """
        if policies_dir is None:
            # Default to policies directory in repo root
            self.policies_dir = Path(__file__).parent.parent.parent / "policies"
        else:
            self.policies_dir = policies_dir

        self._policies_cache = {}

    def load_policy(self, policy_name: str) -> Dict[str, Any]:
        """Load a policy configuration by name.

        Args:
            policy_name: Name of the policy file (without .yaml extension)

        Returns:
            Dictionary containing the policy configuration

        Raises:
            FileNotFoundError: If policy file doesn't exist
            ValueError: If policy file is malformed
        """
        if policy_name in self._policies_cache:
            return self._policies_cache[policy_name]

        policy_file = self.policies_dir / f"{policy_name}.yaml"

        if not policy_file.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_file}")

        try:
            with open(policy_file, "r") as f:
                policy = yaml.safe_load(f)

            self._policies_cache[policy_name] = policy
            return policy

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in policy file {policy_file}: {e}")

    def get_refusal_message(self, policy_name: str, context: str = "general") -> str:
        """Get a contextual refusal message from a policy.

        Args:
            policy_name: Name of the policy to use
            context: Context for the refusal (e.g., 'general', 'formatting', 'translation')

        Returns:
            Formatted refusal message with appropriate alternative
        """
        try:
            policy = self.load_policy(policy_name)

            # Get base template
            template = policy.get("refusal_templates", {}).get(
                "standard", "Sorry—I can't disclose personal data. I can {alternative} instead."
            )

            # Get context-specific alternative
            alternatives = policy.get("refusal_templates", {}).get("alternatives_by_context", {})
            alternative = alternatives.get(
                context, alternatives.get("general", "help with general guidance or templates")
            )

            return template.format(alternative=alternative)

        except (FileNotFoundError, ValueError, KeyError):
            # Fallback to default message if policy loading fails
            return "Sorry—I can't disclose personal data. I can summarize without identifiers or guide you on safe sharing."

    def get_pii_categories(self, policy_name: str = "pii_v1") -> Dict[str, Any]:
        """Get PII categories and detection guidance from policy.

        Args:
            policy_name: Name of the PII policy to use

        Returns:
            Dictionary of PII categories with detection patterns and confidence levels
        """
        try:
            policy = self.load_policy(policy_name)
            return policy.get("pii_categories", {})
        except (FileNotFoundError, ValueError):
            return {}

    def get_masking_guidance(self, policy_name: str = "pii_v1") -> Dict[str, Any]:
        """Get masking guidance from policy.

        Args:
            policy_name: Name of the policy to use

        Returns:
            Dictionary containing masking principles and placeholder formats
        """
        try:
            policy = self.load_policy(policy_name)
            return policy.get("masking_guidance", {})
        except (FileNotFoundError, ValueError):
            return {}


# Global policy manager instance
_policy_manager = PolicyManager()


def get_policy_manager() -> PolicyManager:
    """Get the global policy manager instance."""
    return _policy_manager


def get_contextual_refusal(
    content_type: str = "general", detected_pii: Optional[Dict] = None
) -> str:
    """Get a contextual refusal message based on content type and detected PII.

    Args:
        content_type: Type of content being processed (e.g., 'email', 'phone', 'general')
        detected_pii: Dictionary of detected PII types (from pii scoring)

    Returns:
        Appropriate contextual refusal message
    """
    manager = get_policy_manager()

    # Determine context based on detected PII types
    context = "general"

    if detected_pii:
        if detected_pii.get("email", False):
            context = "formatting"  # Email formatting help
        elif detected_pii.get("phone", False):
            context = "formatting"  # Phone formatting help
        elif detected_pii.get("credit_card", False):
            context = "validation"  # Direct to secure tools
        elif detected_pii.get("address", False):
            context = "examples"  # Provide placeholder examples
        # Can be extended based on specific PII types

    # Override context based on content type if more specific
    content_contexts = {
        "translation": "translation",
        "summarization": "summarization",
        "validation": "validation",
        "formatting": "formatting",
    }

    if content_type in content_contexts:
        context = content_contexts[content_type]

    return manager.get_refusal_message("pii_v1", context)


def should_refuse_content(text: str, pii_result: Dict[str, Any]) -> bool:
    """Determine if content should be refused based on PII detection results.

    Args:
        text: The generated text
        pii_result: PII detection results from scoring

    Returns:
        True if content should be refused, False otherwise
    """
    # Refuse if any PII is detected
    return pii_result.get("leak", False)


def get_safe_alternative_suggestions(detected_pii: Dict[str, bool]) -> str:
    """Generate specific alternative suggestions based on detected PII types.

    Args:
        detected_pii: Dictionary mapping PII types to detection status

    Returns:
        Specific alternative suggestions
    """
    suggestions = []

    if detected_pii.get("email", False):
        suggestions.append("email template creation")

    if detected_pii.get("phone", False):
        suggestions.append("phone number formatting guidelines")

    if detected_pii.get("credit_card", False):
        suggestions.append("secure payment processing guidance")

    if detected_pii.get("address", False):
        suggestions.append("address formatting examples with placeholders")

    if detected_pii.get("uk_ni", False):
        suggestions.append("ID number format explanations using examples")

    if not suggestions:
        suggestions.append("general guidance using non-identifying examples")

    return ", ".join(suggestions[:2])  # Limit to top 2 suggestions for brevity
