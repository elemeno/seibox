"""Prompt specification schema for non-engineer authoring."""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class PromptSpec(BaseModel):
    """Specification for an evaluation prompt with template support.
    
    This schema allows non-engineers to author evaluation items using
    simple templates and helper functions.
    """
    
    id: str = Field(description="Unique identifier for this prompt")
    category: Literal["pii", "injection", "benign", "custom"] = Field(
        description="Category of the prompt"
    )
    template: str = Field(
        description="Template string with {{variables}} for rendering"
    )
    gold: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected outputs/labels for this prompt"
    )
    vars: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variables to use when rendering the template"
    )
    
    # Optional documentation fields for readability
    given: Optional[str] = Field(
        default=None,
        description="Given/context description (for documentation)"
    )
    when: Optional[str] = Field(
        default=None,
        description="When/action description (for documentation)"
    )
    then: Optional[str] = Field(
        default=None,
        description="Then/expected outcome description (for documentation)"
    )
    
    # Optional metadata
    author: Optional[str] = Field(
        default=None,
        description="Author of this prompt"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering/grouping"
    )
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Ensure ID is non-empty and valid."""
        if not v or not v.strip():
            raise ValueError("ID must be non-empty")
        if len(v) > 100:
            raise ValueError("ID must be 100 characters or less")
        return v.strip()
    
    @field_validator('template')
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Ensure template is non-empty."""
        if not v or not v.strip():
            raise ValueError("Template must be non-empty")
        return v
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Ensure category is valid."""
        valid_categories = {"pii", "injection", "benign", "custom"}
        if v not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}")
        return v


class PromptSpecValidationResult(BaseModel):
    """Result of validating a prompt specification."""
    
    valid: bool
    line_number: int
    spec: Optional[PromptSpec] = None
    error: Optional[str] = None
    
    @property
    def status_emoji(self) -> str:
        """Return status emoji for display."""
        return "✅" if self.valid else "❌"