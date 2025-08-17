"""Core data schemas for Safety Evals in a Box."""

from typing import Any, Literal

from pydantic import BaseModel, Field

SuiteId = Literal["pii", "injection", "benign"]


class ProfileConfig(BaseModel):
    """Configuration for mitigation profiles."""

    prompt_hardening: bool = False
    policy_gate: bool = False


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: str  # ISO format timestamp
    redacted: bool = False  # Whether content has been redacted


class AdapterInfo(BaseModel):
    """Information about the adapter used for generation."""

    name: str  # e.g., "openai:gpt-4o-mini"
    model: str  # e.g., "gpt-4o-mini"
    params: dict[str, Any] = Field(default_factory=dict)  # temperature, max_tokens, etc.


class Trace(BaseModel):
    """Execution trace with full conversation details."""

    conversation_id: str
    system_prompt_hash: str
    system_prompt_preview: str | None = None  # First line or first 100 chars
    include_system_full: bool = False  # Whether full system prompt is included
    messages: list[Message] = Field(default_factory=list)
    adapter: AdapterInfo | None = None
    mitigations: list[str] = Field(default_factory=list)
    matches: list[dict[str, Any]] = Field(default_factory=list)  # PII/entity matches
    gold: dict[str, Any] = Field(default_factory=dict)  # Ground truth labels
    redacted: bool = False
    assistant_raw: str | None = None  # Original response before post-processing
    assistant_redacted: str | None = None  # Response after redaction
    profile_name: str | None = None  # Name of profile used (e.g., "baseline", "both")
    prompt_hardening: bool | None = None  # Whether prompt hardening was enabled
    policy_gate: bool | None = None  # Whether policy gate was enabled


class InputRecord(BaseModel):
    """Input record for evaluation.

    Args:
        id: Unique identifier for the record
        suite: The evaluation suite this record belongs to
        prompt: The prompt text to evaluate
        metadata: Additional metadata about the record
        gold: Ground truth labels for evaluation
    """

    id: str
    suite: SuiteId
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    gold: dict[str, Any] = Field(default_factory=dict)  # e.g., {"should_block": true}


class OutputRecord(BaseModel):
    """Output record from evaluation.

    Args:
        id: Unique identifier matching the input record
        model: Model name that generated this output
        text: The generated text response
        judgement: Normalized evaluation labels
        scores: Numeric scores from evaluation
        timing: Timing information including latency
        cost: Cost breakdown including tokens and USD
        trace: Execution trace including mitigations applied
    """

    id: str
    model: str
    text: str
    judgement: dict[
        str, Any
    ]  # normalized labels, e.g., {"blocked": true, "injection_obeyed": false}
    scores: dict[str, float] | dict[str, int]
    timing: dict[str, float]  # {"latency_ms": 812}
    cost: dict[str, float]  # {"input_tokens": 152, "output_tokens": 24, "usd": 0.0009}
    trace: Trace | dict[str, Any] = Field(
        default_factory=dict
    )  # Support both old dict format and new Trace
