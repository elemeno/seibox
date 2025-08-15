"""Core data schemas for Safety Evals in a Box."""

from typing import Any, Dict, List, Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field

SuiteId = Literal["pii", "injection", "benign"]


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
    params: Dict[str, Any] = Field(default_factory=dict)  # temperature, max_tokens, etc.


class Trace(BaseModel):
    """Execution trace with full conversation details."""
    conversation_id: str
    system_prompt_hash: str
    system_prompt_preview: Optional[str] = None  # First line or first 100 chars
    include_system_full: bool = False  # Whether full system prompt is included
    messages: List[Message] = Field(default_factory=list)
    adapter: Optional[AdapterInfo] = None
    mitigations: List[str] = Field(default_factory=list)
    matches: List[Dict[str, Any]] = Field(default_factory=list)  # PII/entity matches
    gold: Dict[str, Any] = Field(default_factory=dict)  # Ground truth labels
    redacted: bool = False
    assistant_raw: Optional[str] = None  # Original response before post-processing
    assistant_redacted: Optional[str] = None  # Response after redaction


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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    gold: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"should_block": true}


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
    judgement: Dict[
        str, Any
    ]  # normalized labels, e.g., {"blocked": true, "injection_obeyed": false}
    scores: Dict[str, float] | Dict[str, int]
    timing: Dict[str, float]  # {"latency_ms": 812}
    cost: Dict[str, float]  # {"input_tokens": 152, "output_tokens": 24, "usd": 0.0009}
    trace: Trace | Dict[str, Any] = Field(default_factory=dict)  # Support both old dict format and new Trace
