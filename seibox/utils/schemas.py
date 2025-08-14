"""Core data schemas for Safety Evals in a Box."""

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

SuiteId = Literal["pii", "injection", "benign"]


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
    judgement: Dict[str, Any]  # normalized labels, e.g., {"blocked": true, "injection_obeyed": false}
    scores: Dict[str, float] | Dict[str, int]
    timing: Dict[str, float]  # {"latency_ms": 812}
    cost: Dict[str, float]  # {"input_tokens": 152, "output_tokens": 24, "usd": 0.0009}
    trace: Dict[str, Any]  # {"system_prompt_hash": "...", "mitigations": ["policy_gate@0.1.0"]}