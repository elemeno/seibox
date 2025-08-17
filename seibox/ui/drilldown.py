"""
Drilldown viewer for individual evaluation records.

Renders detailed HTML views showing prompt-response transcripts,
scores, timing, and cost information for individual evaluation samples.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader
import hashlib


def format_tokens(tokens: float) -> str:
    """Format token count for display."""
    if tokens >= 1000:
        return f"{tokens/1000:.1f}k"
    return str(int(tokens))


def format_cost(usd: float) -> str:
    """Format USD cost for display."""
    if usd < 0.01:
        return f"${usd:.5f}"
    return f"${usd:.4f}"


def format_latency(ms: float) -> str:
    """Format latency in milliseconds."""
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{int(ms)}ms"


def get_score_status(scores: Dict[str, Any]) -> str:
    """Determine overall status from scores."""
    if scores.get("leak", 0) > 0:
        return "blocked"
    if scores.get("injection_success", 0) > 0:
        return "flagged"
    if scores.get("allowed", 0) > 0:
        return "allowed"
    return "unknown"


def extract_system_prompt(messages: list) -> Optional[str]:
    """Extract full system prompt if available."""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            # Check if it's a hash placeholder
            if content.startswith("[Hash:"):
                return None  # Full prompt not included
            return content
    return None


def get_entity_spans(trace: Dict[str, Any]) -> list:
    """Extract PII entity spans from trace matches."""
    spans = []
    matches = trace.get("matches", [])
    
    for match in matches:
        if isinstance(match, dict):
            spans.append({
                "entity": match.get("entity", "unknown"),
                "value": match.get("value", ""),
                "start": match.get("start", 0),
                "end": match.get("end", 0)
            })
    
    return spans


def render_drilldown(record: Dict[str, Any], out_html: str) -> None:
    """
    Render a single evaluation record to an HTML drilldown page.
    
    Args:
        record: The evaluation record dictionary
        out_html: Path to write the HTML output
    """
    # Set up Jinja2 environment
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    # Add custom filters
    env.filters["format_tokens"] = format_tokens
    env.filters["format_cost"] = format_cost
    env.filters["format_latency"] = format_latency
    
    # Load template
    template = env.get_template("drilldown.html.j2")
    
    # Prepare context
    trace = record.get("trace", {})
    messages = trace.get("messages", [])
    
    # Extract user and assistant messages
    user_message = None
    assistant_message = None
    system_prompt = None
    
    for msg in messages:
        role = msg.get("role")
        if role == "user" and user_message is None:
            user_message = msg.get("content", "")
        elif role == "assistant" and assistant_message is None:
            assistant_message = msg.get("content", "")
        elif role == "system":
            # Try to get full prompt if not redacted
            content = msg.get("content", "")
            if not content.startswith("[Hash:"):
                system_prompt = content
            else:
                # Use preview if available
                system_prompt = trace.get("system_prompt_preview", "")
                if system_prompt:
                    system_prompt += "..."
    
    # Prepare score details
    scores = record.get("scores", {})
    score_items = []
    
    # PII scores
    pii_entities = ["email", "phone", "credit_card", "uk_ni", "address", 
                    "iban", "uk_sort_account", "uk_passport", "nhs_number"]
    for entity in pii_entities:
        if entity in scores:
            score_items.append({
                "name": entity.replace("_", " ").title(),
                "value": scores[entity],
                "type": "pii",
                "detected": scores[entity] > 0
            })
    
    # Other scores
    if "injection_success" in scores:
        score_items.append({
            "name": "Injection Success",
            "value": scores["injection_success"],
            "type": "injection",
            "detected": scores["injection_success"] > 0
        })
    
    if "allowed" in scores:
        score_items.append({
            "name": "Allowed",
            "value": scores["allowed"],
            "type": "benign",
            "detected": scores["allowed"] > 0
        })
    
    # Get entity spans
    entity_spans = get_entity_spans(trace)
    
    # Prepare context
    context = {
        "record": record,
        "id": record.get("id", "unknown"),
        "model": record.get("model", "unknown"),
        "category": trace.get("category", "unknown"),
        "profile": trace.get("profile_name", "baseline"),
        "status": get_score_status(scores),
        "user_message": user_message,
        "assistant_message": assistant_message or record.get("text", ""),
        "system_prompt": system_prompt,
        "system_prompt_hash": trace.get("system_prompt_hash", ""),
        "score_items": score_items,
        "entity_spans": entity_spans,
        "mitigations": trace.get("mitigations", []),
        "timing": record.get("timing", {}),
        "cost": record.get("cost", {}),
        "judgement": record.get("judgement", {}),
        "gold": trace.get("gold", {}),
        "conversation_id": trace.get("conversation_id", ""),
        "messages": messages
    }
    
    # Render HTML
    html_content = template.render(**context)
    
    # Write to file
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content)


def create_index(jsonl_path: str, index_path: str) -> None:
    """
    Create an index file for fast record lookup.
    
    Args:
        jsonl_path: Path to the JSONL file
        index_path: Path to write the index JSON
    """
    index = []
    
    with open(jsonl_path, "rb") as f:
        offset = 0
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    record = json.loads(line)
                    index.append({
                        "id": record.get("id", f"record_{line_num}"),
                        "offset": offset,
                        "length": len(line)
                    })
                except json.JSONDecodeError:
                    pass
            offset = f.tell()
    
    # Write index
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def load_record_by_offset(jsonl_path: str, offset: int, length: int) -> Dict[str, Any]:
    """
    Load a specific record from JSONL using byte offset.
    
    Args:
        jsonl_path: Path to the JSONL file
        offset: Byte offset of the record
        length: Length of the record in bytes
        
    Returns:
        The parsed record dictionary
    """
    with open(jsonl_path, "rb") as f:
        f.seek(offset)
        line = f.read(length)
        return json.loads(line)