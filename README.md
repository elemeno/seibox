# Safety Evals in a Box 🛡️

A reproducible harness to evaluate and improve LLM safety without tanking helpfulness.

## Features

- **Deterministic Evaluations**: Reproducible results with response caching and seeded data
- **Multiple Safety Suites**: PII detection, prompt injection, and benign request handling
- **Mitigation Testing**: Toggle mitigations to compare baseline vs protected performance
- **Cost Tracking**: Token accounting and cost-per-call metrics
- **Performance Metrics**: Latency percentiles (p50/p95) and throughput measurement

## Quick Start

### Installation

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Set up environment variables
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Running Evaluations

```bash
# Run PII + Injection evaluation suite
poetry run seibox run --suite pi-injection --model openai:gpt-4o-mini \
  --config configs/eval_pi_injection.yaml --out runs/baseline.jsonl

# Run with mitigation
poetry run seibox run --suite pi-injection --model openai:gpt-4o-mini \
  --config configs/eval_pi_injection.yaml --out runs/mitigated.jsonl \
  --mitigation policy_gate@0.1.0

# Compare results
poetry run seibox compare --a runs/baseline.jsonl --b runs/mitigated.jsonl \
  --report reports/comparison.html

# Launch dashboard
poetry run seibox dashboard --runs runs/
```

## Evaluation Suites

### PII Detection (`pii`)
Tests whether models leak personally identifiable information:
- Email addresses
- Phone numbers  
- Credit card numbers (with Luhn validation)
- UK National Insurance numbers
- Physical addresses

### Prompt Injection (`injection`)
Tests resistance to instruction override attempts:
- System prompt extraction
- Instruction conflicts
- Obfuscation techniques (homoglyphs, zero-width spaces, leet speak)

### Benign Requests (`benign`)
Tests that normal requests are not over-refused:
- Programming help
- Cooking recipes
- Travel planning
- Data analysis
- General knowledge

## Key Metrics

- **Safety Coverage**: Recall on samples that should be blocked
- **Benign Pass Rate**: Percentage of benign requests allowed
- **False Positive Rate**: 1 - benign pass rate
- **Injection Success Rate**: Percentage of successful prompt injections
- **Cost per 1k calls**: USD cost for 1000 API calls
- **Latency p50/p95**: 50th and 95th percentile response times

## Configuration

See `configs/eval_pi_injection.yaml` for configuration options:
- Rate limiting and retry logic
- Dataset sampling
- Scoring parameters
- Available mitigations
- Reporting settings

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black seibox tests

# Lint
poetry run ruff check seibox tests

# Type check
poetry run mypy seibox
```

## Architecture

- `seibox/adapters/`: Model provider adapters (OpenAI, etc.)
- `seibox/scoring/`: Scoring modules for each evaluation type
- `seibox/runners/`: Evaluation orchestration and batch processing
- `seibox/mitigations/`: Pre/post processing mitigations
- `seibox/utils/`: Shared utilities (caching, I/O, schemas)
- `seibox/ui/`: Dashboard and reporting interfaces

## License

MIT