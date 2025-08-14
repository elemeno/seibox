"""Static HTML report generation for Safety Evals in a Box."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from jinja2 import Template

from seibox.utils.io import read_jsonl
from seibox.utils.schemas import OutputRecord
from seibox.scoring.aggregate import aggregate_metrics


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Safety Evals Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-label {
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            margin: 5px 0;
        }
        .metric-good {
            color: #27ae60;
        }
        .metric-bad {
            color: #e74c3c;
        }
        .metric-neutral {
            color: #3498db;
        }
        table {
            width: 100%;
            background: white;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th {
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .example-box {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .prompt {
            color: #2c3e50;
            font-weight: 500;
            margin-bottom: 10px;
        }
        .response {
            color: #555;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            font-family: monospace;
            font-size: 14px;
        }
        .label {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 5px;
        }
        .label-blocked {
            background: #e74c3c;
            color: white;
        }
        .label-allowed {
            background: #27ae60;
            color: white;
        }
        .label-leaked {
            background: #f39c12;
            color: white;
        }
        .comparison-table {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>🛡️ Safety Evals in a Box - Evaluation Report</h1>
    
    <div class="report-meta">
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        <p><strong>Model:</strong> {{ model_name }}</p>
        <p><strong>Mitigation:</strong> {{ mitigation or 'None' }}</p>
    </div>

    <h2>📊 Aggregate Metrics</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Safety Coverage</div>
            <div class="metric-value metric-good">{{ "%.1f" | format(metrics.safety_coverage * 100) }}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Benign Pass Rate</div>
            <div class="metric-value metric-good">{{ "%.1f" | format(metrics.benign_pass_rate * 100) }}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">False Positive Rate</div>
            <div class="metric-value metric-bad">{{ "%.1f" | format(metrics.false_positive_rate * 100) }}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Injection Success</div>
            <div class="metric-value metric-bad">{{ "%.1f" | format(metrics.injection_success_rate * 100) }}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Latency p50</div>
            <div class="metric-value metric-neutral">{{ "%.0f" | format(metrics.latency_p50) }} ms</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Latency p95</div>
            <div class="metric-value metric-neutral">{{ "%.0f" | format(metrics.latency_p95) }} ms</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Cost</div>
            <div class="metric-value metric-neutral">${{ "%.4f" | format(metrics.total_cost_usd) }}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Cost per 1k</div>
            <div class="metric-value metric-neutral">${{ "%.4f" | format(metrics.cost_per_1k) }}</div>
        </div>
    </div>

    {% if comparison %}
    <h2>📈 Comparison with Baseline</h2>
    <div class="comparison-table">
        <div>
            <h3>Baseline</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Safety Coverage</td><td>{{ "%.1f%%" | format(comparison.baseline.safety_coverage * 100) }}</td></tr>
                <tr><td>Benign Pass Rate</td><td>{{ "%.1f%%" | format(comparison.baseline.benign_pass_rate * 100) }}</td></tr>
                <tr><td>False Positive Rate</td><td>{{ "%.1f%%" | format(comparison.baseline.false_positive_rate * 100) }}</td></tr>
                <tr><td>Injection Success</td><td>{{ "%.1f%%" | format(comparison.baseline.injection_success_rate * 100) }}</td></tr>
            </table>
        </div>
        <div>
            <h3>Mitigated</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Safety Coverage</td><td>{{ "%.1f%%" | format(comparison.mitigated.safety_coverage * 100) }}</td></tr>
                <tr><td>Benign Pass Rate</td><td>{{ "%.1f%%" | format(comparison.mitigated.benign_pass_rate * 100) }}</td></tr>
                <tr><td>False Positive Rate</td><td>{{ "%.1f%%" | format(comparison.mitigated.false_positive_rate * 100) }}</td></tr>
                <tr><td>Injection Success</td><td>{{ "%.1f%%" | format(comparison.mitigated.injection_success_rate * 100) }}</td></tr>
            </table>
        </div>
    </div>
    {% endif %}

    <h2>🔍 Example Cases</h2>
    
    <h3>PII Detection Examples</h3>
    {% for example in pii_examples[:5] %}
    <div class="example-box">
        <div class="prompt">
            <strong>Prompt:</strong> {{ example.prompt }}
        </div>
        <div class="response">
            <strong>Response:</strong> {{ example.text[:200] }}{% if example.text|length > 200 %}...{% endif %}
        </div>
        <div style="margin-top: 10px;">
            {% if example.scores.leak %}
                <span class="label label-leaked">LEAKED</span>
            {% else %}
                <span class="label label-allowed">SAFE</span>
            {% endif %}
            {% if example.judgement.blocked %}
                <span class="label label-blocked">BLOCKED</span>
            {% endif %}
        </div>
    </div>
    {% endfor %}

    <h3>Injection Detection Examples</h3>
    {% for example in injection_examples[:5] %}
    <div class="example-box">
        <div class="prompt">
            <strong>Prompt:</strong> {{ example.prompt }}
        </div>
        <div class="response">
            <strong>Response:</strong> {{ example.text[:200] }}{% if example.text|length > 200 %}...{% endif %}
        </div>
        <div style="margin-top: 10px;">
            {% if example.scores.violation %}
                <span class="label label-leaked">VIOLATION</span>
            {% else %}
                <span class="label label-allowed">SAFE</span>
            {% endif %}
            {% if example.judgement.blocked %}
                <span class="label label-blocked">BLOCKED</span>
            {% endif %}
        </div>
    </div>
    {% endfor %}

    <h3>Benign Request Examples</h3>
    {% for example in benign_examples[:5] %}
    <div class="example-box">
        <div class="prompt">
            <strong>Prompt:</strong> {{ example.prompt }}
        </div>
        <div class="response">
            <strong>Response:</strong> {{ example.text[:200] }}{% if example.text|length > 200 %}...{% endif %}
        </div>
        <div style="margin-top: 10px;">
            {% if example.scores.allowed %}
                <span class="label label-allowed">ALLOWED</span>
            {% else %}
                <span class="label label-blocked">REFUSED</span>
            {% endif %}
        </div>
    </div>
    {% endfor %}

    <div class="footer">
        <p>Generated by Safety Evals in a Box | {{ timestamp }}</p>
    </div>
</body>
</html>
"""


def generate_report(
    results_path: str,
    output_path: str,
    comparison_path: Optional[str] = None,
) -> None:
    """Generate an HTML report from evaluation results.

    Args:
        results_path: Path to the results JSONL file
        output_path: Path to save the HTML report
        comparison_path: Optional path to baseline results for comparison
    """
    from datetime import datetime

    # Load results
    records = [OutputRecord(**r) for r in read_jsonl(results_path)]

    if not records:
        raise ValueError(f"No records found in {results_path}")

    # Compute metrics
    metrics = aggregate_metrics(records)

    # Get examples from each suite
    pii_examples = [r for r in records if r.id.startswith("pii")][:5]
    injection_examples = [r for r in records if r.id.startswith("injection")][:5]
    benign_examples = [r for r in records if r.id.startswith("benign")][:5]

    # Create example objects with prompts for the template
    def add_prompts_to_examples(examples):
        enhanced_examples = []
        for ex in examples:
            # Get prompt from trace or create placeholder
            prompt = ex.trace.get("prompt", f"[Prompt for {ex.id}]")

            # Create a dict with all the record data plus prompt
            enhanced_ex = {
                "id": ex.id,
                "prompt": prompt,
                "text": ex.text,
                "judgement": ex.judgement,
                "scores": ex.scores,
                "timing": ex.timing,
                "cost": ex.cost,
                "trace": ex.trace,
                "model": ex.model,
            }
            enhanced_examples.append(enhanced_ex)
        return enhanced_examples

    pii_examples = add_prompts_to_examples(pii_examples)
    injection_examples = add_prompts_to_examples(injection_examples)
    benign_examples = add_prompts_to_examples(benign_examples)

    # Handle comparison if provided
    comparison = None
    if comparison_path:
        baseline_records = [OutputRecord(**r) for r in read_jsonl(comparison_path)]
        comparison = {
            "baseline": aggregate_metrics(baseline_records),
            "mitigated": metrics,
        }

    # Render template
    template = Template(HTML_TEMPLATE)
    html = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_name=records[0].model if records else "Unknown",
        mitigation=(
            records[0].trace.get("mitigations", ["None"])[0]
            if records and records[0].trace.get("mitigations")
            else "None"
        ),
        metrics=metrics,
        comparison=comparison,
        pii_examples=pii_examples,
        injection_examples=injection_examples,
        benign_examples=benign_examples,
    )

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report generated: {output_path}")
