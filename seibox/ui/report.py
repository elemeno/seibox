"""Static HTML report generation for Safety Evals in a Box."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from jinja2 import Template

from seibox.utils.io import read_jsonl
from seibox.utils.schemas import OutputRecord, Trace
from seibox.scoring.aggregate import aggregate_metrics


def _get_trace_mitigations(trace):
    """Extract mitigations from trace, handling both old dict and new Trace formats."""
    if hasattr(trace, "mitigations"):
        # New Trace format
        return trace.mitigations
    elif isinstance(trace, dict):
        # Old dict format
        return trace.get("mitigations", [])
    else:
        return []


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

    {% if entity_metrics or severity_metrics %}
    <h2>🎯 PII Entity & Severity Analysis</h2>
    
    <div class="comparison-table">
        <div>
            <h3>Per-Entity Leak Rates</h3>
            {% if entity_metrics %}
            <table>
                <tr>
                    <th>Entity Type</th>
                    <th>Severity</th>
                    <th>Leak Rate</th>
                    <th>Detected/Total</th>
                </tr>
                {% for entity_type, data in entity_metrics.items() %}
                <tr>
                    <td>{{ entity_type.replace('_', ' ').title() }}</td>
                    <td>
                        <span class="label {% if data.severity == 'high' %}label-leaked{% elif data.severity == 'medium' %}label-blocked{% else %}label-allowed{% endif %}">
                            {{ data.severity.title() }}
                        </span>
                    </td>
                    <td>{{ "%.1f%%" | format(data.leak_rate * 100) }}</td>
                    <td>{{ data.detected_count }}/{{ data.total_count }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>No entity-specific metrics available</p>
            {% endif %}
        </div>
        
        <div>
            <h3>Severity-Level Summary</h3>
            {% if severity_metrics %}
            <table>
                <tr>
                    <th>Severity Level</th>
                    <th>Leak Rate</th>
                    <th>Records with Leaks</th>
                    <th>Entity Types</th>
                </tr>
                {% for severity, data in severity_metrics.items() %}
                {% if data.total_records > 0 %}
                <tr>
                    <td>
                        <span class="label {% if severity == 'high' %}label-leaked{% elif severity == 'medium' %}label-blocked{% else %}label-allowed{% endif %}">
                            {{ severity.title() }}
                        </span>
                    </td>
                    <td>{{ "%.1f%%" | format(data.leak_rate * 100) }}</td>
                    <td>{{ data.detected_records }}/{{ data.total_records }}</td>
                    <td>{{ data.entity_types | length }}</td>
                </tr>
                {% endif %}
                {% endfor %}
            </table>
            
            <h4>Entity Types by Severity</h4>
            {% for severity, data in severity_metrics.items() %}
            {% if data.entity_types %}
            <div class="example-box">
                <div class="prompt">
                    <strong>{{ severity.title() }} Severity:</strong>
                </div>
                <div class="response">
                    {% for entity in data.entity_types %}
                        <span class="label {% if severity == 'high' %}label-leaked{% elif severity == 'medium' %}label-blocked{% else %}label-allowed{% endif %}">
                            {{ entity.replace('_', ' ').title() }}
                        </span>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
            {% endfor %}
            {% else %}
            <p>No severity-level metrics available</p>
            {% endif %}
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
            if hasattr(ex.trace, "messages") and ex.trace.messages:
                # New Trace format - extract user message
                user_messages = [msg for msg in ex.trace.messages if msg.role == "user"]
                prompt = user_messages[0].content if user_messages else f"[Prompt for {ex.id}]"
            elif isinstance(ex.trace, dict):
                # Old dict format
                prompt = ex.trace.get("prompt", f"[Prompt for {ex.id}]")
            else:
                prompt = f"[Prompt for {ex.id}]"

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

    # Extract entity and severity metrics for report
    entity_metrics = metrics.get("entity_metrics", {})
    severity_metrics = metrics.get("severity_metrics", {})

    # Render template
    template = Template(HTML_TEMPLATE)
    html = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_name=records[0].model if records else "Unknown",
        mitigation=(
            _get_trace_mitigations(records[0].trace)[0]
            if records and _get_trace_mitigations(records[0].trace)
            else "None"
        ),
        metrics=metrics,
        comparison=comparison,
        pii_examples=pii_examples,
        injection_examples=injection_examples,
        benign_examples=benign_examples,
        entity_metrics=entity_metrics,
        severity_metrics=severity_metrics,
    )

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report generated: {output_path}")


def generate_model_reports(plan, reports_dir: str) -> None:
    """Generate per-model HTML reports for multiple categories.

    Args:
        plan: Completed evaluation plan
        reports_dir: Directory to save reports in
    """
    from collections import defaultdict

    # Group results by model
    model_results = defaultdict(list)

    for job in plan.jobs:
        if job.status == "completed":
            result_file = Path(job.output_path)
            if result_file.exists():
                model_results[job.model].append(
                    {"category": job.category, "result_file": str(result_file), "job": job}
                )

    # Generate report for each model
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for model, categories in model_results.items():
        # Sanitize model name for filename
        safe_model_name = model.replace(":", "_").replace("/", "_")
        output_path = reports_dir / f"{safe_model_name}.html"

        # Load all results for this model
        all_records = []
        category_summaries = {}

        for category_data in categories:
            records = read_jsonl(category_data["result_file"])
            records = [OutputRecord(**r) for r in records]
            all_records.extend(records)

            # Get summary for this category
            summary_path = Path(category_data["result_file"]).with_suffix(".summary.json")
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    category_summaries[category_data["category"]] = json.load(f)

        # Generate combined metrics
        combined_metrics = aggregate_metrics(all_records)

        # Create multi-category template
        multi_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ model_name }} - Safety Evaluation Report</title>
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
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #5a6c7d; margin-top: 20px; }
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
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .category-section {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .category-header {
            background: #ecf0f1;
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .status-success { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-danger { color: #e74c3c; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>🛡️ {{ model_name }} Safety Evaluation</h1>
    
    <div class="category-section">
        <div class="category-header">
            <h2>📊 Overall Performance Summary</h2>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{{ categories|length }}</div>
                <div class="metric-label">Categories Evaluated</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ combined_metrics.total_calls }}</div>
                <div class="metric-label">Total API Calls</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${{ "%.4f"|format(combined_metrics.get('cost', {}).get('total_usd', 0)) }}</div>
                <div class="metric-label">Total Cost</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ combined_metrics.get('latency_p95', 0)|round }}ms</div>
                <div class="metric-label">P95 Latency</div>
            </div>
        </div>
    </div>
    
    <div class="comparison-grid">
        {% for category, summary in category_summaries.items() %}
        <div class="category-section">
            <div class="category-header">
                <h3>{{ category|title }} Category</h3>
            </div>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                {% if summary.safety_coverage is defined %}
                <tr>
                    <td>Safety Coverage</td>
                    <td class="{% if summary.safety_coverage >= 0.8 %}status-success{% elif summary.safety_coverage >= 0.6 %}status-warning{% else %}status-danger{% endif %}">
                        {{ "%.1f"|format(summary.safety_coverage * 100) }}%
                    </td>
                </tr>
                {% endif %}
                {% if summary.benign_pass_rate is defined %}
                <tr>
                    <td>Benign Pass Rate</td>
                    <td class="{% if summary.benign_pass_rate >= 0.8 %}status-success{% elif summary.benign_pass_rate >= 0.6 %}status-warning{% else %}status-danger{% endif %}">
                        {{ "%.1f"|format(summary.benign_pass_rate * 100) }}%
                    </td>
                </tr>
                {% endif %}
                {% if summary.injection_success_rate is defined %}
                <tr>
                    <td>Injection Success Rate</td>
                    <td class="{% if summary.injection_success_rate <= 0.1 %}status-success{% elif summary.injection_success_rate <= 0.3 %}status-warning{% else %}status-danger{% endif %}">
                        {{ "%.1f"|format(summary.injection_success_rate * 100) }}%
                    </td>
                </tr>
                {% endif %}
                <tr>
                    <td>Cost per 1K</td>
                    <td>${{ "%.4f"|format(summary.cost_per_1k) }}</td>
                </tr>
                <tr>
                    <td>P95 Latency</td>
                    <td>{{ summary.latency_p95|round }}ms</td>
                </tr>
                <tr>
                    <td>Sample Size</td>
                    <td>{{ summary.total_calls }}</td>
                </tr>
            </table>
        </div>
        {% endfor %}
    </div>
    
    <div class="category-section">
        <div class="category-header">
            <h2>📈 Performance Analysis</h2>
        </div>
        
        <h3>Key Insights</h3>
        <ul>
            {% set avg_safety = category_summaries.values() | selectattr('safety_coverage', 'defined') | map(attribute='safety_coverage') | list %}
            {% if avg_safety %}
            <li><strong>Average Safety Coverage:</strong> {{ "%.1f"|format((avg_safety | sum / avg_safety | length) * 100) }}% across {{ avg_safety | length }} categories</li>
            {% endif %}
            
            {% set avg_benign = category_summaries.values() | selectattr('benign_pass_rate', 'defined') | map(attribute='benign_pass_rate') | list %}
            {% if avg_benign %}
            <li><strong>Average Benign Pass Rate:</strong> {{ "%.1f"|format((avg_benign | sum / avg_benign | length) * 100) }}% (helpfulness measure)</li>
            {% endif %}
            
            {% set total_cost = category_summaries.values() | map(attribute='cost.total_usd', default=0) | sum %}
            <li><strong>Total Evaluation Cost:</strong> ${{ "%.4f"|format(total_cost) }}</li>
            
            {% set categories_with_issues = category_summaries.values() | selectattr('safety_coverage', 'defined') | selectattr('safety_coverage', '<', 0.8) | list %}
            {% if categories_with_issues %}
            <li><strong>Categories Needing Attention:</strong> {{ categories_with_issues | length }} categories with safety coverage below 80%</li>
            {% endif %}
        </ul>
    </div>
    
    <div class="category-section">
        <div class="category-header">
            <h2>ℹ️ Report Metadata</h2>
        </div>
        
        <table>
            <tr>
                <td><strong>Model:</strong></td>
                <td>{{ model_name }}</td>
            </tr>
            <tr>
                <td><strong>Generated:</strong></td>
                <td>{{ timestamp }}</td>
            </tr>
            <tr>
                <td><strong>Categories:</strong></td>
                <td>{{ categories | join(', ') }}</td>
            </tr>
        </table>
    </div>
</body>
</html>
        """

        # Render template
        template = Template(multi_template)
        html_content = template.render(
            model_name=model,
            categories=[cat["category"] for cat in categories],
            category_summaries=category_summaries,
            combined_metrics=combined_metrics,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Write report
        with open(output_path, "w") as f:
            f.write(html_content)
