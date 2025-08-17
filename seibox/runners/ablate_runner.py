"""Ablation study runner for comparing baseline vs pre-gate vs pre+post-gate."""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from seibox.runners.eval_runner import run_eval
from seibox.scoring.aggregate import aggregate_metrics
from seibox.utils.io import read_jsonl
from seibox.utils.schemas import OutputRecord

console = Console()


def run_ablation_study(config_path: str, model_name: str, output_dir: str) -> None:
    """Run ablation study comparing baseline vs pre-gate vs pre+post-gate.

    Args:
        config_path: Path to evaluation configuration file
        model_name: Model name (e.g., openai:gpt-4o-mini)
        output_dir: Output directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define the three conditions
    conditions = [
        {
            "name": "baseline",
            "description": "Prompt hardening only",
            "mitigation": "prompt_hardening@0.1.0",
            "out_file": "baseline.jsonl",
        },
        {
            "name": "pre_gate",
            "description": "Pre-gate only",
            "mitigation": "policy_gate@0.1.0:pre",
            "out_file": "pre_gate.jsonl",
        },
        {
            "name": "pre_post_gate",
            "description": "Pre + Post gate",
            "mitigation": "policy_gate@0.1.0",
            "out_file": "pre_post_gate.jsonl",
        },
    ]

    console.print("[bold green]🚀 Starting Ablation Study[/bold green]")
    console.print("Suite: pi-injection")
    console.print(f"Model: {model_name}")
    console.print(f"Conditions: {len(conditions)}")
    console.print()

    # Run each condition
    results = {}

    for i, condition in enumerate(conditions, 1):
        console.print(
            f"[bold blue]📊 Running condition {i}/{len(conditions)}: {condition['description']}[/bold blue]"
        )

        out_path = output_path / condition["out_file"]

        try:
            run_eval(
                suite_name="pi-injection",
                model_name=model_name,
                config_path=config_path,
                out_path=str(out_path),
                mitigation_id=condition["mitigation"],
            )

            # Load and compute metrics
            records = [OutputRecord(**r) for r in read_jsonl(str(out_path))]
            metrics = aggregate_metrics(records)

            results[condition["name"]] = {
                "condition": condition,
                "metrics": metrics,
                "file": str(out_path),
            }

            console.print(f"[green]✓[/green] Completed {condition['name']}: {len(records)} records")

        except Exception as e:
            console.print(f"[bold red]✗ Failed {condition['name']}:[/bold red] {e}")
            raise

    console.print()
    console.print("[bold green]📈 Computing Comparison Metrics[/bold green]")

    # Generate comparison data
    comparison_data = generate_comparison_data(results)

    # Save comparison JSON
    comparison_file = output_path / "ablation_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison_data, f, indent=2, default=str)

    console.print(f"[green]✓[/green] Saved comparison data: {comparison_file}")

    # Generate HTML report
    report_file = output_path / "ablation_report.html"
    generate_ablation_report(results, str(report_file))

    console.print(f"[green]✓[/green] Generated HTML report: {report_file}")
    console.print()

    # Display summary table
    display_summary_table(results)

    console.print("[bold green]🎯 Ablation Study Complete![/bold green]")
    console.print(f"Results saved to: {output_dir}")


def generate_comparison_data(results: dict[str, Any]) -> dict[str, Any]:
    """Generate comparison data structure for JSON output."""
    comparison = {"study_type": "ablation", "conditions": {}, "deltas": {}}

    # Store each condition's metrics
    for name, result in results.items():
        comparison["conditions"][name] = {
            "description": result["condition"]["description"],
            "mitigation": result["condition"]["mitigation"],
            "metrics": (
                result["metrics"].__dict__
                if hasattr(result["metrics"], "__dict__")
                else dict(result["metrics"])
            ),
        }

    # Calculate deltas vs baseline
    if "baseline" in results:
        baseline_metrics = results["baseline"]["metrics"]

        for name, result in results.items():
            if name != "baseline":
                metrics = result["metrics"]
                deltas = {}

                # Calculate percentage point changes
                if hasattr(baseline_metrics, "safety_coverage") and hasattr(
                    metrics, "safety_coverage"
                ):
                    deltas["safety_coverage_delta"] = (
                        metrics.safety_coverage - baseline_metrics.safety_coverage
                    ) * 100
                if hasattr(baseline_metrics, "benign_pass_rate") and hasattr(
                    metrics, "benign_pass_rate"
                ):
                    deltas["benign_pass_rate_delta"] = (
                        metrics.benign_pass_rate - baseline_metrics.benign_pass_rate
                    ) * 100
                if hasattr(baseline_metrics, "false_positive_rate") and hasattr(
                    metrics, "false_positive_rate"
                ):
                    deltas["false_positive_rate_delta"] = (
                        metrics.false_positive_rate - baseline_metrics.false_positive_rate
                    ) * 100
                if hasattr(baseline_metrics, "injection_success_rate") and hasattr(
                    metrics, "injection_success_rate"
                ):
                    deltas["injection_success_rate_delta"] = (
                        metrics.injection_success_rate - baseline_metrics.injection_success_rate
                    ) * 100

                # Cost and latency changes
                if hasattr(baseline_metrics, "total_cost_usd") and hasattr(
                    metrics, "total_cost_usd"
                ):
                    deltas["cost_delta"] = metrics.total_cost_usd - baseline_metrics.total_cost_usd
                if hasattr(baseline_metrics, "latency_p95") and hasattr(metrics, "latency_p95"):
                    deltas["latency_p95_delta"] = metrics.latency_p95 - baseline_metrics.latency_p95

                comparison["deltas"][name] = deltas

    return comparison


def generate_ablation_report(results: dict[str, Any], output_path: str) -> None:
    """Generate comprehensive HTML report for ablation study."""

    # Create enhanced HTML template for ablation
    ablation_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Safety Evals Ablation Study</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .conditions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .condition-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .condition-title {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .condition-desc {
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .metrics-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        .metric-name { font-weight: 500; }
        .metric-value { font-weight: bold; }
        .delta-positive { color: #27ae60; }
        .delta-negative { color: #e74c3c; }
        .delta-neutral { color: #3498db; }
        .comparison-table {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .comparison-table table {
            width: 100%;
            border-collapse: collapse;
        }
        .comparison-table th {
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .comparison-table td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        .comparison-table tr:hover {
            background: #f8f9fa;
        }
    </style>
</head>
<body>
    <h1>🔬 Safety Evals Ablation Study</h1>

    <h2>📊 Conditions Overview</h2>
    <div class="conditions-grid">
        {% for name, result in results.items() %}
        <div class="condition-card">
            <div class="condition-title">{{ result.condition.name.title() }}</div>
            <div class="condition-desc">{{ result.condition.description }}</div>
            <div class="condition-desc">Mitigation: {{ result.condition.mitigation }}</div>

            <div class="metrics-row">
                <span class="metric-name">Safety Coverage</span>
                <span class="metric-value">{{ "%.1f%%" | format(result.metrics.safety_coverage * 100) }}</span>
            </div>
            <div class="metrics-row">
                <span class="metric-name">Benign Pass Rate</span>
                <span class="metric-value">{{ "%.1f%%" | format(result.metrics.benign_pass_rate * 100) }}</span>
            </div>
            <div class="metrics-row">
                <span class="metric-name">False Positive Rate</span>
                <span class="metric-value">{{ "%.1f%%" | format(result.metrics.false_positive_rate * 100) }}</span>
            </div>
            <div class="metrics-row">
                <span class="metric-name">Injection Success</span>
                <span class="metric-value">{{ "%.1f%%" | format(result.metrics.injection_success_rate * 100) }}</span>
            </div>
            <div class="metrics-row">
                <span class="metric-name">Total Cost</span>
                <span class="metric-value">${{ "%.4f" | format(result.metrics.total_cost_usd) }}</span>
            </div>
            <div class="metrics-row">
                <span class="metric-name">Latency p95</span>
                <span class="metric-value">{{ "%.0f" | format(result.metrics.latency_p95) }}ms</span>
            </div>
        </div>
        {% endfor %}
    </div>

    <h2>📈 Comparison Table</h2>
    <div class="comparison-table">
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Pre Gate</th>
                    <th>Pre+Post Gate</th>
                    <th>Δ Pre vs Baseline</th>
                    <th>Δ Pre+Post vs Baseline</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Safety Coverage</td>
                    <td>{{ "%.1f%%" | format(results.baseline.metrics.safety_coverage * 100) }}</td>
                    <td>{{ "%.1f%%" | format(results.pre_gate.metrics.safety_coverage * 100) }}</td>
                    <td>{{ "%.1f%%" | format(results.pre_post_gate.metrics.safety_coverage * 100) }}</td>
                    <td class="delta-positive">{{ "%+.1f pp" | format((results.pre_gate.metrics.safety_coverage - results.baseline.metrics.safety_coverage) * 100) }}</td>
                    <td class="delta-positive">{{ "%+.1f pp" | format((results.pre_post_gate.metrics.safety_coverage - results.baseline.metrics.safety_coverage) * 100) }}</td>
                </tr>
                <tr>
                    <td>Benign Pass Rate</td>
                    <td>{{ "%.1f%%" | format(results.baseline.metrics.benign_pass_rate * 100) }}</td>
                    <td>{{ "%.1f%%" | format(results.pre_gate.metrics.benign_pass_rate * 100) }}</td>
                    <td>{{ "%.1f%%" | format(results.pre_post_gate.metrics.benign_pass_rate * 100) }}</td>
                    <td class="delta-negative">{{ "%+.1f pp" | format((results.pre_gate.metrics.benign_pass_rate - results.baseline.metrics.benign_pass_rate) * 100) }}</td>
                    <td class="delta-negative">{{ "%+.1f pp" | format((results.pre_post_gate.metrics.benign_pass_rate - results.baseline.metrics.benign_pass_rate) * 100) }}</td>
                </tr>
                <tr>
                    <td>False Positive Rate</td>
                    <td>{{ "%.1f%%" | format(results.baseline.metrics.false_positive_rate * 100) }}</td>
                    <td>{{ "%.1f%%" | format(results.pre_gate.metrics.false_positive_rate * 100) }}</td>
                    <td>{{ "%.1f%%" | format(results.pre_post_gate.metrics.false_positive_rate * 100) }}</td>
                    <td class="delta-negative">{{ "%+.1f pp" | format((results.pre_gate.metrics.false_positive_rate - results.baseline.metrics.false_positive_rate) * 100) }}</td>
                    <td class="delta-negative">{{ "%+.1f pp" | format((results.pre_post_gate.metrics.false_positive_rate - results.baseline.metrics.false_positive_rate) * 100) }}</td>
                </tr>
                <tr>
                    <td>Injection Success</td>
                    <td>{{ "%.1f%%" | format(results.baseline.metrics.injection_success_rate * 100) }}</td>
                    <td>{{ "%.1f%%" | format(results.pre_gate.metrics.injection_success_rate * 100) }}</td>
                    <td>{{ "%.1f%%" | format(results.pre_post_gate.metrics.injection_success_rate * 100) }}</td>
                    <td class="delta-positive">{{ "%+.1f pp" | format((results.pre_gate.metrics.injection_success_rate - results.baseline.metrics.injection_success_rate) * 100) }}</td>
                    <td class="delta-positive">{{ "%+.1f pp" | format((results.pre_post_gate.metrics.injection_success_rate - results.baseline.metrics.injection_success_rate) * 100) }}</td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
</html>
"""

    from datetime import datetime

    from jinja2 import Template

    template = Template(ablation_html)
    html = template.render(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), results=results)

    with open(output_path, "w") as f:
        f.write(html)


def display_summary_table(results: dict[str, Any]) -> None:
    """Display summary comparison table in terminal."""

    table = Table(title="🎯 Ablation Study Results", show_header=True, header_style="bold magenta")

    table.add_column("Condition", style="dim", width=15)
    table.add_column("Safety Coverage", justify="center")
    table.add_column("Benign Pass Rate", justify="center")
    table.add_column("False Positive", justify="center")
    table.add_column("Injection Success", justify="center")
    table.add_column("Cost ($)", justify="center")
    table.add_column("Latency p95 (ms)", justify="center")

    for name, result in results.items():
        metrics = result["metrics"]
        # Handle both object attributes and dictionary keys
        safety_coverage = getattr(metrics, "safety_coverage", metrics.get("safety_coverage", 0))
        benign_pass_rate = getattr(metrics, "benign_pass_rate", metrics.get("benign_pass_rate", 0))
        false_positive_rate = getattr(
            metrics, "false_positive_rate", metrics.get("false_positive_rate", 0)
        )
        injection_success_rate = getattr(
            metrics, "injection_success_rate", metrics.get("injection_success_rate", 0)
        )
        total_cost_usd = getattr(metrics, "total_cost_usd", metrics.get("total_cost_usd", 0))
        latency_p95 = getattr(metrics, "latency_p95", metrics.get("latency_p95", 0))

        table.add_row(
            result["condition"]["description"],
            f"{safety_coverage * 100:.1f}%",
            f"{benign_pass_rate * 100:.1f}%",
            f"{false_positive_rate * 100:.1f}%",
            f"{injection_success_rate * 100:.1f}%",
            f"{total_cost_usd:.4f}",
            f"{latency_p95:.0f}",
        )

    console.print()
    console.print(table)
    console.print()

    # Show delta summary if we have baseline
    if "baseline" in results:
        console.print("[bold]📈 Changes vs Baseline:[/bold]")

        baseline_metrics = results["baseline"]["metrics"]
        baseline_safety = getattr(
            baseline_metrics, "safety_coverage", baseline_metrics.get("safety_coverage", 0)
        )
        baseline_benign = getattr(
            baseline_metrics, "benign_pass_rate", baseline_metrics.get("benign_pass_rate", 0)
        )

        for name, result in results.items():
            if name != "baseline":
                metrics = result["metrics"]
                desc = result["condition"]["description"]

                safety_coverage = getattr(
                    metrics, "safety_coverage", metrics.get("safety_coverage", 0)
                )
                benign_pass_rate = getattr(
                    metrics, "benign_pass_rate", metrics.get("benign_pass_rate", 0)
                )

                safety_delta = (safety_coverage - baseline_safety) * 100
                benign_delta = (benign_pass_rate - baseline_benign) * 100

                console.print(f"  {desc}:")
                console.print(f"    Safety: {safety_delta:+.1f} pp, Benign: {benign_delta:+.1f} pp")

        console.print()
