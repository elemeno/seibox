"""Release report generator for comprehensive safety evaluation results."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from jinja2 import Template
import pandas as pd

from seibox.runners.matrix import Plan


def load_release_results(plan: Plan) -> Dict[str, Any]:
    """Load evaluation results from completed release jobs."""
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for job in plan.jobs:
        if job.status != "completed":
            continue

        summary_path = Path(job.output_path).with_suffix(".summary.json")
        if not summary_path.exists():
            continue

        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)

            # Store results by model -> mitigation_combo -> category
            model = job.model
            combo = job.mitigation_combo or "baseline"
            category = job.category

            if model not in results:
                results[model] = {}
            if combo not in results[model]:
                results[model][combo] = {}

            results[model][combo][category] = {"summary": summary, "job": job}

        except Exception:
            continue

    return results


def extract_safety_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract safety state metrics for tables and comparisons."""
    safety_data = {
        "models": list(results.keys()),
        "mitigation_combos": ["baseline", "policy_gate", "prompt_hardening", "both"],
        "categories": ["pii", "injection", "benign"],
        "metrics": {},
        "costs": {},
        "improvements": {},
    }

    # Extract metrics for each model x combo x category
    for model in safety_data["models"]:
        safety_data["metrics"][model] = {}
        safety_data["costs"][model] = {}

        for combo in safety_data["mitigation_combos"]:
            if combo not in results[model]:
                continue

            safety_data["metrics"][model][combo] = {}
            safety_data["costs"][model][combo] = {}

            total_cost = 0.0
            total_calls = 0

            for category in safety_data["categories"]:
                if category not in results[model][combo]:
                    continue

                data = results[model][combo][category]
                summary = data["summary"]

                # Extract key safety metrics
                metrics = {}

                if category == "pii":
                    # PII protection metrics
                    metrics["safety_coverage"] = summary.get("aggregate", {}).get(
                        "safety_coverage", 0.0
                    )
                    metrics["pii_leak_rate"] = summary.get("aggregate", {}).get(
                        "pii_leak_rate", 0.0
                    )
                    metrics["false_positive_rate"] = summary.get("aggregate", {}).get(
                        "false_positive_rate", 0.0
                    )

                elif category == "injection":
                    # Injection resistance metrics
                    metrics["safety_coverage"] = summary.get("aggregate", {}).get(
                        "safety_coverage", 0.0
                    )
                    metrics["injection_success_rate"] = summary.get("aggregate", {}).get(
                        "injection_success_rate", 0.0
                    )
                    metrics["violation_rate"] = summary.get("aggregate", {}).get(
                        "violation_rate", 0.0
                    )

                elif category == "benign":
                    # Benign helpfulness metrics
                    metrics["benign_pass_rate"] = summary.get("aggregate", {}).get(
                        "benign_pass_rate", 0.0
                    )
                    metrics["helpfulness_score"] = summary.get("aggregate", {}).get(
                        "helpfulness_score", 0.0
                    )
                    metrics["false_positive_rate"] = summary.get("aggregate", {}).get(
                        "false_positive_rate", 0.0
                    )

                # Cost and performance data
                cost_data = summary.get("cost", {})
                total_cost += cost_data.get("total_usd", 0.0)
                total_calls += cost_data.get("total_calls", 0)

                metrics["latency_p95"] = summary.get("performance", {}).get("latency_p95_ms", 0.0)
                metrics["cost_per_call"] = cost_data.get("usd_per_call", 0.0)

                safety_data["metrics"][model][combo][category] = metrics

            safety_data["costs"][model][combo] = {
                "total_usd": total_cost,
                "total_calls": total_calls,
                "usd_per_call": total_cost / total_calls if total_calls > 0 else 0.0,
            }

    # Calculate improvements relative to baseline
    for model in safety_data["models"]:
        safety_data["improvements"][model] = {}

        baseline_metrics = safety_data["metrics"][model].get("baseline", {})

        for combo in ["policy_gate", "prompt_hardening", "both"]:
            if combo not in safety_data["metrics"][model]:
                continue

            combo_metrics = safety_data["metrics"][model][combo]
            safety_data["improvements"][model][combo] = {}

            for category in safety_data["categories"]:
                if category not in baseline_metrics or category not in combo_metrics:
                    continue

                baseline = baseline_metrics[category]
                current = combo_metrics[category]
                improvements = {}

                # Calculate percentage point improvements
                for metric in baseline.keys():
                    if metric in current and isinstance(baseline[metric], (int, float)):
                        baseline_val = baseline[metric]
                        current_val = current[metric]

                        if metric in ["safety_coverage", "benign_pass_rate", "helpfulness_score"]:
                            # Higher is better
                            improvements[f"{metric}_improvement"] = (
                                current_val - baseline_val
                            ) * 100
                        elif metric in [
                            "pii_leak_rate",
                            "injection_success_rate",
                            "violation_rate",
                            "false_positive_rate",
                        ]:
                            # Lower is better
                            improvements[f"{metric}_improvement"] = (
                                baseline_val - current_val
                            ) * 100
                        else:
                            # Raw difference for latency, cost
                            improvements[f"{metric}_change"] = current_val - baseline_val

                safety_data["improvements"][model][combo][category] = improvements

    return safety_data


def create_safety_state_tables(safety_data: Dict[str, Any]) -> Dict[str, str]:
    """Create HTML tables showing safety state across models and mitigations."""
    tables = {}

    for category in safety_data["categories"]:
        # Create DataFrame for this category
        table_data = []

        for model in safety_data["models"]:
            for combo in safety_data["mitigation_combos"]:
                if (
                    combo in safety_data["metrics"][model]
                    and category in safety_data["metrics"][model][combo]
                ):

                    metrics = safety_data["metrics"][model][combo][category]
                    cost_data = safety_data["costs"][model][combo]

                    row = {
                        "Model": model,
                        "Mitigation": combo.replace("_", " ").title(),
                        "Cost (USD)": f"${cost_data['total_usd']:.4f}",
                        "Cost/Call": f"${cost_data['usd_per_call']:.6f}",
                    }

                    # Add category-specific metrics
                    if category == "pii":
                        row.update(
                            {
                                "Safety Coverage": f"{metrics.get('safety_coverage', 0):.1%}",
                                "PII Leak Rate": f"{metrics.get('pii_leak_rate', 0):.1%}",
                                "False Positive Rate": f"{metrics.get('false_positive_rate', 0):.1%}",
                            }
                        )
                    elif category == "injection":
                        row.update(
                            {
                                "Safety Coverage": f"{metrics.get('safety_coverage', 0):.1%}",
                                "Injection Success": f"{metrics.get('injection_success_rate', 0):.1%}",
                                "Violation Rate": f"{metrics.get('violation_rate', 0):.1%}",
                            }
                        )
                    elif category == "benign":
                        row.update(
                            {
                                "Benign Pass Rate": f"{metrics.get('benign_pass_rate', 0):.1%}",
                                "Helpfulness Score": f"{metrics.get('helpfulness_score', 0):.2f}",
                                "False Positive Rate": f"{metrics.get('false_positive_rate', 0):.1%}",
                            }
                        )

                    row["Latency P95 (ms)"] = f"{metrics.get('latency_p95', 0):.0f}"
                    table_data.append(row)

        if table_data:
            df = pd.DataFrame(table_data)

            # Convert to HTML with styling
            html_table = df.to_html(
                index=False,
                classes="table table-striped table-hover",
                table_id=f"safety-table-{category}",
                escape=False,
            )

            tables[category] = html_table

    return tables


def create_improvement_tables(safety_data: Dict[str, Any]) -> Dict[str, str]:
    """Create HTML tables showing mitigation effectiveness."""
    tables = {}

    for category in safety_data["categories"]:
        table_data = []

        for model in safety_data["models"]:
            for combo in ["policy_gate", "prompt_hardening", "both"]:
                if (
                    combo in safety_data["improvements"][model]
                    and category in safety_data["improvements"][model][combo]
                ):

                    improvements = safety_data["improvements"][model][combo][category]

                    row = {"Model": model, "Mitigation": combo.replace("_", " ").title()}

                    # Add category-specific improvement metrics
                    for metric, value in improvements.items():
                        if metric.endswith("_improvement"):
                            metric_name = (
                                metric.replace("_improvement", "").replace("_", " ").title()
                            )
                            if value > 0:
                                row[f"{metric_name} Δ"] = f"+{value:.1f}pp"
                            else:
                                row[f"{metric_name} Δ"] = f"{value:.1f}pp"
                        elif metric.endswith("_change"):
                            metric_name = metric.replace("_change", "").replace("_", " ").title()
                            if "cost" in metric.lower():
                                row[f"{metric_name} Δ"] = f"${value:.4f}"
                            else:
                                row[f"{metric_name} Δ"] = f"{value:.1f}"

                    table_data.append(row)

        if table_data:
            df = pd.DataFrame(table_data)
            html_table = df.to_html(
                index=False,
                classes="table table-striped table-hover",
                table_id=f"improvement-table-{category}",
                escape=False,
            )
            tables[category] = html_table

    return tables


def generate_release_report(plan: Plan, output_path: str) -> None:
    """Generate comprehensive release report HTML."""

    # Load and process results
    results = load_release_results(plan)
    if not results:
        raise ValueError("No completed results found in plan")

    safety_data = extract_safety_metrics(results)
    safety_tables = create_safety_state_tables(safety_data)
    improvement_tables = create_improvement_tables(safety_data)

    # Calculate summary statistics
    total_jobs = len(plan.jobs)
    completed_jobs = len([job for job in plan.jobs if job.status == "completed"])
    failed_jobs = len([job for job in plan.jobs if job.status == "failed"])

    total_cost = sum(
        safety_data["costs"][model][combo]["total_usd"]
        for model in safety_data["costs"]
        for combo in safety_data["costs"][model]
    )

    # Generate timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # HTML template
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Release Safety Evaluation Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .metric-card { margin-bottom: 1rem; }
        .improvement-positive { color: #28a745; }
        .improvement-negative { color: #dc3545; }
        .table-container { margin-bottom: 2rem; }
        .category-section { margin-bottom: 3rem; }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <header class="text-center mb-5">
            <h1 class="display-4">Release Safety Evaluation Report</h1>
            <p class="lead text-muted">Comprehensive safety assessment across models and mitigations</p>
            <p class="text-muted">Generated: {{ timestamp }}</p>
        </header>

        <!-- Executive Summary -->
        <section class="mb-5">
            <h2>Executive Summary</h2>
            <div class="row">
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Total Evaluations</h5>
                            <h3 class="text-primary">{{ total_jobs }}</h3>
                            <small class="text-muted">{{ completed_jobs }} completed</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Models Tested</h5>
                            <h3 class="text-info">{{ models|length }}</h3>
                            <small class="text-muted">OpenAI, Anthropic, Gemini</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Total Cost</h5>
                            <h3 class="text-success">${{ "%.4f"|format(total_cost) }}</h3>
                            <small class="text-muted">All evaluations</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Success Rate</h5>
                            <h3 class="text-{{ 'success' if success_rate > 90 else 'warning' if success_rate > 75 else 'danger' }}">{{ "%.1f"|format(success_rate) }}%</h3>
                            <small class="text-muted">{{ completed_jobs }}/{{ total_jobs }}</small>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Safety State Tables -->
        {% for category in categories %}
        <section class="category-section">
            <h2>{{ category.title() }} Protection</h2>
            
            <div class="table-container">
                <h4>Safety State by Model and Mitigation</h4>
                {{ safety_tables[category]|safe }}
            </div>
            
            {% if category in improvement_tables %}
            <div class="table-container">
                <h4>Mitigation Effectiveness (vs Baseline)</h4>
                {{ improvement_tables[category]|safe }}
            </div>
            {% endif %}
        </section>
        {% endfor %}

        <!-- Model Rankings -->
        <section class="mb-5">
            <h2>Model Rankings</h2>
            <p class="text-muted">Based on aggregate safety metrics across all categories</p>
            <!-- Rankings table would go here -->
        </section>

        <!-- Methodology -->
        <section class="mb-5">
            <h2>Methodology</h2>
            <div class="row">
                <div class="col-md-6">
                    <h4>Evaluation Categories</h4>
                    <ul>
                        <li><strong>PII Protection:</strong> Resistance to personally identifiable information disclosure</li>
                        <li><strong>Injection Resistance:</strong> Robustness against prompt injection attacks</li>
                        <li><strong>Benign Helpfulness:</strong> Maintained helpfulness on legitimate requests</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h4>Mitigation Strategies</h4>
                    <ul>
                        <li><strong>Baseline:</strong> No additional safety mitigations</li>
                        <li><strong>Policy Gate:</strong> Post-processing content filtering</li>
                        <li><strong>Prompt Hardening:</strong> Enhanced system prompt instructions</li>
                        <li><strong>Combined:</strong> Both policy gate and prompt hardening</li>
                    </ul>
                </div>
            </div>
        </section>

        <footer class="text-center text-muted mt-5">
            <p>Generated by Safety Evals in a Box</p>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """

    # Render template
    template = Template(html_template)
    html_content = template.render(
        timestamp=timestamp,
        total_jobs=total_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        success_rate=(completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
        total_cost=total_cost,
        models=safety_data["models"],
        categories=safety_data["categories"],
        safety_tables=safety_tables,
        improvement_tables=improvement_tables,
    )

    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
