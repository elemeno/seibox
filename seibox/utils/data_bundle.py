"""Data bundle generation for landscape evaluation results."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy import stats

from seibox.runners.matrix import Plan


def calculate_confidence_interval(
    values: List[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate confidence interval for a list of values.

    Args:
        values: List of numeric values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        return (0.0, 0.0)

    mean = np.mean(values)
    std_err = stats.sem(values)

    # Use t-distribution for small samples
    degrees_freedom = len(values) - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, degrees_freedom)

    margin_error = t_critical * std_err

    return (mean - margin_error, mean + margin_error)


def extract_detailed_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract detailed metrics from evaluation summary."""
    metrics = {}

    # Core metrics
    metrics["safety_coverage"] = summary.get("safety_coverage", 0) * 100
    metrics["benign_pass_rate"] = summary.get("benign_pass_rate", 0) * 100
    metrics["false_positive_rate"] = summary.get("false_positive_rate", 0) * 100
    metrics["injection_success_rate"] = summary.get("injection_success_rate", 0) * 100

    # Cost metrics
    cost_info = summary.get("cost", {})
    metrics["total_cost_usd"] = cost_info.get("total_usd", 0)
    metrics["cost_per_1k"] = summary.get("cost_per_1k", 0)

    # Latency metrics
    metrics["latency_p50"] = summary.get("latency_p50", 0)
    metrics["latency_p95"] = summary.get("latency_p95", 0)
    metrics["latency_mean"] = summary.get("latency_mean", 0)

    # Token usage
    metrics["total_input_tokens"] = cost_info.get("total_input_tokens", 0)
    metrics["total_output_tokens"] = cost_info.get("total_output_tokens", 0)
    metrics["total_tokens"] = metrics["total_input_tokens"] + metrics["total_output_tokens"]

    # Sample size
    metrics["total_calls"] = summary.get("total_calls", 0)

    # Entity-specific metrics (if available)
    entity_metrics = summary.get("entity_metrics", {})
    for entity, data in entity_metrics.items():
        if isinstance(data, dict):
            metrics[f"entity_{entity}_leak_rate"] = data.get("leak_rate", 0) * 100
            metrics[f"entity_{entity}_detected_count"] = data.get("detected_count", 0)

    # Severity metrics (if available)
    severity_metrics = summary.get("severity_metrics", {})
    for severity, data in severity_metrics.items():
        if isinstance(data, dict):
            metrics[f"severity_{severity}_leak_rate"] = data.get("leak_rate", 0) * 100
            metrics[f"severity_{severity}_detected_records"] = data.get("detected_records", 0)

    return metrics


def generate_parquet_bundle(plan: Plan, output_path: str) -> None:
    """Generate normalized Parquet data bundle from evaluation results.

    Args:
        plan: Completed evaluation plan
        output_path: Output path for Parquet file
    """
    records = []

    for job in plan.jobs:
        if job.status != "completed":
            # Include failed jobs with error information
            if job.status == "failed":
                records.append(
                    {
                        "model": job.model,
                        "category": job.category,
                        "metric": "status",
                        "value": 0,  # 0 for failed
                        "n": 0,
                        "ci_low": 0,
                        "ci_high": 0,
                        "cost_per_1k": 0,
                        "p95_ms": 0,
                        "error": job.error or "Unknown error",
                        "sample_size": job.sample_size,
                        "estimated_cost": job.estimated_cost,
                        "actual_duration_seconds": job.actual_duration or 0,
                    }
                )
            continue

        summary_path = Path(job.output_path).with_suffix(".summary.json")
        if not summary_path.exists():
            continue

        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)

            # Extract all metrics
            metrics = extract_detailed_metrics(summary)

            # Add metadata
            base_record = {
                "model": job.model,
                "category": job.category,
                "sample_size": job.sample_size,
                "estimated_cost": job.estimated_cost,
                "actual_cost": job.actual_cost or 0,
                "actual_duration_seconds": job.actual_duration or 0,
                "error": None,
            }

            # Create records for each metric
            for metric_name, value in metrics.items():
                record = base_record.copy()
                record.update(
                    {
                        "metric": metric_name,
                        "value": value,
                        "n": metrics.get("total_calls", 0),
                        "cost_per_1k": metrics.get("cost_per_1k", 0),
                        "p95_ms": metrics.get("latency_p95", 0),
                    }
                )

                # For rate-based metrics, calculate confidence intervals
                if any(rate_term in metric_name for rate_term in ["rate", "coverage", "success"]):
                    n = record["n"]
                    p = value / 100  # Convert percentage to proportion

                    if n > 0 and 0 <= p <= 1:
                        # Wilson confidence interval for proportions
                        z = 1.96  # 95% confidence
                        denominator = 1 + z**2 / n
                        centre_adjusted_probability = p + z**2 / (2 * n)
                        adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

                        lower_bound = (
                            centre_adjusted_probability - z * adjusted_standard_deviation
                        ) / denominator
                        upper_bound = (
                            centre_adjusted_probability + z * adjusted_standard_deviation
                        ) / denominator

                        record["ci_low"] = max(0, lower_bound * 100)
                        record["ci_high"] = min(100, upper_bound * 100)
                    else:
                        record["ci_low"] = value
                        record["ci_high"] = value
                else:
                    # For non-rate metrics, use simple bounds
                    record["ci_low"] = value
                    record["ci_high"] = value

                records.append(record)

        except Exception as e:
            # Add error record
            records.append(
                {
                    "model": job.model,
                    "category": job.category,
                    "metric": "error",
                    "value": 0,
                    "n": 0,
                    "ci_low": 0,
                    "ci_high": 0,
                    "cost_per_1k": 0,
                    "p95_ms": 0,
                    "error": f"Failed to parse results: {str(e)}",
                    "sample_size": job.sample_size,
                    "estimated_cost": job.estimated_cost,
                    "actual_duration_seconds": job.actual_duration or 0,
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Add derived columns
    df["timestamp"] = pd.Timestamp.now()
    df["model_provider"] = df["model"].str.split(":").str[0]
    df["model_name"] = df["model"].str.split(":").str[1]

    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Save to Parquet
    df.to_parquet(output_path_obj, index=False, engine="pyarrow")

    # Also save metadata as JSON
    metadata = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "total_records": len(df),
        "models": sorted(df["model"].unique().tolist()),
        "categories": sorted(df["category"].unique().tolist()),
        "metrics": sorted(df["metric"].unique().tolist()),
        "providers": sorted(df["model_provider"].unique().tolist()),
        "schema": {
            "model": "Model identifier (provider:name)",
            "category": "Evaluation category ID",
            "metric": "Metric name",
            "value": "Metric value",
            "n": "Sample size for this metric",
            "ci_low": "Lower bound of 95% confidence interval",
            "ci_high": "Upper bound of 95% confidence interval",
            "cost_per_1k": "Cost per 1000 API calls (USD)",
            "p95_ms": "P95 latency in milliseconds",
            "error": "Error message if evaluation failed",
            "sample_size": "Number of samples in evaluation",
            "estimated_cost": "Pre-evaluation cost estimate (USD)",
            "actual_cost": "Actual evaluation cost (USD)",
            "actual_duration_seconds": "Wall clock time for evaluation",
            "timestamp": "When this data was generated",
            "model_provider": "Model provider (openai, anthropic, etc.)",
            "model_name": "Model name without provider prefix",
        },
    }

    metadata_path = output_path_obj.with_suffix(".metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated data bundle: {output_path_obj}")
    print(f"Records: {len(df):,}")
    print(f"Models: {len(df['model'].unique())}")
    print(f"Categories: {len(df['category'].unique())}")
    print(f"Metrics: {len(df['metric'].unique())}")


def load_parquet_bundle(parquet_path: str) -> pd.DataFrame:
    """Load and validate Parquet data bundle.

    Args:
        parquet_path: Path to Parquet file

    Returns:
        DataFrame with evaluation results
    """
    df = pd.read_parquet(parquet_path)

    # Basic validation
    required_columns = ["model", "category", "metric", "value"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def get_metric_summary(df: pd.DataFrame, metric_name: str) -> Dict[str, Any]:
    """Get summary statistics for a specific metric across all models/categories.

    Args:
        df: DataFrame from load_parquet_bundle
        metric_name: Name of metric to summarize

    Returns:
        Dictionary with summary statistics
    """
    metric_data = df[df["metric"] == metric_name]

    if metric_data.empty:
        return {"error": f"Metric {metric_name} not found"}

    return {
        "metric": metric_name,
        "count": len(metric_data),
        "mean": metric_data["value"].mean(),
        "std": metric_data["value"].std(),
        "min": metric_data["value"].min(),
        "max": metric_data["value"].max(),
        "median": metric_data["value"].median(),
        "models": sorted(metric_data["model"].unique().tolist()),
        "categories": sorted(metric_data["category"].unique().tolist()),
    }
