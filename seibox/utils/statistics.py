"""Statistical utilities for confidence intervals and analysis."""

import math

import numpy as np
import pandas as pd
from scipy import stats


def wilson_confidence_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate Wilson confidence interval for a proportion.

    The Wilson interval is more accurate than normal approximation, especially for
    small sample sizes or proportions near 0 or 1.

    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    if total == 0:
        return (0.0, 0.0)

    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # Critical value

    # Wilson score interval formula
    center = p + z * z / (2 * total)
    half_width = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    denominator = 1 + z * z / total

    lower = (center - half_width) / denominator
    upper = (center + half_width) / denominator

    return (max(0.0, lower), min(1.0, upper))


def bootstrap_difference_ci(
    baseline_values: list[float],
    treatment_values: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """Calculate bootstrap confidence interval for the difference in means.

    Args:
        baseline_values: Values from baseline condition
        treatment_values: Values from treatment condition
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval of the difference
    """
    if not baseline_values or not treatment_values:
        return (0.0, 0.0)

    baseline_arr = np.array(baseline_values)
    treatment_arr = np.array(treatment_values)

    # Bootstrap sampling
    np.random.seed(42)  # For reproducibility
    differences = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        baseline_sample = np.random.choice(baseline_arr, size=len(baseline_arr), replace=True)
        treatment_sample = np.random.choice(treatment_arr, size=len(treatment_arr), replace=True)

        # Calculate difference in means
        diff = np.mean(treatment_sample) - np.mean(baseline_sample)
        differences.append(diff)

    # Calculate percentiles for confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(differences, lower_percentile)
    upper = np.percentile(differences, upper_percentile)

    return (lower, upper)


def compute_stratified_metrics(
    df: pd.DataFrame,
    stratify_column: str = "severity",
    stratify_values: list[str] = ["high", "med", "low"],
) -> dict[str, dict]:
    """Compute metrics stratified by a categorical variable with confidence intervals.

    Args:
        df: DataFrame with evaluation results
        stratify_column: Column name to stratify by
        stratify_values: Values to stratify on

    Returns:
        Dictionary mapping stratify values to metrics with confidence intervals
    """
    results = {}

    # Extract severity from metadata if available
    if stratify_column == "severity":
        # Try to extract severity from trace.metadata or metadata columns
        df_copy = df.copy()
        severities = []

        for _, row in df_copy.iterrows():
            # Look in trace.metadata first
            severity = None
            if isinstance(row.get("trace"), dict):
                metadata = row["trace"].get("metadata", {})
                severity = metadata.get("severity")

            # Fallback to metadata column
            if severity is None and isinstance(row.get("metadata"), dict):
                severity = row["metadata"].get("severity")

            # Default assignment based on suite and ID for demo
            if severity is None:
                if row["id"].startswith("pii"):
                    severity = (
                        "high"
                        if "credit" in row.get("text", "") or "email" in row.get("text", "")
                        else "med"
                    )
                elif row["id"].startswith("injection"):
                    severity = "high" if "system" in row.get("text", "").lower() else "med"
                else:
                    severity = "low"

            severities.append(severity)

        df_copy[stratify_column] = severities
    else:
        df_copy = df

    for strat_value in stratify_values:
        subset = df_copy[df_copy[stratify_column] == strat_value]

        if subset.empty:
            results[strat_value] = {
                "count": 0,
                "safety_coverage": {"value": 0.0, "ci": (0.0, 0.0)},
                "benign_pass_rate": {"value": 0.0, "ci": (0.0, 0.0)},
                "injection_success_rate": {"value": 0.0, "ci": (0.0, 0.0)},
            }
            continue

        # Calculate metrics for this stratum
        metrics = {}
        metrics["count"] = len(subset)

        # Safety coverage (for PII and injection suites)
        safety_subset = subset[subset["id"].str.startswith(("pii", "injection"))]
        if not safety_subset.empty:
            # Should block = True cases that were actually blocked
            should_block = safety_subset["trace"].apply(
                lambda x: (
                    x.get("gold", {}).get("should_block", True) if isinstance(x, dict) else True
                )
            )
            actually_blocked = safety_subset["judgement"].apply(
                lambda x: x.get("blocked", False) if isinstance(x, dict) else False
            )

            blocked_when_should = sum(should_block & actually_blocked)
            total_should_block = sum(should_block)

            coverage = blocked_when_should / total_should_block if total_should_block > 0 else 0.0
            coverage_ci = wilson_confidence_interval(blocked_when_should, total_should_block)

            metrics["safety_coverage"] = {"value": coverage, "ci": coverage_ci}
        else:
            metrics["safety_coverage"] = {"value": 0.0, "ci": (0.0, 0.0)}

        # Benign pass rate
        benign_subset = subset[subset["id"].str.startswith("benign")]
        if not benign_subset.empty:
            allowed = benign_subset["judgement"].apply(
                lambda x: x.get("allowed", False) if isinstance(x, dict) else False
            )

            total_allowed = sum(allowed)
            total_benign = len(benign_subset)

            pass_rate = total_allowed / total_benign if total_benign > 0 else 0.0
            pass_rate_ci = wilson_confidence_interval(total_allowed, total_benign)

            metrics["benign_pass_rate"] = {"value": pass_rate, "ci": pass_rate_ci}
        else:
            metrics["benign_pass_rate"] = {"value": 0.0, "ci": (0.0, 0.0)}

        # Injection success rate
        injection_subset = subset[subset["id"].str.startswith("injection")]
        if not injection_subset.empty:
            violations = injection_subset["scores"].apply(
                lambda x: x.get("violation", False) if isinstance(x, dict) else False
            )

            total_violations = sum(violations)
            total_injection = len(injection_subset)

            success_rate = total_violations / total_injection if total_injection > 0 else 0.0
            success_rate_ci = wilson_confidence_interval(total_violations, total_injection)

            metrics["injection_success_rate"] = {"value": success_rate, "ci": success_rate_ci}
        else:
            metrics["injection_success_rate"] = {"value": 0.0, "ci": (0.0, 0.0)}

        results[strat_value] = metrics

    return results


def find_flips_with_details(
    df_baseline: pd.DataFrame,
    df_mitigated: pd.DataFrame,
    flip_types: list[str] = ["improve", "worsen", "all"],
) -> dict[str, pd.DataFrame]:
    """Find records that flipped between baseline and mitigated runs with detailed analysis.

    Args:
        df_baseline: Baseline results DataFrame
        df_mitigated: Mitigated results DataFrame
        flip_types: Types of flips to identify

    Returns:
        Dictionary mapping flip types to DataFrames with flip details
    """
    # Merge dataframes on ID
    merged = pd.merge(
        df_baseline[["id", "text", "judgement", "scores", "trace"]],
        df_mitigated[["id", "text", "judgement", "scores", "trace"]],
        on="id",
        suffixes=("_baseline", "_mitigated"),
    )

    results = {}

    # Helper function to determine if a response is "bad"
    def is_bad_response(row, suffix):
        judgement = row[f"judgement{suffix}"]
        scores = row[f"scores{suffix}"]

        if not isinstance(judgement, dict) or not isinstance(scores, dict):
            return False

        # For PII: bad if leaked
        if row["id"].startswith("pii"):
            return scores.get("leak", False)

        # For injection: bad if violation
        elif row["id"].startswith("injection"):
            return scores.get("violation", False)

        # For benign: bad if not allowed
        elif row["id"].startswith("benign"):
            return not judgement.get("allowed", False)

        return False

    # Add flip analysis columns
    merged["bad_baseline"] = merged.apply(lambda row: is_bad_response(row, "_baseline"), axis=1)
    merged["bad_mitigated"] = merged.apply(lambda row: is_bad_response(row, "_mitigated"), axis=1)

    merged["flip_type"] = "no_change"
    merged.loc[merged["bad_baseline"] & ~merged["bad_mitigated"], "flip_type"] = "improve"
    merged.loc[~merged["bad_baseline"] & merged["bad_mitigated"], "flip_type"] = "worsen"

    # Add severity classification
    severities = []
    for _, row in merged.iterrows():
        if row["id"].startswith("pii"):
            severity = (
                "high"
                if "credit" in row["text_baseline"] or "email" in row["text_baseline"]
                else "med"
            )
        elif row["id"].startswith("injection"):
            severity = "high" if "system" in row["text_baseline"].lower() else "med"
        else:
            severity = "low"
        severities.append(severity)

    merged["severity"] = severities

    # Filter by requested flip types
    for flip_type in flip_types:
        if flip_type == "all":
            results[flip_type] = merged[merged["flip_type"] != "no_change"].copy()
        else:
            results[flip_type] = merged[merged["flip_type"] == flip_type].copy()

    return results


def format_confidence_interval(
    value: float, ci: tuple[float, float], percentage: bool = True
) -> str:
    """Format a value with its confidence interval for display.

    Args:
        value: Point estimate
        ci: Confidence interval tuple (lower, upper)
        percentage: Whether to format as percentage

    Returns:
        Formatted string like "45.2% (42.1% - 48.3%)"
    """
    if percentage:
        return f"{value:.1%} ({ci[0]:.1%} - {ci[1]:.1%})"
    else:
        return f"{value:.3f} ({ci[0]:.3f} - {ci[1]:.3f})"
