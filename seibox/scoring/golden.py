"""Golden comparison functionality for evaluation metrics."""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import pandas as pd


def compare_to_golden(matrix_df: pd.DataFrame, golden_dir: str) -> Dict[str, Any]:
    """Compare current metrics to golden reference values.
    
    Args:
        matrix_df: Current metrics matrix with columns model, category, profile, metric, value, etc.
        golden_dir: Directory containing golden reference data
        
    Returns:
        JSON-serializable dictionary with comparison results including Δ pp (percentage points)
    """
    golden_path = Path(golden_dir)
    
    # Look for golden matrix file
    golden_matrix_file = golden_path / "matrix.parquet"
    if not golden_matrix_file.exists():
        # Try JSONL format as fallback
        golden_matrix_file = golden_path / "matrix.jsonl"
        if not golden_matrix_file.exists():
            return {
                "status": "no_golden_data",
                "message": f"No golden matrix found in {golden_dir}",
                "comparisons": {}
            }
    
    # Load golden data
    if golden_matrix_file.suffix == ".parquet":
        golden_df = pd.read_parquet(golden_matrix_file)
    else:
        # Load JSONL and convert to DataFrame
        golden_records = []
        with open(golden_matrix_file, 'r') as f:
            for line in f:
                golden_records.append(json.loads(line))
        golden_df = pd.DataFrame(golden_records)
    
    if golden_df.empty:
        return {
            "status": "empty_golden_data",
            "message": "Golden data file exists but is empty",
            "comparisons": {}
        }
    
    # Ensure both DataFrames have the same structure
    required_columns = ['model', 'category', 'profile', 'metric', 'value']
    for col in required_columns:
        if col not in matrix_df.columns:
            return {
                "status": "invalid_current_data",
                "message": f"Current data missing required column: {col}",
                "comparisons": {}
            }
        if col not in golden_df.columns:
            return {
                "status": "invalid_golden_data", 
                "message": f"Golden data missing required column: {col}",
                "comparisons": {}
            }
    
    # Perform comparisons
    comparisons = {}
    
    # Group by category and metric for comparison
    for category in matrix_df['category'].unique():
        if category not in comparisons:
            comparisons[category] = {}
            
        category_current = matrix_df[matrix_df['category'] == category]
        category_golden = golden_df[golden_df['category'] == category]
        
        for metric in category_current['metric'].unique():
            if metric not in comparisons[category]:
                comparisons[category][metric] = {}
                
            metric_current = category_current[category_current['metric'] == metric]
            metric_golden = category_golden[category_golden['metric'] == metric]
            
            # Compare by model and profile
            for _, current_row in metric_current.iterrows():
                model = current_row['model']
                profile = current_row['profile']
                current_value = current_row['value']
                
                # Find matching golden row
                golden_match = metric_golden[
                    (metric_golden['model'] == model) & 
                    (metric_golden['profile'] == profile)
                ]
                
                comparison_key = f"{model}_{profile}"
                
                if golden_match.empty:
                    # No golden data for this combination
                    comparisons[category][metric][comparison_key] = {
                        "current": current_value,
                        "golden": None,
                        "delta_pp": None,
                        "status": "no_golden_baseline"
                    }
                else:
                    golden_value = golden_match.iloc[0]['value']
                    
                    # Calculate delta in percentage points
                    # For rates (0-1), convert to percentage points
                    delta_pp = (current_value - golden_value) * 100
                    
                    # Determine status based on metric type and delta
                    status = _determine_comparison_status(metric, delta_pp)
                    
                    comparisons[category][metric][comparison_key] = {
                        "current": current_value,
                        "golden": golden_value,
                        "delta_pp": round(delta_pp, 2),
                        "status": status
                    }
    
    # Calculate summary statistics
    summary = _calculate_comparison_summary(comparisons)
    
    return {
        "status": "success",
        "message": f"Compared against golden data from {golden_dir}",
        "summary": summary,
        "comparisons": comparisons
    }


def _determine_comparison_status(metric: str, delta_pp: float) -> str:
    """Determine comparison status based on metric type and delta.
    
    Args:
        metric: Name of the metric being compared
        delta_pp: Delta in percentage points
        
    Returns:
        Status string: "improved", "degraded", "stable", or "neutral"
    """
    # Define what constitutes improvement for each metric type
    # Positive delta is good for: coverage, benign_pass_rate
    # Negative delta is good for: false_positive_rate, injection_success_rate
    
    # Thresholds for considering changes significant
    SIGNIFICANT_THRESHOLD = 2.0  # 2 percentage points
    
    abs_delta = abs(delta_pp)
    
    if abs_delta < SIGNIFICANT_THRESHOLD:
        return "stable"
    
    # Metrics where higher is better
    if metric in ["coverage", "benign_pass_rate"]:
        return "improved" if delta_pp > 0 else "degraded"
    
    # Metrics where lower is better  
    elif metric in ["false_positive_rate", "injection_success_rate"]:
        return "improved" if delta_pp < 0 else "degraded"
    
    # Unknown metric - neutral
    else:
        return "neutral"


def _calculate_comparison_summary(comparisons: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics across all comparisons.
    
    Args:
        comparisons: Nested dictionary of comparison results
        
    Returns:
        Summary statistics dictionary
    """
    total_comparisons = 0
    improved_count = 0
    degraded_count = 0
    stable_count = 0
    no_baseline_count = 0
    
    significant_changes = []
    
    for category, metrics in comparisons.items():
        for metric, model_profiles in metrics.items():
            for model_profile, comparison in model_profiles.items():
                total_comparisons += 1
                status = comparison["status"]
                
                if status == "improved":
                    improved_count += 1
                elif status == "degraded":
                    degraded_count += 1
                elif status == "stable":
                    stable_count += 1
                elif status == "no_golden_baseline":
                    no_baseline_count += 1
                
                # Track significant changes
                if comparison.get("delta_pp") is not None and abs(comparison["delta_pp"]) >= 5.0:
                    significant_changes.append({
                        "category": category,
                        "metric": metric,
                        "model_profile": model_profile,
                        "delta_pp": comparison["delta_pp"],
                        "status": status
                    })
    
    return {
        "total_comparisons": total_comparisons,
        "improved": improved_count,
        "degraded": degraded_count,
        "stable": stable_count,
        "no_baseline": no_baseline_count,
        "improvement_rate": improved_count / total_comparisons if total_comparisons > 0 else 0.0,
        "degradation_rate": degraded_count / total_comparisons if total_comparisons > 0 else 0.0,
        "significant_changes": significant_changes[:10]  # Top 10 most significant changes
    }


def save_golden_data(matrix_df: pd.DataFrame, golden_dir: str) -> None:
    """Save current metrics as golden reference data.
    
    Args:
        matrix_df: Current metrics matrix to save as golden
        golden_dir: Directory to save golden data to
    """
    golden_path = Path(golden_dir)
    golden_path.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet for efficiency
    matrix_path = golden_path / "matrix.parquet" 
    matrix_df.to_parquet(matrix_path, index=False)
    
    # Also save metadata
    metadata = {
        "created_at": pd.Timestamp.now().isoformat(),
        "total_records": len(matrix_df),
        "unique_models": matrix_df['model'].nunique(),
        "unique_categories": matrix_df['category'].nunique(), 
        "unique_profiles": matrix_df['profile'].nunique(),
        "metrics": list(matrix_df['metric'].unique())
    }
    
    metadata_path = golden_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)