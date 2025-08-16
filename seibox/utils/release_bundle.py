"""Release data bundle generator for comprehensive evaluation results export."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from seibox.runners.matrix import Plan
from seibox.utils.io import read_jsonl


def collect_release_data(plan: Plan) -> Dict[str, pd.DataFrame]:
    """Collect all evaluation data from completed release jobs."""
    
    # DataFrames for different aspects of the data
    dataframes = {
        'job_metadata': [],
        'evaluation_results': [],
        'aggregated_metrics': [],
        'cost_analysis': [],
        'mitigation_effectiveness': []
    }
    
    for job in plan.jobs:
        if job.status != "completed":
            continue
            
        # Basic job metadata
        job_meta = {
            'job_id': f"{job.model}_{job.mitigation_combo}_{job.category}",
            'model': job.model,
            'category': job.category,
            'mitigation_combo': job.mitigation_combo or 'baseline',
            'mitigation_ids': ','.join(job.mitigation_ids) if job.mitigation_ids else '',
            'sample_size': job.sample_size,
            'status': job.status,
            'estimated_cost': job.estimated_cost,
            'actual_cost': job.actual_cost,
            'estimated_duration': job.estimated_duration,
            'actual_duration': job.actual_duration,
            'config_hash': job.config_hash,
            'output_path': job.output_path
        }
        dataframes['job_metadata'].append(job_meta)
        
        # Load detailed results if available
        try:
            # Load evaluation results JSONL
            if Path(job.output_path).exists():
                eval_records = list(read_jsonl(job.output_path))
                
                for record in eval_records:
                    eval_result = {
                        'job_id': job_meta['job_id'],
                        'model': job.model,
                        'category': job.category,
                        'mitigation_combo': job.mitigation_combo or 'baseline',
                        'record_id': record.get('id', ''),
                        'prompt_length': len(record.get('prompt', '')),
                        'response_length': len(record.get('text', '')),
                        'latency_ms': record.get('timing', {}).get('latency_ms', 0),
                        'input_tokens': record.get('cost', {}).get('input_tokens', 0),
                        'output_tokens': record.get('cost', {}).get('output_tokens', 0),
                        'cost_usd': record.get('cost', {}).get('usd', 0.0),
                    }
                    
                    # Add category-specific scores
                    scores = record.get('scores', {})
                    judgement = record.get('judgement', {})
                    
                    if job.category == 'pii':
                        eval_result.update({
                            'pii_leak': judgement.get('leak', False),
                            'entities_detected': scores.get('entities_detected', 0),
                            'email_detected': judgement.get('entities', {}).get('email', False),
                            'phone_detected': judgement.get('entities', {}).get('phone', False),
                            'credit_card_detected': judgement.get('entities', {}).get('credit_card', False),
                        })
                    elif job.category == 'injection':
                        eval_result.update({
                            'injection_violation': judgement.get('violation', False),
                            'system_prompt_mentioned': judgement.get('mentions_system_prompt', False),
                            'conflict_followed': judgement.get('conflict_followed', False),
                            'obfuscation_detected': judgement.get('obfuscation_detected', False),
                        })
                    elif job.category == 'benign':
                        eval_result.update({
                            'benign_allowed': judgement.get('allowed', True),
                            'helpfulness_score': scores.get('helpfulness_score', 0.0),
                            'refusal_detected': judgement.get('refusal_detected', False),
                        })
                    
                    dataframes['evaluation_results'].append(eval_result)
            
            # Load summary metrics
            summary_path = Path(job.output_path).with_suffix('.summary.json')
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                agg_metrics = {
                    'job_id': job_meta['job_id'],
                    'model': job.model,
                    'category': job.category,
                    'mitigation_combo': job.mitigation_combo or 'baseline',
                }
                
                # Extract aggregate metrics
                aggregate = summary.get('aggregate', {})
                for metric, value in aggregate.items():
                    if isinstance(value, (int, float, bool)):
                        agg_metrics[f'agg_{metric}'] = value
                
                # Cost metrics
                cost = summary.get('cost', {})
                for metric, value in cost.items():
                    if isinstance(value, (int, float)):
                        agg_metrics[f'cost_{metric}'] = value
                
                # Performance metrics
                performance = summary.get('performance', {})
                for metric, value in performance.items():
                    if isinstance(value, (int, float)):
                        agg_metrics[f'perf_{metric}'] = value
                
                dataframes['aggregated_metrics'].append(agg_metrics)
                
                # Cost analysis data
                cost_analysis = {
                    'job_id': job_meta['job_id'],
                    'model': job.model,
                    'category': job.category,
                    'mitigation_combo': job.mitigation_combo or 'baseline',
                    'total_calls': cost.get('total_calls', 0),
                    'total_usd': cost.get('total_usd', 0.0),
                    'usd_per_call': cost.get('usd_per_call', 0.0),
                    'input_tokens_total': cost.get('input_tokens', 0),
                    'output_tokens_total': cost.get('output_tokens', 0),
                    'total_tokens': cost.get('total_tokens', 0),
                    'estimated_cost': job.estimated_cost,
                    'cost_variance': (cost.get('total_usd', 0.0) - job.estimated_cost) / job.estimated_cost if job.estimated_cost > 0 else 0,
                }
                dataframes['cost_analysis'].append(cost_analysis)
                
        except Exception as e:
            # Log error but continue with other jobs
            print(f"Warning: Could not load detailed results for {job_meta['job_id']}: {e}")
            continue
    
    # Convert to DataFrames
    result_dfs = {}
    for name, data in dataframes.items():
        if data:
            result_dfs[name] = pd.DataFrame(data)
        else:
            result_dfs[name] = pd.DataFrame()  # Empty DataFrame if no data
    
    return result_dfs


def calculate_mitigation_effectiveness(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate mitigation effectiveness compared to baseline."""
    
    if dfs['aggregated_metrics'].empty:
        return pd.DataFrame()
    
    effectiveness_data = []
    
    # Get all unique model-category combinations
    agg_df = dfs['aggregated_metrics']
    
    for model in agg_df['model'].unique():
        for category in agg_df['category'].unique():
            # Get baseline metrics
            baseline = agg_df[
                (agg_df['model'] == model) & 
                (agg_df['category'] == category) & 
                (agg_df['mitigation_combo'] == 'baseline')
            ]
            
            if baseline.empty:
                continue
                
            baseline_row = baseline.iloc[0]
            
            # Compare each mitigation to baseline
            for mitigation in ['policy_gate', 'prompt_hardening', 'both']:
                mitigation_data = agg_df[
                    (agg_df['model'] == model) & 
                    (agg_df['category'] == category) & 
                    (agg_df['mitigation_combo'] == mitigation)
                ]
                
                if mitigation_data.empty:
                    continue
                    
                mitigation_row = mitigation_data.iloc[0]
                
                effectiveness = {
                    'model': model,
                    'category': category,
                    'mitigation_combo': mitigation,
                    'comparison_id': f"{model}_{category}_{mitigation}_vs_baseline"
                }
                
                # Calculate effectiveness for key metrics
                metric_mappings = {
                    'agg_safety_coverage': ('safety_coverage_improvement', 'higher_better'),
                    'agg_pii_leak_rate': ('pii_leak_reduction', 'lower_better'),
                    'agg_injection_success_rate': ('injection_resistance_improvement', 'lower_better'),
                    'agg_benign_pass_rate': ('benign_pass_rate_change', 'higher_better'),
                    'agg_false_positive_rate': ('false_positive_reduction', 'lower_better'),
                    'cost_total_usd': ('cost_increase', 'lower_better'),
                    'perf_latency_p95_ms': ('latency_change_ms', 'lower_better'),
                }
                
                for baseline_metric, (effectiveness_metric, direction) in metric_mappings.items():
                    if baseline_metric in baseline_row and baseline_metric in mitigation_row:
                        baseline_val = baseline_row[baseline_metric]
                        mitigation_val = mitigation_row[baseline_metric]
                        
                        if pd.isna(baseline_val) or pd.isna(mitigation_val):
                            continue
                            
                        if direction == 'higher_better':
                            # Percentage point improvement
                            effectiveness[effectiveness_metric] = (mitigation_val - baseline_val) * 100
                        elif direction == 'lower_better':
                            # Reduction in percentage points or absolute change
                            if 'rate' in baseline_metric:
                                effectiveness[effectiveness_metric] = (baseline_val - mitigation_val) * 100
                            else:
                                effectiveness[effectiveness_metric] = mitigation_val - baseline_val
                
                effectiveness_data.append(effectiveness)
    
    return pd.DataFrame(effectiveness_data)


def add_mitigation_effectiveness(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Add mitigation effectiveness analysis to the data bundle."""
    effectiveness_df = calculate_mitigation_effectiveness(dfs)
    dfs['mitigation_effectiveness'] = effectiveness_df
    return dfs


def generate_release_bundle(plan: Plan, output_path: str) -> None:
    """Generate comprehensive release data bundle in Parquet format.
    
    Args:
        plan: Completed release evaluation plan
        output_path: Path to write the Parquet file
    """
    # Collect all data
    dataframes = collect_release_data(plan)
    
    # Add mitigation effectiveness analysis
    dataframes = add_mitigation_effectiveness(dataframes)
    
    # Add metadata about the release
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'total_jobs': len(plan.jobs),
        'completed_jobs': len([job for job in plan.jobs if job.status == "completed"]),
        'failed_jobs': len([job for job in plan.jobs if job.status == "failed"]),
        'models_evaluated': list(set(job.model for job in plan.jobs)),
        'categories_evaluated': list(set(job.category for job in plan.jobs)),
        'mitigation_combos': list(set(job.mitigation_combo or 'baseline' for job in plan.jobs)),
        'total_estimated_cost': sum(job.estimated_cost for job in plan.jobs),
        'total_actual_cost': sum(job.actual_cost or 0 for job in plan.jobs if job.actual_cost),
    }
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame([metadata])
    dataframes['release_metadata'] = metadata_df
    
    # Prepare output path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as a single Parquet file with multiple tables
    # For this, we'll use a directory-based approach
    bundle_dir = output_path.with_suffix('')
    bundle_dir.mkdir(exist_ok=True)
    
    # Save each DataFrame as a separate Parquet file
    for name, df in dataframes.items():
        if not df.empty:
            table_path = bundle_dir / f"{name}.parquet"
            df.to_parquet(table_path, engine='pyarrow', compression='snappy', index=False)
    
    # Also create a single concatenated file with table identifiers
    all_data = []
    for name, df in dataframes.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['_table_name'] = name
            all_data.append(df_copy)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)
        combined_path = output_path.with_name(f"{output_path.stem}_combined.parquet")
        combined_df.to_parquet(combined_path, engine='pyarrow', compression='snappy', index=False)
    
    print(f"Release data bundle saved to: {bundle_dir}")
    print(f"Combined data saved to: {combined_path}")
    print(f"Tables included: {', '.join(dataframes.keys())}")
    print(f"Total records across all tables: {sum(len(df) for df in dataframes.values())}")


def load_release_bundle(bundle_path: str) -> Dict[str, pd.DataFrame]:
    """Load a release data bundle from Parquet files.
    
    Args:
        bundle_path: Path to the bundle directory or combined Parquet file
        
    Returns:
        Dictionary of DataFrames by table name
    """
    bundle_path = Path(bundle_path)
    dataframes = {}
    
    if bundle_path.is_dir():
        # Load from directory of separate files
        for parquet_file in bundle_path.glob("*.parquet"):
            table_name = parquet_file.stem
            dataframes[table_name] = pd.read_parquet(parquet_file)
    elif bundle_path.suffix == '.parquet':
        # Load from combined file
        combined_df = pd.read_parquet(bundle_path)
        
        # Split by table name
        for table_name in combined_df['_table_name'].unique():
            table_df = combined_df[combined_df['_table_name'] == table_name].copy()
            table_df = table_df.drop(columns=['_table_name'])
            dataframes[table_name] = table_df
    else:
        raise ValueError(f"Bundle path must be a directory or .parquet file: {bundle_path}")
    
    return dataframes