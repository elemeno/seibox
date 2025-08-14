"""Streamlit dashboard for Safety Evals in a Box."""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
from streamlit import session_state as ss

from seibox.utils.io import read_jsonl
from seibox.scoring.aggregate import aggregate_metrics


def load_results(file_path: Path) -> pd.DataFrame:
    """Load evaluation results from JSONL or Parquet file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        DataFrame with evaluation results
    """
    if file_path.suffix == ".jsonl":
        records = list(read_jsonl(file_path))
        return pd.DataFrame(records)
    elif file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def display_metric_cards(metrics: dict):
    """Display metrics as cards in columns.
    
    Args:
        metrics: Dictionary of aggregate metrics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Safety Coverage",
            f"{metrics.get('safety_coverage', 0):.1%}",
            help="Recall on samples that should be blocked"
        )
    
    with col2:
        st.metric(
            "Benign Pass Rate",
            f"{metrics.get('benign_pass_rate', 0):.1%}",
            help="Percentage of benign requests allowed"
        )
    
    with col3:
        st.metric(
            "False Positive Rate",
            f"{metrics.get('false_positive_rate', 0):.1%}",
            delta=f"-{metrics.get('false_positive_rate', 0):.1%}",
            delta_color="inverse",
            help="1 - benign pass rate"
        )
    
    with col4:
        st.metric(
            "Injection Success",
            f"{metrics.get('injection_success_rate', 0):.1%}",
            delta=f"-{metrics.get('injection_success_rate', 0):.1%}",
            delta_color="inverse",
            help="Percentage of successful prompt injections"
        )
    
    # Second row of metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Latency p50",
            f"{metrics.get('latency_p50', 0):.0f} ms",
            help="Median response time"
        )
    
    with col6:
        st.metric(
            "Latency p95",
            f"{metrics.get('latency_p95', 0):.0f} ms",
            help="95th percentile response time"
        )
    
    with col7:
        st.metric(
            "Total Cost",
            f"${metrics.get('total_cost_usd', 0):.4f}",
            help="Total cost in USD"
        )
    
    with col8:
        st.metric(
            "Cost per 1k",
            f"${metrics.get('cost_per_1k', 0):.4f}",
            help="Cost per 1000 API calls"
        )


def display_confusion_matrix(df: pd.DataFrame, suite: str):
    """Display confusion matrix for a specific suite.
    
    Args:
        df: DataFrame with evaluation results
        suite: Suite name to filter by
    """
    suite_df = df[df['id'].str.startswith(suite)]
    
    if suite_df.empty:
        st.warning(f"No data found for suite: {suite}")
        return
    
    if suite == "pii":
        # PII confusion matrix: leaked vs blocked
        leaked = suite_df['scores'].apply(lambda x: x.get('leak', False))
        blocked = suite_df['judgement'].apply(lambda x: x.get('blocked', False))
        
        confusion = pd.crosstab(leaked, blocked, rownames=['Leaked'], colnames=['Blocked'])
        st.dataframe(confusion)
    
    elif suite == "injection":
        # Injection confusion matrix: violation vs blocked
        violation = suite_df['scores'].apply(lambda x: x.get('violation', False))
        blocked = suite_df['judgement'].apply(lambda x: x.get('blocked', False))
        
        confusion = pd.crosstab(violation, blocked, rownames=['Violation'], colnames=['Blocked'])
        st.dataframe(confusion)
    
    elif suite == "benign":
        # Benign confusion matrix: allowed vs expected
        allowed = suite_df['scores'].apply(lambda x: x.get('allowed', False))
        
        st.metric("Allowed", f"{allowed.mean():.1%}")
        st.metric("Refused", f"{(1 - allowed.mean()):.1%}")


def display_flips_viewer(df_baseline: pd.DataFrame, df_mitigated: pd.DataFrame):
    """Display cases where results flipped between baseline and mitigated.
    
    Args:
        df_baseline: Baseline results DataFrame
        df_mitigated: Mitigated results DataFrame
    """
    # Merge on ID to find flips
    merged = pd.merge(
        df_baseline[['id', 'text', 'judgement', 'scores']],
        df_mitigated[['id', 'text', 'judgement', 'scores']],
        on='id',
        suffixes=('_baseline', '_mitigated')
    )
    
    # Find flips in PII suite
    pii_flips = merged[merged['id'].str.startswith('pii')]
    pii_flips = pii_flips[
        pii_flips['scores_baseline'].apply(lambda x: x.get('leak', False)) !=
        pii_flips['scores_mitigated'].apply(lambda x: x.get('leak', False))
    ]
    
    if not pii_flips.empty:
        st.subheader("PII Detection Flips")
        for _, row in pii_flips.iterrows():
            with st.expander(f"ID: {row['id']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Baseline:**")
                    st.write(row['text_baseline'][:500])
                    st.write(f"Leak: {row['scores_baseline'].get('leak', False)}")
                with col2:
                    st.write("**Mitigated:**")
                    st.write(row['text_mitigated'][:500])
                    st.write(f"Leak: {row['scores_mitigated'].get('leak', False)}")


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Safety Evals Dashboard",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ Safety Evals in a Box Dashboard")
    
    # Sidebar for file selection
    with st.sidebar:
        st.header("Load Results")
        
        # Option to specify runs directory
        import os
        default_runs = os.environ.get("SEIBOX_RUNS_DIR", "runs")
        runs_dir = st.text_input("Runs Directory", value=default_runs)
        runs_path = Path(runs_dir)
        
        if runs_path.exists():
            # List available result files
            result_files = list(runs_path.glob("*.jsonl")) + list(runs_path.glob("*.parquet"))
            
            if result_files:
                selected_files = st.multiselect(
                    "Select result files",
                    options=result_files,
                    format_func=lambda x: x.name
                )
                
                if selected_files:
                    # Load selected files
                    if 'dataframes' not in ss:
                        ss.dataframes = {}
                    
                    for file_path in selected_files:
                        if str(file_path) not in ss.dataframes:
                            try:
                                ss.dataframes[str(file_path)] = load_results(file_path)
                                st.success(f"Loaded {file_path.name}")
                            except Exception as e:
                                st.error(f"Error loading {file_path.name}: {e}")
            else:
                st.warning(f"No result files found in {runs_dir}")
        else:
            st.warning(f"Directory {runs_dir} does not exist")
    
    # Main content area
    if 'dataframes' in ss and ss.dataframes:
        tabs = st.tabs(["Overview", "Confusion Matrices", "Flips Viewer", "Raw Data"])
        
        with tabs[0]:  # Overview
            st.header("Evaluation Overview")
            
            for file_path, df in ss.dataframes.items():
                st.subheader(Path(file_path).name)
                
                # Convert DataFrame records to OutputRecord format for aggregation
                from seibox.utils.schemas import OutputRecord
                records = [OutputRecord(**row) for _, row in df.iterrows()]
                metrics = aggregate_metrics(records)
                
                display_metric_cards(metrics)
                st.divider()
        
        with tabs[1]:  # Confusion Matrices
            st.header("Confusion Matrices")
            
            for file_path, df in ss.dataframes.items():
                st.subheader(Path(file_path).name)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**PII Suite**")
                    display_confusion_matrix(df, "pii")
                
                with col2:
                    st.write("**Injection Suite**")
                    display_confusion_matrix(df, "injection")
                
                with col3:
                    st.write("**Benign Suite**")
                    display_confusion_matrix(df, "benign")
                
                st.divider()
        
        with tabs[2]:  # Flips Viewer
            st.header("Flips Viewer")
            
            if len(ss.dataframes) >= 2:
                file_paths = list(ss.dataframes.keys())
                
                col1, col2 = st.columns(2)
                with col1:
                    baseline_file = st.selectbox("Baseline", options=file_paths)
                with col2:
                    mitigated_file = st.selectbox("Mitigated", options=file_paths)
                
                if baseline_file and mitigated_file and baseline_file != mitigated_file:
                    display_flips_viewer(
                        ss.dataframes[baseline_file],
                        ss.dataframes[mitigated_file]
                    )
            else:
                st.info("Load at least 2 result files to compare flips")
        
        with tabs[3]:  # Raw Data
            st.header("Raw Data")
            
            for file_path, df in ss.dataframes.items():
                st.subheader(Path(file_path).name)
                st.dataframe(df, use_container_width=True)
    else:
        st.info("👈 Please load result files from the sidebar to begin")


if __name__ == "__main__":
    main()