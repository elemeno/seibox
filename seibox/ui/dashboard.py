"""Streamlit dashboard for Safety Evals in a Box."""

import json
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit import session_state as ss

from seibox.utils.io import read_jsonl
from seibox.scoring.aggregate import aggregate_metrics
from seibox.utils.statistics import (
    wilson_confidence_interval,
    bootstrap_difference_ci,
    compute_stratified_metrics,
    find_flips_with_details,
    format_confidence_interval,
)


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


def display_metric_cards_with_ci(df: pd.DataFrame, file_name: str):
    """Display enhanced metric cards with confidence intervals.

    Args:
        df: DataFrame with evaluation results
        file_name: Name of the file for display
    """
    # Convert DataFrame records to OutputRecord format for aggregation
    from seibox.utils.schemas import OutputRecord

    records = [OutputRecord(**row) for _, row in df.iterrows()]
    metrics = aggregate_metrics(records)

    st.subheader(f"📊 {file_name}")

    # Calculate confidence intervals for key metrics
    # Safety coverage
    safety_records = df[df["id"].str.startswith(("pii", "injection"))]
    if not safety_records.empty:
        should_block = safety_records["trace"].apply(
            lambda x: x.get("gold", {}).get("should_block", True) if isinstance(x, dict) else True
        )
        actually_blocked = safety_records["judgement"].apply(
            lambda x: x.get("blocked", False) if isinstance(x, dict) else False
        )

        blocked_when_should = sum(should_block & actually_blocked)
        total_should_block = sum(should_block)
        safety_coverage = (
            blocked_when_should / total_should_block if total_should_block > 0 else 0.0
        )
        safety_ci = wilson_confidence_interval(blocked_when_should, total_should_block)
    else:
        safety_coverage = 0.0
        safety_ci = (0.0, 0.0)

    # Benign pass rate
    benign_records = df[df["id"].str.startswith("benign")]
    if not benign_records.empty:
        allowed = benign_records["judgement"].apply(
            lambda x: x.get("allowed", False) if isinstance(x, dict) else False
        )
        total_allowed = sum(allowed)
        total_benign = len(benign_records)
        benign_pass_rate = total_allowed / total_benign if total_benign > 0 else 0.0
        benign_ci = wilson_confidence_interval(total_allowed, total_benign)
    else:
        benign_pass_rate = 0.0
        benign_ci = (0.0, 0.0)

    # Injection success rate
    injection_records = df[df["id"].str.startswith("injection")]
    if not injection_records.empty:
        violations = injection_records["scores"].apply(
            lambda x: x.get("violation", False) if isinstance(x, dict) else False
        )
        total_violations = sum(violations)
        total_injection = len(injection_records)
        injection_success = total_violations / total_injection if total_injection > 0 else 0.0
        injection_ci = wilson_confidence_interval(total_violations, total_injection)
    else:
        injection_success = 0.0
        injection_ci = (0.0, 0.0)

    # Display cards with confidence intervals
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Safety Coverage",
            format_confidence_interval(safety_coverage, safety_ci),
            help="Recall on samples that should be blocked (95% Wilson CI)",
        )

    with col2:
        st.metric(
            "Benign Pass Rate",
            format_confidence_interval(benign_pass_rate, benign_ci),
            help="Percentage of benign requests allowed (95% Wilson CI)",
        )

    with col3:
        false_positive_rate = 1.0 - benign_pass_rate
        fp_ci = (1.0 - benign_ci[1], 1.0 - benign_ci[0])
        st.metric(
            "False Positive Rate",
            format_confidence_interval(false_positive_rate, fp_ci),
            delta=f"-{false_positive_rate:.1%}",
            delta_color="inverse",
            help="1 - benign pass rate (95% Wilson CI)",
        )

    with col4:
        st.metric(
            "Injection Success",
            format_confidence_interval(injection_success, injection_ci),
            delta=f"-{injection_success:.1%}",
            delta_color="inverse",
            help="Percentage of successful prompt injections (95% Wilson CI)",
        )

    # Second row of metrics (without CI for performance metrics)
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "Latency p50", f"{metrics.get('latency_p50', 0):.0f} ms", help="Median response time"
        )

    with col6:
        st.metric(
            "Latency p95",
            f"{metrics.get('latency_p95', 0):.0f} ms",
            help="95th percentile response time",
        )

    with col7:
        st.metric(
            "Total Cost", f"${metrics.get('total_cost_usd', 0):.4f}", help="Total cost in USD"
        )

    with col8:
        st.metric(
            "Cost per 1k", f"${metrics.get('cost_per_1k', 0):.4f}", help="Cost per 1000 API calls"
        )


def display_confusion_matrix(df: pd.DataFrame, suite: str):
    """Display confusion matrix for a specific suite.

    Args:
        df: DataFrame with evaluation results
        suite: Suite name to filter by
    """
    suite_df = df[df["id"].str.startswith(suite)]

    if suite_df.empty:
        st.warning(f"No data found for suite: {suite}")
        return

    if suite == "pii":
        # PII confusion matrix: leaked vs blocked
        leaked = suite_df["scores"].apply(lambda x: x.get("leak", False))
        blocked = suite_df["judgement"].apply(lambda x: x.get("blocked", False))

        confusion = pd.crosstab(leaked, blocked, rownames=["Leaked"], colnames=["Blocked"])
        st.dataframe(confusion)

    elif suite == "injection":
        # Injection confusion matrix: violation vs blocked
        violation = suite_df["scores"].apply(lambda x: x.get("violation", False))
        blocked = suite_df["judgement"].apply(lambda x: x.get("blocked", False))

        confusion = pd.crosstab(violation, blocked, rownames=["Violation"], colnames=["Blocked"])
        st.dataframe(confusion)

    elif suite == "benign":
        # Benign confusion matrix: allowed vs expected
        allowed = suite_df["scores"].apply(lambda x: x.get("allowed", False))

        st.metric("Allowed", f"{allowed.mean():.1%}")
        st.metric("Refused", f"{(1 - allowed.mean()):.1%}")


def display_advanced_flips_viewer(df_baseline: pd.DataFrame, df_mitigated: pd.DataFrame):
    """Display enhanced flips viewer with filtering and navigation.

    Args:
        df_baseline: Baseline results DataFrame
        df_mitigated: Mitigated results DataFrame
    """
    st.subheader("🔄 Flip Explorer")

    # Find flips with detailed analysis
    flip_data = find_flips_with_details(df_baseline, df_mitigated, ["improve", "worsen", "all"])

    # Summary stats
    total_records = len(pd.merge(df_baseline[["id"]], df_mitigated[["id"]], on="id"))
    improvements = len(flip_data.get("improve", pd.DataFrame()))
    worsenings = len(flip_data.get("worsen", pd.DataFrame()))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Improvements", improvements, help="Bad → Good flips")
    with col2:
        st.metric("🔴 Worsenings", worsenings, help="Good → Bad flips", delta=f"-{worsenings}")
    with col3:
        flip_rate = (improvements + worsenings) / total_records if total_records > 0 else 0
        st.metric(
            "📊 Total Flip Rate", f"{flip_rate:.1%}", help="Percentage of records that changed"
        )

    # Filters
    st.write("**Filters:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        flip_type_filter = st.selectbox(
            "Flip Type",
            options=["all", "improve", "worsen"],
            format_func=lambda x: {
                "all": "All Changes",
                "improve": "Improvements",
                "worsen": "Worsenings",
            }[x],
        )

    with col2:
        severity_filter = st.selectbox(
            "Severity",
            options=["all", "high", "med", "low"],
            format_func=lambda x: x.title() if x != "all" else "All Severities",
        )

    with col3:
        suite_filter = st.selectbox(
            "Suite",
            options=["all", "pii", "injection", "benign"],
            format_func=lambda x: x.upper() if x != "all" else "All Suites",
        )

    # Apply filters
    filtered_flips = flip_data.get(flip_type_filter, pd.DataFrame())
    if not filtered_flips.empty:
        if severity_filter != "all":
            filtered_flips = filtered_flips[filtered_flips["severity"] == severity_filter]
        if suite_filter != "all":
            filtered_flips = filtered_flips[filtered_flips["id"].str.startswith(suite_filter)]

    if filtered_flips.empty:
        st.info("No flips found with the selected filters.")
        return

    # Pagination for navigation
    items_per_page = 5
    total_items = len(filtered_flips)
    total_pages = (total_items - 1) // items_per_page + 1

    if total_pages > 1:
        page = st.selectbox(
            f"Page (showing {items_per_page} of {total_items} items)", range(1, total_pages + 1)
        )
    else:
        page = 1

    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    page_items = filtered_flips.iloc[start_idx:end_idx]

    # Display flip examples
    st.write(f"**Showing items {start_idx + 1}-{end_idx} of {total_items}**")

    for idx, (_, row) in enumerate(page_items.iterrows(), start=start_idx + 1):
        flip_emoji = "🟢" if row["flip_type"] == "improve" else "🔴"
        severity_emoji = {"high": "🔥", "med": "⚠️", "low": "📝"}[row["severity"]]

        with st.expander(
            f"{flip_emoji} {severity_emoji} #{idx}: {row['id']} ({row['flip_type'].title()})"
        ):
            # Summary
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Suite:** {row['id'].split('_')[0].upper()}")
                st.write(f"**Severity:** {row['severity'].title()}")
            with col2:
                st.write(f"**Flip:** {row['flip_type'].title()}")
                st.write(
                    f"**Change:** {'Bad → Good' if row['flip_type'] == 'improve' else 'Good → Bad'}"
                )

            st.divider()

            # Side-by-side comparison
            col1, col2 = st.columns(2)
            with col1:
                st.write("**🟦 Baseline**")
                st.code(
                    row["text_baseline"][:400] + ("..." if len(row["text_baseline"]) > 400 else ""),
                    language="text",
                )

                # Show relevant scores
                if row["id"].startswith("pii"):
                    leak = row["scores_baseline"].get("leak", False)
                    st.write(f"**Leak:** {'🔴 Yes' if leak else '🟢 No'}")
                elif row["id"].startswith("injection"):
                    violation = row["scores_baseline"].get("violation", False)
                    st.write(f"**Violation:** {'🔴 Yes' if violation else '🟢 No'}")
                else:
                    allowed = row["judgement_baseline"].get("allowed", False)
                    st.write(f"**Allowed:** {'🟢 Yes' if allowed else '🔴 No'}")

            with col2:
                st.write("**🟩 Mitigated**")
                st.code(
                    row["text_mitigated"][:400]
                    + ("..." if len(row["text_mitigated"]) > 400 else ""),
                    language="text",
                )

                # Show relevant scores
                if row["id"].startswith("pii"):
                    leak = row["scores_mitigated"].get("leak", False)
                    st.write(f"**Leak:** {'🔴 Yes' if leak else '🟢 No'}")
                elif row["id"].startswith("injection"):
                    violation = row["scores_mitigated"].get("violation", False)
                    st.write(f"**Violation:** {'🔴 Yes' if violation else '🟢 No'}")
                else:
                    allowed = row["judgement_mitigated"].get("allowed", False)
                    st.write(f"**Allowed:** {'🟢 Yes' if allowed else '🔴 No'}")


def main():
    """Main dashboard application."""
    st.set_page_config(page_title="Safety Evals Dashboard", page_icon="🛡️", layout="wide")

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
                    "Select result files", options=result_files, format_func=lambda x: x.name
                )

                if selected_files:
                    # Load selected files
                    if "dataframes" not in ss:
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
    if "dataframes" in ss and ss.dataframes:
        tabs = st.tabs(
            ["📈 Overview", "📊 Stratified Analysis", "🔄 Flips Explorer", "📋 Raw Data"]
        )

        with tabs[0]:  # Overview
            st.header("📈 Evaluation Overview")

            for file_path, df in ss.dataframes.items():
                display_metric_cards_with_ci(df, Path(file_path).name)
                st.divider()

        with tabs[1]:  # Stratified Analysis
            st.header("📊 Stratified Analysis by Severity")

            for file_path, df in ss.dataframes.items():
                st.subheader(Path(file_path).name)

                # Compute stratified metrics
                stratified = compute_stratified_metrics(df)

                # Create visualization
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=(
                        "Safety Coverage by Severity",
                        "Benign Pass Rate by Severity",
                        "Injection Success Rate by Severity",
                        "Sample Counts",
                    ),
                    specs=[
                        [{"secondary_y": False}, {"secondary_y": False}],
                        [{"secondary_y": False}, {"type": "bar"}],
                    ],
                )

                severities = ["high", "med", "low"]
                colors = ["#ff4444", "#ffaa44", "#44ff44"]

                for i, severity in enumerate(severities):
                    data = stratified.get(severity, {})

                    # Safety Coverage
                    if "safety_coverage" in data:
                        val = data["safety_coverage"]["value"]
                        ci_lower, ci_upper = data["safety_coverage"]["ci"]
                        fig.add_trace(
                            go.Scatter(
                                x=[severity],
                                y=[val],
                                error_y=dict(
                                    type="data",
                                    symmetric=False,
                                    array=[ci_upper - val],
                                    arrayminus=[val - ci_lower],
                                ),
                                mode="markers+lines",
                                name=f"Safety Coverage",
                                marker=dict(color=colors[i], size=10),
                                showlegend=(i == 0),
                            ),
                            row=1,
                            col=1,
                        )

                    # Benign Pass Rate
                    if "benign_pass_rate" in data:
                        val = data["benign_pass_rate"]["value"]
                        ci_lower, ci_upper = data["benign_pass_rate"]["ci"]
                        fig.add_trace(
                            go.Scatter(
                                x=[severity],
                                y=[val],
                                error_y=dict(
                                    type="data",
                                    symmetric=False,
                                    array=[ci_upper - val],
                                    arrayminus=[val - ci_lower],
                                ),
                                mode="markers+lines",
                                name=f"Benign Pass Rate",
                                marker=dict(color=colors[i], size=10),
                                showlegend=(i == 0),
                            ),
                            row=1,
                            col=2,
                        )

                    # Injection Success Rate
                    if "injection_success_rate" in data:
                        val = data["injection_success_rate"]["value"]
                        ci_lower, ci_upper = data["injection_success_rate"]["ci"]
                        fig.add_trace(
                            go.Scatter(
                                x=[severity],
                                y=[val],
                                error_y=dict(
                                    type="data",
                                    symmetric=False,
                                    array=[ci_upper - val],
                                    arrayminus=[val - ci_lower],
                                ),
                                mode="markers+lines",
                                name=f"Injection Success",
                                marker=dict(color=colors[i], size=10),
                                showlegend=(i == 0),
                            ),
                            row=2,
                            col=1,
                        )

                    # Sample counts
                    count = data.get("count", 0)
                    fig.add_trace(
                        go.Bar(
                            x=[severity],
                            y=[count],
                            name="Sample Count",
                            marker=dict(color=colors[i]),
                            showlegend=(i == 0),
                        ),
                        row=2,
                        col=2,
                    )

                fig.update_layout(height=600, showlegend=True)
                fig.update_yaxes(range=[0, 1], row=1, col=1, title_text="Coverage Rate")
                fig.update_yaxes(range=[0, 1], row=1, col=2, title_text="Pass Rate")
                fig.update_yaxes(range=[0, 1], row=2, col=1, title_text="Success Rate")
                fig.update_yaxes(row=2, col=2, title_text="Count")
                fig.update_xaxes(title_text="Severity", row=2, col=1)
                fig.update_xaxes(title_text="Severity", row=2, col=2)

                st.plotly_chart(fig, use_container_width=True)

                # Summary table
                st.write("**Detailed Metrics with Confidence Intervals**")
                summary_data = []
                for severity in severities:
                    data = stratified.get(severity, {})
                    row = {"Severity": severity.title(), "Count": data.get("count", 0)}

                    for metric in ["safety_coverage", "benign_pass_rate", "injection_success_rate"]:
                        if metric in data:
                            val = data[metric]["value"]
                            ci = data[metric]["ci"]
                            row[metric.replace("_", " ").title()] = format_confidence_interval(
                                val, ci
                            )
                        else:
                            row[metric.replace("_", " ").title()] = "N/A"

                    summary_data.append(row)

                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                st.divider()

        with tabs[2]:  # Enhanced Flips Viewer
            if len(ss.dataframes) >= 2:
                file_paths = list(ss.dataframes.keys())

                col1, col2 = st.columns(2)
                with col1:
                    baseline_file = st.selectbox("Baseline", options=file_paths, key="baseline")
                with col2:
                    mitigated_file = st.selectbox("Mitigated", options=file_paths, key="mitigated")

                if baseline_file and mitigated_file and baseline_file != mitigated_file:
                    display_advanced_flips_viewer(
                        ss.dataframes[baseline_file], ss.dataframes[mitigated_file]
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
