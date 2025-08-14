"""Streamlit labeling interface for evaluation results."""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
from seibox.utils.schemas import OutputRecord


def load_run_data(run_path: str) -> List[OutputRecord]:
    """Load evaluation results from JSONL file.

    Args:
        run_path: Path to JSONL file with evaluation results

    Returns:
        List of OutputRecord objects
    """
    records = []
    with open(run_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            records.append(OutputRecord(**data))
    return records


def load_existing_labels(labels_path: str) -> Dict[str, Dict[str, Any]]:
    """Load existing labels from JSONL file.

    Args:
        labels_path: Path to labels JSONL file

    Returns:
        Dictionary mapping record IDs to label data
    """
    labels = {}
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                labels[data["record_id"]] = data
    return labels


def save_label(
    labels_path: str,
    record_id: str,
    label: str,
    rationale: str,
    model_output: str,
    current_judgement: Dict[str, Any],
):
    """Save a single label to JSONL file.

    Args:
        labels_path: Path to labels JSONL file
        record_id: ID of the record being labeled
        label: Label (Correct, Incorrect, Unsure)
        rationale: Text rationale for the label
        model_output: Model's output text
        current_judgement: Current automated judgement
    """
    # Ensure directory exists
    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)

    label_data = {
        "record_id": record_id,
        "label": label,
        "rationale": rationale,
        "model_output": model_output,
        "current_judgement": current_judgement,
        "timestamp": str(st.session_state.get("timestamp", "unknown")),
    }

    # Load existing labels
    existing_labels = load_existing_labels(labels_path)
    existing_labels[record_id] = label_data

    # Write all labels back
    with open(labels_path, "w") as f:
        for label_entry in existing_labels.values():
            f.write(json.dumps(label_entry) + "\n")


def main():
    """Main Streamlit app for labeling evaluation results."""
    st.set_page_config(page_title="Safety Eval Labeler", page_icon="🏷️", layout="wide")

    st.title("🏷️ Safety Evaluation Labeler")
    st.markdown("Label evaluation results to create golden datasets.")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Check for environment variable first
    env_run_path = os.environ.get("SEIBOX_RUN_PATH")

    # File upload or path input
    uploaded_file = st.sidebar.file_uploader(
        "Upload evaluation results",
        type=["jsonl"],
        help="Upload a JSONL file with evaluation results",
    )

    run_path = None
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        run_path = temp_path
    elif env_run_path and os.path.exists(env_run_path):
        # Use environment variable path
        run_path = env_run_path
        st.sidebar.info(f"Using path from CLI: {env_run_path}")
    else:
        # Path input
        run_path_input = st.sidebar.text_input(
            "Or enter path to JSONL file",
            placeholder="runs/example.jsonl",
            help="Path to JSONL file with evaluation results",
        )
        if run_path_input and os.path.exists(run_path_input):
            run_path = run_path_input
        elif run_path_input:
            st.sidebar.error(f"File not found: {run_path_input}")

    if not run_path:
        st.info("Please upload a file or enter a valid path to begin labeling.")
        return

    # Load data
    try:
        records = load_run_data(run_path)
        st.sidebar.success(f"Loaded {len(records)} records")
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return

    if not records:
        st.warning("No records found in the file.")
        return

    # Labels output path
    labels_path = st.sidebar.text_input(
        "Labels output path", value="golden/v2/labels.jsonl", help="Where to save the labels"
    )

    # Load existing labels
    existing_labels = load_existing_labels(labels_path)

    # Record selection
    st.sidebar.subheader("Record Navigation")

    # Filter options
    suite_filter = st.sidebar.selectbox(
        "Filter by suite", ["All"] + list(set(r.id.split("_")[0] for r in records)), index=0
    )

    # Filter records
    filtered_records = records
    if suite_filter != "All":
        filtered_records = [r for r in records if r.id.startswith(suite_filter)]

    # Show labeling progress
    labeled_count = sum(1 for r in filtered_records if r.id in existing_labels)
    st.sidebar.metric("Labeling Progress", f"{labeled_count}/{len(filtered_records)}")

    # Record selection
    record_idx = st.sidebar.selectbox(
        "Select record",
        range(len(filtered_records)),
        format_func=lambda x: f"{filtered_records[x].id} {'✓' if filtered_records[x].id in existing_labels else '○'}",
    )

    current_record = filtered_records[record_idx]

    # Main labeling interface
    st.header(f"Record: {current_record.id}")

    # Show record details in columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")

        # Extract prompt from trace if available
        prompt = current_record.trace.get("prompt", "[Prompt not available]")
        st.text_area("Prompt", value=prompt, height=100, disabled=True)

        st.subheader("Model Output")
        st.text_area("Response", value=current_record.text, height=150, disabled=True)

        # Model info
        st.caption(f"Model: {current_record.model}")
        st.caption(f"Latency: {current_record.timing.get('latency_ms', 'N/A')} ms")
        if current_record.cost.get("usd"):
            st.caption(f"Cost: ${current_record.cost['usd']:.4f}")

    with col2:
        st.subheader("Current Automated Judgement")

        # Show judgement in a nice format
        judgement_df_data = []
        for key, value in current_record.judgement.items():
            judgement_df_data.append({"Metric": key, "Value": str(value)})

        if judgement_df_data:
            st.dataframe(judgement_df_data, hide_index=True)

        st.subheader("Scores")
        scores_df_data = []
        for key, value in current_record.scores.items():
            scores_df_data.append({"Score": key, "Value": str(value)})

        if scores_df_data:
            st.dataframe(scores_df_data, hide_index=True)

        # Show trace info if available
        if current_record.trace:
            st.subheader("Trace")
            trace_df_data = []
            for key, value in current_record.trace.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, indent=2)
                trace_df_data.append({"Key": key, "Value": str(value)})

            if trace_df_data:
                st.dataframe(trace_df_data, hide_index=True)

    # Labeling section
    st.divider()
    st.subheader("Human Label")

    # Check if already labeled
    existing_label = existing_labels.get(current_record.id)

    # Labeling form
    with st.form("labeling_form"):
        col1, col2 = st.columns([1, 2])

        with col1:
            label = st.selectbox(
                "Label",
                ["Correct", "Incorrect", "Unsure"],
                index=(
                    ["Correct", "Incorrect", "Unsure"].index(existing_label["label"])
                    if existing_label
                    else 0
                ),
            )

        with col2:
            rationale = st.text_area(
                "Rationale",
                value=existing_label["rationale"] if existing_label else "",
                placeholder="Explain why you chose this label...",
                height=100,
            )

        submitted = st.form_submit_button("Save Label", type="primary")

        if submitted:
            if not rationale.strip():
                st.error("Please provide a rationale for your label.")
            else:
                # Add timestamp to session state
                import time

                st.session_state.timestamp = time.time()

                save_label(
                    labels_path=labels_path,
                    record_id=current_record.id,
                    label=label,
                    rationale=rationale.strip(),
                    model_output=current_record.text,
                    current_judgement=current_record.judgement,
                )
                st.success(f"Label saved for record {current_record.id}")
                st.rerun()

    # Show existing label if available
    if existing_label:
        st.info(f"Previously labeled as: **{existing_label['label']}**")
        st.caption(f"Rationale: {existing_label['rationale']}")
        st.caption(f"Timestamp: {existing_label.get('timestamp', 'N/A')}")

    # Navigation buttons
    st.divider()
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

    with nav_col1:
        if st.button("← Previous", disabled=(record_idx == 0)):
            st.session_state.record_idx = record_idx - 1
            st.rerun()

    with nav_col2:
        # Jump to next unlabeled
        unlabeled_indices = [
            i
            for i, r in enumerate(filtered_records)
            if r.id not in existing_labels and i > record_idx
        ]
        if unlabeled_indices and st.button("Next Unlabeled"):
            st.session_state.record_idx = unlabeled_indices[0]
            st.rerun()

    with nav_col3:
        if st.button("Next →", disabled=(record_idx == len(filtered_records) - 1)):
            st.session_state.record_idx = record_idx + 1
            st.rerun()


if __name__ == "__main__":
    main()
