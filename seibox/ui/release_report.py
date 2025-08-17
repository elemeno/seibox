"""
Release report renderer.

Loads the flattened matrix parquet (model × category × profile × metric),
optionally the golden comparison json, prepares a few common views, and
renders a single HTML report via Jinja2.

Usage:
    from seibox.ui.release_report import render_release_report
    render_release_report(
        matrix_parquet="out/release/<tag>/aggregates/matrix.parquet",
        golden_json="out/release/<tag>/aggregates/golden_compare.json",
        out_html="out/release/<tag>/reports/release.html",
    )
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ---------- Public API ----------


def render_release_report(
    matrix_parquet: str, golden_json: str | None, out_html: str, drilldown_map: str | None = None
) -> None:
    """
    Load data, compute derived views, render with Jinja2.

    Args:
        matrix_parquet: Path to aggregates/matrix.parquet produced by the release runner.
        golden_json: Optional path to aggregates/golden_compare.json.
        out_html: Destination HTML file path.
        drilldown_map: Optional path to drilldown_map.json for linking to detailed views.

    Side effects:
        Writes a single HTML file at out_html.
    """
    mp = Path(matrix_parquet)
    if not mp.exists():
        raise FileNotFoundError(f"matrix parquet not found: {mp}")

    df = pd.read_parquet(mp)

    if df.empty:
        raise ValueError("matrix parquet is empty; nothing to render")

    golden_data = _load_json(golden_json) if golden_json else None
    drilldown_data = _load_json(drilldown_map) if drilldown_map else None

    # Build report data structures
    metadata = _build_metadata()
    summary = _build_summary(df)
    profiles = sorted(df["profile"].unique().tolist())
    categories = sorted(df["category"].unique().tolist())
    models = sorted(df["model"].unique().tolist())
    heatmap_data = _build_heatmap_data(df, profiles, categories, models)
    profile_tables = _build_profile_tables(df, profiles)
    cost_table = _build_cost_table(df)

    # Build drilldown links if map is available
    drilldown_links = (
        _build_drilldown_links(drilldown_data, models, categories, profiles)
        if drilldown_data
        else None
    )

    # Bundle context for the template
    context = {
        "metadata": metadata,
        "summary": summary,
        "profiles": profiles,
        "categories": categories,
        "models": models,
        "heatmap_data": heatmap_data,
        "profile_tables": profile_tables,
        "cost_table": cost_table,
        "golden_data": golden_data,
        "drilldown_links": drilldown_links,
    }

    html = _render_template("release.html.j2", context)

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


# ---------- Builders ----------


def _build_metadata() -> dict[str, Any]:
    """Build metadata section."""
    return {
        "tag": "dev",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": "N/A",
        "seed": "N/A",
    }


def _build_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build summary cards data."""

    def best_metric(metric: str, higher_is_better: bool = True) -> tuple[str, float]:
        sub = df[df["metric"] == metric][["model", "value"]]
        if sub.empty:
            return ("—", 0.0)
        row = sub.loc[sub["value"].idxmax()] if higher_is_better else sub.loc[sub["value"].idxmin()]
        return (str(row["model"]), float(row["value"]))

    best_cov_model, best_cov = best_metric("coverage", True)
    best_benign_model, best_benign = best_metric("benign_pass_rate", True)
    best_inj_model, best_inj = best_metric("injection_success_rate", False)

    # Calculate totals
    total_cost = df["cost_total_usd"].sum()
    total_tokens = int(df["tokens_in"].sum() + df["tokens_out"].sum())
    p95_latency = df["p95_ms"].median()

    return {
        "best_coverage": best_cov,
        "best_coverage_model": best_cov_model.replace("openai:", ""),
        "best_benign_pass": best_benign,
        "best_benign_model": best_benign_model.replace("openai:", ""),
        "lowest_injection": best_inj,
        "lowest_injection_model": best_inj_model.replace("openai:", ""),
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "p95_latency": p95_latency,
    }


def _build_heatmap_data(
    df: pd.DataFrame, profiles: list[str], categories: list[str], models: list[str]
) -> dict[str, Any]:
    """Build heatmap data for all profiles."""
    heatmap_data = {}

    for profile in profiles:
        profile_data = {}

        for category in categories:
            category_data = {}

            for model in models:
                # Get coverage data for this model/category/profile
                subset = df[
                    (df["profile"] == profile)
                    & (df["category"] == category)
                    & (df["model"] == model)
                    & (df["metric"] == "coverage")
                ]

                if not subset.empty:
                    row = subset.iloc[0]
                    coverage = float(row["value"])
                    ci_low = float(row["ci_low"]) if pd.notna(row["ci_low"]) else coverage
                    ci_high = float(row["ci_high"]) if pd.notna(row["ci_high"]) else coverage
                    ci_width = (ci_high - ci_low) / 2

                    # Color coding based on coverage (0-10 scale)
                    color_scale = min(10, max(0, int(coverage * 10)))
                    css_class = f"coverage-{color_scale}"

                    category_data[model] = {
                        "coverage": coverage,
                        "coverage_ci_width": ci_width,
                        "css_class": css_class,
                    }
                else:
                    category_data[model] = {
                        "coverage": 0.0,
                        "coverage_ci_width": 0.0,
                        "css_class": "coverage-0",
                    }

            profile_data[category] = category_data

        heatmap_data[profile] = profile_data

    return heatmap_data


def _build_profile_tables(df: pd.DataFrame, profiles: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Build detailed tables for each profile."""
    profile_tables = {}

    for profile in profiles:
        profile_df = df[df["profile"] == profile]

        # Group by model and category to create rows
        grouped = profile_df.groupby(["model", "category"])

        rows = []
        for (model, category), group in grouped:
            row = {"model": model, "category": category}

            # Extract metrics
            for metric in [
                "coverage",
                "benign_pass_rate",
                "false_positive_rate",
                "injection_success_rate",
            ]:
                metric_data = group[group["metric"] == metric]
                if not metric_data.empty:
                    metric_row = metric_data.iloc[0]
                    value = float(metric_row["value"])
                    ci_low = (
                        float(metric_row["ci_low"]) if pd.notna(metric_row["ci_low"]) else value
                    )
                    ci_high = (
                        float(metric_row["ci_high"]) if pd.notna(metric_row["ci_high"]) else value
                    )
                    ci_text = f"±{((ci_high - ci_low) * 50):.1f}pp"

                    row[metric] = value
                    row[f"{metric}_ci"] = ci_text
                else:
                    row[metric] = 0.0
                    row[f"{metric}_ci"] = ""

            rows.append(row)

        profile_tables[profile] = rows

    return profile_tables


def _build_cost_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build cost and token usage table."""

    # Group by model to aggregate costs and tokens
    grouped = (
        df.groupby("model")
        .agg(
            {
                "cost_total_usd": "sum",
                "tokens_in": "sum",
                "tokens_out": "sum",
            }
        )
        .reset_index()
    )

    rows = []
    for _, row in grouped.iterrows():
        total_tokens = int(row["tokens_in"] + row["tokens_out"])
        cost_per_1k = (row["cost_total_usd"] / total_tokens * 1000) if total_tokens > 0 else 0.0

        rows.append(
            {
                "model": row["model"],
                "total_cost": row["cost_total_usd"],
                "input_tokens": int(row["tokens_in"]),
                "output_tokens": int(row["tokens_out"]),
                "total_tokens": total_tokens,
                "cost_per_1k": cost_per_1k,
            }
        )

    return sorted(rows, key=lambda x: x["total_cost"], reverse=True)


# ---------- Template rendering ----------


def _render_template(template_name: str, context: dict[str, Any]) -> str:
    """Render template with context."""
    # Template dir: seibox/ui/templates/
    base_dir = Path(__file__).parent
    tmpl_dir = base_dir / "templates"
    env = Environment(
        loader=FileSystemLoader(str(tmpl_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Add custom filters for formatting
    env.filters["format_pct"] = _format_pct
    env.filters["format_money"] = _format_money
    env.filters["format_ci"] = _format_ci

    template = env.get_template(template_name)
    return template.render(**context)


# ---------- Helpers ----------


def _load_json(path: str | None) -> dict[str, Any]:
    """Load JSON file safely."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _build_drilldown_links(
    drilldown_data: list[dict[str, Any]],
    models: list[str],
    categories: list[str],
    profiles: list[str],
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    """
    Build nested structure of drilldown links organized by model/category/profile.

    Returns:
        Nested dict: model -> category -> profile -> list of sample links
    """
    links = {}

    # Initialize structure
    for model in models:
        links[model] = {}
        for category in categories:
            links[model][category] = {}
            for profile in profiles:
                links[model][category][profile] = []

    # Group drilldown entries
    for entry in drilldown_data:
        model = entry.get("model", "unknown")
        category = entry.get("category", "unknown")
        profile = entry.get("profile", "unknown")

        # Ensure the keys exist
        if model not in links:
            links[model] = {}
        if category not in links[model]:
            links[model][category] = {}
        if profile not in links[model][category]:
            links[model][category][profile] = []

        # Add link with metadata for filtering/sorting
        link_info = {
            "id": entry.get("id"),
            "path": entry.get("path"),
            "blocked": entry.get("judgement", {}).get("blocked", False),
            "scores": entry.get("scores", {}),
            "cost": entry.get("cost", 0),
            "latency_ms": entry.get("latency_ms", 0),
        }
        links[model][category][profile].append(link_info)

    # Sort each list by blocked status (failures first) and then by ID
    for model in links:
        for category in links[model]:
            for profile in links[model][category]:
                links[model][category][profile].sort(
                    key=lambda x: (not x.get("blocked", False), x.get("id", ""))
                )

    return links


def _format_pct(x: float | None) -> str:
    """Format value as percentage."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _format_money(x: float | None) -> str:
    """Format value as currency."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"${float(x):.4f}"


def _format_ci(ci_low: float | None, ci_high: float | None) -> str:
    """Format confidence interval as ± range."""
    if ci_low is None or ci_high is None:
        return ""
    if math.isnan(ci_low) or math.isnan(ci_high) or math.isinf(ci_low) or math.isinf(ci_high):
        return ""
    width = (ci_high - ci_low) * 50  # Convert to percentage points
    return f"±{width:.1f}pp"
