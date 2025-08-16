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
        tag="v0.2.0",
        commit="abc1234",
        seed=42,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json
import math
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape


# ---------- Public API ----------


def render_release_report(
    matrix_parquet: str,
    golden_json: Optional[str],
    out_html: str,
    *,
    tag: Optional[str] = None,
    commit: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Render the release HTML report.

    Args:
        matrix_parquet: Path to aggregates/matrix.parquet produced by the release runner.
        golden_json: Optional path to aggregates/golden_compare.json.
        out_html: Destination HTML file path.
        tag: Optional git tag string (e.g., "v0.2.0").
        commit: Optional short commit SHA.
        seed: Optional global seed used for the run.

    Side effects:
        Writes a single HTML file at out_html.
    """
    mp = Path(matrix_parquet)
    if not mp.exists():
        raise FileNotFoundError(f"matrix parquet not found: {mp}")

    df = pd.read_parquet(mp)

    if df.empty:
        raise ValueError("matrix parquet is empty; nothing to render")

    golden = _load_json(golden_json) if golden_json else None

    # Basic fields we expect in df:
    # ['model','category','profile','metric','value','n','ci_low','ci_high',
    #  'cost_total_usd','tokens_in','tokens_out','p95_ms','config_hash','errors?']
    # TODO: firm up with a schema check if helpful.

    meta = _collect_meta(df, tag=tag, commit=commit, seed=seed)

    # Summary cards
    cards = _build_summary_cards(df)

    # Landscape (baseline) pivot – primary “at a glance” table.
    baseline_profile = "baseline"
    landscape = _build_landscape(df, profile=baseline_profile)

    # Per-profile tables
    profiles = sorted(df["profile"].unique().tolist())
    per_profile_tables = {p: _build_landscape(df, profile=p) for p in profiles}

    # Per-model rollups (costs/tokens/latency)
    per_model_costs = _build_model_costs(df)

    # Bundle context for the template
    ctx = ReportContext(
        meta=meta,
        cards=cards,
        baseline_profile=baseline_profile,
        profiles=profiles,
        landscape=landscape,
        per_profile_tables=per_profile_tables,
        per_model_costs=per_model_costs,
        golden=golden or {},
    )

    html = _render_template("release.html.j2", ctx.to_jinja())

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


# ---------- Data structures ----------


@dataclass
class SummaryCard:
    label: str
    value: str
    subtitle: str | None = None
    good: bool | None = None  # True/False for green/red badge, None neutral


@dataclass
class ReportMeta:
    tag: str | None
    commit: str | None
    seed: int | None


@dataclass
class ReportContext:
    meta: ReportMeta
    cards: List[SummaryCard]
    baseline_profile: str
    profiles: List[str]
    landscape: Dict[str, Any]
    per_profile_tables: Dict[str, Dict[str, Any]]
    per_model_costs: pd.DataFrame
    golden: Dict[str, Any]

    def to_jinja(self) -> Dict[str, Any]:
        return {
            "meta": self.meta.__dict__,
            "cards": [c.__dict__ for c in self.cards],
            "baseline_profile": self.baseline_profile,
            "profiles": self.profiles,
            "landscape": self.landscape,
            "per_profile_tables": self.per_profile_tables,
            "per_model_costs": _df_to_records(self.per_model_costs),
            "golden": self.golden,
            "fmt": {
                "pct": fmt_pct,
                "money": fmt_money,
                "ms": fmt_ms,
                "ci": fmt_ci,
            },
        }


# ---------- Builders ----------


def _collect_meta(
    df: pd.DataFrame, *, tag: str | None, commit: str | None, seed: int | None
) -> ReportMeta:
    return ReportMeta(tag=tag, commit=commit, seed=seed)


def _build_summary_cards(df: pd.DataFrame) -> List[SummaryCard]:
    """
    Very lightweight “best of” summary. Refine once metrics are finalized.
    Expects metrics: 'safety_coverage', 'benign_pass_rate', 'injection_success_rate' (lower better),
    and globals like cost/latency in rows or computed from df.
    """

    def best(metric: str, higher_is_better: bool = True) -> tuple[str, float]:
        sub = df[df["metric"] == metric][["model", "value"]]
        if sub.empty:
            return ("—", float("nan"))
        row = sub.loc[sub["value"].idxmax()] if higher_is_better else sub.loc[sub["value"].idxmin()]
        return (str(row["model"]), float(row["value"]))

    best_cov_model, best_cov = best("safety_coverage", True)
    best_benign_model, best_benign = best("benign_pass_rate", True)
    best_inj_model, best_inj = best("injection_success_rate", False)

    total_cost = (
        df.drop_duplicates(subset=["model", "category", "profile"])
        .groupby("model")["cost_total_usd"]
        .sum()
        .sum()
    )
    p95 = df["p95_ms"].median() if "p95_ms" in df else float("nan")

    cards = [
        SummaryCard("Best safety coverage", f"{fmt_pct(best_cov)} ({best_cov_model})", good=True),
        SummaryCard("Best benign pass", f"{fmt_pct(best_benign)} ({best_benign_model})", good=True),
        SummaryCard(
            "Lowest injection success", f"{fmt_pct(best_inj)} ({best_inj_model})", good=True
        ),
        SummaryCard("Total cost (approx)", fmt_money(total_cost)),
        SummaryCard("Median p95 latency", fmt_ms(p95)),
    ]
    return cards


def _build_landscape(df: pd.DataFrame, *, profile: str) -> Dict[str, Any]:
    """
    Returns a dict with:
      - rows: list of categories
      - cols: list of models
      - cells: mapping[(category, model)] -> dict of key metrics + CI
    """
    subset = df[df["profile"] == profile].copy()
    categories = sorted(subset["category"].unique().tolist())
    models = sorted(subset["model"].unique().tolist())

    key_metrics = ("safety_coverage", "benign_pass_rate", "injection_success_rate")
    cells: Dict[tuple[str, str], Dict[str, Any]] = {}

    for cat in categories:
        for mod in models:
            row = subset[(subset["category"] == cat) & (subset["model"] == mod)]
            cell: Dict[str, Any] = {}
            for m in key_metrics:
                metric_row = row[row["metric"] == m]
                if metric_row.empty:
                    cell[m] = None
                    cell[m + "_ci"] = None
                    continue
                # If multiple rows per metric exist, take the last (or mean)
                metric_row = metric_row.iloc[-1]
                cell[m] = float(metric_row["value"])
                ci = (metric_row.get("ci_low"), metric_row.get("ci_high"))
                cell[m + "_ci"] = (float(ci[0]), float(ci[1])) if all(pd.notna(ci)) else None
            cells[(cat, mod)] = cell

    return {"rows": categories, "cols": models, "cells": cells}


def _build_model_costs(df: pd.DataFrame) -> pd.DataFrame:
    # One row per model with total cost/tokens/p95 (aggregate across categories & profiles)
    grp = (
        df.drop_duplicates(subset=["model", "category", "profile"])
        .groupby("model", as_index=False)
        .agg(
            cost_usd=("cost_total_usd", "sum"),
            tokens_in=("tokens_in", "sum"),
            tokens_out=("tokens_out", "sum"),
            p95_ms=("p95_ms", "median"),
        )
    )
    # Sort by cost descending for visibility
    return grp.sort_values("cost_usd", ascending=False)


# ---------- Template rendering ----------


def _render_template(template_name: str, context: Dict[str, Any]) -> str:
    # Template dir: seibox/ui/templates/
    base_dir = Path(__file__).parent
    tmpl_dir = base_dir / "templates"
    env = Environment(
        loader=FileSystemLoader(str(tmpl_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = fmt_pct
    env.filters["money"] = fmt_money
    env.filters["ms"] = fmt_ms
    env.filters["ci"] = fmt_ci
    template = env.get_template(template_name)
    return template.render(**context)


# ---------- Helpers ----------


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def fmt_money(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"${float(x):.2f}"


def fmt_ms(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{float(x):.0f} ms"


def fmt_ci(ci: Optional[tuple[float, float]]) -> str:
    if not ci or any(
        c is None or (isinstance(c, float) and (math.isnan(c) or math.isinf(c))) for c in ci
    ):
        return ""
    low, high = ci
    return f"± {((high - low) * 50):.1f} pp"  # rough symmetric pp width (for compact display)
