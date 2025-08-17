"""
Batch generator for creating drilldown HTML pages from evaluation runs.

Processes all JSONL run files and generates individual HTML pages for each record,
along with an index mapping for quick lookup.
"""

import json
import re
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from seibox.ui.drilldown import create_index, render_drilldown

console = Console()


def parse_run_filename(filename: str) -> tuple[str, str, str]:
    """
    Parse model, category, and profile from run filename.

    Expected formats:
    - openai_gpt-4o-mini_pii_baseline.jsonl
    - model_category_profile.jsonl
    - baseline_gpt4o_mini.jsonl (fallback patterns)
    """
    # Remove .jsonl extension
    base = filename.replace(".jsonl", "")

    # Try to parse components
    parts = base.split("_")

    # Find known categories
    known_categories = ["pii", "injection", "benign"]
    category_idx = -1
    for i, part in enumerate(parts):
        if part in known_categories:
            category_idx = i
            break

    if category_idx > 0 and len(parts) > category_idx:
        # Found a category - reconstruct around it
        model = "_".join(parts[:category_idx])
        category = parts[category_idx]
        profile = (
            "_".join(parts[category_idx + 1 :]) if category_idx + 1 < len(parts) else "baseline"
        )

        # Convert underscores in model name to colons where appropriate
        if model.startswith("openai_"):
            model = model.replace("openai_", "openai:")
        elif model.startswith("anthropic_"):
            model = model.replace("anthropic_", "anthropic:")
        elif model.startswith("gemini_"):
            model = model.replace("gemini_", "gemini:")

        return model, category, profile

    # Check for profile_model pattern (e.g., baseline_gpt4o_mini)
    known_profiles = ["baseline", "both", "policy_gate", "prompt_hardening"]
    if parts[0] in known_profiles:
        profile = parts[0]
        model = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
        # Try to infer category from file content
        return model, None, profile

    # Fallback: try to extract from the content
    return "unknown", None, None


def process_run_file(run_file: Path, out_dir: Path, base_runs_dir: Path) -> list[dict[str, Any]]:
    """
    Process a single run file and generate drilldown pages.

    Returns:
        List of mappings for the drilldown index
    """
    mappings = []

    # Parse filename to get model, category, profile
    model, category, profile = parse_run_filename(run_file.name)

    # Try to get actual values from first record if parsing failed
    try:
        with open(run_file) as f:
            first_line = f.readline()
            if first_line:
                first_record = json.loads(first_line)
                # Get model from record
                if "model" in first_record:
                    model = first_record["model"]
                # Get category and profile from trace
                trace = first_record.get("trace", {})
                if "profile_name" in trace:
                    profile = trace["profile_name"]
                # Try to infer category from the record
                if "scores" in first_record:
                    scores = first_record["scores"]
                    if "email" in scores or "phone" in scores:
                        category = "pii"
                    elif "injection_success" in scores:
                        category = "injection"
                    elif "allowed" in scores:
                        category = "benign"
    except (OSError, json.JSONDecodeError):
        pass

    # Clean model name for directory
    model_dir = model.replace(":", "_").replace("/", "_")

    # Ensure we have valid values for directory structure
    if not category or category == "unknown":
        category = "unknown"
    if not profile or profile == "baseline":
        profile = "baseline"

    # Create output directory structure
    drilldown_dir = out_dir / model_dir / category / profile
    drilldown_dir.mkdir(parents=True, exist_ok=True)

    # Create index file path
    index_path = drilldown_dir / "index.json"

    # Generate drilldown pages
    records_processed = 0
    with open(run_file) as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                record_id = record.get("id", f"record_{line_num}")

                # Generate HTML filename
                safe_id = re.sub(r"[^\w\-_]", "_", record_id)
                html_filename = f"{safe_id}.html"
                html_path = drilldown_dir / html_filename

                # Render drilldown page
                render_drilldown(record, str(html_path))

                # Calculate relative path from base output directory
                relative_path = html_path.relative_to(out_dir)

                # Add to mappings
                mapping = {
                    "id": record_id,
                    "model": model,
                    "category": category,
                    "profile": profile,
                    "path": str(relative_path),
                    "judgement": record.get("judgement", {}),
                    "scores": record.get("scores", {}),
                    "cost": record.get("cost", {}).get("usd", 0),
                    "latency_ms": record.get("timing", {}).get("latency_ms", 0),
                }
                mappings.append(mapping)
                records_processed += 1

            except json.JSONDecodeError as e:
                console.print(
                    f"[yellow]Warning: Failed to parse line {line_num} in {run_file}: {e}[/yellow]"
                )

    # Create index file for this run
    create_index(str(run_file), str(index_path))

    return mappings


@click.command()
@click.option("--runs", required=True, help="Directory containing run JSONL files")
@click.option("--out", required=True, help="Output directory for drilldown HTML files")
@click.option(
    "--max-per-run",
    default=None,
    type=int,
    help="Maximum number of drilldowns per run file (for testing)",
)
def main(runs: str, out: str, max_per_run: int = None):
    """
    Generate drilldown HTML pages for evaluation run results.

    Example:
        poetry run python -m seibox.ui.make_drilldowns \\
            --runs out/release/local/runs \\
            --out out/release/local/drilldown
    """
    runs_dir = Path(runs)
    out_dir = Path(out)

    if not runs_dir.exists():
        console.print(f"[red]Error: Runs directory not found: {runs_dir}[/red]")
        raise click.Abort()

    # Find all JSONL files
    run_files = list(runs_dir.glob("*.jsonl"))

    if not run_files:
        console.print(f"[yellow]No JSONL files found in {runs_dir}[/yellow]")
        return

    console.print(f"[green]Found {len(run_files)} run files to process[/green]")

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process all run files
    all_mappings = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing run files...", total=len(run_files))

        for run_file in run_files:
            progress.update(task, description=f"Processing {run_file.name}...")

            mappings = process_run_file(run_file, out_dir, runs_dir)
            all_mappings.extend(mappings)

            progress.advance(task)

    # Write drilldown map
    map_path = out_dir / "drilldown_map.json"
    with open(map_path, "w") as f:
        json.dump(all_mappings, f, indent=2)

    # Print summary
    console.print("\n[bold green]✓ Drilldown generation complete![/bold green]")

    # Create summary table
    table = Table(title="Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Run files processed", str(len(run_files)))
    table.add_row("Total drilldowns created", str(len(all_mappings)))
    table.add_row("Output directory", str(out_dir))
    table.add_row("Drilldown map", str(map_path))

    # Count by category
    by_category = {}
    for mapping in all_mappings:
        cat = mapping["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    for cat, count in sorted(by_category.items()):
        table.add_row(f"  - {cat}", str(count))

    console.print(table)

    # Show example drilldown paths
    if all_mappings:
        console.print("\n[cyan]Example drilldown pages:[/cyan]")
        for mapping in all_mappings[:3]:
            console.print(f"  • {out_dir}/{mapping['path']}")
        if len(all_mappings) > 3:
            console.print(f"  ... and {len(all_mappings) - 3} more")


if __name__ == "__main__":
    main()
