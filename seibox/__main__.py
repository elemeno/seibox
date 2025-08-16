"""CLI entry point for Safety Evals in a Box."""

import click
from pathlib import Path
from rich.console import Console

from seibox.runners.eval_runner import run_eval

console = Console()


@click.group()
def main() -> None:
    """Safety Evals in a Box - Evaluate and improve LLM safety."""
    pass


# Alias for backwards compatibility
cli = main


@cli.command()
@click.option(
    "--suite",
    required=True,
    type=click.Choice(["pii", "injection", "benign", "pi-injection"]),
    help="Evaluation suite to run",
)
@click.option(
    "--model",
    required=True,
    help="Model name (e.g., openai:gpt-4o-mini)",
)
@click.option(
    "--config",
    required=True,
    help="Path to configuration file",
)
@click.option(
    "--out",
    required=True,
    help="Output path for results (JSONL)",
)
@click.option(
    "--mitigation",
    default=None,
    help="Mitigation to apply (e.g., policy_gate@0.1.0)",
)
@click.option(
    "--replay",
    default=None,
    help="Path to existing run results to replay (recompute scores only)",
)
@click.option(
    "--profile",
    default=None,
    help="Profile to apply (baseline, policy_gate, prompt_hardening, both)",
)
@click.option(
    "--limit-per-suite",
    type=int,
    default=None,
    help="Limit number of records per suite (for testing)",
)
def run(
    suite: str,
    model: str,
    config: str,
    out: str,
    mitigation: str,
    replay: str,
    profile: str,
    limit_per_suite: int,
) -> None:
    """Run an evaluation suite."""
    try:
        # Handle limit-per-suite for testing
        original_config = None
        if limit_per_suite:
            import yaml
            from pathlib import Path

            # Load and modify config temporarily
            with open(config, "r") as f:
                config_data = yaml.safe_load(f)

            # Apply limit to all datasets
            for dataset_name in config_data.get("datasets", {}):
                if "sampling" not in config_data["datasets"][dataset_name]:
                    config_data["datasets"][dataset_name]["sampling"] = {}
                config_data["datasets"][dataset_name]["sampling"]["n"] = limit_per_suite

            # Save to temporary file
            temp_config = Path(config).parent / f".temp_{Path(config).name}"
            with open(temp_config, "w") as f:
                yaml.dump(config_data, f)

            original_config = config
            config = str(temp_config)

        try:
            run_eval(
                suite_name=suite,
                model_name=model,
                config_path=config,
                out_path=out,
                mitigation_id=mitigation,
                replay_path=replay,
                profile_name=profile,
            )
        finally:
            # Clean up temporary config if created
            if original_config and limit_per_suite:
                import os

                if os.path.exists(config):
                    os.remove(config)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.option("--a", required=True, help="Path to baseline results")
@click.option("--b", required=True, help="Path to comparison results")
@click.option("--report", required=True, help="Output path for HTML report")
def compare(a: str, b: str, report: str) -> None:
    """Compare two evaluation runs."""
    from seibox.ui.report import generate_report

    try:
        generate_report(b, report, comparison_path=a)
        console.print(f"[bold green]Comparison report saved to:[/bold green] {report}")
    except Exception as e:
        console.print(f"[bold red]Error generating report:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.option("--runs", required=True, help="Directory containing run results")
def dashboard(runs: str) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    from pathlib import Path

    # Check if runs directory exists
    runs_path = Path(runs)
    if not runs_path.exists():
        console.print(f"[bold red]Error:[/bold red] Directory {runs} does not exist")
        raise click.Abort()

    # Set environment variable for dashboard to know runs directory
    import os

    os.environ["SEIBOX_RUNS_DIR"] = runs

    # Launch streamlit
    dashboard_path = Path(__file__).parent / "ui" / "dashboard.py"

    try:
        console.print(f"[bold blue]Launching dashboard...[/bold blue]")
        console.print(f"Runs directory: {runs}")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)], check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error launching dashboard:[/bold red] {e}")
        raise click.Abort()
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")


@cli.command()
@click.option("--run", required=True, help="Path to evaluation results JSONL file")
def labeler(run: str) -> None:
    """Launch the Streamlit labeling interface."""
    import subprocess
    import sys
    from pathlib import Path

    # Check if run file exists
    run_path = Path(run)
    if not run_path.exists():
        console.print(f"[bold red]Error:[/bold red] File {run} does not exist")
        raise click.Abort()

    # Set environment variable for labeler to know run path
    import os

    os.environ["SEIBOX_RUN_PATH"] = run

    # Launch streamlit
    labeler_path = Path(__file__).parent / "ui" / "labeler.py"

    try:
        console.print(f"[bold blue]Launching labeling interface...[/bold blue]")
        console.print(f"Run file: {run}")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(labeler_path)], check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error launching labeler:[/bold red] {e}")
        raise click.Abort()
    except KeyboardInterrupt:
        console.print("\n[yellow]Labeler stopped[/yellow]")


@cli.command()
@click.option("--config", required=True, help="Path to configuration file")
@click.option("--model", required=True, help="Model name (e.g., openai:gpt-4o-mini)")
@click.option("--outdir", required=True, help="Output directory for ablation results")
def ablate(config: str, model: str, outdir: str) -> None:
    """Run ablation study: baseline vs pre-gate vs pre+post-gate."""
    from pathlib import Path
    from seibox.runners.ablate_runner import run_ablation_study

    try:
        # Ensure output directory exists
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold blue]Running ablation study...[/bold blue]")
        console.print(f"Config: {config}")
        console.print(f"Model: {model}")
        console.print(f"Output: {outdir}")

        run_ablation_study(config_path=config, model_name=model, output_dir=str(outdir_path))

    except Exception as e:
        console.print(f"[bold red]Error running ablation:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.option("--run", required=True, help="Path to evaluation results JSONL file")
@click.option("--labels", required=True, help="Path to human labels JSONL file")
def kappa(run: str, labels: str) -> None:
    """Measure agreement between automated judge and human labels using Cohen's kappa."""
    from pathlib import Path
    from rich.table import Table
    from seibox.scoring.calibration import (
        load_judge_labels,
        load_human_labels,
        compute_kappa,
        interpret_kappa,
    )

    # Validate input files
    run_path = Path(run)
    labels_path = Path(labels)

    if not run_path.exists():
        console.print(f"[bold red]Error:[/bold red] Run file not found: {run}")
        raise click.Abort()

    if not labels_path.exists():
        console.print(f"[bold red]Error:[/bold red] Labels file not found: {labels}")
        raise click.Abort()

    try:
        # Load labels
        console.print(f"[blue]Loading judge labels from:[/blue] {run}")
        judge_labels = load_judge_labels(str(run_path))

        console.print(f"[blue]Loading human labels from:[/blue] {labels}")
        human_labels = load_human_labels(str(labels_path))

        console.print(f"Judge labels: {len(judge_labels)} records")
        console.print(f"Human labels: {len(human_labels)} records")

        # Compute kappa
        result = compute_kappa(judge_labels, human_labels)

        # Display results
        console.print("\n[bold green]Agreement Analysis[/bold green]")

        # Main metrics
        kappa_value = result["kappa"]
        if kappa_value is not None:
            console.print(
                f"Cohen's κ: [bold]{kappa_value:.3f}[/bold] ({interpret_kappa(kappa_value)})"
            )
        else:
            console.print("Cohen's κ: [bold red]Cannot compute[/bold red] (insufficient data)")

        # Counts table
        counts = result["agreement_counts"]
        console.print(f"\nCounts:")
        console.print(f"  Agree: [green]{counts['agree']}[/green]")
        console.print(f"  Disagree: [red]{counts['disagree']}[/red]")
        console.print(f"  Human Unsure: [yellow]{counts['unsure']}[/yellow]")
        console.print(f"  Total Compared: {result['total_compared']} (excludes Unsure)")

        # Confusion matrix
        cm = result["confusion_matrix"]
        if result["total_compared"] > 0:
            console.print(f"\n[bold]Confusion Matrix[/bold]")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Judge → Human ↓", style="dim", width=20)
            table.add_column("Judge: Correct", justify="center")
            table.add_column("Judge: Incorrect", justify="center")

            table.add_row(
                "Human: Correct",
                str(cm["judge_correct_human_correct"]),
                str(cm["judge_incorrect_human_correct"]),
            )
            table.add_row(
                "Human: Incorrect",
                str(cm["judge_correct_human_incorrect"]),
                str(cm["judge_incorrect_human_incorrect"]),
            )

            console.print(table)

            if cm["human_unsure"] > 0:
                console.print(f"\nHuman Unsure: {cm['human_unsure']} (excluded from κ calculation)")

        # Summary
        if result["total_compared"] == 0:
            console.print(f"\n[bold yellow]Warning:[/bold yellow] No comparable labels found")
        elif kappa_value is not None and kappa_value >= 0.6:
            console.print(f"\n[bold green]✓[/bold green] Good agreement (κ ≥ 0.6)")
        elif kappa_value is not None and kappa_value >= 0.4:
            console.print(f"\n[bold yellow]~[/bold yellow] Moderate agreement (0.4 ≤ κ < 0.6)")
        else:
            console.print(f"\n[bold red]![/bold red] Poor agreement (κ < 0.4)")

    except Exception as e:
        console.print(f"[bold red]Error computing kappa:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.option("--config", help="Config file to estimate costs for (optional)")
@click.option("--model", help="Model to test connectivity for (optional)")
@click.option("--suite", help="Suite to estimate costs for (optional, requires --config)")
def doctor(config: str, model: str, suite: str) -> None:
    """Run health checks for API keys, connectivity, cache, and cost estimation."""
    from seibox.utils.doctor import run_health_checks

    try:
        run_health_checks(config_path=config, model_name=model, suite_name=suite)
    except Exception as e:
        console.print(f"[bold red]Error running health checks:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.option("--name", required=True, help="Name of the new evaluation suite")
@click.option("--description", help="Description of the evaluation suite (optional)")
def new_suite(name: str, description: str) -> None:
    """Create a new evaluation suite with config and dataset scaffolding."""
    from seibox.utils.scaffold import create_new_suite

    try:
        create_new_suite(name, description)
        console.print(f"[bold green]✓[/bold green] Created new suite: {name}")
    except Exception as e:
        console.print(f"[bold red]Error creating suite:[/bold red] {e}")
        raise click.Abort()


@cli.command("validate-prompts")
@click.option("--path", required=True, help="Path to prompts JSONL file(s), supports wildcards")
def validate_prompts(path: str) -> None:
    """Validate prompt specification files for correctness.

    Examples:
      poetry run seibox validate-prompts --path prompts.jsonl
      poetry run seibox validate-prompts --path "seibox/datasets/*/prompts.jsonl"
    """
    import json
    from pathlib import Path
    from glob import glob
    from rich.table import Table
    from seibox.utils.prompt_spec import PromptSpec, PromptSpecValidationResult
    from seibox.datasets.dsl import validate_template_syntax

    # Find all matching files
    files = glob(path, recursive=True)
    if not files:
        console.print(f"[bold red]No files found matching:[/bold red] {path}")
        raise click.Abort()

    total_valid = 0
    total_invalid = 0

    for file_path in sorted(files):
        console.print(f"\n[bold blue]Validating:[/bold blue] {file_path}")

        results = []
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    spec = PromptSpec(**data)

                    # Validate template syntax
                    is_valid, error = validate_template_syntax(spec.template)
                    if not is_valid:
                        results.append(
                            PromptSpecValidationResult(
                                valid=False, line_number=line_num, error=f"Template error: {error}"
                            )
                        )
                        total_invalid += 1
                    else:
                        results.append(
                            PromptSpecValidationResult(valid=True, line_number=line_num, spec=spec)
                        )
                        total_valid += 1

                except json.JSONDecodeError as e:
                    results.append(
                        PromptSpecValidationResult(
                            valid=False, line_number=line_num, error=f"Invalid JSON: {str(e)}"
                        )
                    )
                    total_invalid += 1
                except Exception as e:
                    results.append(
                        PromptSpecValidationResult(valid=False, line_number=line_num, error=str(e))
                    )
                    total_invalid += 1

        # Display results table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Line", style="dim", width=6)
        table.add_column("Status", width=8)
        table.add_column("ID", width=30)
        table.add_column("Category", width=10)
        table.add_column("Error", style="red")

        for result in results:
            if result.valid and result.spec:
                table.add_row(
                    str(result.line_number),
                    result.status_emoji,
                    result.spec.id[:30],
                    result.spec.category,
                    "",
                )
            else:
                table.add_row(
                    str(result.line_number),
                    result.status_emoji,
                    "-",
                    "-",
                    result.error or "Unknown error",
                )

        console.print(table)

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Valid prompts: [green]{total_valid}[/green]")
    console.print(f"  Invalid prompts: [red]{total_invalid}[/red]")

    if total_invalid > 0:
        raise click.Abort()


@cli.command()
@click.option("--path", required=True, help="Path to prompts JSONL file")
@click.option("--n", default=5, help="Number of examples to render")
@click.option("--out", required=True, help="Output path for rendered JSONL")
def render(path: str, n: int, out: str) -> None:
    """Render prompt templates for preview and human review.

    Examples:
      poetry run seibox render --path prompts.jsonl --n 3 --out preview.jsonl
      poetry run seibox render --path seibox/datasets/pii/prompts.jsonl --n 10 --out pii_preview.jsonl
    """
    import json
    from pathlib import Path
    from rich.progress import track
    from seibox.utils.prompt_spec import PromptSpec
    from seibox.datasets.dsl import to_input_record
    from seibox.utils.io import write_jsonl

    # Check input file
    input_path = Path(path)
    if not input_path.exists():
        console.print(f"[bold red]File not found:[/bold red] {path}")
        raise click.Abort()

    # Load and validate specs
    specs = []
    with open(input_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                spec = PromptSpec(**data)
                specs.append(spec)
            except Exception as e:
                console.print(f"[bold red]Error parsing prompt:[/bold red] {e}")
                continue

    if not specs:
        console.print(f"[bold red]No valid prompts found in:[/bold red] {path}")
        raise click.Abort()

    # Limit to requested number
    specs_to_render = specs[:n]

    console.print(f"[bold blue]Rendering {len(specs_to_render)} prompts...[/bold blue]")

    # Render each spec
    rendered_records = []
    for spec in track(specs_to_render, description="Rendering"):
        record = to_input_record(spec)
        rendered_records.append(record)

    # Save to output
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), rendered_records)

    console.print(f"[bold green]✓[/bold green] Rendered {len(rendered_records)} prompts to: {out}")

    # Show preview of first few
    console.print("\n[bold]Preview of rendered prompts:[/bold]")
    for i, record in enumerate(rendered_records[:3], 1):
        console.print(f"\n[dim]#{i} ({record.id}):[/dim]")
        preview = record.prompt[:200] + "..." if len(record.prompt) > 200 else record.prompt
        console.print(f"  {preview}")


@cli.command("export-review")
@click.option(
    "--runs", required=True, help="Path(s) to evaluation results JSONL file(s), supports wildcards"
)
@click.option("--out", required=True, help="Output path for review file")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "jsonl"]),
    default="csv",
    help="Output format",
)
def export_review(runs: str, out: str, output_format: str) -> None:
    """Export evaluation results for human review and labeling."""
    from seibox.tools.export_review import export_review as export_func

    try:
        # Handle multiple paths
        run_paths = [p.strip() for p in runs.split(",")]

        console.print(f"[blue]Exporting results for human review...[/blue]")
        console.print(f"Input: {runs}")
        console.print(f"Output: {out}")
        console.print(f"Format: {output_format}")

        summary = export_func(run_paths, out, format=output_format)

        console.print(f"\n[bold green]✓[/bold green] Export completed")
        console.print(f"Files processed: {summary['files_processed']}")
        console.print(f"Records exported: {summary['records_exported']}")
        console.print(f"Output: {summary['output_path']}")

        if output_format == "csv":
            console.print(f"Columns: {summary['columns']}")
            console.print(f"\n[dim]Open in spreadsheet app for human labeling[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error exporting for review:[/bold red] {e}")
        raise click.Abort()


@cli.command("import-review")
@click.option("--labels", required=True, help="Path to human labels file (CSV or JSONL)")
@click.option("--out", required=True, help="Output path for normalized golden labels")
def import_review(labels: str, out: str) -> None:
    """Import human review labels and create normalized golden labels."""
    from seibox.tools.import_review import import_review as import_func

    try:
        console.print(f"[blue]Importing human labels...[/blue]")
        console.print(f"Input: {labels}")
        console.print(f"Output: {out}")

        summary = import_func(labels, out)

        console.print(f"\n[bold green]✓[/bold green] Import completed")
        console.print(f"Format: {summary['format']}")
        console.print(f"Rows processed: {summary['rows_processed']}")
        if summary["rows_skipped"] > 0:
            console.print(f"Rows skipped: [yellow]{summary['rows_skipped']}[/yellow]")
        console.print(f"Golden labels created: {summary['total_labels']}")
        console.print(f"Output: {summary['output_path']}")

        if summary["format"] == "csv":
            console.print(f"\n[dim]Used columns:[/dim]")
            console.print(f"  Label: {summary['label_column']}")
            if summary["reviewer_column"]:
                console.print(f"  Reviewer: {summary['reviewer_column']}")
            if summary["notes_column"]:
                console.print(f"  Notes: {summary['notes_column']}")

    except Exception as e:
        console.print(f"[bold red]Error importing labels:[/bold red] {e}")
        raise click.Abort()


@cli.command("sanitize-run")
@click.option("--run", required=True, help="Path to evaluation results JSONL file to sanitize")
@click.option("--out", required=True, help="Output path for sanitized results")
@click.option("--redact-system", is_flag=True, help="Redact system prompts (replace with hash)")
@click.option("--redact-raw", is_flag=True, help="Remove raw responses before post-processing")
def sanitize_run(run: str, out: str, redact_system: bool, redact_raw: bool) -> None:
    """Sanitize evaluation results for public sharing by redacting sensitive information."""
    import json
    from pathlib import Path
    from seibox.utils.io import read_jsonl, write_jsonl
    from seibox.utils.schemas import OutputRecord, Trace, Message

    try:
        # Load results
        console.print(f"[blue]Loading results from:[/blue] {run}")
        records = [OutputRecord(**r) for r in read_jsonl(run)]

        if not records:
            console.print(f"[bold red]No records found in {run}[/bold red]")
            raise click.Abort()

        console.print(f"Found {len(records)} records to sanitize")

        # Sanitize each record
        sanitized_records = []
        for record in records:
            # Create a copy of the record
            sanitized_record = record.model_copy(deep=True)

            # Handle trace sanitization
            if isinstance(record.trace, dict):
                # Old format - convert to new format if possible
                trace_data = record.trace.copy()

                # Redact system prompt if requested
                if redact_system:
                    trace_data.pop("system_prompt_preview", None)
                    if "messages" in trace_data:
                        for msg in trace_data["messages"]:
                            if msg.get("role") == "system":
                                msg["content"] = (
                                    f"[Hash: {trace_data.get('system_prompt_hash', 'unknown')}]"
                                )
                                msg["redacted"] = True

                # Remove raw responses if requested
                if redact_raw:
                    trace_data.pop("assistant_raw", None)

                sanitized_record.trace = trace_data

            elif isinstance(record.trace, Trace):
                # New format - work with Trace object
                trace = record.trace.model_copy(deep=True)

                # Redact system prompt if requested
                if redact_system:
                    trace.system_prompt_preview = None
                    trace.include_system_full = False

                    # Redact system messages
                    for msg in trace.messages:
                        if msg.role == "system":
                            msg.content = f"[Hash: {trace.system_prompt_hash}]"
                            msg.redacted = True

                # Remove raw responses if requested
                if redact_raw:
                    trace.assistant_raw = None

                sanitized_record.trace = trace

            sanitized_records.append(sanitized_record)

        # Save sanitized results
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(str(out_path), sanitized_records)

        console.print(f"[bold green]✓[/bold green] Sanitized {len(sanitized_records)} records")
        console.print(f"[green]Saved to:[/green] {out}")

        # Show sanitization summary
        console.print("\n[bold]Sanitization applied:[/bold]")
        console.print(f"  System prompts redacted: {'Yes' if redact_system else 'No'}")
        console.print(f"  Raw responses removed: {'Yes' if redact_raw else 'No'}")

    except Exception as e:
        console.print(f"[bold red]Error sanitizing run:[/bold red] {e}")
        raise click.Abort()


@cli.group()
def packs() -> None:
    """Manage portable prompt packs."""
    pass


@packs.command("list")
def packs_list() -> None:
    """List available prompt packs."""
    from rich.table import Table
    from seibox.packs import discover_packs

    packs = discover_packs()

    if not packs:
        console.print("[yellow]No packs found in packs/ directory[/yellow]")
        return

    table = Table(title="Available Prompt Packs")
    table.add_column("ID", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Name", style="white")
    table.add_column("Categories", style="magenta")
    table.add_column("Prompts", justify="right")
    table.add_column("License", style="dim")

    for pack in packs:
        categories = ", ".join(cat.id for cat in pack.categories)
        table.add_row(
            pack.id,
            pack.version,
            pack.name or "-",
            categories,
            str(pack.prompt_count),
            pack.license or "-",
        )

    console.print(table)
    console.print(f"\n[dim]Total packs: {len(packs)}[/dim]")


@packs.command("import")
@click.option("--id", "pack_id", required=True, help="Pack ID to import")
@click.option("--category", required=True, help="Category to import")
@click.option("--dest", required=True, help="Destination directory")
@click.option("--no-dedupe", is_flag=True, help="Don't deduplicate prompts by ID")
@click.option("--preview", is_flag=True, help="Preview import without writing files")
def packs_import(pack_id: str, category: str, dest: str, no_dedupe: bool, preview: bool) -> None:
    """Import prompts from a pack into a dataset."""
    from seibox.packs import import_pack

    try:
        # Import the pack
        summary = import_pack(
            pack_id=pack_id, category=category, dest=dest, dedupe=not no_dedupe, preview=preview
        )

        # Display summary
        if preview:
            console.print("\n[bold yellow]PREVIEW MODE - No files written[/bold yellow]")

        console.print(f"\n[bold]Import Summary:[/bold]")
        console.print(f"  Pack: {summary['pack_id']} v{summary['pack_version']}")
        console.print(f"  Category: {summary['category']}")
        console.print(f"  Destination: {summary['destination']}")
        console.print(f"  Existing prompts: {summary['existing_count']}")
        console.print(f"  New prompts: [green]{summary['imported_count']}[/green]")

        if summary["duplicate_count"] > 0:
            console.print(f"  Duplicates skipped: [yellow]{summary['duplicate_count']}[/yellow]")
            if summary["duplicates"]:
                console.print(f"    First few: {', '.join(summary['duplicates'][:5])}")

        console.print(f"  Total prompts: [bold]{summary['total_count']}[/bold]")

        if "backup_created" in summary:
            console.print(f"  Backup created: {summary['backup_created']}")

        if not preview:
            console.print(
                f"\n[green]✓[/green] Successfully imported {summary['imported_count']} prompts"
            )

    except Exception as e:
        console.print(f"[bold red]Error importing pack:[/bold red] {e}")
        raise click.Abort()


@packs.command("validate")
@click.option("--id", "pack_id", help="Pack ID to validate")
@click.option("--path", help="Path to pack directory")
def packs_validate(pack_id: str, path: str) -> None:
    """Validate a pack's structure and contents."""
    from pathlib import Path
    from seibox.packs import discover_packs
    from seibox.packs.loader import validate_pack

    # Find pack path
    if path:
        pack_path = path
    elif pack_id:
        packs = discover_packs()
        pack = next((p for p in packs if p.id == pack_id), None)
        if not pack:
            console.print(f"[bold red]Pack not found:[/bold red] {pack_id}")
            raise click.Abort()
        pack_path = str(pack.path)
    else:
        console.print("[bold red]Either --id or --path must be specified[/bold red]")
        raise click.Abort()

    # Validate pack
    results = validate_pack(pack_path)

    # Display results
    if results["valid"]:
        console.print(f"[green]✓[/green] Pack is valid: {pack_path}")
    else:
        console.print(f"[red]✗[/red] Pack validation failed: {pack_path}")

    if results["errors"]:
        console.print("\n[bold red]Errors:[/bold red]")
        for error in results["errors"]:
            console.print(f"  • {error}")

    if results["warnings"]:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in results["warnings"]:
            console.print(f"  • {warning}")

    if results["stats"]:
        console.print("\n[bold]Statistics:[/bold]")
        for key, value in results["stats"].items():
            console.print(f"  {key}: {value}")

    if not results["valid"]:
        raise click.Abort()


@cli.command()
@click.option("--sample", default="SMOKE", help="Sample mode: SMOKE, FULL, or N=<number>")
@click.option("--out", default="runs/landscape", help="Output directory for results")
@click.option("--plan", is_flag=True, help="Show execution plan without running")
@click.option("--models", help="Comma-separated list of models to include")
@click.option("--categories", help="Comma-separated list of categories to include")
@click.option("--workers", default=3, help="Maximum concurrent evaluation jobs")
def landscape(
    sample: str, out: str, plan: bool, models: str, categories: str, workers: int
) -> None:
    """Run evaluation matrix across all models and safety categories."""
    from datetime import datetime
    from pathlib import Path
    from seibox.runners.matrix import MatrixOrchestrator

    try:
        # Parse model and category filters
        model_list = models.split(",") if models else None
        category_list = categories.split(",") if categories else None

        # Create output directory with timestamp if not specified
        if out == "runs/landscape":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = f"runs/landscape/{timestamp}"

        orchestrator = MatrixOrchestrator()

        # Create execution plan
        console.print("[bold blue]Creating execution plan...[/bold blue]")
        execution_plan = orchestrator.plan(
            models=model_list, categories=category_list, sample_mode=sample, outdir=out
        )

        # Always show the plan
        orchestrator.print_plan(execution_plan)

        if plan:
            # Dry run - just show the plan
            console.print("\n[yellow]Dry run complete. Use without --plan to execute.[/yellow]")
            return

        # Execute the plan
        console.print(f"\n[bold green]Executing plan...[/bold green]")
        completed_plan = orchestrator.execute(execution_plan, resume=True, max_workers=workers)

        # Generate reports after execution
        console.print(f"\n[bold blue]Generating reports...[/bold blue]")

        # Create reports directory
        reports_dir = Path("out/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate per-model reports
        from seibox.ui.report import generate_model_reports

        try:
            generate_model_reports(completed_plan, str(reports_dir))
            console.print(f"[green]✓[/green] Per-model reports generated in {reports_dir}")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate per-model reports: {e}[/yellow]")

        # Generate landscape report
        from seibox.ui.landscape_report import generate_landscape_report

        try:
            landscape_path = reports_dir / "landscape.html"
            generate_landscape_report(completed_plan, str(landscape_path))
            console.print(f"[green]✓[/green] Landscape report generated: {landscape_path}")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate landscape report: {e}[/yellow]")

        # Generate data bundle
        try:
            from seibox.utils.data_bundle import generate_parquet_bundle

            bundle_path = Path(out) / "all_runs.parquet"
            generate_parquet_bundle(completed_plan, str(bundle_path))
            console.print(f"[green]✓[/green] Data bundle saved: {bundle_path}")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate data bundle: {e}[/yellow]")

        console.print(f"\n[bold green]Landscape evaluation complete![/bold green]")
        console.print(f"Results: {out}")
        console.print(f"Reports: {reports_dir}")

    except Exception as e:
        console.print(f"[bold red]Error running landscape:[/bold red] {e}")
        raise click.Abort()


if __name__ == "__main__":
    main()
