"""CLI entry point for Safety Evals in a Box."""

import click
from pathlib import Path
from rich.console import Console

from seibox.runners.eval_runner import run_eval

console = Console()


@click.group()
def main():
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
def run(suite: str, model: str, config: str, out: str, mitigation: str, replay: str):
    """Run an evaluation suite."""
    try:
        run_eval(
            suite_name=suite,
            model_name=model,
            config_path=config,
            out_path=out,
            mitigation_id=mitigation,
            replay_path=replay,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.option("--a", required=True, help="Path to baseline results")
@click.option("--b", required=True, help="Path to comparison results")
@click.option("--report", required=True, help="Output path for HTML report")
def compare(a: str, b: str, report: str):
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
def dashboard(runs: str):
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
def labeler(run: str):
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
def ablate(config: str, model: str, outdir: str):
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
        
        run_ablation_study(
            config_path=config,
            model_name=model,
            output_dir=str(outdir_path)
        )
        
    except Exception as e:
        console.print(f"[bold red]Error running ablation:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.option("--run", required=True, help="Path to evaluation results JSONL file")
@click.option("--labels", required=True, help="Path to human labels JSONL file")
def kappa(run: str, labels: str):
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
def doctor(config: str, model: str, suite: str):
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
def new_suite(name: str, description: str):
    """Create a new evaluation suite with config and dataset scaffolding."""
    from seibox.utils.scaffold import create_new_suite
    
    try:
        create_new_suite(name, description)
        console.print(f"[bold green]✓[/bold green] Created new suite: {name}")
    except Exception as e:
        console.print(f"[bold red]Error creating suite:[/bold red] {e}")
        raise click.Abort()


if __name__ == "__main__":
    main()
