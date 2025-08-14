"""CLI entry point for Safety Evals in a Box."""

import click
from pathlib import Path
from rich.console import Console

from seibox.runners.eval_runner import run_eval

console = Console()


@click.group()
def cli():
    """Safety Evals in a Box - Evaluate and improve LLM safety."""
    pass


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
def run(suite: str, model: str, config: str, out: str, mitigation: str):
    """Run an evaluation suite."""
    try:
        run_eval(
            suite_name=suite,
            model_name=model,
            config_path=config,
            out_path=out,
            mitigation_id=mitigation,
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
    console.print("[yellow]Compare command not yet implemented[/yellow]")
    console.print(f"Would compare {a} vs {b} and save to {report}")


@cli.command()
@click.option("--runs", required=True, help="Directory containing run results")
def dashboard(runs: str):
    """Launch the Streamlit dashboard."""
    console.print("[yellow]Dashboard command not yet implemented[/yellow]")
    console.print(f"Would launch dashboard with runs from {runs}")
    console.print("Run: streamlit run seibox/ui/dashboard.py")


if __name__ == "__main__":
    cli()