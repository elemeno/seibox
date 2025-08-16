"""Matrix orchestrator for running evaluations across all models and categories."""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from seibox.runners.eval_runner import run_eval, load_config
from seibox.utils.schemas import SuiteId

console = Console()


@dataclass
class JobSpec:
    """Specification for a single evaluation job."""

    model: str
    category: str
    config_path: str
    sample_size: int
    output_path: str
    config_hash: str
    estimated_calls: int = 0
    estimated_cost: float = 0.0
    estimated_duration: int = 0  # seconds
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None
    actual_duration: Optional[int] = None
    actual_cost: Optional[float] = None


@dataclass
class Plan:
    """Execution plan for matrix evaluation."""

    jobs: List[JobSpec] = field(default_factory=list)
    total_jobs: int = 0
    total_calls: int = 0
    total_cost: float = 0.0
    total_duration: int = 0  # seconds
    outdir: Path = field(default_factory=lambda: Path("runs"))

    def add_job(self, job: JobSpec) -> None:
        """Add a job to the plan."""
        self.jobs.append(job)
        self.total_jobs += 1
        self.total_calls += job.estimated_calls
        self.total_cost += job.estimated_cost
        # Duration estimates account for parallelism
        self.total_duration = max(self.total_duration, job.estimated_duration)

    def get_pending_jobs(self) -> List[JobSpec]:
        """Get jobs that haven't completed successfully."""
        return [job for job in self.jobs if job.status in ["pending", "failed"]]

    def get_completed_jobs(self) -> List[JobSpec]:
        """Get successfully completed jobs."""
        return [job for job in self.jobs if job.status == "completed"]


class MatrixOrchestrator:
    """Orchestrates evaluation runs across multiple models and categories."""

    def __init__(self):
        self.console = console
        self._rate_limiters = {}  # Model name -> rate limiter state
        self._lock = RLock()

    def load_models(self, models_path: str = "configs/models.yaml") -> List[Dict[str, Any]]:
        """Load enabled models from configuration."""
        with open(models_path, "r") as f:
            config = yaml.safe_load(f)

        enabled_models = []
        for model in config.get("models", []):
            if model.get("enabled", False):
                enabled_models.append(model)

        return enabled_models

    def load_categories(
        self, categories_path: str = "configs/categories.yaml"
    ) -> List[Dict[str, Any]]:
        """Load evaluation categories from configuration."""
        with open(categories_path, "r") as f:
            config = yaml.safe_load(f)

        return config.get("categories", [])

    def estimate_job_metrics(
        self, model: Dict[str, Any], category: Dict[str, Any], sample_size: int
    ) -> Tuple[int, float, int]:
        """Estimate API calls, cost, and duration for a job.

        Args:
            model: Model configuration
            category: Category configuration
            sample_size: Number of samples to run

        Returns:
            Tuple of (estimated_calls, estimated_cost_usd, estimated_duration_seconds)
        """
        # Base estimates - these could be improved with historical data
        avg_input_tokens = 150  # Average prompt length
        avg_output_tokens = 50  # Average response length

        # Calculate costs
        cost_config = model.get("cost", {})
        input_cost_per_1k = cost_config.get("input_per_1k", 0.001)
        output_cost_per_1k = cost_config.get("output_per_1k", 0.002)

        estimated_cost = (
            (avg_input_tokens * input_cost_per_1k / 1000)
            + (avg_output_tokens * output_cost_per_1k / 1000)
        ) * sample_size

        # Estimate duration based on rate limits
        rate_limit = model.get("rate_limit", {})
        rpm = rate_limit.get("rpm", 60)  # Default 1 per second

        # Add some overhead for processing time
        base_duration = (sample_size * 60) / rpm  # Convert RPM to seconds
        overhead_factor = 1.2  # 20% overhead for processing
        estimated_duration = int(base_duration * overhead_factor)

        return sample_size, estimated_cost, estimated_duration

    def create_config_hash(self, config_path: str, model: str, sample_size: int) -> str:
        """Create a hash for the configuration to detect changes."""
        try:
            with open(config_path, "r") as f:
                config_content = f.read()

            # Include model name and sample size in hash
            hash_content = f"{config_content}{model}{sample_size}"
            return hashlib.sha256(hash_content.encode()).hexdigest()[:12]
        except FileNotFoundError:
            # Return a default hash if config doesn't exist
            return "missing_config"

    def plan(
        self,
        models: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        sample_mode: str = "SMOKE",
        outdir: str = "runs/landscape",
    ) -> Plan:
        """Create execution plan for matrix evaluation.

        Args:
            models: List of model names to include (None = all enabled)
            categories: List of category IDs to include (None = all)
            sample_mode: SMOKE, FULL, or N=<number>
            outdir: Output directory for results

        Returns:
            Plan with estimated costs and timing
        """
        plan = Plan(outdir=Path(outdir))

        # Load configurations
        all_models = self.load_models()
        all_categories = self.load_categories()

        # Filter models and categories
        if models:
            all_models = [m for m in all_models if m["name"] in models]
        if categories:
            all_categories = [c for c in all_categories if c["id"] in categories]

        # Parse sample mode
        if sample_mode.startswith("N="):
            sample_size = int(sample_mode.split("=")[1])
            sample_sizes = {cat["id"]: sample_size for cat in all_categories}
        else:
            sample_sizes = {}
            for cat in all_categories:
                modes = cat.get("sample_modes", {})
                if sample_mode in modes:
                    sample_sizes[cat["id"]] = modes[sample_mode]
                else:
                    # Fallback to defaults
                    defaults = yaml.safe_load(open("configs/categories.yaml", "r")).get(
                        "defaults", {}
                    )
                    default_modes = defaults.get("sample_modes", {})
                    sample_sizes[cat["id"]] = default_modes.get(sample_mode, 10)

        # Create jobs for each model-category combination
        for model in all_models:
            model_name = model["name"]
            for category in all_categories:
                category_id = category["id"]
                config_path = category["config_path"]

                # Skip if config file doesn't exist
                if not Path(config_path).exists():
                    console.print(f"[yellow]Warning: Config file not found: {config_path}[/yellow]")
                    continue

                sample_size = sample_sizes[category_id]

                # Create output path
                output_path = plan.outdir / model_name.replace(":", "_") / f"{category_id}.jsonl"

                # Calculate estimates
                calls, cost, duration = self.estimate_job_metrics(model, category, sample_size)

                # Create config hash
                config_hash = self.create_config_hash(config_path, model_name, sample_size)

                job = JobSpec(
                    model=model_name,
                    category=category_id,
                    config_path=config_path,
                    sample_size=sample_size,
                    output_path=str(output_path),
                    config_hash=config_hash,
                    estimated_calls=calls,
                    estimated_cost=cost,
                    estimated_duration=duration,
                )

                plan.add_job(job)

        return plan

    def check_existing_results(self, plan: Plan) -> None:
        """Check for existing results and mark jobs as completed if unchanged."""
        for job in plan.jobs:
            output_path = Path(job.output_path)
            summary_path = output_path.with_suffix(".summary.json")

            if output_path.exists() and summary_path.exists():
                try:
                    # Check if config hash matches
                    with open(summary_path, "r") as f:
                        summary = json.load(f)

                    stored_hash = summary.get("metadata", {}).get("config_hash")
                    if stored_hash == job.config_hash:
                        job.status = "completed"
                        # Load actual metrics if available
                        job.actual_cost = summary.get("cost", {}).get("total_usd", 0.0)
                        job.actual_duration = summary.get("metadata", {}).get("duration_seconds", 0)

                except (json.JSONDecodeError, KeyError):
                    # If we can't read the summary, assume we need to re-run
                    pass

    def execute_job(self, job: JobSpec) -> JobSpec:
        """Execute a single evaluation job."""
        start_time = time.time()
        job.status = "running"

        try:
            # Create output directory
            output_path = Path(job.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Update config with sample size
            config = load_config(job.config_path)

            # Update sample sizes for all datasets
            for dataset_name in config.get("datasets", {}):
                if "sampling" in config["datasets"][dataset_name]:
                    config["datasets"][dataset_name]["sampling"]["n"] = job.sample_size

            # Save modified config temporarily
            temp_config_path = output_path.parent / f"temp_config_{job.category}.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(config, f)

            try:
                # Run the evaluation
                run_eval(
                    suite_name=job.category,
                    model_name=job.model,
                    config_path=str(temp_config_path),
                    out_path=job.output_path,
                    mitigation_id=None,
                )

                job.status = "completed"
                job.actual_duration = int(time.time() - start_time)

                # Load actual cost from summary if available
                summary_path = Path(job.output_path).with_suffix(".summary.json")
                if summary_path.exists():
                    with open(summary_path, "r") as f:
                        summary = json.load(f)
                    job.actual_cost = summary.get("cost", {}).get("total_usd", 0.0)

                    # Add config hash to summary for future checks
                    summary.setdefault("metadata", {})["config_hash"] = job.config_hash
                    summary["metadata"]["duration_seconds"] = job.actual_duration

                    with open(summary_path, "w") as f:
                        json.dump(summary, f, indent=2)

            finally:
                # Clean up temp config
                if temp_config_path.exists():
                    temp_config_path.unlink()

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.actual_duration = int(time.time() - start_time)
            console.print(f"[red]Job failed: {job.model} x {job.category}: {e}[/red]")

        return job

    def execute(self, plan: Plan, resume: bool = True, max_workers: int = 3) -> Plan:
        """Execute the evaluation plan.

        Args:
            plan: Execution plan to run
            resume: Skip completed jobs if True
            max_workers: Maximum concurrent jobs

        Returns:
            Updated plan with results
        """
        if resume:
            self.check_existing_results(plan)

        pending_jobs = plan.get_pending_jobs()

        if not pending_jobs:
            console.print("[green]All jobs already completed![/green]")
            return plan

        console.print(f"[bold blue]Executing {len(pending_jobs)} jobs...[/bold blue]")

        # Execute jobs with controlled concurrency
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            task = progress.add_task("Running evaluations", total=len(pending_jobs))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_job = {
                    executor.submit(self.execute_job, job): job for job in pending_jobs
                }

                # Process completed jobs
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        completed_job = future.result()
                        # Update the job in the plan
                        for i, plan_job in enumerate(plan.jobs):
                            if (
                                plan_job.model == completed_job.model
                                and plan_job.category == completed_job.category
                            ):
                                plan.jobs[i] = completed_job
                                break

                        progress.advance(task)

                        if completed_job.status == "completed":
                            console.print(
                                f"[green]✓[/green] {completed_job.model} x {completed_job.category}"
                            )
                        else:
                            console.print(
                                f"[red]✗[/red] {completed_job.model} x {completed_job.category}: {completed_job.error}"
                            )

                    except Exception as e:
                        console.print(f"[red]Unexpected error: {e}[/red]")

        # Print summary
        completed = len(plan.get_completed_jobs())
        failed = len([job for job in plan.jobs if job.status == "failed"])

        console.print(f"\n[bold]Execution Summary:[/bold]")
        console.print(f"  Completed: [green]{completed}[/green]")
        console.print(f"  Failed: [red]{failed}[/red]")
        console.print(f"  Total: {len(plan.jobs)}")

        return plan

    def print_plan(self, plan: Plan) -> None:
        """Print a formatted plan table."""
        table = Table(title="Evaluation Matrix Plan")
        table.add_column("Model", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Samples", justify="right")
        table.add_column("Est. Calls", justify="right")
        table.add_column("Est. Cost", justify="right")
        table.add_column("Est. Duration", justify="right")
        table.add_column("Status", style="bold")

        for job in plan.jobs:
            duration_str = f"{job.estimated_duration // 60}m {job.estimated_duration % 60}s"
            cost_str = f"${job.estimated_cost:.4f}"

            status_color = {
                "pending": "yellow",
                "completed": "green",
                "failed": "red",
                "running": "blue",
            }.get(job.status, "white")

            table.add_row(
                job.model,
                job.category,
                str(job.sample_size),
                str(job.estimated_calls),
                cost_str,
                duration_str,
                f"[{status_color}]{job.status}[/{status_color}]",
            )

        console.print(table)

        # Summary
        total_duration_str = f"{plan.total_duration // 60}m {plan.total_duration % 60}s"
        console.print(f"\n[bold]Plan Summary:[/bold]")
        console.print(f"  Total Jobs: {plan.total_jobs}")
        console.print(f"  Total API Calls: {plan.total_calls:,}")
        console.print(f"  Total Estimated Cost: [green]${plan.total_cost:.4f}[/green]")
        console.print(f"  Estimated Duration: [blue]{total_duration_str}[/blue]")
