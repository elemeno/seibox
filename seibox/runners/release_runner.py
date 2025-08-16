"""Release orchestrator for comprehensive safety evaluation across models and mitigations."""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from rich.console import Console
from rich.table import Table

from seibox.runners.matrix import MatrixOrchestrator, Plan, JobSpec

console = Console()

# Mitigation combinations for release evaluation
MITIGATION_COMBOS = {
    "baseline": {
        "name": "Baseline",
        "description": "No mitigations applied",
        "mitigation_ids": []
    },
    "policy_gate": {
        "name": "Policy Gate Only",
        "description": "Post-processing gate only",
        "mitigation_ids": ["policy_gate@0.1.0"]
    },
    "prompt_hardening": {
        "name": "Prompt Hardening Only", 
        "description": "System prompt hardening only",
        "mitigation_ids": ["prompt_hardening@0.1.0"]
    },
    "both": {
        "name": "Both Mitigations",
        "description": "Policy gate + prompt hardening",
        "mitigation_ids": ["policy_gate@0.1.0", "prompt_hardening@0.1.0"]
    }
}


class ReleaseOrchestrator:
    """Orchestrates comprehensive release evaluation across models and mitigations."""
    
    def __init__(self):
        self.console = console
        self.matrix_orchestrator = MatrixOrchestrator()
    
    def get_enabled_models(self) -> List[Dict]:
        """Get all enabled models from configuration."""
        models = self.matrix_orchestrator.load_models()
        enabled_models = [m for m in models if m.get("enabled", False)]
        
        if not enabled_models:
            console.print("[yellow]Warning: No enabled models found in config[/yellow]")
            
        return enabled_models
    
    def get_safety_categories(self) -> List[Dict]:
        """Get individual safety categories (pii, injection, benign)."""
        all_categories = self.matrix_orchestrator.load_categories()
        safety_categories = [c for c in all_categories if c["id"] in ["pii", "injection", "benign"]]
        
        if len(safety_categories) != 3:
            console.print(f"[yellow]Warning: Expected 3 safety categories, found {len(safety_categories)}[/yellow]")
            
        return safety_categories
    
    def create_release_plan(self, sample_mode: str = "SMOKE", 
                          outdir: str = "releases/latest",
                          models: Optional[List[str]] = None) -> Plan:
        """Create comprehensive release evaluation plan.
        
        Args:
            sample_mode: SMOKE, FULL, or N=<number>
            outdir: Output directory for results
            models: Optional list of model names to filter
            
        Returns:
            Plan with all model x category x mitigation combinations
        """
        enabled_models = self.get_enabled_models()
        safety_categories = self.get_safety_categories()
        
        # Filter models if specified
        if models:
            enabled_models = [m for m in enabled_models if m["name"] in models]
        
        if not enabled_models:
            raise ValueError("No models selected for evaluation")
        
        if not safety_categories:
            raise ValueError("No safety categories found")
        
        plan = Plan(outdir=Path(outdir))
        
        console.print(f"[bold blue]Creating release plan for {len(enabled_models)} models across {len(safety_categories)} categories with {len(MITIGATION_COMBOS)} mitigation configs[/bold blue]")
        
        # Parse sample sizes per category
        sample_sizes = {}
        if sample_mode.startswith("N="):
            sample_size = int(sample_mode.split("=")[1])
            sample_sizes = {cat["id"]: sample_size for cat in safety_categories}
        else:
            for cat in safety_categories:
                modes = cat.get("sample_modes", {})
                if sample_mode in modes:
                    sample_sizes[cat["id"]] = modes[sample_mode]
                else:
                    # Fallback to defaults
                    defaults = yaml.safe_load(open("configs/categories.yaml", 'r')).get("defaults", {})
                    default_modes = defaults.get("sample_modes", {})
                    sample_sizes[cat["id"]] = default_modes.get(sample_mode, 10)
        
        # Create jobs for each model x category x mitigation combination
        for model in enabled_models:
            model_name = model["name"]
            
            for category in safety_categories:
                category_id = category["id"]
                config_path = category["config_path"]
                
                if not Path(config_path).exists():
                    console.print(f"[yellow]Warning: Config file not found: {config_path}[/yellow]")
                    continue
                
                sample_size = sample_sizes[category_id]
                
                for combo_id, combo_config in MITIGATION_COMBOS.items():
                    # Create output path with mitigation combo
                    output_path = (plan.outdir / model_name.replace(":", "_") / 
                                 combo_id / f"{category_id}.jsonl")
                    
                    # Calculate estimates
                    calls, cost, duration = self.matrix_orchestrator.estimate_job_metrics(
                        model, category, sample_size)
                    
                    # Create config hash including mitigations
                    config_hash = self.matrix_orchestrator.create_config_hash(
                        config_path, model_name, sample_size, combo_config["mitigation_ids"])
                    
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
                        mitigation_combo=combo_id,
                        mitigation_ids=combo_config["mitigation_ids"]
                    )
                    
                    plan.add_job(job)
        
        return plan
    
    def print_release_summary(self, plan: Plan) -> None:
        """Print a summary of the release plan."""
        
        # Group by model and mitigation combo
        summary_data = {}
        total_cost = 0.0
        
        for job in plan.jobs:
            model = job.model
            combo = job.mitigation_combo or "baseline"
            
            if model not in summary_data:
                summary_data[model] = {}
            if combo not in summary_data[model]:
                summary_data[model][combo] = {
                    "jobs": 0,
                    "cost": 0.0,
                    "duration": 0
                }
            
            summary_data[model][combo]["jobs"] += 1
            summary_data[model][combo]["cost"] += job.estimated_cost
            summary_data[model][combo]["duration"] = max(
                summary_data[model][combo]["duration"], 
                job.estimated_duration
            )
            total_cost += job.estimated_cost
        
        # Create summary table
        table = Table(title="Release Evaluation Summary")
        table.add_column("Model", style="cyan")
        table.add_column("Mitigation", style="yellow")
        table.add_column("Jobs", justify="right")
        table.add_column("Est. Cost", justify="right")
        table.add_column("Est. Duration", justify="right")
        
        for model in sorted(summary_data.keys()):
            for combo in ["baseline", "policy_gate", "prompt_hardening", "both"]:
                if combo in summary_data[model]:
                    data = summary_data[model][combo]
                    duration_str = f"{data['duration'] // 60}m {data['duration'] % 60}s"
                    cost_str = f"${data['cost']:.4f}"
                    
                    table.add_row(
                        model,
                        MITIGATION_COMBOS[combo]["name"],
                        str(data["jobs"]),
                        cost_str,
                        duration_str
                    )
        
        console.print(table)
        
        # Overall summary
        total_duration_str = f"{plan.total_duration // 60}m {plan.total_duration % 60}s"
        console.print(f"\n[bold]Release Plan Summary:[/bold]")
        console.print(f"  Total Jobs: {plan.total_jobs}")
        console.print(f"  Total API Calls: {plan.total_calls:,}")
        console.print(f"  Total Estimated Cost: [green]${total_cost:.4f}[/green]")
        console.print(f"  Estimated Duration: [blue]{total_duration_str}[/blue]")
        
        # Model breakdown
        console.print(f"\n[bold]Model Coverage:[/bold]")
        for model in sorted(summary_data.keys()):
            model_cost = sum(data["cost"] for data in summary_data[model].values())
            console.print(f"  {model}: ${model_cost:.4f}")
    
    def execute_release(self, plan: Plan, resume: bool = True, 
                       max_workers: int = 2) -> Plan:
        """Execute the release evaluation plan.
        
        Args:
            plan: Release plan to execute
            resume: Skip completed jobs if True
            max_workers: Maximum concurrent jobs (conservative for release)
            
        Returns:
            Updated plan with results
        """
        console.print(f"[bold green]Executing release evaluation...[/bold green]")
        console.print(f"Total jobs: {len(plan.jobs)}")
        console.print(f"Max workers: {max_workers}")
        
        # Use matrix orchestrator for execution
        completed_plan = self.matrix_orchestrator.execute(
            plan, resume=resume, max_workers=max_workers)
        
        # Print final summary
        completed_jobs = len(completed_plan.get_completed_jobs())
        failed_jobs = len([job for job in completed_plan.jobs if job.status == "failed"])
        
        console.print(f"\n[bold green]Release evaluation complete![/bold green]")
        console.print(f"  Completed: [green]{completed_jobs}[/green]")
        console.print(f"  Failed: [red]{failed_jobs}[/red]")
        console.print(f"  Success Rate: {completed_jobs / len(completed_plan.jobs) * 100:.1f}%")
        
        return completed_plan


def run_release_evaluation(sample_mode: str = "SMOKE", 
                         outdir: str = "releases/latest",
                         models: Optional[List[str]] = None,
                         max_workers: int = 2,
                         plan_only: bool = False) -> Plan:
    """Main entry point for release evaluation.
    
    Args:
        sample_mode: SMOKE, FULL, or N=<number>
        outdir: Output directory for results
        models: Optional list of model names to filter
        max_workers: Maximum concurrent jobs
        plan_only: If True, only create and show plan without executing
        
    Returns:
        Execution plan (completed if not plan_only)
    """
    # Create timestamped output directory if using default
    if outdir == "releases/latest":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"releases/{timestamp}"
    
    orchestrator = ReleaseOrchestrator()
    
    # Create plan
    console.print("[bold blue]Creating release evaluation plan...[/bold blue]")
    plan = orchestrator.create_release_plan(
        sample_mode=sample_mode,
        outdir=outdir,
        models=models
    )
    
    # Show plan summary
    orchestrator.print_release_summary(plan)
    
    if plan_only:
        console.print(f"\n[yellow]Plan created. Results would be saved to: {outdir}[/yellow]")
        return plan
    
    # Execute plan
    start_time = time.time()
    completed_plan = orchestrator.execute_release(
        plan, resume=True, max_workers=max_workers)
    
    execution_time = time.time() - start_time
    console.print(f"\n[bold]Total execution time: {execution_time/60:.1f} minutes[/bold]")
    console.print(f"[bold]Results saved to: {outdir}[/bold]")
    
    return completed_plan