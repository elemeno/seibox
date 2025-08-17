"""Health checks and diagnostics for seibox."""

import os
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from seibox.runners.eval_runner import get_adapter, load_config, load_dataset
from seibox.utils import cost

console = Console()


def check_api_keys() -> dict[str, bool]:
    """Check if API keys are configured for different providers.

    Returns:
        Dictionary mapping provider names to availability status
    """
    console.print("[bold blue]Checking API keys...[/bold blue]")

    keys_status = {}

    # Check OpenAI key
    openai_key = os.getenv("OPENAI_API_KEY")
    keys_status["OpenAI"] = bool(openai_key and len(openai_key.strip()) > 0)

    # Check Anthropic key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    keys_status["Anthropic"] = bool(anthropic_key and len(anthropic_key.strip()) > 0)

    # Check vLLM URL (not really an API key, but connectivity setting)
    vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    keys_status["vLLM"] = bool(vllm_url)

    # Check Gemini key (supports both Developer API and Vertex AI)
    gemini_key = os.getenv("GEMINI_API_KEY")
    google_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")

    # Gemini is available if either API key exists OR Vertex AI is configured
    keys_status["Gemini"] = bool(gemini_key or (use_vertexai and google_project))

    # Create results table
    table = Table(title="API Key Status")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Environment Variable")

    for provider, available in keys_status.items():
        if available:
            status = "[green]✓ Found[/green]"
        else:
            status = "[red]✗ Missing[/red]"

        if provider == "OpenAI":
            env_var = "OPENAI_API_KEY"
        elif provider == "Anthropic":
            env_var = "ANTHROPIC_API_KEY"
        elif provider == "Gemini":
            env_var = "GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT+GOOGLE_GENAI_USE_VERTEXAI"
        else:
            env_var = "VLLM_BASE_URL"

        table.add_row(provider, status, env_var)

    console.print(table)

    return keys_status


def check_provider_connectivity(model_name: str | None = None) -> dict[str, bool]:
    """Test connectivity to different providers.

    Args:
        model_name: Optional specific model to test (e.g., "openai:gpt-4o-mini")

    Returns:
        Dictionary mapping provider names to connectivity status
    """
    console.print("\n[bold blue]Checking provider connectivity...[/bold blue]")

    connectivity_status = {}
    models_to_test = []

    if model_name:
        models_to_test.append(model_name)
    else:
        # Test basic connectivity for each provider
        api_keys = check_api_keys()
        if api_keys.get("OpenAI", False):
            models_to_test.append("openai:gpt-4o-mini")
        if api_keys.get("Anthropic", False):
            models_to_test.append("anthropic:claude-3-haiku-20240307")
        if api_keys.get("Gemini", False):
            models_to_test.append("gemini:gemini-2.0-flash-001")
        # Always test vLLM regardless of key (it's a local server)
        models_to_test.append("vllm:test-model")

    # Test each model/provider
    for model in models_to_test:
        provider = model.split(":")[0]
        console.print(f"  Testing {provider}...")

        try:
            if provider == "vllm":
                # For vLLM, just test if we can reach the server
                base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
                try:
                    response = requests.get(f"{base_url}/v1/models", timeout=5)
                    connectivity_status[provider] = response.status_code == 200
                except requests.exceptions.RequestException:
                    connectivity_status[provider] = False
            else:
                # For API providers, try a minimal completion
                adapter = get_adapter(model)
                start_time = time.time()

                # Use very short prompt and low token limit to minimize cost
                result = adapter.complete(system=None, prompt="Hi", temperature=0.0, max_tokens=1)

                latency = (time.time() - start_time) * 1000
                connectivity_status[provider] = True
                console.print(f"    ✓ {provider} responding ({latency:.0f}ms)")

        except Exception as e:
            connectivity_status[provider] = False
            console.print(f"    ✗ {provider} failed: {str(e)[:50]}...")

    return connectivity_status


def check_cache_directory() -> dict[str, any]:
    """Check if cache directory exists and is writable.

    Returns:
        Dictionary with cache directory status information
    """
    console.print("\n[bold blue]Checking cache directory...[/bold blue]")

    # Get cache directory from cache module
    cache_dir = Path(".cache")  # Default cache location

    cache_status = {
        "path": str(cache_dir.absolute()),
        "exists": cache_dir.exists(),
        "writable": False,
        "size_mb": 0,
        "files_count": 0,
    }

    try:
        # Create cache directory if it doesn't exist
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
            cache_status["exists"] = True
            console.print(f"  Created cache directory: {cache_dir}")

        # Test writability by creating a temporary file
        test_file = cache_dir / "test_write.tmp"
        try:
            with open(test_file, "w") as f:
                f.write("test")
            test_file.unlink()  # Clean up
            cache_status["writable"] = True
        except Exception:
            cache_status["writable"] = False

        # Calculate cache size and file count
        if cache_dir.exists():
            total_size = 0
            file_count = 0
            for file_path in cache_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

            cache_status["size_mb"] = total_size / (1024 * 1024)
            cache_status["files_count"] = file_count

    except Exception as e:
        console.print(f"  ✗ Cache directory error: {e}")
        return cache_status

    # Display results
    if cache_status["writable"]:
        console.print(f"  ✓ Cache directory: {cache_status['path']}")
        console.print(
            f"  ✓ Size: {cache_status['size_mb']:.1f} MB ({cache_status['files_count']} files)"
        )
    else:
        console.print(f"  ✗ Cache directory not writable: {cache_status['path']}")

    return cache_status


def estimate_run_cost(config_path: str, suite_name: str | None = None) -> dict[str, any]:
    """Estimate cost for a planned evaluation run.

    Args:
        config_path: Path to configuration file
        suite_name: Optional suite name to estimate for

    Returns:
        Dictionary with cost estimation details
    """
    console.print("\n[bold blue]Estimating run costs...[/bold blue]")

    try:
        # Load configuration
        config = load_config(config_path)
        cost_table = cost.load_cost_table()

        # Determine suites to estimate
        if suite_name:
            suites = [suite_name]
        else:
            # Default to all configured suites
            suites = list(config.get("datasets", {}).keys())

        if not suites:
            suites = ["pii", "injection", "benign"]  # Fallback

        # Load datasets to count records
        total_records = 0
        suite_counts = {}

        for suite in suites:
            try:
                records = load_dataset(suite, config)
                count = len(records)
                suite_counts[suite] = count
                total_records += count
            except Exception as e:
                console.print(f"  Warning: Could not load {suite} dataset: {e}")
                suite_counts[suite] = 3  # Default fallback
                total_records += 3

        console.print(f"  Total records to process: {total_records}")
        for suite, count in suite_counts.items():
            console.print(f"    {suite}: {count} records")

        # Get default model from config
        default_model = config.get("default", {}).get("name", "openai:gpt-4o-mini")

        # Estimate tokens per record (rough estimate)
        avg_input_tokens = 100  # System prompt + user prompt
        avg_output_tokens = 50  # Model response

        total_input_tokens = total_records * avg_input_tokens
        total_output_tokens = total_records * avg_output_tokens

        # Calculate costs
        estimated_cost = cost.compute_cost(
            default_model, total_input_tokens, total_output_tokens, cost_table
        )

        # Create estimation summary
        estimation = {
            "total_records": total_records,
            "suite_counts": suite_counts,
            "model": default_model,
            "estimated_input_tokens": total_input_tokens,
            "estimated_output_tokens": total_output_tokens,
            "estimated_cost_usd": estimated_cost.get("usd", 0.0),
            "cost_per_1k": (
                estimated_cost.get("usd", 0.0) * 1000 / total_records if total_records > 0 else 0
            ),
        }

        # Display cost table
        table = Table(title="Cost Estimation")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Model", default_model)
        table.add_row("Total Records", str(total_records))
        table.add_row("Est. Input Tokens", f"{total_input_tokens:,}")
        table.add_row("Est. Output Tokens", f"{total_output_tokens:,}")
        table.add_row("Est. Total Cost", f"${estimated_cost.get('usd', 0.0):.4f}")
        table.add_row("Cost per 1K calls", f"${estimation['cost_per_1k']:.4f}")

        console.print(table)

        return estimation

    except Exception as e:
        console.print(f"  ✗ Cost estimation failed: {e}")
        return {"error": str(e)}


def run_health_checks(
    config_path: str | None = None,
    model_name: str | None = None,
    suite_name: str | None = None,
):
    """Run all health checks and diagnostics.

    Args:
        config_path: Optional config file to check
        model_name: Optional specific model to test
        suite_name: Optional suite to estimate costs for
    """
    console.print(Panel.fit("[bold green]Safety Evals in a Box - Health Check[/bold green]"))

    # Track overall health
    issues = []

    # 1. Check API keys
    api_status = check_api_keys()
    if not any(api_status.values()):
        issues.append("No API keys configured")

    # 2. Check provider connectivity
    if model_name or any(api_status.values()):
        connectivity = check_provider_connectivity(model_name)
        failed_providers = [p for p, status in connectivity.items() if not status]
        if failed_providers:
            issues.append(f"Connectivity failed for: {', '.join(failed_providers)}")

    # 3. Check cache directory
    cache_status = check_cache_directory()
    if not cache_status.get("writable", False):
        issues.append("Cache directory not writable")

    # 4. Estimate costs if config provided
    if config_path:
        try:
            cost_estimation = estimate_run_cost(config_path, suite_name)
            if "error" in cost_estimation:
                issues.append(f"Cost estimation failed: {cost_estimation['error']}")
        except Exception as e:
            issues.append(f"Could not load config: {e}")

    # Final summary
    console.print("\n[bold blue]Health Check Summary[/bold blue]")

    if not issues:
        console.print(Panel("[bold green]✓ All checks passed![/bold green]", style="green"))
        console.print("Your seibox installation appears to be working correctly.")
    else:
        console.print(Panel("[bold red]Issues found:[/bold red]", style="red"))
        for issue in issues:
            console.print(f"  • {issue}")

        console.print("\n[yellow]Suggestions:[/yellow]")
        if "No API keys configured" in " ".join(issues):
            console.print("  • Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
            console.print("  • Or create a .env file with your API keys")

        if "Cache directory not writable" in " ".join(issues):
            console.print("  • Ensure current directory has write permissions")
            console.print("  • Or run from a directory you own")

        if any("Connectivity failed" in issue for issue in issues):
            console.print("  • Check your internet connection")
            console.print("  • Verify API keys are valid")
            console.print("  • For vLLM: ensure local server is running")
