"""Landscape report generator for cross-model safety evaluation."""

import json
from pathlib import Path
from typing import Dict, List, Any
from jinja2 import Template

from seibox.runners.matrix import Plan


def load_results_data(plan: Plan) -> Dict[str, Any]:
    """Load evaluation results from completed jobs."""
    results: Dict[str, Any] = {}

    for job in plan.jobs:
        if job.status != "completed":
            continue

        summary_path = Path(job.output_path).with_suffix(".summary.json")
        if not summary_path.exists():
            continue

        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)

            # Store results by model and category
            if job.model not in results:
                results[job.model] = {}

            results[job.model][job.category] = {"summary": summary, "job": job}

        except Exception:
            continue

    return results


def extract_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from results for visualization."""
    models_list: List[str] = list(results.keys())
    categories_list: List[str] = []
    heatmap_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    frontier_data: List[Dict[str, Any]] = []
    error_jobs: List[str] = []

    metrics = {
        "models": models_list,
        "categories": categories_list,
        "heatmap_data": heatmap_data,
        "frontier_data": frontier_data,
        "error_jobs": error_jobs,
    }

    # Get all categories
    all_categories = set()
    for model_results in results.values():
        all_categories.update(model_results.keys())
    categories_list[:] = sorted(list(all_categories))

    # Initialize heatmap data structure
    for metric in [
        "safety_coverage",
        "benign_pass_rate",
        "injection_success_rate",
        "cost_per_1k",
        "p95_latency",
    ]:
        heatmap_data[metric] = {}
        for model in models_list:
            heatmap_data[metric][model] = {}

    # Extract metrics for each model-category combination
    for model, categories in results.items():
        model_frontier_data = {
            "model": model,
            "safety_coverage": 0,
            "benign_pass_rate": 0,
            "cost_per_1k": 0,
            "p95_latency": 0,
            "categories_count": 0,
        }

        for category, data in categories.items():
            summary = data["summary"]

            # Extract key metrics
            safety_coverage = summary.get("safety_coverage", 0) * 100
            benign_pass_rate = summary.get("benign_pass_rate", 0) * 100
            injection_success_rate = summary.get("injection_success_rate", 0) * 100
            cost_per_1k = summary.get("cost_per_1k", 0)

            # Get latency from summary (p95)
            latency_p95 = 0
            if "latency_p95" in summary:
                latency_p95 = summary["latency_p95"]
            elif "latency" in summary:
                latency_p95 = summary["latency"].get("p95", 0)

            # Store in heatmap data
            heatmap_data["safety_coverage"][model][category] = safety_coverage
            heatmap_data["benign_pass_rate"][model][category] = benign_pass_rate
            heatmap_data["injection_success_rate"][model][category] = injection_success_rate
            heatmap_data["cost_per_1k"][model][category] = cost_per_1k
            heatmap_data["p95_latency"][model][category] = latency_p95

            # Aggregate for frontier chart (average across categories)
            model_frontier_data["safety_coverage"] += safety_coverage
            model_frontier_data["benign_pass_rate"] += benign_pass_rate
            model_frontier_data["cost_per_1k"] += cost_per_1k
            model_frontier_data["p95_latency"] += latency_p95
            model_frontier_data["categories_count"] += 1

        # Average the frontier data
        if model_frontier_data["categories_count"] > 0:
            count = model_frontier_data["categories_count"]
            model_frontier_data["safety_coverage"] /= count
            model_frontier_data["benign_pass_rate"] /= count
            model_frontier_data["cost_per_1k"] /= count
            model_frontier_data["p95_latency"] /= count

            frontier_data.append(model_frontier_data)

    return metrics


def generate_landscape_report(plan: Plan, output_path: str) -> None:
    """Generate comprehensive landscape HTML report."""

    # Load results data
    results = load_results_data(plan)

    # Extract metrics
    metrics = extract_metrics(results)

    # Add error information
    error_jobs = []
    for job in plan.jobs:
        if job.status == "failed":
            error_jobs.append(
                {
                    "model": job.model,
                    "category": job.category,
                    "error": job.error or "Unknown error",
                }
            )
    metrics["error_jobs"] = error_jobs

    # HTML template
    template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Safety Evaluation Landscape</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
        }
        .summary-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .summary-label {
            color: #6c757d;
            margin-top: 5px;
        }
        .plot-container {
            margin: 30px 0;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            overflow: hidden;
        }
        .metric-selector {
            margin: 20px 0;
        }
        .metric-selector button {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 4px;
            border-radius: 4px;
            cursor: pointer;
        }
        .metric-selector button.active {
            background: #2980b9;
        }
        .metric-selector button:hover {
            background: #2980b9;
        }
        .error-section {
            background: #fff5f5;
            border: 1px solid #fed7d7;
            border-radius: 6px;
            padding: 15px;
            margin: 20px 0;
        }
        .error-item {
            background: white;
            border-left: 4px solid #e53e3e;
            padding: 10px;
            margin: 10px 0;
        }
        .filters {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
        }
        .filter-group {
            display: inline-block;
            margin-right: 20px;
        }
        .filter-group label {
            font-weight: bold;
            color: #495057;
        }
        .filter-group select {
            margin-left: 10px;
            padding: 5px;
            border: 1px solid #ced4da;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Safety Evaluation Landscape</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-value">{{ metrics.models|length }}</div>
                <div class="summary-label">Models Evaluated</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{{ metrics.categories|length }}</div>
                <div class="summary-label">Safety Categories</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{{ (metrics.models|length * metrics.categories|length) - metrics.error_jobs|length }}</div>
                <div class="summary-label">Successful Evaluations</div>
            </div>
            {% if metrics.error_jobs %}
            <div class="summary-card">
                <div class="summary-value" style="color: #e74c3c;">{{ metrics.error_jobs|length }}</div>
                <div class="summary-label">Failed Evaluations</div>
            </div>
            {% endif %}
        </div>

        {% if metrics.error_jobs %}
        <div class="error-section">
            <h3>⚠️ Failed Evaluations</h3>
            {% for error in metrics.error_jobs %}
            <div class="error-item">
                <strong>{{ error.model }}</strong> × <strong>{{ error.category }}</strong>: {{ error.error }}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="filters">
            <div class="filter-group">
                <label>View Mode:</label>
                <select id="viewMode">
                    <option value="all">All Models & Categories</option>
                    <option value="models">Compare Models</option>
                    <option value="categories">Compare Categories</option>
                </select>
            </div>
        </div>
        
        <h2>📊 Performance Heatmaps</h2>
        <div class="metric-selector">
            <button onclick="showHeatmap('safety_coverage')" class="active" id="btn-safety_coverage">Safety Coverage</button>
            <button onclick="showHeatmap('benign_pass_rate')" id="btn-benign_pass_rate">Benign Pass Rate</button>
            <button onclick="showHeatmap('injection_success_rate')" id="btn-injection_success_rate">Injection Success Rate</button>
            <button onclick="showHeatmap('cost_per_1k')" id="btn-cost_per_1k">Cost per 1K</button>
            <button onclick="showHeatmap('p95_latency')" id="btn-p95_latency">P95 Latency</button>
        </div>
        <div id="heatmap" class="plot-container"></div>
        
        <h2>🎯 Performance Frontier</h2>
        <p>Safety Coverage vs Benign Pass Rate. Bubble size = cost per 1K calls, color = P95 latency.</p>
        <div id="frontier" class="plot-container"></div>
        
        <script>
            // Data from template
            const metricsData = {{ metrics|tojson }};
            let currentMetric = 'safety_coverage';
            
            // Create heatmap
            function createHeatmap(metric) {
                const data = metricsData.heatmap_data[metric];
                const models = metricsData.models;
                const categories = metricsData.categories;
                
                // Prepare data for heatmap
                const z = [];
                const x = categories;
                const y = models;
                
                for (let model of models) {
                    const row = [];
                    for (let category of categories) {
                        const value = data[model] && data[model][category] !== undefined ? data[model][category] : null;
                        row.push(value);
                    }
                    z.push(row);
                }
                
                // Configure color scale based on metric
                let colorscale, title;
                if (metric === 'injection_success_rate') {
                    colorscale = 'Reds'; // Lower is better
                    title = 'Injection Success Rate (%)';
                } else if (metric === 'cost_per_1k') {
                    colorscale = 'Oranges'; // Lower is better
                    title = 'Cost per 1K Calls ($)';
                } else if (metric === 'p95_latency') {
                    colorscale = 'Oranges'; // Lower is better  
                    title = 'P95 Latency (ms)';
                } else {
                    colorscale = 'Viridis'; // Higher is better
                    title = metric.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                    if (metric.includes('rate') || metric.includes('coverage')) {
                        title += ' (%)';
                    }
                }
                
                const trace = {
                    z: z,
                    x: x,
                    y: y,
                    type: 'heatmap',
                    colorscale: colorscale,
                    showscale: true,
                    hoverongaps: false,
                    hovertemplate: '<b>%{y}</b><br>' +
                                   '<b>%{x}</b><br>' +
                                   title + ': %{z}<br>' +
                                   '<extra></extra>'
                };
                
                const layout = {
                    title: title,
                    xaxis: { title: 'Safety Categories', side: 'bottom' },
                    yaxis: { title: 'Models' },
                    margin: { l: 150, r: 50, t: 80, b: 80 },
                    height: Math.max(400, models.length * 40 + 160)
                };
                
                Plotly.newPlot('heatmap', [trace], layout, {responsive: true});
            }
            
            function showHeatmap(metric) {
                // Update button states
                document.querySelectorAll('.metric-selector button').forEach(btn => btn.classList.remove('active'));
                document.getElementById('btn-' + metric).classList.add('active');
                
                currentMetric = metric;
                createHeatmap(metric);
            }
            
            // Create frontier chart
            function createFrontier() {
                const data = metricsData.frontier_data;
                
                const trace = {
                    x: data.map(d => d.safety_coverage),
                    y: data.map(d => d.benign_pass_rate),
                    mode: 'markers',
                    type: 'scatter',
                    text: data.map(d => d.model),
                    marker: {
                        size: data.map(d => Math.max(8, Math.min(30, d.cost_per_1k * 10000))), // Scale bubble size
                        color: data.map(d => d.p95_latency),
                        colorscale: 'Viridis',
                        showscale: true,
                        colorbar: { title: 'P95 Latency (ms)' },
                        line: { width: 1, color: 'black' }
                    },
                    hovertemplate: '<b>%{text}</b><br>' +
                                   'Safety Coverage: %{x:.1f}%<br>' +
                                   'Benign Pass Rate: %{y:.1f}%<br>' +
                                   'Cost/1K: $%{marker.size}<br>' +
                                   'P95 Latency: %{marker.color}ms<br>' +
                                   '<extra></extra>'
                };
                
                const layout = {
                    title: 'Safety vs Helpfulness Frontier',
                    xaxis: { 
                        title: 'Safety Coverage (%)',
                        range: [0, 100]
                    },
                    yaxis: { 
                        title: 'Benign Pass Rate (%)',
                        range: [0, 100]
                    },
                    margin: { l: 60, r: 60, t: 80, b: 60 },
                    height: 500
                };
                
                Plotly.newPlot('frontier', [trace], layout, {responsive: true});
            }
            
            // Initialize charts
            createHeatmap('safety_coverage');
            createFrontier();
            
            // View mode filtering (placeholder for future enhancement)
            document.getElementById('viewMode').addEventListener('change', function(e) {
                // Could implement filtering logic here
                console.log('View mode changed to:', e.target.value);
            });
        </script>
    </div>
</body>
</html>
    """

    # Render template
    template = Template(template_str)
    html_content = template.render(metrics=metrics)

    # Write to file
    with open(output_path, "w") as f:
        f.write(html_content)
