#!/bin/bash
# Format all Python code with Black

echo "🎨 Formatting Python code with Black..."
poetry run black seibox tests scripts --line-length=100

echo "✓ Formatting complete!"