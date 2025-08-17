#!/bin/bash
# Local release script for Safety Evals in a Box
# Runs a smoke evaluation and generates an HTML report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Safety Evals in a Box - Local Release${NC}"
echo "================================================"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}❌ Poetry is not installed. Please install it first:${NC}"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check for API keys
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f .env ]; then
        export $(cat .env | xargs)
    fi
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${YELLOW}⚠️  Warning: OPENAI_API_KEY not set. Using cache-only mode.${NC}"
        echo "   To use OpenAI models, set OPENAI_API_KEY in .env or environment"
    fi
fi

# Default values
OUT_DIR="out/release/local"
SAMPLE="SMOKE"
MODELS="openai:*"
PROFILES="baseline,policy_gate,prompt_hardening,both"
GOLDEN_DIR="golden/v1/"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --out)
            OUT_DIR="$2"
            shift 2
            ;;
        --sample)
            SAMPLE="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --profiles)
            PROFILES="$2"
            shift 2
            ;;
        --golden)
            GOLDEN_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --out DIR       Output directory (default: out/release/local)"
            echo "  --sample TYPE   Sample size: SMOKE or FULL (default: SMOKE)"
            echo "  --models PATTERN Models to include (default: openai:*)"
            echo "  --profiles LIST  Comma-separated profiles (default: baseline,policy_gate,prompt_hardening,both)"
            echo "  --golden DIR    Golden baseline directory (default: golden/v1/)"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run with defaults"
            echo "  $0 --sample FULL             # Full evaluation"
            echo "  $0 --models 'anthropic:*'    # Anthropic models only"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Output: $OUT_DIR"
echo "  Sample: $SAMPLE"
echo "  Models: $MODELS"
echo "  Profiles: $PROFILES"
echo "  Golden: $GOLDEN_DIR"
echo ""

# Check if golden directory exists
if [ ! -d "$GOLDEN_DIR" ]; then
    echo -e "${YELLOW}⚠️  Golden directory not found: $GOLDEN_DIR${NC}"
    echo "   Running without golden comparison"
    GOLDEN_ARG=""
else
    GOLDEN_ARG="--golden $GOLDEN_DIR"
fi

# Run the release command
echo -e "${GREEN}▶️  Starting evaluation...${NC}"
echo ""

poetry run seibox release \
    --out "$OUT_DIR" \
    --sample "$SAMPLE" \
    --include-models "$MODELS" \
    --profiles "$PROFILES" \
    $GOLDEN_ARG

# Check if the report was generated
REPORT_PATH="$OUT_DIR/reports/release.html"
if [ -f "$REPORT_PATH" ]; then
    echo ""
    echo -e "${GREEN}✅ Release complete!${NC}"
    echo ""
    echo "📊 Report generated at: $REPORT_PATH"
    echo ""
    
    # Try to open the report in the default browser
    if command -v open &> /dev/null; then
        echo "Opening report in browser..."
        open "$REPORT_PATH"
    elif command -v xdg-open &> /dev/null; then
        echo "Opening report in browser..."
        xdg-open "$REPORT_PATH"
    else
        echo "To view the report, open:"
        echo "  $REPORT_PATH"
    fi
else
    echo -e "${RED}❌ Report generation failed${NC}"
    echo "Check the output above for errors"
    exit 1
fi