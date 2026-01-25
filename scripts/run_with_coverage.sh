#!/bin/bash
# Run tests with coverage - SEPARATE from regular test execution
# This prevents memory explosion from running coverage with xdist
#
# Usage:
#   ./scripts/run_with_coverage.sh          # Run all tests with coverage
#   ./scripts/run_with_coverage.sh unit     # Run unit tests only
#   ./scripts/run_with_coverage.sh group 0  # Run specific group

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}Running Tests with Coverage (Memory-Safe Mode)${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# Ensure coverage is installed
if ! python -m coverage --version >/dev/null 2>&1; then
    echo -e "${RED}Error: pytest-cov not installed${NC}"
    echo "Run: pip install pytest-cov"
    exit 1
fi

# Clean previous coverage data
echo -e "${YELLOW}Cleaning previous coverage data...${NC}"
rm -f .coverage
rm -f .coverage.*
rm -rf htmlcov/
rm -rf .pytest_cache/

# Determine what to run
TEST_TARGET="$1"

if [ -z "$TEST_TARGET" ] || [ "$TEST_TARGET" = "all" ]; then
    # Run all tests without xdist (sequential for coverage)
    echo -e "${YELLOW}Running all tests sequentially (no xdist)...${NC}"
    python -m pytest \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        --cov-report=json:coverage.json \
        -v \
        --timeout=300 \
        --tb=short \
        -m "not debug" \
        tests/

elif [ "$TEST_TARGET" = "unit" ]; then
    # Unit tests only
    echo -e "${YELLOW}Running unit tests...${NC}"
    python -m pytest \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        -v \
        --timeout=300 \
        tests/rag/unit/ \
        tests/test_*.py

elif [ "$TEST_TARGET" = "group" ]; then
    # Run specific group with coverage
    GROUP_NUM="$2"
    python scripts/run_tests_split.py --group "$GROUP_NUM" \
        --cov=src \
        --cov-report=term-missing \
        --cov-append

else
    # Custom target
    echo -e "${YELLOW}Running custom target: $TEST_TARGET${NC}"
    python -m pytest \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        -v \
        --timeout=300 \
        "$TEST_TARGET"
fi

# Check if coverage report was generated
if [ -f "htmlcov/index.html" ]; then
    echo ""
    echo -e "${GREEN}==================================================${NC}"
    echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
    echo -e "${GREEN}==================================================${NC}"

    # Try to open in browser (macOS only)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${YELLOW}Opening coverage report in browser...${NC}"
        open htmlcov/index.html 2>/dev/null || true
    fi
fi

echo ""
echo -e "${GREEN}Done!${NC}"
