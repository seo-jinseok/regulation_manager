#!/bin/bash
# Memory-efficient test execution script
# Usage: ./scripts/run_tests_with_memory_limit.sh [number_of_workers]

set -e

# Default to 4 workers if not specified
WORKERS=${1:-4}

echo "========================================"
echo "Memory-Efficient Test Runner"
echo "========================================"
echo "Workers: $WORKERS"
echo ""

# Create necessary directories
mkdir -p .cache
mkdir -p htmlcov

# Check if pytest-xdist is installed
if ! python -c "import xdist" 2>/dev/null; then
    echo "Installing pytest-xdist..."
    uv pip install pytest-xdist pytest-timeout
fi

echo "Step 1: Running tests with pytest-xdist (parallel execution)..."
echo "This isolates memory per worker process"
echo ""

# Run tests with xdist for parallel execution
# Each worker has its own memory space, preventing accumulation
pytest -n "$WORKERS" \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-report=xml:coverage.xml \
    --timeout=300 \
    -v \
    -m "not debug" \
    --maxfail=5 \
    --tb=short \
    tests/

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo ""
echo "Coverage reports generated:"
echo "  - Terminal: Already displayed above"
echo "  - HTML: htmlcov/index.html"
echo "  - XML: coverage.xml"
echo ""
echo "To view HTML coverage report:"
echo "  open htmlcov/index.html"
echo ""
echo "========================================"
echo "Memory Optimization Tips"
echo "========================================"
echo ""
echo "1. For even better memory isolation, reduce workers:"
echo "   pytest -n 2 --cov=src tests/"
echo ""
echo "2. For quick testing without coverage:"
echo "   pytest -n 4 tests/ -x"
echo ""
echo "3. To run specific test modules:"
echo "   pytest -n 4 tests/test_converter_coverage.py -x"
echo ""
echo "4. Monitor memory usage during test:"
echo "   While tests run, check Activity Monitor or htop"
echo ""
