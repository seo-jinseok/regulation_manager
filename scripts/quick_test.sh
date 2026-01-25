#!/usr/bin/env bash
# Quick test runner for rapid feedback during development
# Runs only the most critical tests to detect regressions

set -e

echo "=== Quick Test Runner ==="
echo "Running critical tests for rapid feedback..."
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Base command with memory-safe settings
BASE_CMD="pytest -n 2 --no-cov --timeout=300 --maxfail=5 --tb=short"

# Most critical tests (fast, high-value)
CRITICAL_TESTS=(
    "tests/rag/unit/application/test_search_usecase.py"
    "tests/rag/unit/application/test_search_usecase_coverage_v2.py"
    "tests/rag/unit/infrastructure/test_chroma_store.py"
    "tests/rag/unit/infrastructure/test_hyde_coverage.py"
)

echo -e "${YELLOW}Phase 1: Critical application tests${NC}"
for test in "${CRITICAL_TESTS[@]}"; do
    if [ -f "${test}" ] || [ -d "${test}" ]; then
        echo "  Running: ${test}"
        if ${BASE_CMD} "${test}" -q; then
            echo -e "    ${GREEN}✓${NC}"
        else
            echo -e "    ${RED}✗ Failed${NC}"
            exit 1
        fi
    fi
done

echo ""
echo -e "${GREEN}=== All critical tests passed! ===${NC}"
echo ""
echo "For full test suite, run: ./scripts/run_tests_batched.sh"
echo "For specific modules: pytest tests/rag/unit/... -v"
