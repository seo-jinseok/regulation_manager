#!/usr/bin/env bash
# Run tests in smaller batches to prevent memory overload
# This script splits the 2198 tests into manageable chunks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Batched Test Runner ===${NC}"
echo "Running 2198 tests in batches to prevent memory overload"
echo ""

# Configuration
BATCH_SIZE=${BATCH_SIZE:-200}
DELAY=${DELAY:-5}
WORKERS=${WORKERS:-2}
BASE_CMD="pytest -n ${WORKERS} --no-cov --timeout=300 --maxfail=10 --tb=short"

# Count total tests
echo "Counting total tests..."
TOTAL_TESTS=$(pytest --collect-only -q 2>/dev/null | grep -oE '[0-9]+ tests collected' | grep -oE '[0-9]+' || echo "2198")
echo -e "${YELLOW}Total tests: ${TOTAL_TESTS}${NC}"
echo ""

# Test batches to run (most critical first)
BATCHES=(
    "tests/rag/unit/application/"
    "tests/rag/unit/infrastructure/"
    "tests/rag/unit/interface/"
    "tests/rag/integration/"
    "tests/unit/"
    "tests/integration/"
    "tests/test_converter_coverage.py"
    "tests/test_converter_extended.py"
    "tests/test_coverage_boost.py"
    "tests/test_main*.py"
    "tests/test_formatter*.py"
    "tests/test_reference*.py"
    "tests/test_search*.py"
    "tests/test_gradio*.py"
    "tests/test_llm*.py"
    "tests/test_refine*.py"
)

FAILED_BATCHES=()
PASSED_BATCHES=0

# Run each batch
for batch in "${BATCHES[@]}"; do
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running batch: ${batch}${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Get test count for this batch
    BATCH_COUNT=$(pytest --collect-only "${batch}" -q 2>/dev/null | grep -oE '[0-9]+ tests collected' | grep -oE '[0-9]+' || echo "unknown")
    echo -e "Test count: ${YELLOW}${BATCH_COUNT}${NC}"

    # Run the batch
    START_TIME=$(date +%s)
    if ${BASE_CMD} "${batch}"; then
        PASSED_BATCHES=$((PASSED_BATCHES + 1))
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "${GREEN}✓ Batch passed in ${DURATION}s${NC}"
    else
        FAILED_BATCHES+=("${batch}")
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "${RED}✗ Batch failed after ${DURATION}s${NC}"
    fi

    # Memory cooldown between batches
    if [ "${DELAY}" -gt 0 ]; then
        echo -e "${YELLOW}Waiting ${DELAY}s for GC cooldown...${NC}"
        sleep "${DELAY}"
    fi
    echo ""
done

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}=== Test Summary ===${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Passed batches: ${GREEN}${PASSED_BATCHES}${NC}"
echo -e "Failed batches: ${RED}${#FAILED_BATCHES[@]}${NC}"

if [ ${#FAILED_BATCHES[@]} -gt 0 ]; then
    echo -e "${RED}Failed batches:${NC}"
    for batch in "${FAILED_BATCHES[@]}"; do
        echo -e "  ${RED}- ${batch}${NC}"
    done
    exit 1
fi

echo -e "${GREEN}All batches passed!${NC}"
exit 0
