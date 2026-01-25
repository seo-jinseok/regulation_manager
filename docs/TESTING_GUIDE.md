# Testing Guide - Memory-Safe Configuration

This guide explains the memory-safe testing configuration implemented to prevent the 31GB memory leak issue when running tests with coverage.

## Problem Summary

**Original Issue**:
- pytest-xdist worker processes accumulated 31GB+ memory
- Coverage execution caused memory explosion
- 2111 tests across 3500+ test files (1.37M+ lines of code)

**Root Causes**:
1. Coverage data accumulation in each xdist worker
2. Large test files (e.g., `test_query_handler_extended.py` with 2114 lines)
3. Thousands of parametrized test cases staying in memory
4. Insufficient garbage collection between tests

## Solution Overview

### 1. Coverage Separation (CRITICAL)

**Default behavior**: `pytest` now runs with `--no-cov` by default
- Coverage is completely separated from regular test execution
- Prevents memory explosion from coverage + xdist interaction

**To run coverage separately**:
```bash
# Run all tests with coverage (sequential, no xdist)
./scripts/run_with_coverage.sh

# Run unit tests only with coverage
./scripts/run_with_coverage.sh unit

# Run specific group with coverage
./scripts/run_with_coverage.sh group 0
```

### 2. Test Grouping

Tests are divided into 10 groups to prevent memory buildup:

| Group | Description | Example Tests |
|-------|-------------|---------------|
| 0 | Core functionality | converter, formatter, main |
| 1 | Parsing | converter_extended, parsing_coverage |
| 2 | RAG Core | domain entities, infrastructure |
| 3 | RAG Search | search_usecase, query_analyzer |
| 4 | RAG Interface | query_handler, web integration |
| 5 | Automation | automation framework tests |
| 6 | Coverage Files | focused_coverage, priority_modules |
| 7 | Extended Tests | enhance_for_rag, search_extended2 |
| 8 | Gradio | gradio_app tests |
| 9 | Misc | utils, analysis, cache_manager |

**Run tests by group**:
```bash
# Run all groups sequentially (memory-safe)
python scripts/run_tests_split.py

# Run specific group
python scripts/run_tests_split.py --group 0

# List all groups
python scripts/run_tests_split.py --list-groups

# Run with xdist per group (4 workers)
python scripts/run_tests_split.py --workers 4
```

### 3. Enhanced Memory Management

**conftest.py improvements**:
- Aggressive GC thresholds: `(100, 5, 5)` instead of default `(700, 10, 10)`
- Automatic GC after each test teardown
- xdist worker-specific GC settings
- Vector store cleanup tracking with weakref
- BM25 cache cleanup

**pyproject.toml settings**:
- `--no-cov` in default addopts
- xdist scheduler set to `loadscope` for better memory isolation
- coverage parallel execution disabled
- `--maxfail=5` to stop early on persistent failures

## Usage Patterns

### Development Workflow

```bash
# 1. Quick test run during development (no coverage, fast)
pytest                          # Uses --no-cov by default

# 2. Run specific test file
pytest tests/test_main.py

# 3. Run with xdist (parallel)
pytest -n 4                    # 4 workers, each with isolated memory

# 4. Run specific group
python scripts/run_tests_split.py --group 3

# 5. Generate coverage report (when needed)
./scripts/run_with_coverage.sh
```

### CI/CD Pipeline

```bash
# 1. Run all tests without coverage (fast feedback)
python scripts/run_tests_split.py --workers 4

# 2. If tests pass, run coverage separately
./scripts/run_with_coverage.sh

# 3. Check coverage threshold
coverage report --fail-under=80
```

### Debugging Memory Issues

```bash
# Run specific problematic test group
python scripts/run_tests_split.py --group 7 -v

# Check which tests use most memory
pytest -v -m memory_intensive

# Run without xdist to isolate memory issues
pytest --no-cov tests/test_focused_coverage_improvement.py
```

## Configuration Files

### pyproject.toml
```toml
[tool.pytest.ini_options]
# CRITICAL: --no-cov is DEFAULT
addopts = '--no-cov -m "not debug" --timeout=300 --maxfail=5 --tb=short'

[tool.pytest.xdist]
scheduler = "loadscope"  # Group tests by scope for better memory isolation

[tool.coverage.run]
parallel = false  # CRITICAL: Disable parallel coverage
```

### tests/conftest.py
- Aggressive GC thresholds
- Post-test garbage collection
- Vector store cleanup tracking
- xdist worker GC optimization

## Troubleshooting

### Memory Still High

1. **Check specific groups**:
   ```bash
   python scripts/run_tests_split.py --list-groups
   python scripts/run_tests_split.py --group 7 -v
   ```

2. **Run without xdist**:
   ```bash
   pytest --no-cov tests/test_specific_file.py
   ```

3. **Check for large fixtures**:
   ```bash
   pytest --fixtures
   ```

### Coverage Issues

1. **Coverage not updating**:
   ```bash
   rm -rf .coverage htmlcov/
   ./scripts/run_with_coverage.sh
   ```

2. **Coverage with xdist** (not recommended):
   ```bash
   # Only if necessary, runs sequential coverage per worker
   pytest -n 4 --cov=src --cov-append
   ```

### Test Failures

1. **Stop on first failure**:
   ```bash
   pytest -x                    # Stop on first failure
   python scripts/run_tests_split.py --stop-on-failure
   ```

2. **Verbose output**:
   ```bash
   pytest -vv                   # Extra verbose
   python scripts/run_tests_split.py -v
   ```

## Best Practices

1. **Always use `--no-cov` for regular testing**
   - Only run coverage when needed (PR, pre-commit)
   - Coverage runs are slower and memory-intensive

2. **Use test groups for large test suites**
   - Divide large test files into smaller groups
   - Run groups sequentially to allow memory cleanup

3. **Leverage xdist with proper settings**
   - Use `loadscope` scheduler for better memory isolation
   - Each worker has its own memory space

4. **Monitor memory usage**
   - Check header output for available memory
   - Use `-v` flag to see which tests are slow

5. **Clean up resources**
   - Use `vector_store` fixture for automatic cleanup
   - Use `clean_mock` fixture for mock cleanup
   - Call `gc.collect()` in test teardown if needed

## Quick Reference

```bash
# Regular testing (fast, memory-safe)
pytest                              # All tests, no coverage
pytest -n 4                         # With 4 workers
pytest tests/test_main.py           # Specific file

# Split test execution
python scripts/run_tests_split.py               # All groups
python scripts/run_tests_split.py --group 0     # Specific group
python scripts/run_tests_split.py --workers 4   # With xdist

# Coverage (separate execution)
./scripts/run_with_coverage.sh                 # All tests with coverage
./scripts/run_with_coverage.sh unit            # Unit tests only

# List groups
python scripts/run_tests_split.py --list-groups
```

## Technical Details

### Memory Management Strategy

1. **Prevention**:
   - `--no-cov` by default prevents coverage data accumulation
   - Aggressive GC thresholds (100, 5, 5)
   - Test grouping prevents large-scale memory buildup

2. **Isolation**:
   - xdist workers have separate memory spaces
   - `loadscope` scheduler groups related tests
   - Each group runs in fresh process

3. **Cleanup**:
   - Automatic GC after each test
   - Weakref tracking for vector stores
   - BM25 cache cleanup between tests

4. **Monitoring**:
   - Memory info in pytest header
   - Warning on remaining references
   - Session cleanup report

### File Structure

```
scripts/
├── run_tests_split.py          # Split test runner (Python)
└── run_with_coverage.sh        # Coverage execution (Bash)

tests/
└── conftest.py                 # Enhanced memory management hooks

pyproject.toml                  # Test configuration with --no-cov default
```
