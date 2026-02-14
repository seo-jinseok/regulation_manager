# Implementation Plan: SPEC-TEST-COV-001

## Overview

This plan outlines the multi-iteration approach for improving test coverage from 6.94% to 85% across the Regulation Manager project. The implementation follows a phased strategy prioritizing critical business logic while maintaining existing test quality.

---

## Milestones

### Phase 1: Critical Modules (Priority High)

**Priority**: Critical (Primary Goal)

**Objective**: Achieve 90% coverage for the two most critical modules that form the core business logic.

**Target Modules**:

| Module                    | Lines  | Target Coverage | Est. Tests |
| ------------------------- | ------ | --------------- | ---------- |
| search_usecase.py         | 3,719  | 90%             | ~150 tests |
| query_analyzer.py         | 1,897  | 90%             | ~80 tests  |

**Tasks**:

1. **search_usecase.py Analysis**
   - Map all public methods and their dependencies
   - Identify mockable external dependencies (LLM, vector DB)
   - Document business logic branches and edge cases

2. **search_usecase.py Implementation**
   - Create test file structure with fixture setup
   - Implement unit tests for core search methods
   - Implement tests for error handling paths
   - Implement tests for edge cases (empty queries, no results)

3. **query_analyzer.py Analysis**
   - Analyze query parsing logic
   - Document intent classification branches
   - Identify Korean language processing edge cases

4. **query_analyzer.py Implementation**
   - Create tests for query normalization
   - Create tests for intent classification
   - Create tests for entity extraction
   - Create tests for ambiguity detection

**Deliverables**:
- `tests/unit/application/test_search_usecase.py` (150+ tests)
- `tests/unit/infrastructure/test_query_analyzer.py` (80+ tests)
- Coverage report showing 90%+ for both modules

**Success Criteria**:
- All tests pass
- Coverage >= 90% for both modules
- No regression in existing tests

---

### Phase 2: High Priority Modules

**Priority**: High (Secondary Goal)

**Objective**: Achieve 85% coverage for high-priority modules in Application and Infrastructure layers.

**Target Modules**:

| Module                          | Layer         | Lines  | Est. Tests |
| ------------------------------- | ------------- | ------ | ---------- |
| multi_hop_handler.py            | Application   | 778    | ~50 tests  |
| conversation_memory.py          | Application   | 729    | ~45 tests  |
| synonym_generator_service.py    | Application   | 267    | ~25 tests  |
| llm_client.py                   | Infrastructure| 246    | ~30 tests  |
| llm_judge.py                    | Domain        | 710    | ~50 tests  |

**Tasks**:

1. **multi_hop_handler.py**
   - Test multi-turn conversation flow
   - Test context accumulation
   - Test query refinement based on history

2. **conversation_memory.py**
   - Test session creation and retrieval
   - Test memory persistence
   - Test context window management

3. **synonym_generator_service.py**
   - Test synonym expansion logic
   - Test cache behavior
   - Test fallback mechanisms

4. **llm_client.py**
   - Test connection pooling
   - Test retry logic
   - Test circuit breaker states
   - Test error handling

5. **llm_judge.py**
   - Test evaluation criteria
   - Test scoring logic
   - Test result formatting

**Deliverables**:
- Test files for all 5 modules
- Coverage >= 85% for each module
- Integration with CI pipeline

**Success Criteria**:
- All tests pass
- Average coverage >= 85% for high-priority modules
- Mock strategies documented and reusable

---

### Phase 3: Medium Priority Modules

**Priority**: Medium (Tertiary Goal)

**Objective**: Achieve 85% coverage for medium-priority modules.

**Target Modules**:

| Module                      | Layer         | Lines  | Est. Tests |
| --------------------------- | ------------- | ------ | ---------- |
| experiment_service.py       | Application   | 739    | ~40 tests  |
| full_view_usecase.py        | Application   | 532    | ~35 tests  |
| query_expansion.py          | Application   | 412    | ~30 tests  |
| reranker.py                 | Infrastructure| 384    | ~25 tests  |
| vector_index_builder.py     | Infrastructure| 275    | ~20 tests  |
| quality_evaluator.py        | Domain        | 724    | ~45 tests  |
| custom_judge.py             | Domain        | 537    | ~35 tests  |

**Tasks**:

1. Analyze each module's structure
2. Identify test scenarios
3. Implement unit tests
4. Add integration tests where appropriate

**Deliverables**:
- Test files for all 7 modules
- Coverage >= 85% for each module

**Success Criteria**:
- All tests pass
- Overall coverage approaching 80%

---

### Phase 4: Low Priority and Remaining Modules

**Priority**: Low (Final Goal)

**Objective**: Complete coverage for remaining modules to achieve 85% overall.

**Target Modules**:

| Module          | Layer       | Lines  | Est. Tests |
| --------------- | ----------- | ------ | ---------- |
| auto_learn.py   | Application | 489    | ~30 tests  |
| evaluate.py     | Application | 315    | ~20 tests  |

**Additional Tasks**:
- Coverage gap analysis
- Edge case testing
- Documentation of test patterns

**Deliverables**:
- Test files for remaining modules
- Final coverage report showing >= 85%

**Success Criteria**:
- Overall coverage >= 85%
- All tests pass
- Test execution time < 5 minutes

---

## Technical Approach

### Testing Methodology

This SPEC uses a **Hybrid** testing approach:

- **For new test files**: TDD (Test-Driven Development) pattern
  - Write test first describing expected behavior
  - Verify test fails (confirms it tests something)
  - Implementation follows test requirements

- **For existing code**: DDD (Characterization Tests) pattern
  - Analyze existing behavior first
  - Write tests that capture current behavior
  - Use tests as safety net for future refactoring

### Mock Strategy

**Layer-Specific Mocking**:

| Layer         | Primary Mock Strategy                      |
| ------------- | ------------------------------------------ |
| Application   | Mock Infrastructure and Domain dependencies|
| Infrastructure| Mock external services (LLM, DB, Network) |
| Domain        | Mock only external state, test logic fully |

**Mock Implementation Patterns**:

```python
# Pattern 1: Fixture-based mocking
@pytest.fixture
def mock_llm():
    with patch("src.infrastructure.llm_client.LLMClient") as mock:
        mock.return_value.generate = AsyncMock(return_value="response")
        yield mock

# Pattern 2: Context manager for temporary mocking
def test_with_temporary_mock():
    with patch.object(TargetClass, "method", return_value="value"):
        # test code

# Pattern 3: Decorator for class-level mocking
@patch("module.dependency", autospec=True)
class TestClass:
    def test_method(self, mock_dependency):
        # test code
```

### Test Organization

```
tests/
  unit/
    application/       # Application layer tests
      test_search_usecase.py
      test_multi_hop_handler.py
      ...
    infrastructure/    # Infrastructure layer tests
      test_query_analyzer.py
      test_llm_client.py
      ...
    domain/            # Domain layer tests
      test_llm_judge.py
      test_quality_evaluator.py
      ...
  integration/         # Integration tests (minimal)
    test_search_flow.py
  conftest.py          # Shared fixtures
```

### Shared Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_chroma_collection():
    """Standard mock for ChromaDB collection."""
    mock = MagicMock()
    mock.query.return_value = {
        "ids": [["test-id"]],
        "documents": [["test document"]],
        "metadatas": [[{"source": "test"}]],
        "distances": [[0.5]]
    }
    return mock

@pytest.fixture
def mock_embedding_function():
    """Standard mock for embedding function."""
    return MagicMock(return_value=[[0.1] * 1024])

@pytest.fixture
def mock_llm_response():
    """Standard mock for LLM responses."""
    return AsyncMock(return_value="Generated response")
```

---

## File Modification Plan

### New Test Files to Create

| File Path                                             | Phase | Est. Lines |
| ----------------------------------------------------- | ----- | ---------- |
| tests/unit/application/test_search_usecase.py         | 1     | ~800       |
| tests/unit/infrastructure/test_query_analyzer.py      | 1     | ~500       |
| tests/unit/application/test_multi_hop_handler.py      | 2     | ~400       |
| tests/unit/application/test_conversation_memory.py    | 2     | ~350       |
| tests/unit/application/test_synonym_generator.py      | 2     | ~200       |
| tests/unit/infrastructure/test_llm_client.py          | 2     | ~250       |
| tests/unit/domain/test_llm_judge.py                   | 2     | ~400       |
| tests/unit/application/test_experiment_service.py     | 3     | ~350       |
| tests/unit/application/test_full_view_usecase.py      | 3     | ~300       |
| tests/unit/application/test_query_expansion.py        | 3     | ~250       |
| tests/unit/infrastructure/test_reranker.py            | 3     | ~200       |
| tests/unit/infrastructure/test_vector_index_builder.py| 3     | ~180       |
| tests/unit/domain/test_quality_evaluator.py           | 3     | ~350       |
| tests/unit/domain/test_custom_judge.py                | 3     | ~300       |
| tests/unit/application/test_auto_learn.py             | 4     | ~250       |
| tests/unit/application/test_evaluate.py               | 4     | ~180       |

### Configuration Updates

| File              | Change                                    |
| ----------------- | ----------------------------------------- |
| pytest.ini        | Update markers, add coverage thresholds   |
| pyproject.toml    | Update coverage configuration             |
| tests/conftest.py | Add shared fixtures for new modules       |

---

## Constraints and Guidelines

### Performance Constraints

- Test execution must complete within 5 minutes
- No actual network calls during test execution
- No actual file I/O to production directories
- Memory usage should not exceed 2GB during test run

### Code Quality Constraints

- All tests must follow AAA pattern (Arrange, Act, Assert)
- Test names must clearly describe the scenario
- No hardcoded test data that could become stale
- Use pytest fixtures for test data generation

### Coverage Constraints

- Minimum 85% line coverage for all modules
- Critical modules require 90% coverage
- No coverage decrease in existing tested modules
- Branch coverage should be considered for complex logic

---

## Risk Mitigation Strategy

### Complex Mocking Challenges

**Risk**: LLM responses and vector DB behavior difficult to mock realistically.

**Mitigation**:
- Create response fixture library based on actual responses
- Use snapshot testing for LLM output validation
- Implement VCR-like recording for integration tests

### Test Non-Determinism

**Risk**: Tests may fail intermittently due to timing or state issues.

**Mitigation**:
- Strict isolation between tests
- No shared mutable state
- Reset mocks before each test
- Use pytest-randomly to detect order dependencies

### Large Scope

**Risk**: 78% coverage gap is substantial, may exceed timeline.

**Mitigation**:
- Prioritize by business impact (critical modules first)
- Use parallel test development for independent modules
- Accept incremental progress with clear milestones

---

## Dependencies

### Internal Dependencies

- Existing test infrastructure in `tests/`
- Source modules in `src/`
- Configuration in `pytest.ini`, `pyproject.toml`

### External Dependencies

- pytest (v9.0+)
- pytest-cov (v7.0+)
- pytest-asyncio (for async tests)
- pytest-xdist (for parallel execution)

---

## Success Criteria Summary

- [ ] Overall test coverage >= 85%
- [ ] Critical module (search_usecase.py) coverage >= 90%
- [ ] Critical module (query_analyzer.py) coverage >= 90%
- [ ] All existing tests continue to pass
- [ ] All new tests follow naming conventions
- [ ] Test execution completes within 5 minutes
- [ ] No actual network calls during test execution
- [ ] All mock strategies documented
