# SPEC-TEST-COV-001: Test Coverage Improvement to 85%

## Metadata

| Field       | Value                         |
| ----------- | ----------------------------- |
| SPEC ID     | SPEC-TEST-COV-001             |
| Title       | Test Coverage Improvement to 85% |
| Created     | 2026-02-14                    |
| Status      | Planned                       |
| Priority    | High                          |
| Assigned    | expert-testing                |
| Lifecycle   | spec-anchored                 |

## TAG BLOCK

```yaml
tags:
  - TEST-COV-001
  - test-coverage
  - pytest
  - unit-testing
  - quality-improvement
dependencies: []
related_spec: []
```

---

## Environment

### System Context

Regulation Manager is an AI-powered search system for university regulations using RAG (Retrieval-Augmented Generation). The project follows Clean Architecture with three main layers: Application, Infrastructure, and Domain. Test coverage is critical for maintaining system reliability and enabling safe refactoring.

### Current State Analysis

| Metric                   | Current Value  | Target Value    | Gap              |
| ------------------------ | -------------- | --------------- | ---------------- |
| Overall Coverage         | 6.94%          | 85%             | ~78%             |
| Application Layer        | 0%             | 85%             | 85%              |
| Infrastructure Layer     | 0%             | 85%             | 85%              |
| Domain Layer             | 0%             | 85%             | 85%              |
| Test Count               | ~200 tests     | ~1000+ tests    | ~800 tests       |

### Modules with 100% Coverage (Already Complete)

| Module                            | Tests  | Status     |
| --------------------------------- | ------ | ---------- |
| period_keyword_detector.py        | 18     | Complete   |
| completeness_validator.py         | 33     | Complete   |
| academic_calendar_service.py      | 27     | Complete   |
| hallucination_filter.py           | 39     | Complete   |
| conversation_service.py           | 40     | Complete   |

### Priority Modules for Coverage

**Application Layer (0% Coverage)**

| File                        | Lines  | Priority  | Reason                              |
| --------------------------- | ------ | --------- | ----------------------------------- |
| search_usecase.py           | 3,719  | Critical  | Core business logic                 |
| multi_hop_handler.py        | 778    | High      | Multi-turn conversation support     |
| conversation_memory.py      | 729    | High      | Session state management            |
| experiment_service.py       | 739    | Medium    | A/B testing infrastructure          |
| full_view_usecase.py        | 532    | Medium    | Full regulation view                |
| query_expansion.py          | 412    | Medium    | Query enhancement                   |
| synonym_generator_service.py| 267    | High      | Synonym expansion                   |
| auto_learn.py               | 489    | Low       | Automatic learning                  |
| evaluate.py                 | 315    | Low       | Evaluation utilities                |

**Infrastructure Layer (0% Coverage)**

| File                   | Lines  | Priority  | Reason                              |
| ---------------------- | ------ | --------- | ----------------------------------- |
| query_analyzer.py      | 1,897  | Critical  | Query understanding                 |
| llm_client.py          | 246    | High      | External dependency (LLM)           |
| reranker.py            | 384    | Medium    | Result reranking                    |
| vector_index_builder.py| 275    | Medium    | Vector indexing                     |

**Domain Layer (0% Coverage)**

| File                   | Lines  | Priority  | Reason                              |
| ---------------------- | ------ | --------- | ----------------------------------- |
| llm_judge.py           | 710    | High      | LLM-based evaluation                |
| quality_evaluator.py   | 724    | Medium    | Quality metrics                     |
| custom_judge.py        | 537    | Medium    | Custom evaluation logic             |

---

## Assumptions

### Technical Assumptions

1. Existing test infrastructure (pytest, pytest-cov) remains the foundation
2. Mock strategies can effectively isolate external dependencies (LLM, vector DB)
3. Test coverage metrics accurately reflect code quality and risk areas
4. Characterization tests can capture existing behavior for legacy code

### Business Assumptions

1. Higher test coverage enables safer refactoring and feature additions
2. Test-first approach for new code improves overall quality
3. Multi-iteration effort is acceptable given scope (0% to 85%)

### Constraint Assumptions

1. Each module can be tested independently with proper mocking
2. External services (LLM, ChromaDB) can be mocked reliably
3. Test execution time should remain under 5 minutes for full suite

---

## Requirements

### REQ-001: Application Layer Coverage (Ubiquitous)

The system **shall** achieve minimum 85% test coverage for all Application layer modules.

**EARS Pattern**: Ubiquitous (Always Active)

**Scope**:
- `src/application/search_usecase.py` - Core search business logic
- `src/application/multi_hop_handler.py` - Multi-turn conversation
- `src/application/conversation_memory.py` - Session state management
- `src/application/synonym_generator_service.py` - Synonym expansion
- `src/application/query_expansion.py` - Query enhancement
- `src/application/full_view_usecase.py` - Full regulation view
- `src/application/experiment_service.py` - A/B testing
- `src/application/auto_learn.py` - Automatic learning
- `src/application/evaluate.py` - Evaluation utilities

**Acceptance Criteria**:
- WHEN running `pytest --cov=src/application` THEN coverage >= 85%
- WHEN running pytest THEN all Application layer tests pass

### REQ-002: Infrastructure Layer Coverage (Ubiquitous)

The system **shall** achieve minimum 85% test coverage for all Infrastructure layer modules.

**EARS Pattern**: Ubiquitous (Always Active)

**Scope**:
- `src/infrastructure/query_analyzer.py` - Query understanding (Critical)
- `src/infrastructure/llm_client.py` - LLM integration (High)
- `src/infrastructure/reranker.py` - Result reranking
- `src/infrastructure/vector_index_builder.py` - Vector indexing

**Mock Requirements**:
- LLM responses must be mocked for deterministic testing
- Vector database operations must be mocked
- Network calls must not occur during test execution

**Acceptance Criteria**:
- WHEN running `pytest --cov=src/infrastructure` THEN coverage >= 85%
- WHEN running tests THEN no actual LLM or network calls occur

### REQ-003: Domain Layer Coverage (Ubiquitous)

The system **shall** achieve minimum 85% test coverage for all Domain layer modules.

**EARS Pattern**: Ubiquitous (Always Active)

**Scope**:
- `src/domain/llm_judge.py` - LLM-based evaluation
- `src/domain/quality_evaluator.py` - Quality metrics
- `src/domain/custom_judge.py` - Custom evaluation logic

**Acceptance Criteria**:
- WHEN running `pytest --cov=src/domain` THEN coverage >= 85%
- WHEN running Domain layer tests THEN all tests pass

### REQ-004: Overall Coverage Target (Event-Driven)

**WHEN** running the complete test suite **THEN** the system **shall** report overall coverage of at least 85%.

**EARS Pattern**: Event-Driven (Trigger-Response)

**Measurement Command**:
```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=85
```

**Acceptance Criteria**:
- WHEN running full test suite THEN overall coverage >= 85%
- WHEN running full test suite THEN all tests pass
- WHEN running full test suite THEN execution completes within 5 minutes

### REQ-005: Critical Module Priority (State-Driven)

**IF** a module is classified as Critical priority **THEN** the system **shall** achieve 90% coverage for that module.

**EARS Pattern**: State-Driven (Conditional)

**Critical Modules**:
- `search_usecase.py` (3,719 lines)
- `query_analyzer.py` (1,897 lines)

**Acceptance Criteria**:
- WHEN running coverage for search_usecase.py THEN coverage >= 90%
- WHEN running coverage for query_analyzer.py THEN coverage >= 90%

### REQ-006: Test Quality Standards (Ubiquitous)

The system **shall** maintain test quality according to TRUST 5 principles.

**EARS Pattern**: Ubiquitous (Always Active)

**Quality Standards**:
- **Tested**: All new tests follow AAA pattern (Arrange, Act, Assert)
- **Readable**: Test names clearly describe the scenario being tested
- **Unified**: Consistent test structure across all test files
- **Secured**: No hardcoded credentials or sensitive data in tests
- **Trackable**: Each test references the SPEC requirement it validates

**Acceptance Criteria**:
- WHEN reviewing tests THEN all follow AAA pattern
- WHEN reviewing tests THEN no sensitive data is present
- WHEN reviewing tests THEN each test has clear purpose in name

### REQ-007: Mock Strategy for External Dependencies (State-Driven)

**IF** a module depends on external services **THEN** the tests **shall** use appropriate mocking strategies.

**EARS Pattern**: State-Driven (Conditional)

**Mock Categories**:

| Dependency Type    | Mock Strategy                          |
| ------------------ | -------------------------------------- |
| LLM Calls          | `unittest.mock.AsyncMock` with fixture |
| Vector DB (Chroma) | In-memory ChromaDB or mock collection  |
| Network Requests   | `pytest-httpx` or `responses` library  |
| File I/O           | `tmp_path` fixture or `pyfakefs`       |

**Acceptance Criteria**:
- WHEN running tests THEN no actual network calls to LLM providers
- WHEN running tests THEN no actual ChromaDB persistence to disk
- WHEN running tests THEN file operations use temporary directories

### REQ-008: No Test Regression (Unwanted Behavior)

The system **shall not** have failing tests or reduced coverage in already-covered modules.

**EARS Pattern**: Unwanted Behavior (Prohibition)

**Protected Areas**:
- `period_keyword_detector.py` (18 tests, 100% coverage)
- `completeness_validator.py` (33 tests, 100% coverage)
- `academic_calendar_service.py` (27 tests, 100% coverage)
- `hallucination_filter.py` (39 tests, 100% coverage)
- `conversation_service.py` (40 tests, 100% coverage)

**Acceptance Criteria**:
- WHEN running existing test suite THEN all tests continue to pass
- WHEN measuring coverage THEN existing modules maintain 100% coverage

---

## Specifications

### Test File Naming Convention

```
tests/
  unit/
    application/
      test_search_usecase.py
      test_multi_hop_handler.py
      test_conversation_memory.py
      ...
    infrastructure/
      test_query_analyzer.py
      test_llm_client.py
      ...
    domain/
      test_llm_judge.py
      test_quality_evaluator.py
      ...
```

### Test Structure Template

```python
"""
Tests for {module_name}.

Reference: SPEC-TEST-COV-001 REQ-{XXX}
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.{layer}.{module} import {ClassName}


class Test{ClassName}:
    """Test suite for {ClassName}."""

    @pytest.fixture
    def {dependency}_mock(self):
        """Create mock for {dependency}."""
        return MagicMock()

    @pytest.fixture
    def sut(self, {dependency}_mock):
        """Create system under test."""
        return {ClassName}({dependency}={dependency}_mock)

    class Test{MethodName}:
        """Tests for {method_name} method."""

        def test_{scenario}_returns_expected_result(self, sut):
            """Test that {scenario} returns expected result."""
            # Arrange
            input_data = {...}
            expected = {...}

            # Act
            result = sut.{method_name}(input_data)

            # Assert
            assert result == expected

        def test_{edge_case}_handles_gracefully(self, sut):
            """Test that {edge_case} is handled gracefully."""
            # Arrange
            input_data = {...}

            # Act & Assert
            with pytest.raises({ExpectedException}):
                sut.{method_name}(input_data)
```

### Mock Fixture Patterns

```python
# LLM Mock Pattern
@pytest.fixture
def llm_client_mock():
    """Mock LLM client for deterministic testing."""
    mock = AsyncMock()
    mock.generate.return_value = "Mocked LLM response"
    return mock


# ChromaDB Mock Pattern
@pytest.fixture
def chroma_collection_mock():
    """Mock ChromaDB collection."""
    mock = MagicMock()
    mock.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{}, {}]],
        "distances": [[0.1, 0.2]]
    }
    return mock


# Embedding Mock Pattern
@pytest.fixture
def embedding_mock():
    """Mock embedding function."""
    return MagicMock(return_value=[[0.1] * 1024])
```

---

## Success Metrics

| Metric                      | Current     | Target      | Measurement Method              |
| --------------------------- | ----------- | ----------- | ------------------------------- |
| Overall Coverage            | 6.94%       | 85%         | `pytest --cov=src --cov-report` |
| Application Layer Coverage  | 0%          | 85%         | `pytest --cov=src/application`  |
| Infrastructure Coverage     | 0%          | 85%         | `pytest --cov=src/infrastructure`|
| Domain Layer Coverage       | 0%          | 85%         | `pytest --cov=src/domain`       |
| Critical Module Coverage    | 0%          | 90%         | Per-module coverage report      |
| Test Count                  | ~200        | ~1000+      | `pytest --collect-only`         |
| Test Execution Time         | ~30s        | <5 min      | `pytest --durations=0`          |

---

## Risks and Mitigations

| Risk                                | Probability | Impact | Mitigation                              |
| ----------------------------------- | ----------- | ------ | --------------------------------------- |
| Complex mocking for LLM             | High        | Medium | Use AsyncMock with realistic fixtures   |
| Large test scope                    | High        | Medium | Phased implementation, prioritize       |
| Test execution time                 | Medium      | Medium | Use pytest-xdist, parallel execution    |
| Non-deterministic tests             | Medium      | High   | Strict mock isolation, no real I/O      |
| Coverage gaps in edge cases         | Medium      | Low    | Add mutation testing in future phases   |
| Regression in existing tests        | Low         | High   | Run full suite before each commit       |

---

## References

- Test Configuration: `pytest.ini`, `pyproject.toml`
- Existing Test Patterns: `tests/unit/`
- TRUST 5 Framework: MoAI-ADK Foundation Core
- EARS Specification: MoAI-ADK Foundation Core
