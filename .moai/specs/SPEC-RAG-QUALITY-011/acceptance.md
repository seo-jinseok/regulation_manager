# Acceptance Criteria: RAG System Quality Improvement

**SPEC ID:** SPEC-RAG-QUALITY-011
**Created:** 2026-02-24
**Status:** Planned

---

## Quality Gate Overview

| Gate | Threshold | Verification Method |
|------|-----------|---------------------|
| Query Pass Rate | >= 80% | Evaluation script |
| Average Score | >= 0.700 | Evaluation script |
| Classification Accuracy | >= 95% | Unit tests |
| Test Coverage | >= 85% | pytest-cov |
| Lint Errors | 0 | ruff |
| Type Errors | 0 | mypy/pyright |

---

## Scenario 1: Self-RAG Prompt Improvement

### AC-001.1: Regulation Queries Return True

**Given** a Self-RAG evaluator with improved prompt
**When** user asks "휴학 신청 방법이 뭐야?"
**Then** `needs_retrieval()` returns `True`
**And** retrieval is performed

```gherkin
Scenario: Regulation query triggers retrieval
  Given a SelfRAGEvaluator instance with improved prompt
  And an LLM client is configured
  When the query "휴학 신청 방법이 뭐야?" is evaluated
  Then needs_retrieval() should return True
```

### AC-001.2: Greeting Queries Return False

**Given** a Self-RAG evaluator with improved prompt
**When** user says "안녕하세요"
**Then** `needs_retrieval()` returns `False`
**And** no retrieval is performed

```gherkin
Scenario: Simple greeting does not trigger retrieval
  Given a SelfRAGEvaluator instance with improved prompt
  And an LLM client is configured
  When the query "안녕하세요" is evaluated
  Then needs_retrieval() should return False
```

### AC-001.3: Ambiguous Queries Default to True

**Given** a Self-RAG evaluator with improved prompt
**When** user asks an ambiguous question like "학교 관련 질문이에요"
**Then** `needs_retrieval()` returns `True`
**And** retrieval is performed (conservative approach)

```gherkin
Scenario: Ambiguous query defaults to retrieval
  Given a SelfRAGEvaluator instance with improved prompt
  And an LLM client is configured
  When the query "학교 관련 질문이에요" is evaluated
  Then needs_retrieval() should return True
```

### AC-001.4: Classification Accuracy >= 95%

**Given** a test set of 100 regulation queries
**When** all queries are evaluated by Self-RAG
**Then** at least 95 queries return `True`
**And** classification accuracy is >= 95%

```gherkin
Scenario: High classification accuracy for regulation queries
  Given a SelfRAGEvaluator instance with improved prompt
  And a test set of 100 regulation queries
  When all queries are evaluated
  Then at least 95 queries should return True
  And classification accuracy should be >= 0.95
```

---

## Scenario 2: Keyword Pre-Filtering

### AC-002.1: Keywords Detected in Query

**Given** a query containing "규정"
**When** `_has_regulation_keywords()` is called
**Then** returns `True`
**And** completes within 1ms

```gherkin
Scenario: Keyword detection succeeds
  Given a SelfRAGEvaluator instance
  And the keyword list includes "규정"
  When _has_regulation_keywords("이 규정에 대해 알려주세요") is called
  Then it should return True
  And execution time should be less than 1ms
```

### AC-002.2: No Keywords in Query

**Given** a query without regulation keywords
**When** `_has_regulation_keywords()` is called
**Then** returns `False`

```gherkin
Scenario: No keyword detection
  Given a SelfRAGEvaluator instance
  When _has_regulation_keywords("오늘 날씨 어때?") is called
  Then it should return False
```

### AC-002.3: Bypass LLM on Keyword Match

**Given** a query with regulation keywords
**When** `needs_retrieval()` is called
**Then** LLM is NOT called
**And** returns `True` immediately

```gherkin
Scenario: LLM bypassed when keywords detected
  Given a SelfRAGEvaluator instance with mocked LLM client
  And the query contains "학칙"
  When needs_retrieval() is called
  Then the LLM client should not be called
  And it should return True
```

### AC-002.4: Keyword List Configurable

**Given** a custom keyword file at `data/config/custom_keywords.json`
**When** Self-RAG evaluator is initialized
**Then** keywords are loaded from the file
**And** custom keywords are used for matching

```gherkin
Scenario: Custom keywords loaded from file
  Given a keyword file at data/config/custom_keywords.json
  And the file contains ["커스텀키워드"]
  When SelfRAGEvaluator is initialized with keywords_path
  Then _has_regulation_keywords("커스텀키워드 관련 질문") should return True
```

---

## Scenario 3: Fallback Mechanism

### AC-003.1: Override Activates on Keyword Mismatch

**Given** LLM returns `[RETRIEVE_NO]`
**And** query contains regulation keywords
**When** `needs_retrieval()` completes
**Then** override activates
**And** returns `True`

```gherkin
Scenario: Override when LLM says no but keywords exist
  Given a SelfRAGEvaluator instance
  And LLM client is mocked to return "[RETRIEVE_NO]"
  And query contains "장학금"
  When needs_retrieval() is called
  Then it should return True (override activated)
```

### AC-003.2: Override is Logged

**Given** override condition is met
**When** override activates
**Then** override event is logged
**And** log includes query hash and reason

```gherkin
Scenario: Override logging
  Given a SelfRAGEvaluator instance with logging enabled
  And override condition is met
  When override activates
  Then a log entry should be created
  And log should contain "override" and reason
```

### AC-003.3: Override Rate Tracked

**Given** multiple queries with override conditions
**When** `get_metrics()` is called
**Then** override count is returned
**And** override rate is calculated

```gherkin
Scenario: Override metrics tracked
  Given a SelfRAGEvaluator instance
  And 10 queries processed with 2 overrides
  When get_metrics() is called
  Then override_count should be 2
  And override_rate should be 0.2
```

---

## Scenario 4: Health Verification

### AC-004.1: Empty Collection Warning

**Given** ChromaDB collection is empty
**When** `health_check()` is called
**Then** status is "unhealthy"
**And** issues list contains "Empty collection"

```gherkin
Scenario: Empty collection detected
  Given a ChromaVectorStore with empty collection
  When health_check() is called
  Then status should be "unhealthy"
  And issues should contain "Empty collection"
```

### AC-004.2: Missing Embedding Function Warning

**Given** embedding function is None
**When** `health_check()` is called
**Then** status is "unhealthy"
**And** issues list contains "Embedding function unavailable"

```gherkin
Scenario: Missing embedding function detected
  Given a ChromaVectorStore with no embedding function
  When health_check() is called
  Then status should be "unhealthy"
  And issues should contain "Embedding function unavailable"
```

### AC-004.3: Healthy Status

**Given** ChromaDB has documents
**And** embedding function is available
**When** `health_check()` is called
**Then** status is "healthy"
**And** issues list is empty

```gherkin
Scenario: Healthy status returned
  Given a ChromaVectorStore with 100 documents
  And embedding function is available
  When health_check() is called
  Then status should be "healthy"
  And issues should be empty
```

---

## Scenario 5: Data Ingestion Verification

### AC-005.1: Empty Collection Error Message

**Given** ChromaDB collection is empty
**When** user submits a query
**Then** informative error message is returned
**And** message suggests running data ingestion

```gherkin
Scenario: Empty collection returns helpful error
  Given a ChromaVectorStore with empty collection
  When user queries "휴학 규정이 뭐야?"
  Then error message should include "데이터가 없습니다"
  And error message should suggest "데이터 동기화"
```

### AC-005.2: Distinguish from No Results

**Given** ChromaDB has documents
**And** search returns no relevant results
**When** user submits a query
**Then** error message indicates "no relevant results"
**And** does NOT suggest data ingestion

```gherkin
Scenario: No results vs empty collection distinction
  Given a ChromaVectorStore with documents
  And search returns no results
  When user queries "존재하지않는키워드12345"
  Then error message should NOT suggest data ingestion
  And should indicate "관련 규정을 찾을 수 없습니다"
```

---

## Scenario 6: Metrics & Logging

### AC-006.1: Classification Counters

**Given** Self-RAG processes queries
**When** `get_metrics()` is called
**Then** counters are accurate
**And** include retrieval_yes, retrieval_no, bypass, override

```gherkin
Scenario: Classification counters tracked
  Given a SelfRAGEvaluator instance
  And 10 queries processed: 5 retrieval_yes, 3 retrieval_no, 2 bypass
  When get_metrics() is called
  Then retrieval_yes_count should be 5
  And retrieval_no_count should be 3
  And bypass_count should be 2
```

### AC-006.2: Structured Logging

**Given** logging is enabled
**When** classification decision is made
**Then** log entry includes query hash
**And** includes classification result
**And** includes latency

```gherkin
Scenario: Structured logging for classification
  Given a SelfRAGEvaluator instance with logging
  When classification decision is made
  Then log should contain query_hash
  And log should contain classification result
  And log should contain latency_ms
```

---

## Scenario 7: Configuration

### AC-008.1: Environment Variable Override

**Given** `ENABLE_SELF_RAG=false`
**When** RAGConfig is initialized
**Then** `enable_self_rag` is False
**And** Self-RAG is disabled

```gherkin
Scenario: Self-RAG disabled via environment
  Given environment variable ENABLE_SELF_RAG is set to "false"
  When RAGConfig is initialized
  Then enable_self_rag should be False
```

### AC-008.2: Keywords Path Configuration

**Given** `SELF_RAG_KEYWORDS_PATH=/custom/path/keywords.json`
**When** Self-RAG evaluator is initialized
**Then** keywords are loaded from custom path

```gherkin
Scenario: Custom keywords path
  Given environment variable SELF_RAG_KEYWORDS_PATH is set
  When SelfRAGEvaluator is initialized
  Then keywords should be loaded from custom path
```

---

## Scenario 8: LLM Response Parsing

### AC-007.1: JSON Parsing Error Handled

**Given** LLM returns malformed JSON
**When** evaluation processes response
**Then** error is caught
**And** fallback to rule-based evaluation
**And** evaluation continues

```gherkin
Scenario: Malformed JSON handled gracefully
  Given an LLM judge that returns malformed JSON
  When evaluation processes the response
  Then no exception should be raised
  And fallback evaluation should be used
```

### AC-007.2: Error Rate Tracked

**Given** multiple JSON parsing errors occur
**When** metrics are collected
**Then** JSON error count is available
**And** error rate is calculated

```gherkin
Scenario: JSON error metrics tracked
  Given 10 evaluations with 2 JSON parsing errors
  When metrics are collected
  Then json_error_count should be 2
  And json_error_rate should be 0.2
```

---

## End-to-End Test Scenarios

### E2E-001: Complete Query Flow

```gherkin
Feature: RAG Query Processing

  Scenario: Student asks about leave of absence
    Given the RAG system is initialized
    And ChromaDB contains regulation documents
    And Self-RAG is enabled
    When a student asks "휴학 신청은 어떻게 하나요?"
    Then retrieval should be performed
    And relevant regulation chunks should be retrieved
    And a helpful answer should be generated
    And sources should be cited
```

### E2E-002: Quality Evaluation Pass Rate

```gherkin
Feature: Quality Evaluation

  Scenario: Pass rate threshold met
    Given the evaluation script is configured
    And 30 test queries are prepared
    When evaluation is executed
    Then pass rate should be >= 0.80
    And average score should be >= 0.700
    And no query should receive rejection message
```

---

## Regression Test Checklist

- [ ] Existing unit tests pass
- [ ] Existing integration tests pass
- [ ] Query processing latency not increased
- [ ] Memory usage not increased
- [ ] API backward compatibility maintained
- [ ] Configuration backward compatibility maintained

---

## Definition of Done

### Code Quality
- [ ] All code follows project style guide
- [ ] All code has type hints
- [ ] All code has docstrings
- [ ] No lint errors (ruff)
- [ ] No type errors (mypy/pyright)

### Test Coverage
- [ ] Unit test coverage >= 85%
- [ ] All acceptance criteria have passing tests
- [ ] Edge cases covered
- [ ] Error scenarios covered

### Documentation
- [ ] Updated docstrings
- [ ] Updated configuration documentation
- [ ] Updated troubleshooting guide

### Integration
- [ ] Changes integrated with existing code
- [ ] No breaking changes
- [ ] Backward compatible

### Verification
- [ ] Quality evaluation shows improvement
- [ ] Pass rate >= 80%
- [ ] Average score >= 0.700
- [ ] Manual testing confirms fixes

---

**Document Version:** 1.0
**Last Updated:** 2026-02-24
**Author:** MoAI System Architect Agent
