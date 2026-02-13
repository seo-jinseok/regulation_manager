# Acceptance Criteria: SPEC-EMBED-001

---
spec_id: SPEC-EMBED-001
title: Embedding Function Consistency Fix
created: 2026-02-13
status: Planned
---

## Overview

This document defines the acceptance criteria and test scenarios for verifying the embedding function consistency fix.

---

## Acceptance Criteria

### AC-001: Consistent Embedding Model Usage

**Given** the RAG system is initialized with default configuration
**When** `get_default_embedding_function()` is called
**Then** it returns an `EmbeddingFunctionWrapper` using the configured model `jhgan/ko-sbert-sts`
**And** the model produces 768-dimensional embeddings

### AC-002: No Silent Fallback on Config Import Error

**Given** the `src.rag.config` module cannot be imported
**When** `get_default_embedding_function()` is called
**Then** it raises an `ImportError` with a descriptive message
**And** the error message contains "RAG configuration unavailable"
**And** no fallback model is used

### AC-003: No Silent Fallback on Config Loading Error

**Given** the `get_config()` function raises an exception
**When** `get_default_embedding_function()` is called
**Then** it raises a `RuntimeError` with a descriptive message
**And** the error message contains actionable guidance
**And** no fallback model is used

### AC-004: EmbeddingFunctionWrapper Config Resolution

**Given** an `EmbeddingFunctionWrapper` is created without a model_name
**When** `__call__()` or `embed_query()` is invoked
**Then** it resolves the model name from configuration
**And** raises `RuntimeError` if configuration is unavailable

### AC-005: Existing Tests Pass

**Given** the existing test suite
**When** all tests are executed
**Then** all tests pass without modification
**And** no regressions are introduced

---

## Test Scenarios

### Scenario 1: Happy Path - Default Embedding Function

```gherkin
Feature: Default Embedding Function

  Scenario: Uses configured model by default
    Given the RAG configuration is available
    And the configured model is "jhgan/ko-sbert-sts"
    When I call get_default_embedding_function()
    Then it returns an EmbeddingFunctionWrapper
    And the model name is "jhgan/ko-sbert-sts"
    And embeddings have 768 dimensions

  Scenario: Embedding dimensions are consistent
    Given an EmbeddingFunctionWrapper with model "jhgan/ko-sbert-sts"
    When I generate embeddings for ["Hello", "World"]
    Then each embedding has exactly 768 dimensions
    And embeddings are normalized (unit vectors)
```

### Scenario 2: Error Handling - Config Import Failure

```gherkin
Feature: Error Handling

  Scenario: Raises ImportError when config module unavailable
    Given the src.rag.config module cannot be imported
    When I call get_default_embedding_function()
    Then it raises an ImportError
    And the error message contains "RAG configuration unavailable"
    And the error message contains "Ensure src.rag.config is importable"

  Scenario: Raises RuntimeError on config loading failure
    Given the get_config() function raises ValueError("Invalid config")
    When I call get_default_embedding_function()
    Then it raises a RuntimeError
    And the error message contains "Failed to load embedding model configuration"
    And the error message contains "Invalid config"
```

### Scenario 3: EmbeddingFunctionWrapper Behavior

```gherkin
Feature: EmbeddingFunctionWrapper

  Scenario: Uses provided model name
    Given an EmbeddingFunctionWrapper with model_name "custom/model"
    When I call __call__(["test"])
    Then it loads the SentenceTransformer with "custom/model"

  Scenario: Resolves model from config when model_name is None
    Given an EmbeddingFunctionWrapper with model_name None
    And the RAG configuration returns model "jhgan/ko-sbert-sts"
    When I call __call__(["test"])
    Then it loads the SentenceTransformer with "jhgan/ko-sbert-sts"

  Scenario: Raises error when model_name is None and config unavailable
    Given an EmbeddingFunctionWrapper with model_name None
    And the RAG configuration is unavailable
    When I call __call__(["test"])
    Then it raises a RuntimeError
    And the error message contains "No embedding model specified"
```

### Scenario 4: ChromaDB Integration

```gherkin
Feature: ChromaDB Integration

  Scenario: ChromaVectorStore uses consistent embedding
    Given a ChromaVectorStore with default embedding function
    And an existing collection created with ko-sbert-sts
    When I search for "test query"
    Then the search succeeds without dimension errors
    And results are returned with correct similarity scores

  Scenario: Sync creates correct embeddings
    Given a ChromaVectorStore with default embedding function
    When I add chunks to the store
    Then the embeddings are created with 768 dimensions
    And subsequent queries find the added chunks
```

### Scenario 5: Regression Prevention

```gherkin
Feature: Regression Prevention

  Scenario: No silent fallback to 384-dimension model
    Given the configuration is temporarily unavailable
    When I attempt to create an embedding function
    Then it raises an error
    And it does NOT use "paraphrase-multilingual-MiniLM-L12-v2"
    And it does NOT produce 384-dimensional embeddings

  Scenario: Error message is actionable
    Given any configuration error occurs
    When get_default_embedding_function() raises an error
    Then the error message suggests at least one remediation action
```

---

## Verification Methods

### Automated Tests

| Test File | Test Name | Criterion |
|-----------|-----------|-----------|
| test_embedding_function.py | test_get_default_uses_config | AC-001 |
| test_embedding_function.py | test_get_default_raises_import_error | AC-002 |
| test_embedding_function.py | test_get_default_raises_runtime_error | AC-003 |
| test_embedding_function.py | test_wrapper_uses_config_default | AC-004 |
| test_embedding_function.py | test_wrapper_raises_without_config | AC-004 |
| pytest (all) | All existing tests | AC-005 |

### Manual Verification

1. **Dimension Check**
   ```bash
   python -c "
   from src.rag.infrastructure.embedding_function import get_default_embedding_function
   ef = get_default_embedding_function()
   emb = ef(['test'])
   assert len(emb[0]) == 768, f'Expected 768 dims, got {len(emb[0])}'
   print('PASS: Embedding dimensions are 768')
   "
   ```

2. **Error Handling Check**
   ```bash
   # This should raise ImportError or RuntimeError
   python -c "
   import sys
   # Simulate config unavailable
   sys.modules['src.rag.config'] = None
   from src.rag.infrastructure.embedding_function import get_default_embedding_function
   get_default_embedding_function()
   " 2>&1 | grep -E "(ImportError|RuntimeError)"
   ```

3. **ChromaDB Integration Check**
   ```bash
   # Run existing integration tests
   pytest tests/integration/rag/test_chroma_store.py -v -k "embedding"
   ```

---

## Quality Gates

### Code Quality

- [ ] No ruff lint errors
- [ ] No mypy type errors
- [ ] Code formatted with black
- [ ] All docstrings updated

### Test Quality

- [ ] Test coverage >= 90% for modified code
- [ ] All edge cases covered
- [ ] Error messages verified in tests
- [ ] No flaky tests

### Integration Quality

- [ ] ChromaDB queries succeed
- [ ] Sync operations succeed
- [ ] No dimension mismatch errors
- [ ] Existing functionality preserved

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] All test scenarios passing
- [ ] Code review completed
- [ ] Documentation updated
- [ ] No regressions in existing tests
- [ ] Quality gates passed
- [ ] Manual verification successful

---

## Notes

### Why 768 Dimensions?

The `jhgan/ko-sbert-sts` model produces 768-dimensional embeddings. This is the standard dimension for BERT-base models. The existing ChromaDB collection was created with this model, so all subsequent operations must use a model with matching dimensions.

### Why No Fallback?

Silent fallback to a different model (like `paraphrase-multilingual-MiniLM-L12-v2` with 384 dimensions) creates hidden bugs that are difficult to diagnose. It is better to fail fast with a clear error message than to continue with a broken configuration.

### Alternative Approaches Considered

1. **Fallback to same-dimension model**: Rejected because it still creates unexpected behavior
2. **Auto-detect collection dimensions**: Rejected as too complex for this bug fix
3. **Configuration validation at startup**: Good idea, but out of scope for this SPEC
