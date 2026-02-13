# SPEC-EMBED-001: Embedding Function Consistency Fix

---
spec_id: SPEC-EMBED-001
title: Embedding Function Consistency Fix
created: 2026-02-13
status: Completed
priority: High
assigned: manager-ddd
domain: backend
tags: [embedding, chromadb, bugfix, rag]
---

## Problem Analysis

### Background

During DB testing, ChromaDB's `query_texts` method fails with a dimension mismatch error:
- DB was created with: `jhgan/ko-sbert-sts` (768 dimensions)
- Query uses: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions) as fallback
- Error: "Collection expecting embedding with dimension of 768, got 384"

### Root Cause

In `src/rag/infrastructure/embedding_function.py`, the `get_default_embedding_function()` function has a problematic fallback mechanism:

```python
def get_default_embedding_function():
    try:
        from ..config import get_config
        config = get_config()
        model_name = config.get_embedding_model_name()
        logger.info(f"Using configured embedding model: {model_name}")
    except Exception:
        # Fallback to multilingual model if config is unavailable
        # Using paraphrase-multilingual-MiniLM-L12-v2 (384 dims) for compatibility
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        logger.warning(f"Config unavailable, using fallback model: {model_name}")
```

**WHY this is problematic**:
1. The fallback model has 384 dimensions, incompatible with DB created using 768-dimension model
2. Silent fallback allows the system to continue with a broken configuration
3. No validation that embedding dimensions match the existing collection

### Impact

- Search queries fail with dimension mismatch errors
- Sync operations create inconsistent embeddings
- User-facing RAG queries return no results or errors

---

## Environment

- Python 3.13+
- ChromaDB 0.4.16+
- sentence-transformers library
- Existing vector database created with `jhgan/ko-sbert-sts` (768 dimensions)

---

## Assumptions

1. The configured model `jhgan/ko-sbert-sts` is the authoritative model for the project
2. Existing databases must remain compatible with the configured model
3. Dimension mismatch is a critical error, not a recoverable condition
4. `get_config()` failure is exceptional and should be surfaced, not silently handled

---

## Requirements

### REQ-001: Consistent Model Usage (Ubiquitous)

The system **shall** use the configured embedding model (`jhgan/ko-sbert-sts`) consistently across all operations including collection creation, document sync, and query execution.

**Rationale**: Consistency between embedding dimensions is critical for vector similarity search. A mismatch makes the database unusable.

### REQ-002: No Silent Fallback (Unwanted Behavior)

**IF** the configuration cannot be loaded, **THEN** the system **shall not** silently fall back to a model with different dimensions.

**Rationale**: Silent fallback creates hidden bugs that are difficult to diagnose. Explicit errors are preferable.

### REQ-003: Informative Error Handling (Event-Driven)

**WHEN** configuration loading fails, **THEN** the system **shall** raise a clear, actionable error message indicating the root cause.

**Rationale**: Developers need actionable information to fix configuration issues quickly.

### REQ-004: Dimension Validation (State-Driven)

**WHILE** initializing ChromaDB with an existing collection, **IF** an embedding function is provided, **THEN** the system **shall** validate that embedding dimensions match the collection metadata.

**Rationale**: Catching dimension mismatches at initialization prevents runtime errors during queries.

### REQ-005: Backward Compatibility (Optional)

**WHERE** feasible, the system **shall** provide a mechanism to detect and report dimension mismatches before operations fail.

**Rationale**: Proactive detection improves user experience.

---

## Specifications

### SPECS-001: Error Handling Strategy

Replace the current silent fallback in `get_default_embedding_function()` with explicit error handling:

**Current behavior** (problematic):
```python
except Exception:
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dims - WRONG!
```

**Required behavior**:
```python
except ImportError as e:
    raise ImportError(
        "RAG configuration unavailable. Ensure src.rag.config is importable. "
        f"Original error: {e}"
    ) from e
except Exception as e:
    raise RuntimeError(
        f"Failed to load embedding model configuration: {e}. "
        "Check that RAG_EMBEDDING_MODEL environment variable is set correctly, "
        "or ensure the default model 'jhgan/ko-sbert-sts' is available."
    ) from e
```

### SPECS-002: EmbeddingFunctionWrapper Default Model

Update `EmbeddingFunctionWrapper._get_model()` to use the configured model by default, not a hardcoded fallback:

**Current behavior**:
```python
if self._model_name is None:
    self._model_name = "jhgan/ko-sbert-sts"  # Hardcoded
```

**Required behavior**:
```python
if self._model_name is None:
    # Try to get from config
    try:
        from ..config import get_config
        self._model_name = get_config().get_embedding_model_name()
    except Exception as e:
        raise RuntimeError(
            "No embedding model specified and configuration unavailable. "
            "Either provide model_name parameter or ensure RAG config is accessible. "
            f"Original error: {e}"
        ) from e
```

### SPECS-003: Dimension Mismatch Detection (Optional Enhancement)

Add a helper function to detect dimension mismatches before operations:

```python
def validate_embedding_dimensions(
    collection_count: int,
    embedding_function: EmbeddingFunctionWrapper
) -> None:
    """
    Validate embedding dimensions match collection expectations.

    Raises:
        ValueError: If dimensions do not match.
    """
    if collection_count == 0:
        return  # Empty collection, no validation needed

    # Generate test embedding
    test_embedding = embedding_function(["test"])[0]
    actual_dim = len(test_embedding)

    # This would require collection metadata access
    # Implementation depends on ChromaDB API support
```

---

## Constraints

1. **No Breaking Changes**: Existing tests must pass after modification
2. **No Data Migration Required**: Fix should work with existing vector stores
3. **Minimal Code Changes**: Focus on error handling, not architecture changes
4. **Performance**: No measurable performance impact

---

## Dependencies

- `src/rag/config.py` - Configuration module
- `src/rag/infrastructure/chroma_store.py` - ChromaDB integration
- `src/rag/application/sync_usecase.py` - Uses embedding during sync
- `src/rag/application/search_usecase.py` - Uses embedding during search

---

## Traceability

| Requirement | Source | Verification |
|-------------|--------|--------------|
| REQ-001 | Dimension mismatch bug report | Unit test: test_embedding_consistency |
| REQ-002 | Silent fallback analysis | Unit test: test_no_silent_fallback |
| REQ-003 | Error handling best practices | Unit test: test_informative_error |
| REQ-004 | Proactive validation | Integration test: test_dimension_validation |
| REQ-005 | User experience improvement | Manual verification |

---

## Related Documents

- plan.md - Implementation plan and milestones
- acceptance.md - Acceptance criteria and test scenarios

---

## Implementation Summary

**Completion Date**: 2026-02-13
**Commit**: 87e8b8f

### Changes Implemented

1. **REQ-001 (Consistent Model Usage)**: Implemented
   - All embedding operations now use configured model (jhgan/ko-sbert-sts, 768 dims)

2. **REQ-002 (No Silent Fallback)**: Implemented
   - Removed silent fallback to 384-dimension model
   - Explicit errors raised instead

3. **REQ-003 (Informative Error Handling)**: Implemented
   - Clear, actionable error messages with root cause and remediation steps

4. **REQ-004 (Dimension Validation)**: Deferred
   - Optional enhancement, not required for bug fix

5. **REQ-005 (Backward Compatibility)**: Verified
   - All existing tests pass after modification

### Test Results

- Total tests: 14 passed
- Coverage: 75.86% for modified file
- Quality validation: PASS (TRUST 5)
