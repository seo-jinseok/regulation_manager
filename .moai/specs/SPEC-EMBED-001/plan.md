# Implementation Plan: SPEC-EMBED-001

---
spec_id: SPEC-EMBED-001
title: Embedding Function Consistency Fix
created: 2026-02-13
status: Planned
---

## Overview

This plan outlines the implementation approach for fixing the embedding function dimension mismatch issue in the RAG system.

---

## Priority Milestones

### Primary Goal: Fix Critical Bug

1. **Error Handling Enhancement** (Priority: Critical)
   - Modify `get_default_embedding_function()` to raise informative errors
   - Remove silent fallback to incompatible model
   - Files: `src/rag/infrastructure/embedding_function.py`

2. **EmbeddingFunctionWrapper Update** (Priority: Critical)
   - Update `_get_model()` to use config properly
   - Add explicit error for missing configuration
   - Files: `src/rag/infrastructure/embedding_function.py`

### Secondary Goal: Test Coverage

3. **Unit Tests** (Priority: High)
   - Test error handling paths
   - Test embedding consistency
   - Files: `tests/unit/rag/infrastructure/test_embedding_function.py`

4. **Integration Tests** (Priority: Medium)
   - Test with actual ChromaDB collection
   - Verify dimension compatibility
   - Files: `tests/integration/rag/test_chroma_store.py`

### Final Goal: Documentation

5. **Code Comments and Docstrings** (Priority: Low)
   - Update docstrings to reflect new error behavior
   - Add inline comments explaining the rationale
   - Files: `src/rag/infrastructure/embedding_function.py`

---

## File Modification Plan

### File 1: `src/rag/infrastructure/embedding_function.py`

**Changes Required**:

| Line Range | Current | Proposed |
|------------|---------|----------|
| 100-113 | Silent fallback to `paraphrase-multilingual-MiniLM-L12-v2` | Raise `ImportError` or `RuntimeError` with actionable message |
| 148-159 | Hardcoded default `jhgan/ko-sbert-sts` | Use `get_config().get_embedding_model_name()` with error handling |

**New Functions** (Optional):
- `validate_embedding_dimensions()` - Proactive dimension mismatch detection

### File 2: `tests/unit/rag/infrastructure/test_embedding_function.py` (New/Existing)

**Test Cases Required**:

| Test Name | Purpose | Priority |
|-----------|---------|----------|
| `test_get_default_embedding_function_uses_config` | Verify config model is used | Critical |
| `test_get_default_embedding_function_raises_on_config_import_error` | Verify ImportError is raised | Critical |
| `test_get_default_embedding_function_raises_on_config_error` | Verify RuntimeError is raised | Critical |
| `test_embedding_function_wrapper_uses_config_default` | Verify wrapper uses config | High |
| `test_embedding_function_wrapper_raises_without_config` | Verify wrapper error | High |
| `test_embedding_dimensions_consistency` | Verify 768 dims for ko-sbert-sts | Medium |

---

## Technical Approach

### Error Handling Strategy

**Pattern**: Explicit error handling with actionable messages

```python
def get_default_embedding_function():
    """
    Get the default embedding function based on RAG configuration.

    Returns:
        ChromaDB-compatible EmbeddingFunctionWrapper instance.

    Raises:
        ImportError: If RAG configuration module cannot be imported.
        RuntimeError: If configuration loading fails for other reasons.
    """
    try:
        from ..config import get_config
        config = get_config()
        model_name = config.get_embedding_model_name()
        logger.info(f"Using configured embedding model: {model_name}")
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

    return EmbeddingFunctionWrapper(model_name)
```

### EmbeddingFunctionWrapper Enhancement

**Pattern**: Lazy configuration resolution with explicit errors

```python
def _get_model(self):
    """Lazy-load the SentenceTransformer model."""
    if self._model is None:
        try:
            from sentence_transformers import SentenceTransformer

            if self._model_name is None:
                # Try to get from config instead of hardcoding
                try:
                    from ..config import get_config
                    self._model_name = get_config().get_embedding_model_name()
                except Exception as e:
                    raise RuntimeError(
                        "No embedding model specified and configuration unavailable. "
                        "Either provide model_name parameter or ensure RAG config is accessible."
                    ) from e

            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded SentenceTransformer model: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return self._model
```

---

## Success Criteria

### Functional Criteria

- [ ] `get_default_embedding_function()` raises `ImportError` when config module unavailable
- [ ] `get_default_embedding_function()` raises `RuntimeError` with actionable message on config errors
- [ ] `EmbeddingFunctionWrapper._get_model()` resolves model from config when model_name is None
- [ ] No silent fallback to models with different dimensions
- [ ] All existing tests pass after modification

### Quality Criteria

- [ ] Test coverage for new error paths >= 90%
- [ ] Error messages contain actionable guidance
- [ ] No regression in existing functionality
- [ ] Code follows project style guidelines (ruff, black)

### Integration Criteria

- [ ] ChromaDB operations work correctly with existing collections
- [ ] Sync operations create embeddings with correct dimensions (768)
- [ ] Query operations return results without dimension errors

---

## Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Existing code depends on silent fallback | Low | High | Search codebase for try/except patterns around embedding functions |
| Config import fails in unexpected ways | Medium | Medium | Add comprehensive error types and messages |
| Tests rely on fallback behavior | Low | Medium | Review and update affected tests |

---

## Dependencies

### Upstream
- None (bug fix is self-contained)

### Downstream
- `src/rag/infrastructure/chroma_store.py` - Uses `get_default_embedding_function()`
- `src/rag/application/sync_usecase.py` - Uses embedding during sync
- `src/rag/application/search_usecase.py` - Uses embedding during search

---

## Rollback Plan

If issues arise:

1. Revert changes to `embedding_function.py`
2. Original silent fallback will be restored
3. Document the dimension mismatch issue for future resolution

---

## Verification Commands

```bash
# Run unit tests
pytest tests/unit/rag/infrastructure/test_embedding_function.py -v

# Run integration tests
pytest tests/integration/rag/test_chroma_store.py -v

# Verify embedding dimensions
python -c "
from src.rag.infrastructure.embedding_function import get_default_embedding_function
ef = get_default_embedding_function()
embedding = ef(['test'])
print(f'Embedding dimension: {len(embedding[0])}')  # Should be 768
"

# Verify error handling
python -c "
import sys
sys.modules['src.rag.config'] = None  # Simulate import failure
from src.rag.infrastructure.embedding_function import get_default_embedding_function
try:
    get_default_embedding_function()
except (ImportError, RuntimeError) as e:
    print(f'Error raised correctly: {type(e).__name__}')
"
```
