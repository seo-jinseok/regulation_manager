# AGENTS.md - AI Agent Context

> Context for AI coding agents (Gemini CLI, Cursor, GitHub Copilot, Claude, Codex, etc.)

## Quick Context

```
Project: University Regulation Manager (HWP → JSON → RAG Search)
Structure: src/rag/ with Clean Architecture (domain/ → application/ → infrastructure/ → interface/)
Testing: TDD required - write tests before implementation
Runtime: Python 3.11+, uv package manager (pip/conda forbidden)
Entry: `regulation` CLI (convert, sync, search, serve)

Key Constraints:
1. No external library imports in Domain layer
2. No features without tests
3. Never manually edit data/chroma_db/ or sync_state.json
```

## Build & Test Commands

```bash
# Environment setup
uv venv && uv sync
cp .env.example .env

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/rag/unit/application/test_search_usecase.py -v

# Run single test by name
uv run pytest -k "test_keyword_bonus_applied" -v

# Run tests matching pattern
uv run pytest -k "search" -v

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Lint and format with ruff
uv run ruff check src/ tests/
uv run ruff check --fix src/  # auto-fix
uv run ruff format src/       # auto-format
```

## Code Style Guidelines

### Naming Conventions
- **Functions/Variables**: `snake_case` (e.g., `search_regulations`, `top_k`)
- **Classes**: `CamelCase` (e.g., `SearchUseCase`, `ChromaVectorStore`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `RELEVANCE_THRESHOLD`)
- **Private**: `_leading_underscore` (e.g., `_compute_confidence`)

### Import Order (ruff isort)
```python
# 1. Standard library
import json
from dataclasses import dataclass

# 2. Third-party
import pytest
from chromadb import Client

# 3. First-party (src/)
from src.rag.domain.entities import Chunk, SearchResult
```

### Type Hints (Required)
```python
def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
    ...
```

### Docstrings (Google Style)
```python
def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
    """Search for relevant regulation chunks.

    Args:
        query: Search query text.
        top_k: Maximum number of results.

    Returns:
        List of SearchResult sorted by relevance.

    Raises:
        SearchError: If vector store unavailable.
    """
```

### Error Handling
Use domain exceptions from `src/exceptions.py`:
```python
from src.exceptions import SearchError, LLMError, VectorStoreError

raise SearchError("Vector store not initialized")
raise LLMError("ollama", "Connection refused")
```

### Path Handling
```python
from pathlib import Path
config_path = Path("data/config/synonyms.json")  # Always use pathlib
```

## Architecture Rules

```
[Interface] → [Application] → [Domain] ← [Infrastructure]
   CLI/Web      Use Cases     Entities    ChromaDB/LLM
```

| Layer | Location | Dependencies |
|-------|----------|--------------|
| Domain | `src/rag/domain/` | None (pure Python stdlib only) |
| Application | `src/rag/application/` | Domain only |
| Infrastructure | `src/rag/infrastructure/` | Implements Domain interfaces |
| Interface | `src/rag/interface/` | Calls Application |

### Forbidden Patterns
```python
# ❌ Domain importing external libraries
from chromadb import Client  # FORBIDDEN in domain/

# ❌ Application importing Infrastructure directly
from src.rag.infrastructure.chroma_store import ChromaVectorStore  # FORBIDDEN

# ✅ Use interfaces instead
from src.rag.domain.repositories import IVectorStore  # OK
```

## Testing Patterns

```python
# Use Fake classes for dependencies
class FakeStore:
    def __init__(self, results):
        self._results = results
    
    def search(self, query, filter=None, top_k: int = 10):
        return self._results

def test_keyword_bonus_applied():
    """Korean: 키워드 보너스가 점수에 적용되는지 테스트"""
    chunk = make_chunk("내용", keywords=[Keyword(term="교원", weight=1.0)])
    store = FakeStore([SearchResult(chunk=chunk, score=0.4, rank=1)])
    usecase = SearchUseCase(store, use_reranker=False)

    results = usecase.search("교원", top_k=1)

    assert results[0].score == pytest.approx(0.45)
```

- Test files: `test_<module>.py`
- Test functions: `test_<behavior>` or `test_<scenario>_<expected>`
- Korean comments allowed for context

## CLI Reference

```bash
uv run regulation                          # Interactive mode
uv run regulation search "query" -n 5      # Search with limit
uv run regulation search "query" -a        # Search + LLM answer
uv run regulation sync data/output/규정집.json
uv run regulation status
uv run regulation reset --confirm
uv run regulation serve --web              # Gradio UI
uv run regulation serve --mcp              # MCP Server
```

## Related Files

- `.github/copilot-instructions.md` - GitHub Copilot specific
- `QUICKSTART.md` - User guide
- `LLM_GUIDE.md` - LLM configuration
- `SCHEMA_REFERENCE.md` - JSON schema spec

## Security & Input Validation

### Query Validation
All user queries are validated in `QueryHandler.validate_query()`:
- **Max length**: 500 characters
- **Forbidden patterns**: XSS, SQL injection, template injection
- **Control characters**: Blocked (except newlines)

```python
# Validation patterns (src/rag/interface/query_handler.py)
FORBIDDEN_PATTERNS = [
    r"<script",           # XSS
    r"javascript:",       # JavaScript URL
    r"on\w+\s*=",        # Event handlers
    r"<iframe",          # Iframe injection
    r"DROP\s+TABLE",     # SQL injection
    r"\$\{.*\}",         # Template injection
    r"\{\{.*\}\}",       # Jinja2 template
]
```

### Prompt Management
LLM prompts are externalized to `data/config/prompts.json`:
```python
# Load prompt (src/rag/application/search_usecase.py)
from src.rag.application.search_usecase import _load_prompt
prompt = _load_prompt("regulation_qa")
```

### Logging Best Practices
- Use `logging.getLogger(__name__)` instead of `print()`
- Log levels: `debug` for dev, `info` for operations, `warning/error` for issues
- Never log sensitive data (API keys, user PII)

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Processing query: %s", query[:50])  # Truncate for safety
logger.info("Search completed in %.2fs", elapsed)
logger.error("LLM connection failed: %s", error)
```
