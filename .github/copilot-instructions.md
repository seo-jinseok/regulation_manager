# GitHub Copilot Instructions

This file provides project-specific context to GitHub Copilot.

## Project Context

**See [AGENTS.md](../AGENTS.md) for comprehensive project context including:**
- Project structure and Clean Architecture
- Query processing pipeline (Tool Calling, Corrective RAG, Self-RAG)
- TDD and coding rules
- Command references

## Quick Reference

### System Overview

**대학 규정 관리 시스템 (Regulation Manager)**
- HWP 규정집 → 구조화된 JSON → Hybrid RAG 기반 AI 검색/Q&A
- Clean Architecture: `domain/` → `application/` → `infrastructure/` → `interface/`

### Key Components

| Component | Location | Role |
|-----------|----------|------|
| `QueryHandler` | `interface/query_handler.py` | Query routing & result aggregation |
| `SearchUseCase` | `application/search_usecase.py` | Search logic + Corrective RAG + Self-RAG + HyDE |
| `FunctionGemmaAdapter` | `infrastructure/function_gemma_adapter.py` | LLM Tool Calling |
| `QueryAnalyzer` | `infrastructure/query_analyzer.py` | Intent/audience detection |
| `RetrievalEvaluator` | `infrastructure/retrieval_evaluator.py` | Corrective RAG trigger (dynamic thresholds) |
| `SelfRAGEvaluator` | `infrastructure/self_rag.py` | Retrieval necessity & relevance evaluation |
| `HyDEGenerator` | `infrastructure/hyde.py` | Hypothetical document generation |
| `HybridSearcher` | `infrastructure/hybrid_search.py` | BM25 (KoNLPy) + Dense search |

### Development Principles

- **TDD**: Write tests first (RED → GREEN → REFACTOR)
- **Clean Architecture**: Dependency flows inward (Domain has no external deps)
- **Package Manager**: Use `uv` (not pip or conda)

### Key Commands

```bash
uv run pytest                           # Run tests
uv run regulation                       # Interactive mode
uv run regulation search "query" -a     # Search with LLM answer
uv run regulation serve --web           # Web UI
```

### Coding Style

- Python 3.11+
- `snake_case` for functions/variables
- `CamelCase` for classes
- Type hints required
- Google-style docstrings

### Forbidden

- ❌ Do not modify `domain/` layer to depend on external libraries
- ❌ Do not add features without tests
- ❌ Do not manually edit `sync_state.json` or `data/chroma_db/`

### Query Pipeline Quick Reference

```
User Query → NFC Normalize → Route by Type → [Tool Calling | Traditional Search]
                                                    ↓                ↓
                                           QueryAnalyzer      Self-RAG Check
                                           → Intent Hints     → HyDE (vague queries)
                                           → ToolExecutor     → HybridSearcher
                                           → LLM Answer       → Corrective RAG
                                                              → BGE Reranker
                                                              → LLM Answer
```

### Advanced RAG Features (enabled by default)

| Feature | Env Variable | Description |
|---------|-------------|-------------|
| Self-RAG | `ENABLE_SELF_RAG=true` | LLM evaluates retrieval necessity |
| HyDE | `ENABLE_HYDE=true` | Hypothetical doc for vague queries |
| BM25 KoNLPy | `BM25_TOKENIZE_MODE=konlpy` | Morpheme-based tokenization |
| Corrective RAG | Dynamic thresholds | simple: 0.3, medium: 0.4, complex: 0.5 |

---

For detailed pipeline documentation, see [QUERY_PIPELINE.md](../QUERY_PIPELINE.md).
