# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Regulation Manager** converts HWP university regulation files to structured JSON and provides RAG-powered AI search/Q&A. The system uses Clean Architecture with Python 3.11+.

### Three-Stage Pipeline

```
HWP File → JSON → Vector DB → Hybrid Search → LLM Answer
  (1)       (2)        (3)          (4)           (5)
```

## Development Commands

**Package Manager:** `uv` (not pip, not conda)

```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package>

# Run tests (excludes debug tests by default)
uv run pytest
uv run pytest --cov
uv run pytest -m "not debug"

# Linting and formatting
uv run ruff check
uv run ruff format

# Run the CLI
uv run regulation                    # Interactive mode
uv run regulation convert "file.hwp" # HWP to JSON
uv run regulation sync <json>        # Sync to vector DB
uv run regulation search "query"     # Search
uv run regulation search "query" -a  # Search + LLM answer
uv run regulation serve --web        # Web UI (Gradio)
uv run regulation serve --mcp        # MCP server for AI agents
```

## Architecture

### Clean Architecture Layers

```
src/rag/
├── domain/           # Core entities & interfaces (pure Python only)
├── application/      # Use cases (business logic)
├── infrastructure/   # External system implementations (ChromaDB, LLM)
└── interface/        # CLI, Web UI, MCP Server
```

**Critical Rule:** `domain/` layer must NOT import external libraries. Keep it pure Python.

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `QueryHandler` | `interface/query_handler.py` | Query routing, mode selection |
| `SearchUseCase` | `application/search_usecase.py` | Core search/answer logic, Corrective RAG |
| `QueryAnalyzer` | `infrastructure/query_analyzer.py` | Intent detection, audience analysis |
| `FunctionGemmaAdapter` | `infrastructure/function_gemma_adapter.py` | LLM tool calling |
| `HybridSearch` | `infrastructure/hybrid_search.py` | BM25 + Dense vector fusion with RRF |
| `Reranker` | `infrastructure/reranker.py` | BGE-Reranker-v2-m3 for precision |
| `RetrievalEvaluator` | `infrastructure/retrieval_evaluator.py` | Corrective RAG trigger |
| `SelfRAGEvaluator` | `infrastructure/self_rag.py` | Self-RAG evaluation |

### Query Processing Pipeline

```
User Query → NFC normalization → Query type detection
                                  ↓
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
Regulation name only     Tool Calling (Agentic)    Traditional search
    (Overview)              (Agentic RAG)             (Search/Ask)
    │                         │                         │
    └─────────→ QueryAnalyzer analysis ←──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              Intent expansion    Corrective RAG
              (Intent)             (if low relevance)
                    │                   │
                    └─────────┬─────────┘
                              ↓
                     BGE Reranker → LLM answer
```

### RAG Features

- **Hybrid Search:** BM25 (sparse) + Dense vector fusion with Reciprocal Rank Fusion (RRF)
- **Audience-Aware Search:** Penalizes mismatched student/faculty/staff queries
- **Corrective RAG:** Dynamic thresholds based on query complexity
- **Self-RAG:** Evaluates retrieval necessity and relevance
- **HyDE:** Hypothetical document embeddings for vague/intent-based queries
- **Korean NLP:** KoNLPy Komoran tokenizer for morphological analysis

## Development Principles

### TDD Required

```
RED → GREEN → REFACTOR
```

Write tests before implementing features. Use `pytest` with markers:
- `@pytest.mark.debug` - for debug tests (excluded by default)

### Prohibited Actions

- Do NOT import external libraries in `domain/` layer
- Do NOT add features without tests
- Do NOT manipulate `sync_state.json` or `data/chroma_db/` directly

### Change Checklist

| Change Target | Required Action |
|---------------|-----------------|
| `SearchUseCase` | Run integration tests |
| `QueryAnalyzer` | Verify search test cases |
| `FunctionGemmaAdapter` | Test tool calling scenarios |
| `RetrievalEvaluator` | Verify search quality when adjusting thresholds |
| `QueryHandler` | Test both CLI and Web interfaces |

## Key Dependencies

- **RAG Framework:** `llama-index` (>=0.14.10)
- **Vector Store:** `chromadb` (>=1.4.0)
- **Embedding:** `flagembedding` (bge-m3, bge-reranker-v2-m3) - Korean-optimized
- **Web UI:** `gradio` (>=6.2.0)
- **MCP:** `mcp[cli]` (>=1.9.0) - Model Context Protocol server
- **HWP Processing:** `pyhwp` (>=0.1b15)
- **Korean NLP:** `konlpy` (Komoran tokenizer)

## Related Documentation

- `README.md` - User documentation (Korean)
- `AGENTS.md` - MoAI-ADK workflow instructions (Korean)
- `QUERY_PIPELINE.md` - Detailed query processing pipeline
- `SCHEMA_REFERENCE.md` - JSON schema documentation
- `.cursor/rules/regulation_manager.mdc` - Cursor AI rules
