# Project Overview: Regulation Manager
Convert University Regulation HWP files to structured JSON for Hybrid RAG.

## Purpose
- HWP to JSON conversion (preserving hierarchy)
- Vector DB sync (ChromaDB + BGE-M3)
- Hybrid Search (BM25 + Dense) + BGE Reranker
- LLM Q&A

## Tech Stack
- **Language**: Python 3.11+
- **Environment**: `uv`
- **Search/RAG**: ChromaDB, BGE-M3, BGE-Reranker
- **Interfaces**: CLI (`regulation`), Gradio (Web), FastMCP (MCP)
- **Hwp Conversion**: `hwp5html` (pyhwp)

## Codebase Structure
- `src/parsing`: HWP/HTML parsing and hierarchy extraction
- `src/rag/domain`: Entities and Repository interfaces
- `src/rag/application`: Use Cases (Search, Sync)
- `src/rag/infrastructure`: Implementation (Chroma, LLM)
- `src/rag/interface`: CLI, Web, MCP
- `tests/`: Pytest suite

## Rules
- Clean Architecture (Domain-centered)
- TDD (Test-Driven Development)
- Python type hints and Google-style docstrings
- `uv` for all dependency management
