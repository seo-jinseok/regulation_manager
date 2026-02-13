# Project Structure

## Directory Layout

```
regulation_manager/
├── .moai/                      # MoAI-ADK configuration
│   ├── config/                 # Project and workflow configs
│   ├── specs/                  # SPEC documents
│   ├── agents/                 # Agent definitions
│   ├── skills/                 # Skill definitions
│   └── rules/                  # Development rules
├── src/                        # Source code
│   ├── main.py                 # Entry point
│   ├── converter.py            # HWP conversion
│   ├── formatter.py            # JSON formatting
│   ├── enhance_for_rag.py      # RAG optimization
│   ├── commands/               # CLI commands
│   │   ├── __init__.py
│   │   └── reparse_hwpx.py     # HWPX full reparse command
│   ├── analysis/               # Quality analysis
│   │   ├── __init__.py
│   │   └── quality_reporter.py # Quality report generator
│   ├── parsing/                # Document parsing
│   │   ├── regulation_parser.py
│   │   ├── reference_resolver.py
│   │   └── table_extractor.py
│   ├── rag/                    # RAG System (Clean Architecture)
│   │   ├── interface/          # CLI, Web UI, MCP Server
│   │   │   ├── unified_cli.py
│   │   │   ├── gradio_app.py
│   │   │   └── mcp_server.py
│   │   ├── application/        # Use Cases
│   │   │   ├── search_usecase.py
│   │   │   ├── sync_usecase.py
│   │   │   ├── query_expansion.py
│   │   │   └── evaluation/
│   │   ├── domain/             # Domain models
│   │   │   ├── entities.py
│   │   │   ├── repositories.py
│   │   │   ├── value_objects.py
│   │   │   ├── citation/
│   │   │   ├── conversation/
│   │   │   ├── llm/
│   │   │   ├── evaluation/
│   │   │   └── experiment/
│   │   ├── infrastructure/     # External integrations
│   │   │   ├── chroma_store.py
│   │   │   ├── dense_retriever.py
│   │   │   ├── reranker.py
│   │   │   ├── llm_client.py
│   │   │   ├── cache.py
│   │   │   └── storage/
│   │   └── automation/         # RAG Testing Automation
│   │       ├── domain/
│   │       ├── application/
│   │       ├── infrastructure/
│   │       └── interface/
│   └── tools/                  # Utility tools
│       └── merger.py
├── data/                       # Data directory
│   ├── input/                  # HWP files
│   ├── output/                 # JSON files
│   ├── chroma_db/              # Vector database
│   ├── cache/                  # Query cache
│   ├── config/                 # Synonyms, intents
│   ├── test_sessions/          # Test sessions
│   ├── test_reports/           # Test reports
│   └── evaluations/            # Quality evaluations
├── tests/                      # Pytest tests
│   ├── rag/
│   │   ├── unit/               # Unit tests
│   │   ├── integration/        # Integration tests
│   │   └── e2e/                # End-to-end tests
│   └── conftest.py             # Pytest configuration
├── scripts/                    # Utility scripts
│   ├── quick_test.sh           # Fast tests
│   └── run_tests_batched.sh   # Batched tests
├── .venv/                      # Virtual environment
├── pyproject.toml              # Project config
├── README.md                   # Main documentation
└── CHANGELOG.md                # Version history
```

## Architecture Layers

### Clean Architecture (RAG System)

```
┌─────────────────────────────────────────────┐
│           Interface Layer                   │
│  (CLI, Web UI, MCP Server, Automation CLI)  │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│         Application Layer                    │
│  (Search, Sync, Evaluation, Conversation)    │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│           Domain Layer                      │
│  (Entities, Value Objects, Repositories)     │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│       Infrastructure Layer                  │
│  (ChromaDB, LLM, Cache, Storage, Reranker)  │
└─────────────────────────────────────────────┘
```

## Key Components

### Parsing Pipeline
- `regulation_parser.py`: HWP → JSON with hierarchy
- `reference_resolver.py`: Cross-reference resolution
- `table_extractor.py`: Table data extraction

### RAG Pipeline
- `query_expansion.py`: Query enhancement (synonyms, intents)
- `chroma_store.py`: Vector database operations
- `dense_retriever.py`: Semantic search
- `reranker.py`: Result re-ranking
- `llm_client.py`: LLM integration

### Quality Evaluation
- `evaluation/`: LLM-as-Judge system
- `automation/`: Automated testing framework
- 6 persona types for comprehensive testing

### Domain Services
- `citation/`: Article number extraction
- `conversation/`: Multi-turn dialogue
- `llm/`: Circuit breaker, ambiguity, emotion
- `experiment/`: A/B testing framework
