# Technical Stack

## Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.11+ | Core runtime |
| **Package Manager** | uv | Fast dependency management |
| **Testing** | pytest | Test framework |

## RAG Components

### Vector Database
- **ChromaDB** (v1.4+): Local persistent vector store
- Location: `data/chroma_db/`
- Collections: regulations, embeddings

### Embedding Models
| Model | Dimensions | Purpose |
|-------|-----------|---------|
| BAAI/bge-m3 | 1024 | Korean-optimized multilingual embeddings |
| ko-sbert-sts | 768 | Korean sentence similarity (backup) |

### Reranker
- **BAAI/bge-reranker-v2-m3**: Cross-encoder for precision ranking
- Conditional reranking based on query complexity

### Keyword Search
- **BM25**: Okapi BM25 algorithm (k1=1.5, b=0.75)
- **KiwiPiePy**: Korean morpheme analyzer (replaces KoNLPy)
- RRF Fusion: Reciprocal Rank Fusion (k=60)

## LLM Integration

### Supported Providers
| Provider | Models | Notes |
|----------|--------|-------|
| Ollama | gemma2, llama2 | Local inference |
| LMStudio | exaone-4.0-32b-mlx | Local with MLX |
| OpenAI | gpt-4, gpt-3.5 | Cloud API |
| Gemini | gemini-pro | Google API |

### LLM Features
- **Circuit Breaker**: 3-state failure detection (CLOSED, OPEN, HALF_OPEN)
- **Connection Pooling**: Redis (50), HTTP (100)
- **Multi-layer Caching**: L1 (memory), L2 (Redis), L3 (ChromaDB)

## Web Interface

### Gradio (v6.2+)
- ChatGPT-style interface
- Example query cards
- Regulation detail view
- Target audience selection

### MCP Server
- **FastMCP** (v1.9+): Model Context Protocol
- Tools: search_regulations, ask_regulations, view_article
- Resources: sync status, regulation list

## Advanced RAG Techniques

### Agentic RAG
- Tool selection by LLM
- Tools: search_regulations, get_regulation_detail, generate_answer

### Corrective RAG (CRAG)
- Relevance evaluation
- Query expansion
- Automatic re-retrieval
- Dynamic threshold (0.3-0.5)

### Self-RAG
- Retrieval necessity evaluation
- Result quality assessment
- Groundedness verification

### HyDE (Hypothetical Document Embeddings)
- Virtual document generation
- Ambiguous query detection
- LRU cache with zlib compression

## Development Tools

### Testing
- **pytest** (v9.0+): Test framework
- **pytest-xdist** (v3.6+): Parallel execution (max 2 workers)
- **pytest-cov** (v7.0+): Coverage reporting
- **pytest-asyncio**: Async test support
- **pytest-timeout**: 300s limit
- **pytest-benchmark**: Performance testing

### Code Quality
- **ruff** (v0.8+): Linting and formatting
- **pyright**: Type checking
- **mypy**: Static analysis (alternative)

### Evaluation Frameworks
- **RAGAS** (v0.4+): LLM-as-Judge evaluation
- **DeepEval** (v3.8+): Alternative evaluation

## Security & Validation

### Input Validation
- **Pydantic** (v2.0+): Schema validation
- Query length limit: 1000 characters
- Malicious pattern detection
- top_k range: 1-100

### Encryption
- **cryptography** (v41.0+): AES-256 for sensitive cache
- API key format validation
- Expiration warnings (7 days)

## Performance Features

### Caching Strategy
- L1: In-memory (fastest, size-limited)
- L2: Redis (distributed, persistent)
- L3: ChromaDB (vector similarity)

### Cache Warming
- Top 100 regulations pre-embedded
- Scheduled warming (default: 2 AM)
- Progressive warming by query frequency

### Memory Management
- Maximum 2 xdist workers
- --no-cov default for pytest
- 12GB memory limit with auto-abort
- Batched testing for large test suites

## Dependencies (Key)

```toml
# Core
llama-index>=0.14.10
chromadb>=1.4.0
gradio>=6.2.0

# NLP
flagembedding>=1.3.5  # BGE models
kiwipiepy>=0.20.0     # Korean morpheme
sentence-transformers>=2.2.0

# LLM
openai>=2.11.0
mlx-lm>=0.29.1

# Quality
ragas>=0.4.3
deepeval>=3.8.1

# Security
pydantic>=2.0.0
cryptography>=41.0.0
```

## Configuration Files

| File | Purpose |
|------|---------|
| pyproject.toml | Project config, dependencies |
| .env | Environment variables |
| pytest.ini | Test configuration |
| .moai/config/ | MoAI-ADK settings |
| data/config/synonyms.json | Synonym dictionary (167 terms) |
| data/config/intents.json | Intent rules (51 rules) |
