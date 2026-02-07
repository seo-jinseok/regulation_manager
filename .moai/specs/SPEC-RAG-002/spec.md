# SPEC-RAG-002: RAG System Quality and Maintainability Improvements

## TAG BLOCK

```yaml
spec_id: SPEC-RAG-002
title: RAG System Quality and Maintainability Improvements
status: Planned
priority: High
created: 2026-02-07
assigned: manager-spec
lifecycle: spec-first
estimated_effort: 37 hours
labels: [quality, maintainability, testing, security, refactoring]
related_specs: [SPEC-RAG-001]
```

## Environment

### Current System Context

**Project**: University Regulation Manager (대학 규정 관리 시스템)

**Technology Stack**:
- Python 3.11+
- llama-index >= 0.14.10 (RAG framework)
- chromadb >= 1.4.0 (Vector database)
- flagembedding >= 1.3.5 (BGE-M3 embeddings)
- FastAPI (REST API)
- Redis (Caching)

**Current Architecture**:
- Clean Architecture with domain-driven design
- Hybrid RAG (BM25 + Dense retrieval)
- Multiple LLM providers (Ollama, OpenAI, Gemini)
- Advanced capabilities: Self-RAG, CRAG, T-Fix re-retrieval, HyDE, Query Expansion

### System Scope

**In Scope**:
- Code quality improvements (duplicate code removal, magic numbers to constants)
- Performance optimizations (tokenizer, caching, connection pooling)
- Testing infrastructure setup (pytest, integration tests, benchmarks)
- Security hardening (API key validation, input validation, Redis security)
- Maintainability enhancements (refactoring long functions, configuration management)

**Out of Scope**:
- New feature development (covered in SPEC-RAG-001)
- HWP file processing improvements
- Database schema changes
- New interface development

## Assumptions

### Technical Assumptions

- **High Confidence**: Current codebase has duplicate docstrings and comments in self_rag.py
- **High Confidence**: pytest is not installed in the current environment
- **High Confidence**: Magic numbers exist throughout the codebase (max_context_chars=4000, top_k=10)
- **Medium Confidence**: BM25 index caching using pickle can be optimized with msgpack
- **Evidence**: Quality agent analysis identified these issues in self_rag.py, query_analyzer.py, and tool_executor.py

### Business Assumptions

- **High Confidence**: Code quality improvements will reduce technical debt
- **High Confidence**: Testing infrastructure will prevent future regressions
- **Medium Confidence**: Performance optimizations will improve user experience
- **Risk if Wrong**: Time spent on improvements may not directly translate to user-visible features
- **Validation Method**: Measure test coverage, lint warnings, and performance benchmarks

### Integration Assumptions

- **High Confidence**: Existing codebase follows Clean Architecture principles
- **Medium Confidence**: Circular dependencies exist between search_usecase.py and hybrid_search.py
- **Medium Confidence**: Long functions (_search_general method 1000+ lines) need refactoring
- **Risk if Wrong**: Refactoring may introduce unexpected behavior changes
- **Validation Method**: Comprehensive characterization tests before refactoring

## Requirements

### Priority 1: Code Quality Improvements (7 hours)

#### Ubiquitous Requirements

**REQ-CQ-001**: The system shall not contain duplicate docstrings or comments.
**REQ-CQ-002**: The system shall use named constants instead of magic numbers.
**REQ-CQ-003**: The system shall have consistent type hints across all modules.
**REQ-CQ-004**: The system shall use consistent error messages (unified language).

#### Event-Driven Requirements

**REQ-CQ-005**: WHEN duplicate code is detected, the system SHALL extract common functionality into shared functions.
**REQ-CQ-006**: WHEN magic numbers are found, the system SHALL replace them with named constants in config.py.
**REQ-CQ-007**: WHEN function calls are incorrect (e.g., _extended.warmup_reranker() instead of warmup_all_models()), the system SHALL be corrected.

#### State-Driven Requirements

**REQ-CQ-008**: IF a function exceeds 100 lines, the system SHALL consider refactoring into smaller functions.
**REQ-CQ-009**: IF circular dependencies are detected, the system SHALL be refactored to use dependency injection.

#### Unwanted Behavior Requirements

**REQ-CQ-010**: The system shall NOT have duplicate docstrings (e.g., self_rag.py:129-154).
**REQ-CQ-011**: The system shall NOT have duplicate comments (e.g., self_rag.py:124-127).
**REQ-CQ-012**: The system shall NOT use magic numbers without clear documentation.

---

### Priority 2: Performance Optimizations (14 hours)

#### Ubiquitous Requirements

**REQ-PO-001**: The system shall optimize Kiwi tokenizer initialization.
**REQ-PO-002**: The system shall use msgpack instead of pickle for BM25 index caching.
**REQ-PO-003**: The system shall implement connection pool monitoring for RAGQueryCache.
**REQ-PO-004**: The system shall implement LRU caching with compression for HyDE.

#### Event-Driven Requirements

**REQ-PO-005**: WHEN Kiwi tokenizer is initialized, the system SHALL use lazy loading pattern.
**REQ-PO-006**: WHEN BM25 index is cached, the system SHALL use msgpack for faster serialization.
**REQ-PO-007**: WHEN connection pool is exhausted, the system SHALL log warning and provide metrics.
**REQ-PO-008**: WHEN HyDE cache is accessed, the system SHALL use LRU eviction policy.

#### State-Driven Requirements

**REQ-PO-009**: IF Redis connection is slow, the system SHALL use connection pool monitoring.
**REQ-PO-010**: IF cache size exceeds threshold, the system SHALL evict least recently used entries.

#### Optional Requirements

**REQ-PO-011**: Where possible, the system MAY implement adaptive compression for cached data.
**REQ-PO-012**: Where possible, the system MAY provide performance metrics dashboard.

---

### Priority 3: Testing and Validation (16 hours)

#### Ubiquitous Requirements

**REQ-TV-001**: The system shall have pytest installed and configured.
**REQ-TV-002**: The system shall have integration tests for RAG pipeline.
**REQ-TV-003**: The system shall have performance benchmarking setup.
**REQ-TV-004**: The system shall achieve 85%+ code coverage.

#### Event-Driven Requirements

**REQ-TV-005**: WHEN pytest is installed, the system SHALL configure pytest.ini with asyncio_mode=auto.
**REQ-TV-006**: WHEN integration tests are created, the system SHALL test end-to-end RAG pipeline.
**REQ-TV-007**: WHEN benchmarks are run, the system SHALL measure latency, throughput, and memory usage.

#### State-Driven Requirements

**REQ-TV-008**: IF code coverage drops below 85%, the system SHALL fail CI/CD pipeline.
**REQ-TV-009**: IF performance regresses by >10%, the system SHALL alert developers.

#### Unwanted Behavior Requirements

**REQ-TV-010**: The system shall NOT have untested critical paths.
**REQ-TV-011**: The system shall NOT have flaky tests.

---

### Priority 4: Security Hardening (7 hours)

#### Ubiquitous Requirements

**REQ-SH-001**: The system shall validate API keys before use.
**REQ-SH-002**: The system shall alert on API key expiration.
**REQ-SH-003**: The system shall strengthen input validation.
**REQ-SH-004**: The system shall enforce Redis password authentication.

#### Event-Driven Requirements

**REQ-SH-005**: WHEN API key is invalid or expired, the system SHALL log error and alert administrators.
**REQ-SH-006**: WHEN input contains malicious patterns, the system SHALL reject with error message.
**REQ-SH-007**: WHEN Redis connection is established, the system SHALL require password authentication.

#### State-Driven Requirements

**REQ-SH-008**: IF API key expires within 7 days, the system SHALL send warning notification.
**REQ-SH-009**: IF input validation fails, the system SHALL return 400 Bad Request with clear message.

#### Unwanted Behavior Requirements

**REQ-SH-010**: The system shall NOT log API keys in plain text.
**REQ-SH-011**: The system shall NOT allow unauthenticated Redis connections.

---

### Priority 5: Maintainability Enhancements (Optional, Low Priority)

#### Ubiquitous Requirements

**REQ-ME-001**: The system shall have configuration centralized in config.py.
**REQ-ME-002**: The system shall have logging level standardized.
**REQ-ME-003**: The system shall have architecture documentation updated.

#### Event-Driven Requirements

**REQ-ME-004**: WHEN configuration is needed, the system SHALL read from config.py.
**REQ-ME-005**: WHEN logging occurs, the system SHALL use consistent levels (DEBUG, INFO, WARNING, ERROR).

#### State-Driven Requirements

**REQ-ME-006**: IF circular dependency exists, the system SHALL be refactored using dependency injection.
**REQ-ME-007**: IF function exceeds 1000 lines, the system SHALL be split into smaller functions.

---

## Specifications

### Architecture Design

#### Code Quality Improvements (REQ-CQ-001 to REQ-CQ-012)

**Component**: Refactoring across multiple files

**Duplicate Code Removal**:
- Remove duplicate docstrings in self_rag.py:129-154
- Remove duplicate comments in self_rag.py:124-127
- Extract common functionality into shared functions

**Magic Numbers to Constants**:
```python
# config.py
MAX_CONTEXT_CHARS = 4000
DEFAULT_TOP_K = 10
CACHE_TTL_SECONDS = 3600
AMBIGUITY_THRESHOLD = 0.7
```

**Type Hints Improvement**:
- Add type hints to all function signatures
- Use Protocol for structural typing
- Add TypeVar for generic types

**Error Message Consistency**:
- Standardize on Korean error messages
- Create error message constants
- Use consistent error format

#### Performance Optimizations (REQ-PO-001 to REQ-PO-012)

**Kiwi Tokenizer Optimization**:
```python
# Lazy loading pattern
class KiwiTokenizer:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Kiwi()
        return cls._instance
```

**BM25 Caching with msgpack**:
```python
import msgpack

def save_bm25_index(index, path):
    with open(path, 'wb') as f:
        msgpack.dump(index.to_dict(), f)

def load_bm25_index(path):
    with open(path, 'rb') as f:
        return BM25.from_dict(msgpack.load(f))
```

**Connection Pool Monitoring**:
```python
class RAGQueryCache:
    def __init__(self):
        self.pool = redis.ConnectionPool(max_connections=50)
        self.metrics = ConnectionPoolMetrics(self.pool)

    def check_pool_health(self):
        if self.pool.available_connections < 5:
            logger.warning("Connection pool nearly exhausted")
```

**HyDE Caching Strategy**:
```python
from functools import lru_cache
import zlib

@lru_cache(maxsize=1000)
def get_hyde_query(query: str) -> str:
    cached = cache.get(f"hyde:{query}")
    if cached:
        return zlib.decompress(cached).decode()
    # Generate HyDE query
    result = generate_hyde_query(query)
    compressed = zlib.compress(result.encode())
    cache.set(f"hyde:{query}", compressed)
    return result
```

#### Testing Infrastructure (REQ-TV-001 to REQ-TV-011)

**pytest Configuration**:
```ini
# pytest.ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=85
```

**Integration Test Structure**:
```python
# tests/integration/test_rag_pipeline.py
import pytest

@pytest.mark.asyncio
async def test_end_to_end_rag_pipeline():
    query = "학생 휴학 절차"
    results = await search_service.search(query)
    assert len(results) > 0
    assert all(r.citation for r in results)
```

**Performance Benchmarking**:
```python
# tests/benchmarks/test_performance.py
import pytest

@pytest.mark.benchmark
def test_bm25_retrieval_latency(benchmark):
    query = "휴학 규정"
    results = benchmark(bm25_retriever.retrieve, query)
    assert len(results) > 0
```

#### Security Hardening (REQ-SH-001 to REQ-SH-011)

**API Key Validation**:
```python
class APIKeyValidator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.expiry_date = self._extract_expiry(api_key)

    def validate(self) -> bool:
        if not self.api_key:
            raise ValueError("API key is missing")
        if self.is_expired():
            raise ValueError("API key has expired")
        return True

    def is_expiring_soon(self, days: int = 7) -> bool:
        return (self.expiry_date - datetime.now()).days <= days
```

**Input Validation**:
```python
from pydantic import BaseModel, validator

class SearchQuery(BaseModel):
    query: str
    top_k: int = 10

    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:
            raise ValueError('Query too long (max 1000 characters)')
        # Check for malicious patterns
        if any(pattern in v.lower() for pattern in ['<script', 'javascript:', 'eval(']):
            raise ValueError('Query contains invalid characters')
        return v

    @validator('top_k')
    def validate_top_k(cls, v):
        if v < 1 or v > 100:
            raise ValueError('top_k must be between 1 and 100')
        return v
```

**Redis Password Enforcement**:
```python
# config.py
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
if not REDIS_PASSWORD:
    raise ValueError("REDIS_PASSWORD environment variable must be set")

# redis_client.py
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    password=REDIS_PASSWORD,
    decode_responses=True
)
```

### File Structure

```
src/
├── config.py                           # NEW/MODIFIED: Centralized constants
├── domain/
│   ├── llm/
│   │   ├── self_rag.py                 # MODIFIED: Remove duplicates
│   │   └── api_key_validator.py        # NEW: API key validation
│   └── query/
│       └── query_analyzer.py           # MODIFIED: Fix function calls
├── infrastructure/
│   ├── cache/
│   │   ├── bm25_cache.py               # MODIFIED: Use msgpack
│   │   ├── hyde_cache.py               # NEW: LRU + compression
│   │   └── pool_monitor.py             # NEW: Connection pool monitoring
│   └── nlp/
│       └── kiwi_tokenizer.py           # MODIFIED: Lazy loading
├── application/
│   ├── rag/
│   │   └── search_usecase.py           # MODIFIED: Refactor long methods
│   └── services/
│       └── tool_executor.py            # MODIFIED: Implement TODOs
└── tests/
    ├── unit/                           # NEW: Unit tests
    ├── integration/                    # NEW: Integration tests
    └── benchmarks/                     # NEW: Performance benchmarks
```

### Dependencies

**New Dependencies**:
```toml
[tool.poetry.dependencies]
msgpack = "^1.0.0"  # Faster serialization
pydantic = "^2.0.0"  # Input validation

[tool.poetry.dev-dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0"
pytest-cov = "^5.0.0"
pytest-benchmark = "^4.0.0"
```

## Traceability

### Requirements to Components Mapping

| Requirement ID | Component | File |
|---------------|-----------|------|
| REQ-CQ-001 ~ REQ-CQ-012 | Code Quality Refactoring | domain/llm/self_rag.py, application/rag/search_usecase.py |
| REQ-PO-001 ~ REQ-PO-012 | Performance Optimizations | infrastructure/cache/, infrastructure/nlp/ |
| REQ-TV-001 ~ REQ-TV-011 | Testing Infrastructure | tests/unit/, tests/integration/, tests/benchmarks/ |
| REQ-SH-001 ~ REQ-SH-011 | Security Hardening | domain/llm/api_key_validator.py, infrastructure/cache/ |

### Components to Test Cases Mapping

| Component | Test File | Test Coverage Target |
|-----------|-----------|---------------------|
| KiwiTokenizer | tests/unit/test_kiwi_tokenizer.py | 85% |
| BM25Cache | tests/unit/test_bm25_cache.py | 90% |
| HyDECache | tests/unit/test_hyde_cache.py | 85% |
| APIKeyValidator | tests/unit/test_api_key_validator.py | 90% |
| RAGPipeline | tests/integration/test_rag_pipeline.py | 80% |

### Dependencies

**External Dependencies**:
- msgpack for faster serialization
- pytest, pytest-asyncio, pytest-cov for testing
- pydantic for input validation

**Internal Dependencies**:
- config.py for all constant definitions
- existing RAG pipeline components

---

## Appendix

### Glossary

- **Magic Numbers**: Hard-coded numeric values without named constants
- **msgpack**: Binary serialization format faster than pickle
- **LRU Cache**: Least Recently Used caching strategy
- **Kiwi Tokenizer**: Korean morphological analyzer
- **BM25**: Ranking function for information retrieval
- **HyDE**: Hypothetical Document Embeddings

### References

- pytest Documentation: https://docs.pytest.org/
- msgpack Documentation: https://msgpack.org/
- Pydantic Documentation: https://docs.pydantic.dev/
- OWASP Security Guidelines: https://owasp.org/

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-07 | manager-spec | Initial SPEC creation for quality improvements |

---

**SPEC Status**: Planned
**Next Phase**: /moai:2-run SPEC-RAG-002 (Implementation with DDD)
**Estimated Completion**: 2026-02-21 (approximately 2 weeks)
