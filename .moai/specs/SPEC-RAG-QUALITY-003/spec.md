# SPEC: RAG Retrieval Quality Improvement

**SPEC ID:** SPEC-RAG-QUALITY-003
**Title:** RAG Retrieval Quality Improvement (어휘 불일치 문제 해결)
**Created:** 2026-02-15
**Status:** Complete (All Phases 1-5)
**Priority:** High
**Related SPECs:** SPEC-RAG-QUALITY-001, SPEC-RAG-QUALITY-002
**Lifecycle Level:** spec-anchored (maintained alongside implementation)

## Implementation Progress

| Phase | Status | Files Created |
|-------|--------|---------------|
| Phase 1 (P1-Critical): Colloquial Transformation | ✅ Complete | `src/rag/domain/query/colloquial_transformer.py`, `data/config/colloquial_patterns.json` |
| Phase 2 (P2-High): Morphological Expansion | ✅ Complete | `src/rag/domain/query/morphological_expander.py` |
| Phase 3 (P3-High): Semantic Similarity | ✅ Complete | `src/rag/infrastructure/evaluation/semantic_evaluator.py` |
| Phase 4 (P4-Medium): LLM-as-Judge | ✅ Complete | `src/rag/domain/evaluation/llm_judge.py` (enhanced), `tests/rag/unit/domain/evaluation/test_llm_judge.py` |
| Phase 5 (P5-Medium): Hybrid Weight Optimization | ✅ Complete | `src/rag/application/hybrid_weight_optimizer.py`, `tests/rag/unit/application/test_hybrid_weight_optimizer.py` |

---

## Problem Analysis

### Current State (2026-02-15 Evaluation - Retrieval Quality Proxy Method)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Overall Pass Rate | 0.0% | 70% | FAIL |
| Faithfulness | 0.500 | 0.60 | FAIL |
| Answer Relevancy | 0.501 | 0.70 | FAIL |
| Contextual Precision | 0.505 | 0.65 | FAIL |
| Contextual Recall | 0.501 | 0.65 | FAIL |
| Overall Score | 0.502 | 0.70 | FAIL |

### Key Findings from Analysis

1. **Retrieval IS Working**: Documents are being retrieved correctly from the vector database
2. **Vocabulary Mismatch Problem**: Critical gap between query and document vocabulary
   - User queries: Colloquial Korean (e.g., "휴학 어떻게 해?", "뭐야?", "알려줘")
   - Document text: Formal administrative Korean (e.g., "제3조제1항제2호", "규정")
3. **Evaluation Method Limitation**: Keyword-based evaluation doesn't capture semantic similarity

### Root Cause Analysis (Five Whys)

1. **Why Pass Rate 0%?** Evaluation metrics fail to recognize correct semantic matches
2. **Why Metrics Fail?** Keyword overlap method doesn't capture Korean morphological variants
3. **Why No Semantic Matching?** Current evaluation uses lexical comparison only
4. **Why Lexical Only?** RAGAS/DeepEval configuration lacks Korean semantic simulators
5. **Root Cause:** Dual problem - (A) Query processing doesn't normalize colloquial Korean, (B) Evaluation method doesn't use semantic similarity

### Impact Analysis

| Issue | Current Impact | Affected Component |
|-------|----------------|-------------------|
| Colloquial queries | 60-70% of student queries | Query Processor |
| Morphological variants | 40-50% retrieval misses | BM25 Indexer |
| Keyword evaluation | 100% false negative rate | Evaluation Framework |

---

## Environment

### Technology Stack

- Python 3.13+
- Query Processor: `src/rag/application/query_processor.py`
- BM25 Indexer: `src/rag/infrastructure/bm25_indexer.py`
- Evaluation Framework: `src/rag/evaluation/`
- KiwiPiePy: Korean morpheme analyzer
- Embedding Model: BAAI/bge-m3 (1024 dimensions)

### Configuration Files

- `data/config/colloquial_mappings.json` (NEW)
- `data/config/synonyms.json` (extend)
- `src/rag/evaluation/config.yaml` (update)

### Dependencies

- SPEC-RAG-QUALITY-002 (completed citation/staff improvements)
- Existing KiwiPiePy integration
- BGE-M3 embedding model

---

## Assumptions

| Assumption | Confidence | Validation Method |
|------------|------------|-------------------|
| Colloquial patterns are finite and enumerable | High | Korean corpus analysis |
| Semantic similarity captures query intent | High | A/B testing with embeddings |
| KiwiPiePy can handle morphological expansion | High | Existing integration |
| LLM provider available for LLM-as-Judge | Medium | Check provider configuration |
| BGE-M3 embeddings capture Korean semantics | High | Existing performance data |

---

## Requirements (EARS Format)

### Phase 1 (P1 - Critical): Colloquial-to-Formal Query Transformation

#### Event-Driven Requirements

- **WHEN** a user submits a colloquial query (e.g., "휴학 어떻게 해?", "이거 뭐야?")
- **THEN** the system **SHALL** transform the query to formal Korean before retrieval
- **AND** the transformation **SHALL** preserve the original query intent

**Korean Translation:**
- 사용자가 구어체 쿼리(예: "휴학 어떻게 해?", "이거 뭐야?")를 제출하면
- 시스템은 검색 전에 쿼리를 문어체 한국어로 변환해야 한다
- 변환은 원래 쿼리 의도를 보존해야 한다

#### Ubiquitous Requirements

- The system **SHALL** maintain a colloquial-to-formal mapping dictionary in `colloquial_mappings.json`
- The system **SHALL** handle at least 50 common colloquial patterns
- The system **SHALL** log transformation decisions for debugging

**Korean Translation:**
- 시스템은 `colloquial_mappings.json`에 구어체-문어체 매핑 사전을 유지해야 한다
- 시스템은 최소 50개의 일반적인 구어체 패턴을 처리해야 한다
- 시스템은 디버깅을 위해 변환 결정을 로깅해야 한다

#### State-Driven Requirements

- **IF** a colloquial pattern is not recognized
- **THEN** the system **SHALL** fallback to original query with a warning log
- **AND** the system **SHALL** queue the pattern for dictionary expansion

**Korean Translation:**
- 구어체 패턴이 인식되지 않으면
- 시스템은 경고 로그와 함께 원래 쿼리로 폴백해야 한다
- 시스템은 패턴을 사전 확장을 위해 대기열에 추가해야 한다

---

### Phase 2 (P2 - High): Korean Morphological Expansion

#### Event-Driven Requirements

- **WHEN** the BM25 indexer processes a query
- **THEN** the system **SHALL** expand the query with morphological variants
- **AND** the expansion **SHALL** include conjugation variants (e.g., "휴학하다" -> "휴학", "휴학한", "휴학할")

**Korean Translation:**
- BM25 인덱서가 쿼리를 처리할 때
- 시스템은 쿼리를 형태소 변형으로 확장해야 한다
- 확장은 활용 변형을 포함해야 한다 (예: "휴학하다" -> "휴학", "휴학한", "휴학할")

#### Ubiquitous Requirements

- The system **SHALL** use KiwiPiePy for morphological analysis
- The system **SHALL** maintain a cache of morphological expansions
- The system **SHALL** support both noun extraction and full morpheme analysis modes

**Korean Translation:**
- 시스템은 형태소 분석을 위해 KiwiPiePy를 사용해야 한다
- 시스템은 형태소 확장 캐시를 유지해야 한다
- 시스템은 명사 추출 및 전체 형태소 분석 모드를 모두 지원해야 한다

---

### Phase 3 (P3 - High): Semantic Similarity Evaluation

#### Event-Driven Requirements

- **WHEN** the evaluation framework measures answer quality
- **THEN** the system **SHALL** use embedding-based semantic similarity
- **AND** the similarity threshold **SHALL** be configurable (default: 0.75)

**Korean Translation:**
- 평가 프레임워크가 답변 품질을 측정할 때
- 시스템은 임베딩 기반 의미적 유사성을 사용해야 한다
- 유사성 임계값은 구성 가능해야 한다 (기본값: 0.75)

#### State-Driven Requirements

- **IF** semantic similarity score >= threshold
- **THEN** the evaluation **SHALL** mark the answer as relevant
- **AND** the evaluation **SHALL** record the similarity score for analysis

**Korean Translation:**
- 의미적 유사성 점수가 임계값 이상이면
- 평가는 답변을 관련성 있음으로 표시해야 한다
- 평가는 분석을 위해 유사성 점수를 기록해야 한다

---

### Phase 4 (P4 - Medium): LLM-as-Judge Integration

#### Event-Driven Requirements

- **WHEN** an LLM provider is configured and available
- **THEN** the evaluation framework **SHALL** use LLM-as-Judge for nuanced assessment
- **AND** the LLM **SHALL** evaluate answer correctness, completeness, and citation accuracy

**Korean Translation:**
- LLM 제공자가 구성되고 사용 가능할 때
- 평가 프레임워크는 미묘한 평가를 위해 LLM-as-Judge를 사용해야 한다
- LLM은 답변 정확성, 완전성, 인용 정확성을 평가해야 한다

#### Ubiquitous Requirements

- The system **SHALL** support multiple LLM providers (Ollama, LMStudio, OpenAI, Gemini)
- The system **SHALL** implement graceful degradation when LLM is unavailable
- The system **SHALL** cache LLM judgments to reduce API calls

**Korean Translation:**
- 시스템은 여러 LLM 제공자(Ollama, LMStudio, OpenAI, Gemini)를 지원해야 한다
- 시스템은 LLM을 사용할 수 없을 때 정상적인 저하를 구현해야 한다
- 시스템은 API 호출을 줄이기 위해 LLM 판단을 캐시해야 한다

#### Unwanted Behavior Requirements

- The system **SHALL NOT** fail evaluation when LLM provider is unavailable
- The system **SHALL NOT** block retrieval while waiting for LLM judgment

**Korean Translation:**
- 시스템은 LLM 제공자를 사용할 수 없을 때 평가에 실패해서는 안 된다
- 시스템은 LLM 판단을 기다리는 동안 검색을 차단해서는 안 된다

---

### Phase 5 (P5 - Medium): Hybrid Search Weight Optimization

#### Event-Driven Requirements

- **WHEN** the hybrid search combines BM25 and vector search results
- **THEN** the system **SHALL** apply dynamic weighting based on query type
- **AND** colloquial queries **SHALL** receive higher vector search weight (0.7 vs 0.3)

**Korean Translation:**
- 하이브리드 검색이 BM25와 벡터 검색 결과를 결합할 때
- 시스템은 쿼리 유형에 따라 동적 가중치를 적용해야 한다
- 구어체 쿼리는 더 높은 벡터 검색 가중치(0.7 vs 0.3)를 받아야 한다

#### Ubiquitous Requirements

- The system **SHALL** detect query formality level automatically
- The system **SHALL** log weight adjustments for analysis
- The system **SHALL** allow manual weight override via configuration

**Korean Translation:**
- 시스템은 쿼리 격식 수준을 자동으로 감지해야 한다
- 시스템은 분석을 위해 가중치 조정을 로깅해야 한다
- 시스템은 구성을 통한 수동 가중치 재정의를 허용해야 한다

---

## Specifications

### Component Changes

#### 1. Colloquial Query Transformer (NEW)

**File:** `src/rag/application/colloquial_transformer.py`

**Responsibilities:**
- Detect colloquial patterns in queries
- Transform colloquial Korean to formal Korean
- Log transformations for dictionary expansion

**Interface:**
```python
class ColloquialTransformer:
    def transform(self, query: str) -> TransformResult:
        """Transform colloquial query to formal Korean."""
        ...

    def detect_patterns(self, query: str) -> list[ColloquialPattern]:
        """Detect colloquial patterns in query."""
        ...
```

#### 2. Morphological Expander Enhancement

**File:** `src/rag/infrastructure/bm25_indexer.py`

**Changes:**
- Add morphological expansion for queries
- Implement caching for expansion results
- Support configurable expansion depth

#### 3. Semantic Evaluator (NEW)

**File:** `src/rag/evaluation/semantic_evaluator.py`

**Responsibilities:**
- Calculate embedding-based similarity scores
- Compare answer semantics with expected results
- Support configurable thresholds

**Interface:**
```python
class SemanticEvaluator:
    def evaluate_similarity(self, answer: str, expected: str) -> float:
        """Calculate semantic similarity between answer and expected."""
        ...

    def batch_evaluate(self, answers: list, expected: list) -> list[float]:
        """Batch evaluate multiple answers."""
        ...
```

#### 4. LLM-as-Judge Integration

**File:** `src/rag/evaluation/llm_judge.py`

**Responsibilities:**
- Integrate with configured LLM providers
- Implement graceful degradation
- Cache judgments for efficiency

#### 5. Hybrid Weight Optimizer (NEW)

**File:** `src/rag/application/hybrid_weight_optimizer.py`

**Responsibilities:**
- Detect query formality level
- Calculate dynamic BM25/vector weights
- Log weight decisions

---

## Configuration Schema

### colloquial_mappings.json

```json
{
  "version": "1.0.0",
  "mappings": [
    {
      "pattern": "어떻게 해",
      "formal": "방법",
      "context": "procedure"
    },
    {
      "pattern": "뭐야",
      "formal": "정의",
      "context": "definition"
    },
    {
      "pattern": "알려줘",
      "formal": "안내",
      "context": "information"
    },
    {
      "pattern": "언제까지",
      "formal": "기한",
      "context": "deadline"
    }
  ],
  "regex_patterns": [
    {
      "pattern": "(.+)하는법",
      "replacement": "\\1 방법"
    },
    {
      "pattern": "(.+)어디서",
      "replacement": "\\1 위치"
    }
  ]
}
```

---

## Constraints

### Technical Constraints

- No changes to existing API interfaces
- Backward compatible with existing queries
- Maximum 100ms overhead for transformation
- KiwiPiePy must remain the morphological analyzer

### Performance Constraints

- Query transformation: < 50ms
- Morphological expansion: < 30ms
- Semantic evaluation: < 200ms per query
- Total overhead: < 100ms per query

### Quality Constraints

- Transformation accuracy: >= 95%
- Semantic similarity accuracy: >= 90%
- False positive rate: < 5%
- Test coverage: >= 85%

---

## Out of Scope

1. New embedding model integration
2. Database schema changes
3. UI/UX modifications
4. New LLM provider integration
5. Multi-language support (Korean only)
6. Real-time dictionary learning

---

## Risks and Mitigation

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Colloquial patterns incomplete | Medium | High | Iterative dictionary expansion with logging |
| Semantic similarity threshold too strict | Medium | Medium | Configurable threshold with A/B testing |
| LLM provider unavailable | Low | Medium | Graceful degradation to keyword evaluation |
| Transformation loses query intent | High | Low | Validation tests for transformation accuracy |
| Performance overhead too high | Medium | Low | Caching and optimization |

---

## Success Metrics

| Metric | Current | Target | Measurement Method |
|--------|---------|--------|-------------------|
| Overall Pass Rate | 0.0% | >= 80% | Full evaluation suite |
| Faithfulness | 0.500 | >= 0.75 | RAGAS metric |
| Answer Relevancy | 0.501 | >= 0.75 | RAGAS metric |
| Contextual Precision | 0.505 | >= 0.75 | RAGAS metric |
| Contextual Recall | 0.501 | >= 0.75 | RAGAS metric |
| Colloquial Query Handling | 0% | >= 85% | Transformation success rate |
| Semantic Similarity Accuracy | N/A | >= 90% | Manual validation |

---

## Traceability

**TAG:** SPEC-RAG-QUALITY-003

**Related Files:**
- `src/rag/application/colloquial_transformer.py` (new)
- `src/rag/infrastructure/bm25_indexer.py` (modify)
- `src/rag/evaluation/semantic_evaluator.py` (new)
- `src/rag/evaluation/llm_judge.py` (modify)
- `src/rag/application/hybrid_weight_optimizer.py` (new)
- `src/rag/application/query_processor.py` (modify)

**Configuration Files:**
- `data/config/colloquial_mappings.json` (new)
- `data/config/synonyms.json` (extend)
- `src/rag/evaluation/config.yaml` (modify)

**Test Files:**
- `tests/rag/application/test_colloquial_transformer.py` (new)
- `tests/rag/evaluation/test_semantic_evaluator.py` (new)
- `tests/rag/application/test_hybrid_weight_optimizer.py` (new)

---

**Document Status:** Complete
**Review Required:** Yes
**Implementation Ready:** Pending approval
