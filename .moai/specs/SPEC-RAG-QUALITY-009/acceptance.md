# Acceptance Criteria: SPEC-RAG-QUALITY-009

## Metadata

| Field | Value |
|-------|-------|
| SPEC ID | SPEC-RAG-QUALITY-009 |
| Created | 2026-02-21 |
| Status | Planned |

---

## Overview

본 문서는 RAG 시스템 품질 종합 개선에 대한 상세 인수 기준을 정의합니다. 모든 테스트 시나리오는 Given-When-Then (Gherkin) 형식으로 작성됩니다.

---

## REQ-001: Enable Faithfulness Validation by Default

### AC-001-01: Validation Enabled by Default

**Given** SearchUseCase가 기본 설정으로 초기화될 때
**When** 시스템이 인스턴스를 생성하면
**Then** `use_faithfulness_validation`이 `True`로 설정되어야 한다

```python
# Test Case
def test_validation_enabled_by_default():
    usecase = SearchUseCase(...)
    assert usecase.use_faithfulness_validation == True
```

### AC-001-02: All Answers Validated

**Given** RAG 시스템이 답변을 생성할 때
**When** `search()` 또는 `search_with_qa()` 메서드가 호출되면
**Then** 모든 답변이 `FaithfulnessValidator`를 통해 검증되어야 한다

```python
# Test Case
async def test_all_answers_validated():
    usecase = SearchUseCase(...)
    with mock.patch.object(FaithfulnessValidator, 'validate_answer') as mock_validate:
        mock_validate.return_value = FaithfulnessValidationResult(
            score=0.8, is_acceptable=True
        )
        await usecase.search_with_qa("휴학 규정 알려주세요")
        assert mock_validate.called
```

### AC-001-03: Low Faithfulness Triggers Regeneration

**Given** 답변의 Faithfulness 점수가 0.6 미만일 때
**When** 답변이 생성되면
**Then** 시스템은 최대 2회 재생성을 시도해야 한다

```python
# Test Case
async def test_low_faithfulness_triggers_regeneration():
    usecase = SearchUseCase(...)
    with mock.patch.object(usecase, '_llm_generate') as mock_generate:
        # First call returns low faithfulness answer
        mock_generate.side_effect = [
            Answer(content="낮은 품질 답변", confidence=0.3),
            Answer(content="개선된 답변", confidence=0.8),
        ]
        with mock.patch.object(FaithfulnessValidator, 'validate_answer') as mock_validate:
            mock_validate.side_effect = [
                FaithfulnessValidationResult(score=0.4, is_acceptable=False),
                FaithfulnessValidationResult(score=0.7, is_acceptable=True),
            ]
            result = await usecase.search_with_qa("휴학 규정")
            assert mock_generate.call_count == 2
```

### AC-001-04: Fallback Message on Regeneration Failure

**Given** 모든 재생성 시도가 실패했을 때
**When** 최대 재시도 횟수에 도달하면
**Then** 명확한 fallback 메시지가 반환되어야 한다

```python
# Test Case
async def test_fallback_on_regeneration_failure():
    usecase = SearchUseCase(...)
    with mock.patch.object(FaithfulnessValidator, 'validate_answer') as mock_validate:
        mock_validate.return_value = FaithfulnessValidationResult(
            score=0.4, is_acceptable=False
        )
        result = await usecase.search_with_qa("휴학 규정")
        assert "찾을 수 없습니다" in result.content
```

### Success Criteria

| Criteria | Target |
|----------|--------|
| Faithfulness Score | >= 0.60 |
| Validation Rate | 100% |
| Regeneration Rate | 20-40% |
| Fallback Accuracy | 100% |

---

## REQ-002: Fix RAGAS Library Compatibility

### AC-002-01: RAGAS Version Compatible

**Given** RAGAS 라이브러리가 설치되어 있을 때
**When** 시스템이 RAGAS를 import하면
**Then** 버전이 0.4.13 이상이어야 한다

```python
# Test Case
def test_ragas_version():
    import ragas
    from packaging import version
    assert version.parse(ragas.__version__) >= version.parse("0.4.13")
```

### AC-002-02: RunConfig Initialization Success

**Given** RAGAS 평가가 초기화될 때
**When** `RunConfig`가 생성되면
**Then** `on_chain_start` 속성 오류가 발생하지 않아야 한다

```python
# Test Case
def test_runconfig_initialization():
    from ragas import RunConfig
    config = RunConfig(max_wait=60, max_workers=4)
    # Should not raise AttributeError
    assert config is not None
```

### AC-002-03: ProgressInfo Compatibility

**Given** 평가가 진행 중일 때
**When** 진행률이 업데이트되면
**Then** `ProgressInfo.completed` 속성 오류가 발생하지 않아야 한다

```python
# Test Case
def test_progressinfo_compatibility():
    from ragas.evaluation import ProgressInfo
    progress = ProgressInfo(completed=0, total=10)
    progress.completed += 1  # Should not raise AttributeError
    assert progress.completed == 1
```

### AC-002-04: Evaluation Pipeline Success

**Given** 평가 파이프라인이 실행될 때
**When** `evaluate()` 메서드가 호출되면
**Then** 4개 메트릭 점수가 정상 반환되어야 한다

```python
# Test Case
async def test_evaluation_pipeline_success():
    evaluator = RAGQualityEvaluator(use_ragas=True)
    sample = SingleTurnSample(
        user_input="휴학 규정",
        response="휴학은...",
        retrieved_contexts=["휴학규정 제1조..."],
    )
    result = await evaluator.evaluate(sample)
    assert result.faithfulness_score is not None
    assert result.answer_relevancy_score is not None
    assert result.contextual_precision_score is not None
    assert result.contextual_recall_score is not None
```

### AC-002-05: Fallback to DeepEval

**Given** RAGAS 평가가 실패할 때
**When** `AttributeError` 또는 `ImportError`가 발생하면
**Then** deepeval로 자동 전환되어야 한다

```python
# Test Case
async def test_fallback_to_deepeval():
    evaluator = RAGQualityEvaluator(use_ragas=True)
    with mock.patch.object(evaluator, '_evaluate_with_ragas') as mock_ragas:
        mock_ragas.side_effect = AttributeError("Test error")
        with mock.patch.object(evaluator, '_evaluate_with_deepeval') as mock_deepeval:
            mock_deepeval.return_value = EvaluationResult(...)
            result = await evaluator.evaluate(sample)
            assert mock_deepeval.called
```

### Success Criteria

| Criteria | Target |
|----------|--------|
| RAGAS Error Rate | 0% |
| Successful Evaluations | 100% |
| Fallback Success Rate | 100% |

---

## REQ-003: Optimize Reranker for Korean Regulations

### AC-003-01: Document Type Weights Applied

**Given** 검색 결과가 반환될 때
**When** reranking이 수행되면
**Then** 문서 타입별 가중치가 적용되어야 한다

```python
# Test Case
def test_document_type_weights():
    reranker = Reranker()
    docs = [
        ScoredDocument(content="규정 본문", score=0.8, metadata={"document_type": "regulation"}),
        ScoredDocument(content="서식", score=0.8, metadata={"document_type": "form"}),
    ]
    result = reranker.apply_type_weights(docs)
    # Regulation should have higher score than form
    assert result[0].score > result[1].score
```

### AC-003-02: Transitional Documents Suppressed

**Given** 검색 결과에 경과조치 문서가 포함될 때
**When** reranking이 수행되면
**Then** 경과조치 문서의 점수가 낮아져야 한다

```python
# Test Case
def test_transitional_documents_suppressed():
    reranker = Reranker()
    docs = [
        ScoredDocument(content="휴학규정 제5조", score=0.7, metadata={"document_type": "regulation"}),
        ScoredDocument(content="휴학규정 경과조치", score=0.8, metadata={"document_type": "transitional"}),
    ]
    result = reranker.apply_type_weights(docs)
    # Regulation should rank higher than transitional
    assert result[0].metadata["document_type"] == "regulation"
```

### AC-003-03: Query Type Aware Reranking

**Given** 절차 관련 질문이 입력될 때
**When** 검색이 수행되면
**Then** 절차 문서가 우선순위를 가져야 한다

```python
# Test Case
async def test_procedure_query_boost():
    usecase = SearchUseCase(...)
    result = await usecase.search("휴학 신청 방법")
    # First result should be procedure document
    assert "절차" in result[0].content or "신청" in result[0].content
```

### Success Criteria

| Criteria | Target |
|----------|--------|
| Contextual Precision | >= 0.65 |
| Irrelevant Doc Rate | < 20% |
| Transitional Suppression | 50%+ reduction |

---

## REQ-004: Enhance Query Intent Detection

### AC-004-01: Ambiguous Query Classification

**Given** 모호한 질문이 입력될 때 (예: "XX가 뭐예요?")
**When** 의도 분류가 수행되면
**Then** 적절한 의도 카테고리로 분류되어야 한다

```python
# Test Case
def test_ambiguous_query_classification():
    classifier = IntentClassifier()
    result = classifier.classify("휴학이 뭐예요?")
    assert result.category in [IntentCategory.DEFINITION, IntentCategory.GENERAL]
    assert result.confidence >= 0.7
```

### AC-004-02: Intent-Based Search Configuration

**Given** 의도가 분류될 때
**When** 검색이 수행되면
**Then** 의도에 맞는 검색 설정이 적용되어야 한다

```python
# Test Case
async def test_intent_based_search_config():
    usecase = SearchUseCase(...)
    with mock.patch.object(usecase.searcher, 'search') as mock_search:
        mock_search.return_value = []
        await usecase.search("휴학 신청 방법")  # PROCEDURE intent
        call_args = mock_search.call_args
        assert call_args.kwargs.get('top_k') >= 10
```

### Success Criteria

| Criteria | Target |
|----------|--------|
| Intent Classification Accuracy | >= 85% |
| Answer Relevancy | >= 0.70 |
| Ambiguous Query Handling | >= 80% |

---

## REQ-005: Remove Evasive Response Patterns

### AC-005-01: Evasive Pattern Detection

**Given** 답변에 회피성 패턴이 포함될 때
**When** 패턴 감지가 수행되면
**Then** 회피성 답변으로 분류되어야 한다

```python
# Test Case
def test_evasive_pattern_detection():
    detector = EvasivePatternDetector()
    is_evasive, pattern = detector.detect("정확한 정보는 홈페이지를 참고하세요")
    assert is_evasive == True
    assert "홈페이지" in pattern or "참고" in pattern
```

### AC-005-02: Regeneration on Evasive Response

**Given** 회피성 답변이 감지될 때
**When** 답변이 생성되면
**Then** 재생성이 시도되어야 한다

```python
# Test Case
async def test_regeneration_on_evasive():
    usecase = SearchUseCase(...)
    with mock.patch.object(EvasivePatternDetector, 'detect') as mock_detect:
        mock_detect.side_effect = [
            (True, "홈페이지"),  # First answer is evasive
            (False, ""),  # Second answer is not
        ]
        result = await usecase.search_with_qa("휴학 규정")
        assert "홈페이지" not in result.content
```

### Success Criteria

| Criteria | Target |
|----------|--------|
| Evasive Response Rate | < 5% |
| Detection Accuracy | >= 90% |
| Regeneration Success | >= 80% |

---

## REQ-006: Implement Persona-Based Evaluation System

### AC-006-01: Persona Definitions Complete

**Given** 페르소나 평가 시스템이 초기화될 때
**When** 페르소나가 로드되면
**Then** 6가지 페르소나가 모두 정의되어 있어야 한다

```python
# Test Case
def test_persona_definitions():
    from src.rag.domain.evaluation.persona_definitions import PERSONAS
    expected_personas = ["freshman", "student", "professor", "staff", "parent", "international"]
    for persona in expected_personas:
        assert persona in PERSONAS
        assert "name" in PERSONAS[persona]
        assert "characteristics" in PERSONAS[persona]
```

### AC-006-02: Persona Query Templates

**Given** 각 페르소나에 대해
**When** 쿼리 템플릿이 로드되면
**Then** 최소 10개의 쿼리가 정의되어 있어야 한다

```python
# Test Case
def test_persona_query_templates():
    import json
    with open("src/rag/domain/evaluation/persona_queries.json") as f:
        queries = json.load(f)
    for persona, query_list in queries.items():
        assert len(query_list) >= 10
```

### AC-006-03: Persona Evaluation Execution

**Given** 페르소나 평가가 실행될 때
**When** `evaluate_persona()`가 호출되면
**Then** 해당 페르소나의 평균 점수가 반환되어야 한다

```python
# Test Case
async def test_persona_evaluation():
    evaluator = PersonaEvaluator(base_evaluator=mock_evaluator)
    result = await evaluator.evaluate_persona(
        persona_id="freshman",
        queries=["휴학이 뭐예요?", "복학 어떻게 해요?"],
        search_usecase=mock_usecase
    )
    assert result.persona_id == "freshman"
    assert 0 <= result.avg_score <= 1
```

### Success Criteria

| Criteria | Target |
|----------|--------|
| Persona Coverage | 6 personas |
| Queries per Persona | >= 10 |
| Evaluation Time | < 5 min |

---

## Integration Test Scenarios

### E2E-001: Full Pipeline Quality Improvement

**Given** RAG 시스템이 Phase 1-4 개선 사항이 적용되었을 때
**When** 30개 테스트 쿼리로 평가를 실행하면
**Then** 다음 기준을 달성해야 한다:
- Faithfulness >= 0.60
- Answer Relevancy >= 0.70
- Contextual Precision >= 0.65
- Pass Rate >= 40%

```python
# Test Case
async def test_full_pipeline_improvement():
    # Run evaluation with 30 test queries
    results = await run_full_evaluation(queries=TEST_QUERIES)

    metrics = calculate_average_metrics(results)
    assert metrics.faithfulness >= 0.60
    assert metrics.answer_relevancy >= 0.70
    assert metrics.contextual_precision >= 0.65

    pass_rate = sum(1 for r in results if r.passed) / len(results)
    assert pass_rate >= 0.40
```

### E2E-002: No Regression in Contextual Recall

**Given** Phase 1-4 개선 사항이 적용되었을 때
**When** 평가를 실행하면
**Then** Contextual Recall이 0.85 이상을 유지해야 한다

```python
# Test Case
async def test_no_regression_in_recall():
    results = await run_full_evaluation(queries=TEST_QUERIES)
    avg_recall = sum(r.contextual_recall for r in results) / len(results)
    assert avg_recall >= 0.85
```

### E2E-003: Response Time Acceptable

**Given** Faithfulness Validation이 활성화되었을 때
**When** 답변을 생성하면
**Then** 평균 응답 시간이 3초를 초과하지 않아야 한다

```python
# Test Case
async def test_response_time_acceptable():
    usecase = SearchUseCase(...)

    times = []
    for query in TEST_QUERIES[:10]:
        start = time.time()
        await usecase.search_with_qa(query)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    assert avg_time < 3.0  # 3 seconds
```

---

## Quality Gates Summary

| Gate | Metric | Threshold | Priority |
|------|--------|-----------|----------|
| QG-001 | Faithfulness | >= 0.60 | Critical |
| QG-002 | RAGAS Error Rate | = 0% | Critical |
| QG-003 | Contextual Precision | >= 0.65 | High |
| QG-004 | Answer Relevancy | >= 0.70 | High |
| QG-005 | Pass Rate | >= 40% | High |
| QG-006 | Contextual Recall | >= 0.85 | Medium |
| QG-007 | Response Time | < 3s | Medium |
| QG-008 | Evasive Response Rate | < 5% | Medium |

---

## Test Execution Plan

### Phase 1 Tests (Week 1)

- [ ] AC-001-01 ~ AC-001-04: Faithfulness Validation
- [ ] AC-002-01 ~ AC-002-05: RAGAS Compatibility
- [ ] E2E-001: Full Pipeline (Partial)

### Phase 2 Tests (Week 2)

- [ ] AC-003-01 ~ AC-003-03: Reranker Optimization
- [ ] E2E-001: Full Pipeline (Complete)
- [ ] E2E-002: No Regression

### Phase 3 Tests (Week 3-4)

- [ ] AC-004-01 ~ AC-004-02: Query Intent
- [ ] AC-005-01 ~ AC-005-02: Evasive Patterns
- [ ] E2E-003: Response Time

### Phase 4 Tests (Week 5+)

- [ ] AC-006-01 ~ AC-006-03: Persona Evaluation
- [ ] Full regression suite

---

## Verification Commands

```bash
# Run Phase 1 tests
uv run pytest tests/rag/unit/application/test_faithfulness_validation_enabled.py -v
uv run pytest tests/rag/unit/domain/evaluation/test_ragas_compatibility.py -v

# Run full evaluation
uv run python scripts/verify_evaluation_metrics.py --full-eval

# Run integration tests
uv run pytest tests/integration/test_quality_improvement.py -v

# Verify quality gates
uv run python scripts/check_quality_gates.py --spec SPEC-RAG-QUALITY-009
```

---

## References

- Gherkin Syntax: https://cucumber.io/docs/gherkin/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- RAGAS Documentation: https://docs.ragas.io/
