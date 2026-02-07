# SPEC-RAG-EVAL-001: 구현 계획

## TAG BLOCK

```yaml
spec_id: SPEC-RAG-EVAL-001
related_spec: spec.md
related_acceptance: acceptance.md
implementation_phase: Planned
```

## Implementation Milestones

### Primary Goal (주 1)

**목표**: RAGAS 라이브러리 호환성 수정 및 실제 평가 활성화

**주요 작업**:
1. RAGAS import 오류 수정
2. Fallback 평가 프레임워크 구현
3. 실제 LLM-as-Judge 평가 활성화
4. 30개 시나리오 실제 평가 실행

**성공 기준**:
- RAGAS import 오류 해결
- 실제 평가로 점수 생성 확인
- Mock 결과와 실제 결과 비교 보고서

### Secondary Goal (주 2)

**목표**: 임계값 재설정 및 첫 번째 실제 평가 분석

**주요 작업**:
1. 현실적인 임계값 설정
2. 실제 평가 결과 상세 분석
3. 페르소나/카테고리별 성능 분석
4. 개선 필요 영역 식별

**성공 기준**:
- 단계별 임계값 문서화
- 성능 분석 보고서
- 개선 우선순위 목록

### Final Goal (주 3)

**목표**: RAG 파이프라인 개선 권장사항 도출

**주요 작업**:
1. 개선 권장사항 생성기 구현
2. 우선순위별 개선 계획 수립
3. 예상 영향 분석
4. 실험 계획 수립

**성공 기준**:
- 개선 권장사항 문서
- A/B 테스트 계획
- 재평가 일정

## Technical Approach

### 단계 1: RAGAS 라이브러리 수정

**문제 분석**:
```python
# 현재 오류 발생 위치
# src/rag/domain/evaluation/quality_evaluator.py:17
from ragas.embeddings import RagasEmbeddings  # ImportError
```

**해결 방안 순서**:

1. **RAGAS 버전 확인**:
   ```bash
   poetry show ragas
   # 또는
   pip show ragas
   ```

2. **버전별 호환성 확인**:
   - RAGAS 0.0.22: RagasEmbeddings 존재 (구버전)
   - RAGAS 0.1.x+: RagasEmbeddings 제거됨

3. **선택 사항**:
   - **옵션 A**: 버전 다운그레이드 (안정적, 기능 제한)
   - **옵션 B**: API 업데이트 (최신 기능, 개발 필요)
   - **옵션 C**: Fallback 구현 (유연성, 유지보수)

**추천**: 옵션 C (Fallback 구현)
- 최신 RAGAS 기능 활용
- 호환성 문제 우회
- 장기적 유지보수 용이

### 단계 2: 실제 평가 실행

**평가 실행 파이프라인**:
```python
# scripts/real_evaluation_test.py
async def run_real_evaluation():
    # 1. 환경 설정 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required for real evaluation")

    # 2. 평가기 초기화
    evaluator = RAGQualityEvaluator(
        judge_model="gpt-4o",
        judge_api_key=api_key,
        use_ragas=True,
        enable_real_evaluation=True  # 실제 평가 활성화
    )

    # 3. 파일럿 테스트 시나리오 로드
    scenarios = load_pilot_test_scenarios()

    # 4. 실제 평가 실행
    results = []
    for scenario in scenarios:
        result = await evaluator.evaluate(
            query=scenario["query"],
            answer=scenario["answer"],
            contexts=scenario["contexts"],
            ground_truth=scenario["ground_truth"]
        )
        results.append(result)

    # 5. 결과 분석
    comparison = compare_mock_vs_real(results)
    report = generate_evaluation_report(results, comparison)

    return report
```

**비용 추정**:
- 쿼리당 1,600 토큰
- 30개 시나리오 = 48,000 토큰
- GPT-4o ($5/1M input, $15/1M output)
- 예상 비용: $0.50-$1.00

### 단계 3: 임계값 재설정

**단계적 접근**:
```python
# 주 1: 초기 목표 (보수적)
INITIAL_THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
    "contextual_precision": 0.65,
    "contextual_recall": 0.65,
}

# 주 2-3: 중간 목표
INTERMEDIATE_THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.80,
    "contextual_precision": 0.75,
    "contextual_recall": 0.75,
}

# 주 4+: 최종 목표
TARGET_THRESHOLDS = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.85,
    "contextual_precision": 0.80,
    "contextual_recall": 0.80,
}
```

**보정 절차**:
1. 실제 평가 실행으로 현재 성능 측정
2. 현재 성능 기준으로 초기 목표 설정
3. 점진적 향상을 위한 중간 목표 설정
4. 산업 표준 기준으로 최종 목표 설정

### 단계 4: 개선 권장사항 도출

**개선 영역 식별**:
```python
class ImprovementSuggester:
    def analyze_results(self, results: List[EvaluationResult]) -> List[Suggestion]:
        suggestions = []

        # 1. Faithfulness 분석
        faithfulness_scores = [r.faithfulness for r in results]
        if mean(faithfulness_scores) < 0.7:
            suggestions.append(self._suggest_hallucination_fixes())

        # 2. Contextual Precision 분석
        precision_scores = [r.contextual_precision for r in results]
        if mean(precision_scores) < 0.7:
            suggestions.append(self._suggest_retrieval_fixes())

        # 3. Answer Relevancy 분석
        relevancy_scores = [r.answer_relevancy for r in results]
        if mean(relevancy_scores) < 0.7:
            suggestions.append(self._suggest_generation_fixes())

        # 4. 우선순위 지정
        suggestions.sort(key=lambda s: s.impact_score, reverse=True)

        return suggestions
```

## Architecture Design

### 컴포넌트 구조

```
┌─────────────────────────────────────────────────────┐
│            RAG Evaluation System                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────┐      ┌─────────────────────┐ │
│  │ Query Handler    │─────▶│ RAG Quality         │ │
│  │ (RAG Pipeline)   │      │ Evaluator           │ │
│  └──────────────────┘      │                     │ │
│                             │ ┌─────────────────┐ │ │
│  ┌──────────────────┐      │ │ RAGAS Framework │ │ │
│  │ Vector Store     │      │ │ (Primary)       │ │ │
│  │ (ChromaDB)       │──────▶│ └─────────────────┘ │ │
│  └──────────────────┘      │ ┌─────────────────┐ │ │
│                             │ │ DeepEval        │ │ │
│  ┌──────────────────┐      │ │ (Fallback)      │ │ │
│  │ Judge LLM        │      │ └─────────────────┘ │ │
│  │ (GPT-4o)         │──────▶│                     │ │
│  └──────────────────┘      │ ┌─────────────────┐ │ │
│                             │ │ Custom Eval     │ │ │
│  ┌──────────────────┐      │ │ (Last Resort)  │ │ │
│  │ Improvement      │◀─────│ └─────────────────┘ │ │
│  │ Suggester        │      │                     │ │
│  └──────────────────┘      └─────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Fallback 체인

```python
class EvaluationFramework:
    """다중 Fallback 평가 프레임워크"""

    async def evaluate(self, query, answer, contexts):
        # 1단계: RAGAS (선호)
        try:
            return await self._ragas_evaluate(query, answer, contexts)
        except Exception as e:
            logger.warning(f"RAGAS failed: {e}, falling back to DeepEval")

        # 2단계: DeepEval (첫 번째 Fallback)
        try:
            return await self._deepeval_evaluate(query, answer, contexts)
        except Exception as e:
            logger.warning(f"DeepEval failed: {e}, falling back to Custom")

        # 3단계: Custom (최후 수단)
        return await self._custom_evaluate(query, answer, contexts)
```

## Risk Management

### 식별된 위험

**위험 1: 높은 API 비용**
- 확률: 중간
- 영향: 중간
- 완화: 소규모 테스트 후 비용 확인, 캐싱 구현

**위험 2: 실제 평가 후에도 낮은 점수**
- 확률: 높음
- 영향: 높음
- 완화: 단계적 임계값 설정, 점진적 개선

**위험 3: RAGAS 호환성 지속적 문제**
- 확률: 낮음
- 영향: 중간
- 완화: 강건한 Fallback 체인 구현

**위험 4: 개선 효과 측정 어려움**
- 확률: 중간
- 영향: 중간
- 완화: A/B 테스트, 통계적 유의성 검정

## Timeline

| 주차 | 주요 작업 | 산출물 | 성공 기준 |
|------|----------|--------|-----------|
| 1 | RAGAS 수정, 실제 평가 실행 | 수정된 코드, 평가 결과 | Import 오류 해결, 실제 점수 확인 |
| 2 | 임계값 재설정, 결과 분석 | 임계값 문서, 분석 보고서 | 현실적 임계값, 성능 분석 완료 |
| 3 | 개선 권장사항 도출 | 개선 계획, A/B 테스트 계획 | 우선순위별 개선 목록 |

## Testing Strategy

### 단위 테스트
```python
# tests/evaluation/test_quality_evaluator.py
def test_ragas_import_success():
    """RAGAS import 성공 확인"""
    evaluator = RAGQualityEvaluator(use_ragas=True)
    assert evaluator.ragas_available is True

def test_fallback_activation():
    """Fallback 메커니즘 테스트"""
    evaluator = RAGQualityEvaluator(use_ragas=False)
    result = await evaluator.evaluate(query, answer, contexts)
    assert result is not None

def test_real_vs_mock_evaluation():
    """실제 평가 vs Mock 평가 비교"""
    real_result = await evaluator.evaluate_real(query, answer, contexts)
    mock_result = await evaluator.evaluate_mock(query, answer, contexts)
    assert real_result.evaluation_type == "real"
    assert mock_result.evaluation_type == "mock"
```

### 통합 테스트
```python
# tests/evaluation/test_evaluation_pipeline.py
async def test_full_evaluation_pipeline():
    """전체 평가 파이프라인 테스트"""
    # 1. 쿼리 실행
    result = query_handler.ask(question)

    # 2. 평가 실행
    evaluation = await evaluator.evaluate(
        query=result.query,
        answer=result.answer,
        contexts=result.contexts
    )

    # 3. 검증
    assert evaluation.faithfulness >= 0.0
    assert evaluation.faithfulness <= 1.0
    assert isinstance(evaluation.passed, bool)
```

## Dependencies

### 선행 조건
- OpenAI API 키 설정
- 충분한 API 크레딧 ($1-2)
- RAG 파이프라인 정상 작동

### 외부 의존성
- ragas >= 0.0.22 또는 deepeval >= 0.21.0
- openai >= 1.0.0
- pytest >= 7.4.0

### 내부 의존성
- src/rag/domain/evaluation/quality_evaluator.py
- src/rag/interface/query_handler.py
- src/rag/infrastructure/chroma_store.py

## Success Metrics

### 주 1 성공 기준
- [ ] RAGAS import 오류 해결
- [ ] 실제 평가로 30개 시나리오 실행 완료
- [ ] Mock vs 실제 결과 비교 보고서

### 주 2 성공 기준
- [ ] 단계별 임계값 문서화
- [ ] 페르소나별 성능 분석 완료
- [ ] 개선 필요 영역 식별

### 주 3 성공 기준
- [ ] 개선 권장사항 목록
- [ ] 우선순위 지정
- [ ] A/B 테스트 계획 수립

## Next Steps

1. **RAGAS 버전 확인**: `poetry show ragas`
2. **API 키 설정**: `export OPENAI_API_KEY=sk-...`
3. **실제 평가 실행**: `python scripts/real_evaluation_test.py`
4. **결과 분석**: 분석 보고서 작성
5. **개선 계획**: 우선순위별 개선 목록 작성

---

**계획 상태**: 준비 완료
**다음 단계**: /moai:2-run SPEC-RAG-EVAL-001
**예상 완료**: 2026-02-28
