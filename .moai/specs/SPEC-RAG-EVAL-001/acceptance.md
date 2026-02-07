# SPEC-RAG-EVAL-001: 인수 기준

## TAG BLOCK

```yaml
spec_id: SPEC-RAG-EVAL-001
related_spec: spec.md
related_plan: plan.md
acceptance_status: Pending
```

## Acceptance Criteria Overview

### Overall Success Criteria

SPEC-RAG-EVAL-001의 성공적인 완료를 위해 다음 기준을 모두 충족해야 한다:

1. **기술적 완성**: RAGAS 라이브러리 호환성 문제 해결
2. **기능적 완성**: 실제 LLM-as-Judge 평가 작동
3. **품질 완성**: 현실적인 임계값 설정 및 문서화
4. **분석 완성**: 개선 권장사항 도출 및 우선순위 지정

---

## Priority 1: RAGAS 라이브러리 수정

### AC-RAGAS-001: RAGAS Import 오류 해결

**Given**: RAG 라이브러리가 프로젝트에 설치되어 있음
**When**: `quality_evaluator.py` 모듈을 임포트할 때
**Then**: `RagasEmbeddings` import 오류가 발생하지 않아야 한다

```python
# 테스트 코드
def test_ragas_import_no_error():
    from src.rag.domain.evaluation.quality_evaluator import RAGQualityEvaluator
    evaluator = RAGQualityEvaluator(use_ragas=True)
    assert evaluator is not None
    assert evaluator.ragas_available is True  # 또는 fallback 사용 중 명시
```

### AC-RAGAS-002: Fallback 메커니즘 작동

**Given**: RAGAS 라이브러리를 사용할 수 없는 상황
**When**: 평가를 실행할 때
**Then**: DeepEval 또는 커스텀 평가로 자동 전환되어야 한다

```python
# 테스트 코드
async def test_fallback_mechanism():
    evaluator = RAGQualityEvaluator(use_ragas=True)
    # RAGAS unavailable 시뮬레이션
    evaluator._ragas_available = False

    result = await evaluator.evaluate(
        query="테스트 질의",
        answer="테스트 답변",
        contexts=["컨텍스트 1"]
    )

    assert result is not None
    assert result.metadata.get("fallback_used") is True
```

### AC-RAGAS-003: 실제 평가 활성화

**Given**: OpenAI API 키가 설정되어 있음
**When**: `enable_real_evaluation=True`로 평가기를 초기화할 때
**Then**: Mock 평가 대신 실제 LLM-as-Judge 평가가 실행되어야 한다

```python
# 테스트 코드
async def test_real_evaluation_enabled():
    os.environ["OPENAI_API_KEY"] = "test-key"

    evaluator = RAGQualityEvaluator(
        enable_real_evaluation=True,
        judge_api_key="test-key"
    )

    assert evaluator.enable_real_evaluation is True
    assert evaluator.evaluation_mode == "real"
```

### AC-RAGAS-004: Mock vs 실제 결과 구분

**Given**: 평가 결과가 생성됨
**When**: 결과 메타데이터를 확인할 때
**Then**: 평가 타입(real/mock)이 명확히 식별되어야 한다

```python
# 테스트 코드
async def test_evaluation_type_identification():
    # Mock 평가
    mock_result = await evaluator.evaluate(...)
    assert mock_result.metadata["evaluation_type"] == "mock"

    # 실제 평가
    real_result = await evaluator.evaluate_real(...)
    assert real_result.metadata["evaluation_type"] == "real_llm_judge"
```

### AC-RAGAS-005: API 키 미설정 처리

**Given**: OpenAI API 키가 설정되지 않음
**When**: 실제 평가를 시도할 때
**Then**: 명시적 경고 메시지와 함께 Mock 모드로 fallback되어야 한다

```python
# 테스트 코드
async def test_missing_api_key_handling():
    os.environ["OPENAI_API_KEY"] = ""

    evaluator = RAGQualityEvaluator(enable_real_evaluation=True)

    # 경고 로그 확인
    with pytest.warns(UserWarning, match="No judge API key"):
        assert evaluator.enable_real_evaluation is False
```

---

## Priority 2: 임계값 재설정

### AC-THRESH-001: 단계별 임계값 정의

**Given**: 실제 평가 결과가 존재함
**When**: 임계값을 설정할 때
**Then**: Initial, Intermediate, Target 세 단계로 정의되어야 한다

```python
# 테스트 코드
def test_phased_thresholds_defined():
    from src.rag.domain.evaluation.threshold_config import PHASE_THRESHOLDS

    assert "initial" in PHASE_THRESHOLDS
    assert "intermediate" in PHASE_THRESHOLDS
    assert "target" in PHASE_THRESHOLDS

    # 각 단계가 모든 지표를 포함
    for phase in PHASE_THRESHOLDS.values():
        assert "faithfulness" in phase
        assert "answer_relevancy" in phase
        assert "contextual_precision" in phase
        assert "contextual_recall" in phase
```

### AC-THRESH-002: 임계값 단계적 향상

**Given**: 세 단계 임계값이 정의됨
**When**: 단계를 비교할 때
**Then**: Initial < Intermediate < Target 순서여야 한다

```python
# 테스트 코드
def test_threshold_progression():
    thresholds = PHASE_THRESHOLDS

    for metric in ["faithfulness", "answer_relevancy",
                   "contextual_precision", "contextual_recall"]:
        initial = thresholds["initial"][metric]
        intermediate = thresholds["intermediate"][metric]
        target = thresholds["target"][metric]

        assert initial < intermediate < target, \
            f"{metric} thresholds not progressive"
```

### AC-THRESH-003: 현실적인 초기 임계값

**Given**: 실제 평가 결과 평균이 0.515임
**When**: Initial 임계값을 설정할 때
**Then**: 현재 성능보다 높지만 달성 가능한 수준이어야 한다 (0.6-0.7)

```python
# 테스트 코드
def test_initial_thresholds_realistic():
    current_performance = 0.515
    initial_thresholds = PHASE_THRESHOLDS["initial"]

    for metric, threshold in initial_thresholds.items():
        # 현재 성능보다 높지만 달성 가능
        assert threshold > current_performance
        assert threshold <= current_performance + 0.2
```

### AC-THRESH-004: 임계값 문서화

**Given**: 임계값이 설정됨
**When**: 문서를 확인할 때
**Then**: 각 임계값의 설정 근거가 문서화되어야 한다

**문서화 요구사항**:
```markdown
### Faithfulness 임계값

- **Initial (0.70)**: 현재 0.502에서 0.20 향상, 할루시네이션 위험 감소
- **Intermediate (0.80)**: 산업 표준 수준, 대부분 주장 지원
- **Target (0.90)**: 매우 엄격한 기준, 거의 모든 주장 지원

**설정 근거**:
- 현재 Mock 평가 0.502는 실제 평가 시 상승 예상
- RAGAS 벤치마크 평균: 0.75-0.85
- 제품 서비스 권장 수준: 0.90+
```

### AC-THRESH-005: 임계값 동적 조정

**Given**: 평가 결과가 임계값을 초과 달성함
**When**: 다음 평가 주기가 올 때
**Then**: 상향 조정된 임계값이 적용되어야 한다

```python
# 테스트 코드
def test_threshold_adjustment():
    calibrator = ThresholdCalibrator()

    # 모든 지표가 목표 초과 달성
    current_results = {
        "faithfulness": 0.95,
        "answer_relevancy": 0.90,
        "contextual_precision": 0.85,
        "contextual_recall": 0.85
    }

    new_thresholds = calibrator.suggest_adjustment(
        current_results,
        current_phase="target"
    )

    # 상향 조정 확인
    assert all(new_thresholds[m] > current_thresholds["target"][m]
               for m in new_thresholds)
```

---

## Priority 3: 실제 평가 실행 및 분석

### AC-EVAL-001: 30개 시나리오 실제 평가

**Given**: 파일럿 테스트의 30개 시나리오가 존재함
**When**: 실제 평가를 실행할 때
**Then**: 모든 시나리오에 대해 실제 LLM-as-Judge 평가가 완료되어야 한다

```python
# 테스트 코드
async def test_all_scenarios_evaluated():
    scenarios = load_pilot_test_scenarios()
    assert len(scenarios) == 30

    results = []
    for scenario in scenarios:
        result = await evaluator.evaluate(
            query=scenario["query"],
            answer=scenario["answer"],
            contexts=scenario["contexts"]
        )
        results.append(result)

    assert len(results) == 30
    assert all(r.metadata["evaluation_type"] == "real" for r in results)
```

### AC-EVAL-002: Mock vs 실제 결과 비교

**Given**: Mock 결과와 실제 결과가 존재함
**When**: 비교 보고서를 생성할 때
**Then**: 점수 차이, 변화 추이, 통계적 유의성이 분석되어야 한다

```python
# 테스트 코드
def test_mock_vs_real_comparison():
    mock_results = load_mock_results()
    real_results = load_real_results()

    comparison = compare_evaluation_results(mock_results, real_results)

    assert comparison["score_difference"] is not None
    assert comparison["statistical_significance"] is not None
    assert comparison["improvement_percentage"] is not None

    # 보고서 생성 확인
    report = generate_comparison_report(comparison)
    assert "mock_vs_real" in report
    assert len(report) > 0
```

### AC-EVAL-003: 페르소나별 성능 분석

**Given**: 6개 페르소나(freshman, graduate, professor, staff, parent, international) 결과가 존재함
**When**: 페르소나별 분석을 실행할 때
**Then**: 각 페르소나의 평균 점수, 표준편차, 문제 영역이 식별되어야 한다

```python
# 테스트 코드
def test_persona_performance_analysis():
    analyzer = EvaluationAnalyzer()
    results = load_real_results()

    persona_analysis = analyzer.analyze_by_persona(results)

    expected_personas = ["freshman", "graduate", "professor",
                        "staff", "parent", "international"]

    for persona in expected_personas:
        assert persona in persona_analysis
        assert "mean_score" in persona_analysis[persona]
        assert "std_dev" in persona_analysis[persona]
        assert "weaknesses" in persona_analysis[persona]
```

### AC-EVAL-004: 카테고리별 성능 분석

**Given**: Simple, Complex, Edge 카테고리 결과가 존재함
**When**: 카테고리별 분석을 실행할 때
**Then**: 각 카테고리의 평균 점수, 주요 실패 원인이 식별되어야 한다

```python
# 테스트 코드
def test_category_performance_analysis():
    analyzer = EvaluationAnalyzer()
    results = load_real_results()

    category_analysis = analyzer.analyze_by_category(results)

    expected_categories = ["simple", "complex", "edge"]

    for category in expected_categories:
        assert category in category_analysis
        assert "mean_score" in category_analysis[category]
        assert "main_failures" in category_analysis[category]
```

### AC-EVAL-005: 저조 성과 시나리오 식별

**Given**: 평가 결과가 존재함
**When**: 하위 20% 시나리오를 분석할 때
**Then**: 공통 실패 패턴이 식별되어야 한다

```python
# 테스트 코드
def test_low_performance_identification():
    analyzer = EvaluationAnalyzer()
    results = load_real_results()

    bottom_20 = analyzer.identify_low_performance(results, percentile=20)

    assert len(bottom_20) > 0
    assert all("failure_patterns" in r for r in bottom_20)

    # 공통 패턴 추출
    common_patterns = analyzer.extract_common_patterns(bottom_20)
    assert len(common_patterns) > 0
```

---

## Priority 4: RAG 파이프라인 개선 제안

### AC-IMPROVE-001: 할루시네이션 방지 제안

**Given**: Faithfulness 점수가 0.7 미만임
**When**: 개선 제안을 생성할 때
**Then**: temperature 감소, 컨텍스트 고집 강화 등 구체적 제안이 포함되어야 한다

```python
# 테스트 코드
def test_hallucination_prevention_suggestions():
    suggester = ImprovementSuggester()
    results = create_low_faithfulness_results()

    suggestions = suggester.analyze_results(results)

    faithfulness_suggestions = [s for s in suggestions
                                if s.issue_type == "hallucination"]

    assert len(faithfulness_suggestions) > 0

    suggestion = faithfulness_suggestions[0]
    assert "reduce_temperature" in suggestion.recommendation
    assert "context_adherence" in suggestion.recommendation
    assert suggestion.expected_improvement > 0
```

### AC-IMPROVE-002: 검색 정확도 개선 제안

**Given**: Contextual Precision 점수가 0.7 미만임
**When**: 개선 제안을 생성할 때
**Then**: 재순위 임계값 조정, top_k 증가 등 구체적 제안이 포함되어야 한다

```python
# 테스트 코드
def test_retrieval_improvement_suggestions():
    suggester = ImprovementSuggester()
    results = create_low_precision_results()

    suggestions = suggester.analyze_results(results)

    precision_suggestions = [s for s in suggestions
                            if s.issue_type == "irrelevant_retrieval"]

    assert len(precision_suggestions) > 0

    suggestion = precision_suggestions[0]
    assert "reranker_threshold" in suggestion.recommendation
    assert "top_k" in suggestion.recommendation
```

### AC-IMPROVE-003: 우선순위 지정

**Given**: 여러 개선 제안이 존재함
**When**: 우선순위를 지정할 때
**Then**: 영향력(impact)과 노력(effort)을 기반으로 정렬되어야 한다

```python
# 테스트 코드
def test_suggestion_prioritization():
    suggester = ImprovementSuggester()
    results = load_mixed_performance_results()

    suggestions = suggester.analyze_results(results)

    # 우선순위 확인
    for i in range(len(suggestions) - 1):
        current = suggestions[i]
        next_suggestion = suggestions[i + 1]

        # 영향력 순 정렬 확인
        assert current.impact_score >= next_suggestion.impact_score
```

### AC-IMPROVE-004: 예상 영향 제공

**Given**: 개선 제안이 생성됨
**When**: 제안을 확인할 때
**Then**: 각 제안의 예상 영향(점수 향상)이 포함되어야 한다

```python
# 테스트 코드
def test_expected_impact_provided():
    suggester = ImprovementSuggester()
    results = load_low_performance_results()

    suggestions = suggester.analyze_results(results)

    for suggestion in suggestions:
        assert suggestion.expected_impact is not None
        assert suggestion.expected_impact > 0
        assert "+" in str(suggestion.expected_impact)  # "+0.15" 형식
```

### AC-IMPROVE-005: A/B 테스트 계획

**Given**: 개선 제안이 승인됨
**When**: A/B 테스트를 계획할 때
**Then**: 실험 설계, 샘플 크기, 통계적 유의성 기준이 포함되어야 한다

```python
# 테스트 코드
def test_ab_test_plan():
    planner = ABTestPlanner()
    suggestion = create_improvement_suggestion()

    plan = planner.create_test_plan(suggestion)

    assert plan["hypothesis"] is not None
    assert plan["sample_size"] > 0
    assert plan["statistical_significance"] == 0.95
    assert plan["test_duration_days"] > 0
```

---

## Definition of Done

### 기술적 완성 기준

- [ ] RAGAS import 오류 해결 및 테스트 통과
- [ ] Fallback 메커니즘 구현 및 테스트 통과
- [ ] 실제 LLM-as-Judge 평가 30개 시나리오 실행 완료
- [ ] Mock vs 실제 결과 비교 보고서 작성
- [ ] 단계별 임계값 문서화 완료
- [ ] 페르소나/카테고리별 성능 분석 완료
- [ ] 개선 권장사항 목록 작성 및 우선순위 지정

### 품질 기준

- [ ] 모든 단위 테스트 통과 (85%+ 커버리지)
- [ ] 통합 테스트 통과
- [ ] LSP 오류 0개
- [ ] 린트 오류 0개
- [ ] 코드 리뷰 완료

### 문서화 기준

- [ ] API 문서화 완료
- [ ] 임계값 설정 근거 문서화
- [ ] 성능 분석 보고서 작성
- [ ] 개선 권장사항 문서 작성
- [ ] A/B 테스트 계획 작성

### 검증 기준

- [ ] 실제 평가 결과로 목표 임계값 달성 가능성 확인
- [ ] 개선 제안의 예상 영향 타당성 검증
- [ ] stakeholder 승인 획득

---

## Test Scenarios

### 시나리오 1: RAGAS 수정 후 첫 실제 평가

**Given**:
- RAGAS import 오류가 수정됨
- OpenAI API 키가 설정됨

**When**:
- 30개 시나리오에 대해 실제 평가 실행

**Then**:
- Import 오류 없이 평가 완료
- 모든 결과가 "real" 타입으로 식별
- 평균 점수가 0.5 이상으로 측정됨 (Mock 0.515 vs 실제)

### 시나리오 2: 페르소나별 성능 차이 발견

**Given**:
- 실제 평가 결과가 존재함

**When**:
- 페르소나별 분석 실행

**Then**:
- 각 페르소나의 강점/약점 식별
- 특정 페르소나(예: professor)에 맞춘 프롬프트 최적화 제안

### 시나리오 3: 임계값 단계적 달성

**Given**:
- Initial 임계값 설정됨

**When**:
- 첫 번째 개선 적용 후 재평가

**Then**:
- Initial 임계값 달성 확인
- Intermediate 임계값으로 조정
- 다음 개선 사항 제안

---

## Acceptance Testing Process

### 1단계: 단위 테스트
```bash
pytest tests/evaluation/test_quality_evaluator.py -v
pytest tests/evaluation/test_threshold_config.py -v
```

### 2단계: 통합 테스트
```bash
pytest tests/evaluation/test_evaluation_pipeline.py -v
```

### 3단계: 실제 평가 실행
```bash
python scripts/real_evaluation_test.py --scenarios 30
```

### 4단계: 결과 검증
```bash
python scripts/validate_results.py --results data/evaluations/real_evaluation_*.json
```

### 5단계: stakeholder 승인
- 성능 분석 보고서 검토
- 개선 권장사항 승인
- A/B 테스트 계획 승인

---

**인수 상태**: 대기 중
**다음 단계**: /moai:2-run SPEC-RAG-EVAL-001
**예상 완료**: 2026-02-28
