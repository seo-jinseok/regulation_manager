# SPEC-RAG-EVAL-001: 파일럿 테스트 결과 기반 RAG 평가 시스템 개선

## TAG BLOCK

```yaml
spec_id: SPEC-RAG-EVAL-001
title: 파일럿 테스트 결과 기반 RAG 평가 시스템 개선
status: Planned
priority: Critical
created: 2026-02-07
assigned: manager-ddd
lifecycle: spec-anchored
estimated_effort: 3 weeks
labels: [rag, evaluation, ragas, quality-improvement, root-cause-analysis]
related_specs: [SPEC-RAG-QUALITY-001, SPEC-RAG-002]
```

## Environment

### 현재 시스템 상황

**프로젝트**: 대학 규정 관리 시스템 (University Regulation Manager)

**기술 스택**:
- Python 3.11+
- RAGAS >= 0.1.0 (LLM-as-Judge 평가 프레임워크)
- ChromaDB >= 1.4.0 (벡터 데이터베이스)
- BGE Reranker (BAAI/bge-reranker-v2-m3)
- pytest >= 7.4.0

**파일럿 테스트 결과** (2026-02-07):
- 총 시나리오: 30개 (6 페르소나 × 5 시나리오)
- 통과: 0개
- 실패: 30개
- 합격률: 0.0%
- 평균 점수: 0.515 (목표 0.80)

### 범위

**포함**:
- RAGAS 라이브러리 버전 호환성 수정
- Mock 평가를 실제 LLM-as-Judge 평가로 전환
- Faithfulness 점수 개선 (할루시네이션 방지)
- 검색 정확도 개선 (Contextual Precision/Recall)
- 임계값 재설정 및 보정

**제외**:
- 새로운 평가 프레임워크 개발
- 규정 데이터 수정
- UI 개발

## Assumptions

### 기술적 가정

- **높은 신뢰도**: RAGAS 라이브러리 버전 호환성 문제로 인해 import 오류 발생
- **높은 신뢰도**: 현재 Mock 평가 결과는 실제 성능을 반영하지 않음
- **중간 신뢰도**: 실제 LLM-as-Judge 평가 시 점수가 상승할 것으로 예상
- **증거**: `quality_evaluator.py:17`에서 `RagasEmbeddings` import 오류 확인

### 비즈니스 가정

- **높은 신뢰도**: 현재 평가 시스템이 실제 RAG 품질을 정확히 측정하지 못함
- **높은 신뢰도**: 합격률 0%는 평가 시스템 오류일 가능성이 높음
- **중간 신뢰도**: 실제 평가 후에도 개선이 필요할 것으로 예상

### 통합 가정

- **높은 신락도**: OpenAI API 키가 설정되어 있어 GPT-4o를 Judge 모델로 사용 가능
- **위험**: 잘못된 경우 API 비용 과다 발생 가능
- **검증 방법**: 소규모 테스트 후 비용 확인

## Root Cause Analysis

### 문제 1: RAGAS 라이브러리 Import 오류

**발생 위치**: `src/rag/domain/evaluation/quality_evaluator.py:17`

**오류 내용**:
```
cannot import name 'RagasEmbeddings' from 'ragas.embeddings'
```

**근본 원인**:
1. RAGAS 라이브러리 버전 호환성 문제
2. `RagasEmbeddings` 클래스가 RAGAS 0.1.x 버전에서 제거되거나 이름이 변경됨
3. 현재 코드가 구버전 RAGAS API를 참조

**영향**:
- Mock 평가로 대체되어 실제 LLM-as-Judge 평가 불가
- 모든 점수가 고정값 0.5로 반환
- 실제 RAG 품질 측정 불가

### 문제 2: 일관되게 낮은 점수 (0.515 vs 0.80 목표)

**관찰된 패턴**:
- Faithfulness: 0.502 (목표 0.90)
- Answer Relevancy: 0.557 (목표 0.85)
- Contextual Precision: 0.500 (목표 0.80)
- Contextual Recall: 0.500 (목표 0.80)

**근본 원인 분석**:

**가설 1: Mock 평가 영향** (가장 유력)
- 모든 점수가 0.5 부근에 집중
- 실제 평가가 아닌 Mock 반환값 사용
- RAGAS import 실패로 fallback 모델 동작

**가설 2: 임계값 설정 과도**
- Faithfulness 0.90은 매우 엄격한 기준
- 산업 표준: 0.70-0.80 수준
- 현재 시스템 수준을 고려하지 않은 설정

**가설 3: 실제 RAG 파이프라인 문제**
- 검색 관련성 부족 (Contextual Precision 0.500)
- 답변 불완전성 (Contextual Recall 0.500)
- 할루시네이션 위험 (Faithfulness 0.502)

### 문제 3: 페르소나별 성능 차이 미미

**관찰**:
- 모든 페르소나 평균 점수: 0.500-0.530 범위
- 유의미한 차이 없음

**근본 원인**:
- Mock 평가로 인한 실제 차이 반영 안 됨
- 페르소나별 프롬프트 최적화 미흡

## Requirements

### Priority 1: RAGAS 라이브러리 수정 (주 1)

#### Ubiquitous Requirements

**REQ-RAGAS-001**: 시스템은 RAGAS 라이브러리와 호환되는 평가 프레임워크를 제공해야 한다.

**REQ-RAGAS-002**: 시스템은 RAGAS import 오류를 우회하는 fallback 메커니즘을 제공해야 한다.

**REQ-RAGAS-003**: 시스템은 실제 LLM-as-Judge 평가를 실행하여 신뢰할 수 있는 점수를 생성해야 한다.

#### Event-Driven Requirements

**REQ-RAGAS-004**: WHEN RAGAS import 실패 시, 시스템 SHALL DeepEval fallback으로 자동 전환해야 한다.

**REQ-RAGAS-005**: WHEN 평가 프레임워크 선택 안 됨 시, 시스템 SHALL 사용 가능한 프레임워크를 자동 감지해야 한다.

**REQ-RAGAS-006**: WHEN 실제 평가 완료 시, 시스템 SHALL Mock 결과와 실제 결과를 비교 보고해야 한다.

#### State-Driven Requirements

**REQ-RAGAS-007**: IF RAGAS 라이브러리 사용 가능 시, 시스템 SHALL RAGAS를 주요 평가 프레임워크로 사용해야 한다.

**REQ-RAGAS-008**: IF 모든 평가 프레임워크 unavailable 시, 시스템 SHALL 명시적 오류 메시지와 함께 실패 처리해야 한다.

**REQ-RAGAS-009**: IF LLM API 키 미설정 시, 시스템 SHALL API 설정 필요성을 알리고 Mock 모드 사용을 확인해야 한다.

#### Unwanted Behavior Requirements

**REQ-RAGAS-010**: 시스템은 NOT RAGAS import 오류를 무음 처리해서는 안 된다.

**REQ-RAGAS-011**: 시스템은 NOT 사용자에게 Mock 평가 결과를 실제 결과로 표시해서는 안 된다.

---

### Priority 2: 임계값 재설정 (주 1)

#### Ubiquitous Requirements

**REQ-THRESH-001**: 시스템은 실제 RAG 성능을 반영하는 현실적인 임계값을 사용해야 한다.

**REQ-THRESH-002**: 시스템은 평가 지표별로 다른 임계값을 적용해야 한다.

**REQ-THRESH-003**: 시스템은 산업 표준을 기반으로 임계값을 설정해야 한다.

#### Event-Driven Requirements

**REQ-THRESH-004**: WHEN 실제 평가 첫 실행 후, 시스템 SHALL 현재 성능 수준을 기반으로 임계값을 제안해야 한다.

**REQ-THRESH-005**: WHEN 임계값 설정 변경 시, 시스템 SHALL 변경 사유와 영향을 문서화해야 한다.

**REQ-THRESH-006**: WHEN 목표 임계값 도달 시, 시스템 SHALL 점진적 향상을 위한 새로운 목표를 설정해야 한다.

#### State-Driven Requirements

**REQ-THRESH-007**: IF 실제 평가 결과 없음 시, 시스템 SHALL 보수적인 초기 임계값을 사용해야 한다.

**REQ-THRESH-008**: IF 일부 지표만 목표 달성 시, 시스템 SHALL 달성되지 않은 지표에 집중해야 한다.

**REQ-THRESH-009**: IF 모든 지표가 목표 초과 달성 시, 시스템 SHALL 임계값을 상향 조정해야 한다.

#### Optional Requirements

**REQ-THRESH-010**: 가능한 경우, 시스템 MAY 사용자 정의 임계값을 지원해야 한다.

**REQ-THRESH-011**: 가능한 경우, 시스템 MAY 시나리오별로 다른 임계값을 지원해야 한다.

---

### Priority 3: 실제 평가 실행 및 분석 (주 1)

#### Ubiquitous Requirements

**REQ-EVAL-001**: 시스템은 30개 시나리오에 대해 실제 LLM-as-Judge 평가를 실행해야 한다.

**REQ-EVAL-002**: 시스템은 평가 결과를 상세 분석하여 개선 필요 영역을 식별해야 한다.

**REQ-EVAL-003**: 시스템은 페르소나별 성능 차이를 분석해야 한다.

**REQ-EVAL-004**: 시스템은 카테고리별(Simple/Complex/Edge) 성능 차이를 분석해야 한다.

#### Event-Driven Requirements

**REQ-EVAL-005**: WHEN 실제 평가 완료 시, 시스템 SHALL Mock 결과와의 차이를 보고해야 한다.

**REQ-EVAL-006**: WHEN Faithfulness 점수 < 0.7 시, 시스템 SHALL 할루시네이션 방지 프롬프트 개선을 제안해야 한다.

**REQ-EVAL-007**: WHEN Contextual Precision 점수 < 0.7 시, 시스템 SHALL 검색 파라미터 튜닝을 제안해야 한다.

**REQ-EVAL-008**: WHEN Answer Relevancy 점수 < 0.7 시, 시스템 SHALL 답변 생성 프롬프트 개선을 제안해야 한다.

#### State-Driven Requirements

**REQ-EVAL-009**: IF 평가 점수가 여전히 낮음 (< 0.6) 시, 시스템 SHALL RAG 파이프라인 전체 검토를 권장해야 한다.

**REQ-EVAL-010**: IF 특정 페르소나만 낮은 점수 시, 시스템 SHALL 페르소나별 프롬프트 최적화를 제안해야 한다.

**REQ-EVAL-011**: IF 특정 카테고리만 낮은 점수 시, 시스템 SHALL 해당 카테고리 쿼리 패턴을 분석해야 한다.

#### Optional Requirements

**REQ-EVAL-012**: 가능한 경우, 시스템 MAY 평가 결과를 시각화하여 제공해야 한다.

**REQ-EVAL-013**: 가능한 경우, 시스템 MAY 이전 평가와 비교 추이를 보여줘야 한다.

---

### Priority 4: RAG 파이프라인 개선 제안 (주 2-3)

#### Ubiquitous Requirements

**REQ-IMPROVE-001**: 시스템은 평가 결과 기반 RAG 파이프라인 개선 권장사항을 제공해야 한다.

**REQ-IMPROVE-002**: 시스템은 개선 권장사항을 우선순위별로 정렬해야 한다.

**REQ-IMPROVE-003**: 시스템은 각 권장사항의 예상 영향을 제공해야 한다.

#### Event-Driven Requirements

**REQ-IMPROVE-004**: WHEN 할루시네이션 발견 시, 시스템 SHALL temperature 감소 및 컨텍스트 고집 강화를 제안해야 한다.

**REQ-IMPROVE-005**: WHEN 검색 관련성 낮음 시, 시스템 SHALL 재순위 임계값 조정을 제안해야 한다.

**REQ-IMPROVE-006**: WHEN 답변 불완전 시, 시스템 SHALL top_k 증가 및 컨텍스트 윈도우 확대를 제안해야 한다.

#### State-Driven Requirements

**REQ-IMPROVE-007**: IF 여러 개선 사항 있음 시, 시스템 SHALL 순차적 적용을 권장해야 한다.

**REQ-IMPROVE-008**: IF 개선 적용 후 시, 시스템 SHALL 재평가하여 영향을 측정해야 한다.

**REQ-IMPROVE-009**: IF 개선 효과 없음 시, 시스템 SHALL 대안을 제안해야 한다.

#### Optional Requirements

**REQ-IMPROVE-010**: 가능한 경우, 시스템 MAY A/B 테스트를 위한 실험 계획을 제안해야 한다.

**REQ-IMPROVE-011**: 가능한 경우, 시스템 MAY 자동 개선 적용 옵션을 제공해야 한다.

#### Unwanted Behavior Requirements

**REQ-IMPROVE-012**: 시스템은 NOT 충분한 데이터 없이 생산 파이프라인을 수정해서는 안 된다.

**REQ-IMPROVE-013**: 시스템은 NOT 상충하는 개선 권장사항을 제시해서는 안 된다.

## Specifications

### Architecture Design

#### RAGAS 라이브러리 호환성 수정 (REQ-RAGAS-001 ~ REQ-RAGAS-011)

**현재 문제 코드** (`src/rag/domain/evaluation/quality_evaluator.py:17`):
```python
# 현재 (오류 발생)
from ragas.embeddings import RagasEmbeddings
```

**수정 방안 1: RAGAS 최신 버전 API 사용**:
```python
# 수정 후 (RAGAS 0.1.x+)
try:
    from ragas.embeddings import RagasEmbeddings
    RAGAS_AVAILABLE = True
except ImportError:
    # Fallback to custom wrapper
    RAGAS_AVAILABLE = False
    logger.warning("RagasEmbeddings not available, using fallback")
```

**수정 방안 2: 직접 임베딩 래퍼 구현**:
```python
class FallbackRagasEmbeddings:
    """Fallback wrapper when RagasEmbeddings is not available"""

    def __init__(self, model_name: str = "openai"):
        self.model_name = model_name
        self._embedding_function = self._get_embedding_function()

    def _get_embedding_function(self):
        if self.model_name == "openai":
            from openai import OpenAI
            client = OpenAI()
            return lambda texts: [e.embedding for e in
                client.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small"
                ).data]
        # 추가 임베딩 모델 지원

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._embedding_function(texts)
```

**수정 방안 3: RAGAS 버전 고정**:
```toml
# pyproject.toml
[tool.poetry.dependencies]
ragas = "0.0.22"  # RagasEmbeddings가 있는 마지막 버전
```

#### 실제 LLM-as-Judge 평가 실행

**평가 파이프라인 수정**:
```python
class RAGQualityEvaluator:
    def __init__(
        self,
        judge_model: str = "gpt-4o",
        judge_api_key: Optional[str] = None,
        use_ragas: bool = True,
        enable_real_evaluation: bool = True  # NEW: 실제 평가 활성화
    ):
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key or os.getenv("OPENAI_API_KEY")
        self.use_ragas = use_ragas
        self.enable_real_evaluation = enable_real_evaluation

        # API 키 확인
        if not self.judge_api_key and enable_real_evaluation:
            logger.warning(
                "No judge API key found. "
                "Set OPENAI_API_KEY for real evaluation. "
                "Using mock evaluation."
            )
            self.enable_real_evaluation = False

        self._setup_evaluation_framework()

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = ""
    ) -> EvaluationResult:
        if not self.enable_real_evaluation:
            return self._mock_evaluate(query, answer, contexts)

        # 실제 LLM-as-Judge 평가
        return await self._real_evaluate(query, answer, contexts, ground_truth)

    async def _real_evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> EvaluationResult:
        """실제 LLM-as-Judge 평가 실행"""
        results = {}

        # Faithfulness
        results["faithfulness"] = await self._evaluate_faithfulness(
            answer, contexts
        )

        # Answer Relevancy
        results["answer_relevancy"] = await self._evaluate_relevancy(
            query, answer
        )

        # Contextual Precision
        results["contextual_precision"] = await self._evaluate_precision(
            query, contexts
        )

        # Contextual Recall
        results["contextual_recall"] = await self._evaluate_recall(
            query, contexts, ground_truth
        )

        overall = sum(results.values()) / len(results)
        passed = self._check_thresholds(results)

        return EvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            faithfulness=results["faithfulness"],
            answer_relevancy=results["answer_relevancy"],
            contextual_precision=results["contextual_precision"],
            contextual_recall=results["contextual_recall"],
            overall_score=overall,
            passed=passed,
            failure_reasons=self._get_failure_reasons(results),
            metadata={"evaluation_type": "real_llm_judge"}
        )

    async def _evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """할루시네이션 검출을 위한 Faithfulness 평가"""
        if not self.use_ragas:
            return await self._custom_faithfulness_eval(answer, contexts)

        try:
            from ragas.metrics import FaithfulnessMetric
            from ragas.llms import LLMChain

            metric = FaithfulnessMetric(
                llm=LLMChain(model=self.judge_model, api_key=self.judge_api_key)
            )

            score = await metric.evaluate(
                question="",  # Not used for faithfulness
                answer=answer,
                contexts=contexts
            )

            return score

        except Exception as e:
            logger.error(f"RAGAS faithfulness evaluation failed: {e}")
            return await self._custom_faithfulness_eval(answer, contexts)

    async def _custom_faithfulness_eval(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """RAGAS fallback 시 사용할 커스텀 Faithfulness 평가"""
        combined_context = "\n".join(contexts)

        prompt = f"""
        다음 답변이 주어진 컨텍스트를 기반으로 사실적으로 정확한지 평가하세요.

        컨텍스트:
        {combined_context}

        답변:
        {answer}

        답변의 각 주장이 컨텍스트에서 지원되는지 확인하고,
        0.0에서 1.0까지의 점수를 부여하세요 (1.0은 모든 주장이 지원됨).

        점수만 반환하세요 (0.0-1.0):
        """

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.judge_api_key)

            response = client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )

            score_str = response.choices[0].message.content.strip()
            return float(score_str)

        except Exception as e:
            logger.error(f"Custom faithfulness evaluation failed: {e}")
            return 0.5  # Fallback score
```

#### 현실적인 임계값 재설정

**현재 임계값 (과도)**:
```python
CURRENT_THRESHOLDS = {
    "faithfulness": 0.90,      # 매우 엄격
    "answer_relevancy": 0.85,  # 높음
    "contextual_precision": 0.80,
    "contextual_recall": 0.80,
}
```

**제안된 새로운 임계값**:
```python
# 단계별 목표 설정
PHASE_THRESHOLDS = {
    "initial": {      # 초기 목표 (주 1)
        "faithfulness": 0.70,
        "answer_relevancy": 0.70,
        "contextual_precision": 0.65,
        "contextual_recall": 0.65,
    },
    "intermediate": { # 중간 목표 (주 2-3)
        "faithfulness": 0.80,
        "answer_relevancy": 0.80,
        "contextual_precision": 0.75,
        "contextual_recall": 0.75,
    },
    "target": {       # 최종 목표 (주 4+)
        "faithfulness": 0.90,
        "answer_relevancy": 0.85,
        "contextual_precision": 0.80,
        "contextual_recall": 0.80,
    }
}

# Critical 임계값 (할루시네이션 위험)
CRITICAL_THRESHOLDS = {
    "faithfulness": 0.60,  # 0.60 미만 시 critical 경고
}
```

#### RAG 파이프라인 개선 권장사항

**할루시네이션 방지**:
```python
# 현재
llm_temperature = 0.7

# 제안
llm_temperature = 0.1  # 더 결정적 답변

# 프롬프트 추가
anti_hallucination_prompt = """
다음 지침을严格遵守하세요:
1. 제공된 컨텍스트에 명시된 정보만 사용하세요
2. 컨텍스트에 없는 정보는 "알 수 없음"이라고 답하세요
3. 추측하지 말고 확실한 정보만 제공하세요
"""
```

**검색 정확도 개선**:
```python
# 현재
top_k = 5
reranker_threshold = 0.3

# 제안
top_k = 10  # 더 많은 컨텍스트 검색
reranker_threshold = 0.5  # 더 엄격한 필터링

# 하이브리드 검색 가중치
hybrid_weights = {
    "dense": 0.7,  # 의미 검색
    "sparse": 0.3  # 키워드 검색
}
```

### File Structure

```
src/rag/domain/evaluation/
├── quality_evaluator.py          # MODIFIED: RAGAS 호환성 수정
├── metrics.py                    # MODIFIED: 커스텀 지표 추가
├── threshold_config.py           # NEW: 임계값 설정
└── improvement_suggester.py      # NEW: 개선 권장사항 생성기

scripts/
├── real_evaluation_test.py       # NEW: 실제 평가 실행 스크립트
└── threshold_calibrator.py       # NEW: 임계값 보정 스크립트
```

## Traceability

### Requirements to Components Mapping

| Requirement ID | Component | File |
|---------------|-----------|------|
| REQ-RAGAS-001 ~ REQ-RAGAS-011 | RAGQualityEvaluator, FallbackRagasEmbeddings | domain/evaluation/quality_evaluator.py |
| REQ-THRESH-001 ~ REQ-THRESH-011 | ThresholdConfig, ThresholdCalibrator | domain/evaluation/threshold_config.py |
| REQ-EVAL-001 ~ REQ-EVAL-013 | RealEvaluationTest, EvaluationAnalyzer | scripts/real_evaluation_test.py |
| REQ-IMPROVE-001 ~ REQ-IMPROVE-013 | ImprovementSuggester | domain/evaluation/improvement_suggester.py |

### Dependencies

**External Dependencies**:
- RAGAS (버전 고정 또는 업데이트)
- OpenAI API (GPT-4o Judge 모델)

**Internal Dependencies**:
- `domain/retrieval/hybrid_search.py`
- `domain/llm/self_rag.py`
- `application/rag/pipeline.py`

## Appendix

### Glossary

- **LLM-as-Judge**: LLM을 사용하여 다른 LLM 출력의 품질을 평가하는 방법
- **Faithfulness**: 생성된 답변과 검색된 컨텍스트 간의 사실적 일관성 측정
- **Answer Relevancy**: 답변이 원래 사용자 질의를 얼마나 잘 다루는지 측정
- **Contextual Precision**: 관련 문서가 비관련 문서보다 높은 순위로 검색되었는지 측정
- **Contextual Recall**: 관련 정보가 지식 베이스에서 모두 검색되었는지 측정
- **할루시네이션 (Hallucination)**: LLM이 근거 없는 정보를 생성하는 현상

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-07 | manager-spec | 파일럿 테스트 결과 기반 초기 SPEC 생성 |

---

**SPEC Status**: Planned
**Next Phase**: /moai:2-run SPEC-RAG-EVAL-001 (DDD로 구현)
**Estimated Completion**: 2026-02-28 (3주)
