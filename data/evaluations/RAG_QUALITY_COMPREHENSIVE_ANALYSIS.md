# RAG 시스템 품질 종합 분석 보고서
## RAG System Quality Comprehensive Analysis Report

**생성일 (Generated):** 2026-02-07 17:30:00
**평가 기간 (Evaluation Period):** 2026-01-26 ~ 2026-02-07
**분석 버전 (Analysis Version):** 1.0

---

## 📊 요약 (Executive Summary)

### 현재 상태 (Current State)

| 지표 (Metric) | 현재 값 (Current) | 목표 (Target) | Gap | 상태 (Status) |
|---------------|-------------------|----------------|-----|---------------|
| **통과율 (Pass Rate)** | 13.3% | 40.0% | -26.7% | ❌ 미달 |
| **신뢰성 (Faithfulness)** | 50.3% | 55.0% | -4.7% | ❌ 미달 |
| **답변 관련성 (Answer Relevancy)** | 71.0% | 75.0% | -4.0% | ⚠️ 근접 |
| **맥락 정밀도 (Contextual Precision)** | 54.0% | 65.0% | -11.0% | ❌ 미달 |
| **맥락 재현율 (Contextual Recall)** | 32.0% | 50.0% | -18.0% | ❌ 심각 |
| **종합 점수 (Overall)** | 52.6% | 60.0% | -7.4% | ❌ 미달 |

### 주요 발견 (Key Findings)

1. **맥락 재현율이 가장 큰 문제** (32% vs 목표 50%, -18%p)
2. **신뢰성 개선 필요** (환각 문제 지속)
3. **단순 쿼리에서는 개선 observed, 복잡한 쿼리에서는 성능 저하**
4. **6개 페르소나 중 2개만 통과율 > 0%** (Freshman: 40%, Parent: 20%)

---

## 🎯 우선순위별 개선 작업 (Priority Improvement Tasks)

다음 턴에서 AI가 바로 작업할 수 있도록 SPEC 형식으로 정리했습니다.

---

## 🔴 Priority 1: 긴급 (Critical) - 맥락 재현율 개선

### 문제 (Problem)
- 현재: 32.0% vs 목표: 50.0% (Gap: -18.0%)
- 검색된 문서에 필요한 정보가 누락됨

### 근본 원인 (Root Causes)
1. 관련 문서 검색 실패 (Retrieval Failure)
2. 규정 간 연관 정보 부재
3. 검색 쿼리와 문서 간 의미적 간극
4. Top-K=5가 충분하지 않음

---

### SPEC-RAG-RECALL-001: 검색 인덱스 개선

**상태 (Status):** TODO (Ready for Implementation)

#### Requirements

**WHEN** 사용자가 규정과 관련된 쿼리를 검색할 때
**THEN** 시스템은 관련된 모든 필수 정보를 포함하는 문서를 검색해야 함
**SHALL** 최소 50%의 Contextual Recall 점수 달성

#### Acceptance Criteria

1. **형태소 분석기 최적화**
   - [ ] 한국어 특화 형태소 분석기 적용
   - [ ] 조항 번호 패턴 (제X조, 제X항) 인식 강화
   - [ ] 규정 전용 용어 사전 구축

2. **Top-K 동적 조정**
   - [ ] 쿼리 복잡도에 따른 Top-K 조정 (5 → 10 for complex)
   - [ ] 단순 쿼리: Top-K=5
   - [ ] 복잡한 쿼리: Top-K=10
   - [ ] 엣지 쿼리: Top-K=15

3. **하이브리드 검색 강화**
   - [ ] BM25 가중치 튜닝 (현재 0.3 → 0.5로 조정 시도)
   - [ ] Dense 검색 임베딩 모델 재교육 고려
   - [ ] 검색 결과 다양성 확보

#### Implementation Guide

```python
# 파일: src/rag/infrastructure/retrieval_config.py

TOP_K_CONFIG = {
    "simple": 5,
    "complex": 10,
    "edge": 15,
    "multi_turn": 8,
}

# 파일: src/rag/domain/query/analyzer.py
def classify_query_complexity(query: str) -> str:
    """쿼리 복잡도 분류"""
    # 간단: 단일 키워드, 짧은 쿼리
    if len(query.split()) <= 3:
        return "simple"
    # 복잡: 여러 조건, 규정 비교
    if any(word in query for word in ["비교", "차이", "또한", "그리고"]):
        return "complex"
    # 엣지: 모호한 단어
    if len(query.split()) == 1:
        return "edge"
    return "simple"
```

#### Test Cases

1. **Given:** "휴학 절차가 어떻게 되나요?" (simple)
   - **Then:** Top-K=5 사용
   - **Expect:** Recall >= 0.40

2. **Given:** "휴학과 복학 규정의 차이점은?" (complex)
   - **Then:** Top-K=10 사용
   - **Expect:** Recall >= 0.50

#### Success Metrics

- 맥락 재현율: 32% → 50%+
- 복잡한 쿼리 통과율: 10% → 25%+

---

### SPEC-RAG-RECALL-002: 다중 검색 (Multi-Hop Retrieval)

**상태 (Status):** TODO

#### Requirements

**WHEN** 단일 검색으로 충분한 정보를 찾지 못할 때
**THEN** 여러 단계에 걸쳐 관련 규정을 종합 검색해야 함

#### Acceptance Criteria

1. **규정 간 참조 추적**
   - [ ] 조항 참조 ("제X조에 따르면") 자동 연계
   - [ ] 관련 규정 자동 검색

2. **다단계 검색 파이프라인**
   - [ ] 1단계: 관련 규정명 식별
   - [ ] 2단계: 구체 조항 검색
   - [ ] 3단계: 예외/특례 검색

#### Implementation Guide

```python
# 파일: src/rag/domain/retrieval/multi_hop.py

class MultiHopRetriever:
    def retrieve(self, query: str) -> List[Document]:
        # 1단계: 규정명 검색
        regulation_names = self._extract_regulation_names(query)
        # 2단계: 조항 검색
        articles = self._extract_articles(query)
        # 3단계: 결합 검색
        return self._combine_and_retrieve(regulation_names, articles)
```

---

## 🟡 Priority 2: 높음 (High) - 환각 방지

### 문제 (Problem)
- 신뢰성 50.3% vs 목표 55.0% (Gap: -4.7%)
- 검색되지 않은 문서에 대한 답변 생성 시 환각 발생

### 근본 원인 (Root Causes)
1. 불충분한 컨텍스트로 답변 생성 시도
2. "모른다" 응답 회피
3. 인용 검증 부재

---

### SPEC-RAG-HALLUCINATION-001: 인증 검증 강화

**상태 (Status):** TODO

#### Requirements

**WHEN** 검색된 문서가 충분하지 않을 때
**THEN** 시스템은 "알 수 없음" 또는 관련 정보 부족을 명시해야 함
**SHALL** 검색되지 않은 내용으로 답변하지 않음

#### Acceptance Criteria

1. **컨텍스트 충분성 확인**
   - [ ] 검색된 문서 수 < 3개일 때 "알 수 없음" 응답
   - [ ] 검색 점수 < 0.5일 때 주의 표시

2. **인용 검증**
   - [ ] 답변의 모든 규정 인용이 검색된 문서에 존재하는지 확인
   - [ ] 존재하지 않는 인용은 "확인 필요"로 표시

3. **답변 생성 프롬프트 개선**
   - [ ] "규정에 따르면" 시작 시 실제 규정 인용 강제
   - [ ] 불확실한 정보는 "확인이 필요합니다" 추가

#### Implementation Guide

```python
# 파일: src/rag/domain/generation/prompt_templates.py

STRICT_ANSWER_PROMPT = """
You are a university regulation assistant. Answer ONLY based on the retrieved contexts.

CRITICAL RULES:
1. If the answer is not in the contexts, say "죄송합니다. 해당 정보를 찾을 수 없습니다. 관련 부서에 문의해 주세요."
2. Every regulation citation MUST be from the retrieved contexts
3. Format: "규정명 제X조에 따르면..."
4. Never make up contact information

Retrieved Contexts:
{contexts}

Question: {question}
"""
```

---

## 🟢 Priority 3: 중간 (Medium) - 복잡한 쿼리 대응

### 문제 (Problem)
- 복잡한 쿼리 평균 0.486 (개선 전 0.550 감소)
- Professor/Graduate 페르소나 0% 통과율

### 근본 원인 (Root Causes)
1. 전문 용어 처리 미흡
2. 복합 조건 쿼리 이해 부족
3. 규정 간 비교 능력 부족

---

### SPEC-RAG-COMPLEX-001: 쿼리 분류 및 전략 차별화

**상태 (Status):** TODO

#### Requirements

**WHEN** 사용자가 복잡한 쿼리를 입력할 때
**THEN** 시스템은 쿼리 유형을 분류하고 적절한 검색 전략을 적용해야 함

#### Acceptance Criteria

1. **쿼리 유형 자동 분류**
   - [ ] 단순 (Simple): 단일 토픽
   - [ ] 복합 (Complex): 다중 토픽 또는 비교
   - [ ] 엣지 (Edge): 모호하거나 불완전
   - [ ] 대화 (Multi-turn): 맥락 의존

2. **분류별 검색 전략**
   - [ ] Simple: 기본 검색
   - [ ] Complex: Multi-hop 검색 + Top-K=10
   - [ ] Edge: Query Reformulation + Top-K=15

#### Implementation Guide

```python
# 파일: src/rag/domain/query/classifier.py

from enum import Enum

class QueryType(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    EDGE = "edge"
    MULTI_TURN = "multi_turn"

def classify_query(query: str) -> QueryType:
    keywords_count = len(query.split())
    has_comparison = any(word in query for word in ["비교", "차이", "차이점"])
    has_conjunction = any(word in query for word in ["그리고", "또한", "및"])

    if keywords_count == 1:
        return QueryType.EDGE
    elif has_comparison or has_conjunction:
        return QueryType.COMPLEX
    else:
        return QueryType.SIMPLE
```

---

### SPEC-RAG-COMPLEX-002: 전문 용어 사전 구축

**상태 (Status):** TODO

#### Requirements

**WHEN** 전문 용어가 포함된 쿼리가 입력될 때
**THEN** 시스템은 전문 용어를 인식하고 적절히 확장해야 함

#### Acceptance Criteria

1. **전문 용어 사전**
   - [ ] 교원 관련: 승진, 정년, 연구년, 휴직
   - [ ] 직원 관련: 연차, 복무, 급여
   - [ ] 학생 관련: 휴학, 복학, 자퇴, 제적

2. **약어/동의어 확장**
   - [ ] "휴학" → "휴학(休學)", "휴학신청", "휴학절차"
   - [ ] "성적" → "성적정정", "성적이의신청", "성적포기"

#### Implementation Guide

```python
# 파일: src/rag/domain/query/expansion.py

TERMINOLOGY_DICT = {
    "휴학": ["휴학(休學)", "휴학신청", "휴학절차", "휴학신청서"],
    "복학": ["복학(復學)", "복학신청", "재입학"],
    "승진": ["승진(昇進)", "정승", "승진심사"],
}

def expand_query_with_terminology(query: str) -> List[str]:
    """전문 용어 확장"""
    expanded_queries = [query]
    for term, synonyms in TERMINOLOGY_DICT.items():
        if term in query:
            for synonym in synonyms:
                expanded_queries.append(query.replace(term, synonym))
    return expanded_queries
```

---

## 🔵 Priority 4: 낮음 (Low) - 평가 시스템 개선

---

### SPEC-RAG-EVAL-001: 평가 데이터셋 확장

**상태 (Status):** TODO

#### Requirements

**WHEN** RAG 시스템 개선 후 효과를 검증할 때
**THEN** 충분한 크기의 평가 데이터셋을 사용해야 함

#### Acceptance Criteria

1. **데이터셋 규모**
   - [ ] 현재 30개 → 150개 이상 (rag-quality-local 스킬 목표)
   - [ ] 6개 페르소나 × 6개 카테고리 × 4-5개 시나리오

2. **시나리오 다양성**
   - [ ] 단일 턴 (30개)
   - [ ] 다중 턴 (20개)
   - [ ] 엣지 케이 (40개)
   - [ ] 도메인 특화 (25개)
   - [ ] 적대적 (10개)

3. **실제 사용 로그 기반**
   - [ ] 실제 사용자 쿼리 로그 분석
   - [ ] 자주 검색어 Top 100 반영

---

## 📋 개선 작업 실행 가이드 (Implementation Guide for Next Turn)

### 1단계: 맥락 재현율 개선 (가장 높은 우선순위)

```bash
# 1. 쿼리 복잡도 분류기 구현
cd /Users/truestone/Dropbox/repo/University/regulation_manager
# 파일: src/rag/domain/query/classifier.py 생성

# 2. Top-K 동적 조정 구현
# 파일: src/rag/infrastructure/retrieval_config.py 수정

# 3. 테스트
python test_scenarios/rag_quality_evaluator.py --use-llm-judge --limit 5
```

### 2단계: 환각 방지

```bash
# 1. 답변 생성 프롬프트 개선
# 파일: src/rag/domain/generation/prompt_templates.py 수정

# 2. 인용 검증 시스템 구현
# 파일: src/rag/domain/validation/citation_checker.py 생성

# 3. 테스트
python test_scenarios/rag_quality_evaluator.py --use-llm-judge
```

### 3단계: 복잡한 쿼리 대응

```bash
# 1. 전문 용어 사전 구축
# 파일: src/rag/domain/query/expansion.py 생성

# 2. 쿼리 분류기 구현
# 파일: src/rag/domain/query/classifier.py 수정

# 4. 테스트
python test_scenarios/rag_quality_evaluator.py --use-llm-judge --parallel
```

---

## 📈 예상 개선 효과 (Expected Improvements)

| 개선 작업 | 기대 효과 | 작업 난이도 | 예상 소요 시간 |
|----------|----------|------------|--------------|
| **Top-K 동적 조정** | Recall +10~15% | 쉬움 | 1-2시간 |
| **형태소 분석기 최적화** | Precision +5~10% | 보통 | 2-4시간 |
| **환각 방지 프롬프트** | Faithfulness +10~15% | 쉬움 | 1-2시간 |
| **전문 용어 사전** | Complex Query +15~20% | 어려움 | 4-8시간 |
| **Multi-hop Retrieval** | Recall +20~25% | 어려움 | 8-16시간 |

---

## 🎯 다음 턴에서 바로 시작할 수 있는 작업 (Quick Start for Next Turn)

### 1. Top-K 동적 조정 (가장 빠름, 1-2시간)

```python
# 수정 파일: src/rag/interface/query_handler.py

def get_top_k_for_query(query: str) -> int:
    """쿼리 복잡도에 따른 Top-K 결정"""
    words = query.split()

    if len(words) <= 3:
        return 5  # 단순
    elif any(w in query for w in ["비교", "차이", "그리고", "또한"]):
        return 10  # 복잡
    else:
        return 7  # 중간
```

### 2. 환각 방지 프롬프트 개선 (가장 빠름, 1시간)

```python
# 수정 파일: src/rag/domain/generation/rag_prompt.py

RAG_ANSWER_PROMPT_STRICT = """
당신은 대학교 규정 전문가입니다. 검색된 문서를 바탕으로만 답변하세요.

**중요 규칙:**
1. 검색된 문서에 정보가 없으면 "죄송합니다. 해당 정보를 찾을 수 없습니다. 관련 부서에 문의해 주세요."라고 답변하세요.
2. 모든 규정 인용은 "규정명 제X조" 형식을 따르세요.
3. 전화번호나 연락처를 절대로 만들어내지 마세요.
4. 불확실한 정보는 "확인이 필요합니다"라고 표시하세요.

검색된 문서:
{contexts}

질문: {question}
"""
```

---

## 📊 성공 지표 (Success Criteria)

모든 개선 작업 완료 후 기대 성과:

| 지표 | 현재 | 목표 (Phase 1) | 목표 (Phase 2) |
|------|------|-----------------|-----------------|
| 통과율 | 13.3% | 30% | 50% |
| 신뢰성 | 50.3% | 60% | 75% |
| 맥락 재현율 | 32.0% | 45% | 60% |
| 종합 점수 | 52.6% | 60% | 75% |

---

## 🔗 관련 파일 (Related Files)

### 평가 결과
- `data/evaluations/custom_llm_judge_eval_stage1_latest.json` - 최신 평가 데이터
- `data/evaluations/final_evaluation_report_20260207_171420.md` - 비교 보고서
- `data/evaluations/llm_judge_evaluation_summary.md` - 요약 보고서

### 스킬 문서
- `.claude/skills/rag-quality-local/SKILL.md` - 스킬 정의
- `.claude/skills/rag-quality-local/modules/evaluation.md` - 평가 프롬프트
- `.claude/skills/rag-quality-local/modules/metrics.md` - 메트릭 정의

### 구현 파일
- `src/rag/domain/evaluation/llm_judge.py` - LLM-as-Judge 구현
- `src/rag/domain/evaluation/parallel_evaluator.py` - 병렬 평가기
- `test_scenarios/rag_quality_evaluator.py` - 평가 실행 스크립트

---

## 💡 다음 턴 AI를 위한 팁 (Tips for Next Turn AI)

1. **이 보고서를 활용하여** SPEC 형식으로 정의된 작업을 순차적으로 진행하세요
2. **가장 빠르고 효과가 큰 작업**부터 시작하세요 (Top-K 동적 조정, 환각 방지 프롬프트)
3. **각 작업 완료 후 평가**를 실행하여 개선 효과를 확인하세요
4. **목표 달성 시까지** 반복적인 개선-평가 사이클을 진행하세요

---

**보고서 생성자:** Claude Code MoAI System
**분석 기간:** 2026-01-26 ~ 2026-02-07
**다음 개선 주기:** Phase 1 improvements 후 재평가 예정
