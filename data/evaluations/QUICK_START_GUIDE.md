# RAG 시스템 개선 빠른 시작 가이드
## Quick Start Guide for RAG System Improvements

**목적 (Purpose):** 다음 턴 AI가 즉시 시작할 수 있는 구체적인 구현 가이드

---

## 🚀 5분 만에 구현 가능한 개선 (Quick Wins)

### 1. Top-K 동적 조정 (3분)

**파일:** `src/rag/interface/query_handler.py`

```python
def get_top_k_for_query(query: str) -> int:
    """쿼리 복잡도에 따른 Top-K 결정"""
    words = query.split()

    if len(words) <= 3:
        return 5  # 단순
    elif any(w in query for w in ["비교", "차이", "그리고", "또한", "및으며"]):
        return 10  # 복잡
    else:
        return 7  # 중간
```

**적용 위치:** `QueryHandler.process_query()` 메서드

---

### 2. 환각 방지 프롬프트 개선 (2분)

**파일:** `src/rag/domain/generation/rag_prompt.py` (또는 새로 생성)

```python
STRICT_RAG_PROMPT = """
You are a university regulation expert assistant. Answer ONLY based on the retrieved contexts.

CRITICAL RULES:
1. If information is not in the contexts, reply:
   "죄송합니다. 해당 정보를 찾을 수 없습니다. 관련 부서에 문의해 주세요."

2. All regulation citations MUST follow the format: "규정명 제X조"

3. NEVER invent contact information or phone numbers

4. For uncertain information, add: "(확인이 필요합니다)"

Retrieved Contexts:
{contexts}

Question: {question}

Answer:"""
```

---

## ⚡ 30분 만에 구현 가능한 개선

### 1. 쿼리 복잡도 분류기 (15분)

**파일:** `src/rag/domain/query/classifier.py` (새로 생성)

```python
from enum import Enum
from dataclasses import dataclass

class QueryType(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    EDGE = "edge"

@dataclass
class QueryAnalysis:
    query_type: QueryType
    complexity_score: float
    keywords: list[str]

def analyze_query(query: str) -> QueryAnalysis:
    """쿼리 분석"""
    words = query.split()

    # 복잡도 키워드
    complexity_keywords = ["비교", "차이", "그리고", "또한", "및으며"]

    # 분류
    if len(words) == 1:
        query_type = QueryType.EDGE
        complexity = 0.2
    elif any(kw in query for kw in complexity_keywords):
        query_type = QueryType.COMPLEX
        complexity = 0.8
    else:
        query_type = QueryType.SIMPLE
        complexity = 0.4

    # 키워드 추출
    keywords = [w for w in words if len(w) > 1]

    return QueryAnalysis(
        query_type=query_type,
        complexity_score=complexity,
        keywords=keywords
    )
```

### 2. 전문 용어 확장 (15분)

**파일:** `src/rag/domain/query/expansion.py` (새로 생성)

```python
# 규정 전문 용어 사전
REGULATION_TERMS = {
    # 휴학 관련
    "휴학": ["휴학(休學)", "휴학신청", "휴학절차", "휴학신청서", "휴학허가"],
    "복학": ["복학(復學)", "복학신청", "재입학", "복학허가"],
    "자퇴": ["자퇴(自退)", "자퇴신청", "제적"],

    # 성적 관련
    "성적": ["성적정정", "성적이의신청", "성적포기", "학점"],
    "등록": ["등록", "수강신청", "과목등록", "학점등록"],

    # 교원 관련
    "승진": ["승진(昇進)", "정승", "승진심사", "승진임용"],
    "정년": ["정년", "연구년", "안식", "안식년"],
    "휴직": ["휴직", "교원휴직"],

    # 장학금 관련
    "장학금": ["장학금", "성적장학금", "근로장학금", "기숀장학금"],
}

def expand_query(query: str) -> list[str]:
    """전문 용어 확장"""
    expanded = [query]

    for term, synonyms in REGULATION_TERMS.items():
        if term in query:
            for synonym in synonyms:
                expanded.append(query.replace(term, synonym))

    return list(set(expanded))
```

---

## 🔧 구현 순서 (Implementation Order)

### Step 1: Top-K 동적 조정 (가장 먼저, 가장 쉬움)

1. 파일 열기: `src/rag/interface/query_handler.py`
2. `get_top_k_for_query()` 함수 추가
3. `process_query()` 메서드에서 `top_k` 매개변수를 동적으로 설정
4. 테스트: `python test_scenarios/rag_quality_evaluator.py --use-llm-judge --limit 3`

### Step 2: 환각 방지 프롬프트

1. 파일 열기 또는 생성: `src/rag/domain/generation/rag_prompt.py`
2. `STRICT_RAG_PROMPT` 정의
3. LLM 호출 시 프롬프트 적용
4. 테스트: 환각이 많던 시나리오 재평가

### Step 3: 쿼리 분류기

1. 파일 생성: `src/rag/domain/query/classifier.py`
2. `QueryType` enum과 `analyze_query()` 함수 구현
3. 검색 시 쿼리 분류 결과 활용
4. 테스트: 복잡한 쿼리 자동 분류 확인

### Step 4: 전문 용어 확장

1. 파일 생성: `src/rag/domain/query/expansion.py`
2. `REGULATION_TERMS` 사전 구축
3. 검색 전 쿼리 확장 적용
4. 테스트: 전문 용어 쿼리 검색 품질 확인

---

## 📝 테스트 명령어 (Test Commands)

```bash
# 기본 평가 (3개 쿼리만)
python test_scenarios/rag_quality_evaluator.py --use-llm-judge --limit 3

# 전체 평가
python test_scenarios/rag_quality_evaluator.py --use-llm-judge

# 병렬 페르소나 평가
python test_scenarios/rag_quality_evaluator.py --use-llm-judge --parallel

# 결과 저장
python test_scenarios/rag_quality_evaluator.py --use-llm-judge --output data/evaluations/test_report.md
```

---

## 🎯 성공 확인 방법 (How to Verify)

### 1. Top-K 동적 조정 확인
```bash
# 복잡한 쿼리 검색 시 Top-K=10 사용되는지 로그 확인
python -c "
from src.rag.interface.query_handler import QueryHandler
qh = QueryHandler(...)
print(qh.get_top_k_for_query('휴학과 복학의 차이점은?'))  # 10이어야 함
print(qh.get_top_k_for_query('휴학 방법'))  # 5이어야 함
"
```

### 2. 환각 방지 확인
- 이전에 환각이 많던 시나리오 재평가
- "알 수 없음" 응답이 나오는지 확인

### 3. 전체 평가 점수 확인
```bash
python test_scenarios/rag_quality_evaluator.py --use-llm-judge
# 출력: Pass Rate, Overall Score 등 확인
# 목표: Pass Rate 13% → 30% 이상
```

---

## 🔗 빠른 참고 (Quick References)

### 수정해야 할 주요 파일
- `src/rag/interface/query_handler.py` - Top-K 동적 조정
- `src/rag/domain/generation/rag_prompt.py` - 환각 방지 프롬프트
- `src/rag/domain/query/classifier.py` - 쿼리 분류기 (새 파일)
- `src/rag/domain/query/expansion.py` - 용어 확장 (새 파일)

### 관련 문서
- `data/evaluations/RAG_QUALITY_COMPREHENSIVE_ANALYSIS.md` - 종합 분석 보고서
- `.claude/skills/rag-quality-local/modules/evaluation.md` - 평가 기준
- `.claude/skills/rag-quality-local/modules/metrics.md` - 메트릭 정의

---

**생성일:** 2026-02-07 17:30:00
**버전:** 1.0
**다음 업데이트:** Phase 1 개선 완료 후
