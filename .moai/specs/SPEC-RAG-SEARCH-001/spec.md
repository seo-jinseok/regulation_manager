# SPEC-RAG-SEARCH-001: RAG 검색 시스템 Contextual Recall 개선

## TAG BLOCK

```yaml
spec_id: SPEC-RAG-SEARCH-001
title: RAG 검색 시스템 Contextual Recall 개선
status: Planned
priority: Critical
created: 2026-02-07
assigned: manager-strategy
lifecycle: spec-first
estimated_effort: 2 weeks
labels: [rag, search, recall, entity-recognition, query-expansion, multi-hop]
related_specs: [SPEC-RAG-002, SPEC-RAG-EVAL-001]
```

## Environment

### 현재 시스템 상황

**프로젝트**: 대학 규정 관리 시스템 (University Regulation Manager)

**기술 스택**:
- Python 3.11+
- RAG Framework: llama-index >= 0.14.10
- Vector Database: ChromaDB >= 1.4.0
- Embedding: BGE Reranker (BAAI/bge-reranker-v2-m3)
- Hybrid Search: BM25 + Dense Retrieval
- Query Expansion: DynamicQueryExpander
- Evaluation: RAGAS (LLM-as-Judge)

**현재 성능 지표** (2026-02-07 기준):
| 지표 | 현재 값 | 목표 값 | 간격 |
|------|---------|---------|------|
| Contextual Recall | 32% | 65% | -33% |
| Faithfulness | 51.7% | 90% | -38.3% |
| Answer Relevancy | 73.3% | 85% | -11.7% |
| Contextual Precision | N/A | 80% | - |
| Overall Score | ~54% | 80% | -26% |

### 범위

**포함**:
- 엔티티 인식 강화 (6개 새로운 엔티티 유형)
- 쿼리 확장 파이프라인 개선 (3단계 확장)
- 적응형 Top-K 구현 (동적 결과 수 조정)
- 기본 멀티-홉 검색 (인용 추적)

**제외**:
- 규정 데이터 수정 (데이터 변경 없음)
- 완전한 재색인 (기존 인덱스 활용)
- UI 개발 (백엔드 검색 로직에만 집중)
- LLM 프롬프트 변경 (검색에만 집중)

## Assumptions

### 기술적 가정

- **높은 신뢰도**: 현재 엔티티 인식이 명시적 키워드만 인식
  - **증거**: query_analyzer.py에서 ACADEMIC_KEYWORDS만 확인
  - **위험**: 잘못된 경우 시간 낭비
  - **검증**: 실제 쿼리 로그 분석으로 엔티티 패턴 확인

- **높은 신뢰도**: DynamicQueryExpander가 존재하지만 불충분
  - **증거**: query_expander.py 파일 존재 확인
  - **위험**: 중복 기능 구현 가능성
  - **검증**: 기존 확장 패턴 분석 및 보완점 식별

- **중간 신뢰도**: Top-K=10이 복잡한 쿼리에 불충분
  - **증거**: 사용자 피드백 "관련 정보 누락"
  - **위험**: 성능 저하 없이 개선 불가능할 수 있음
  - **검증**: 쿼리 복잡도 분류 및 Top-K 효과 측정

- **중간 신뢰도**: 멀티-홉 검색이 Contextual Recall 개선에 기여
  - **증거**: 규정 간 인용 관계 존재 (제X조 참조)
  - **위험**: 인용이 항상 관련성 있다는 보장 없음
  - **검증**: 인용 그래프 분석 및 관련성 측정

### 비즈니스 가정

- **높은 신뢰도**: 낮은 Contextual Recall이 사용자 만족도에 부정적 영향
  - **위험**: 다른 지표(Answer Relevancy)가 더 중요할 수 있음
  - **검증**: 사용자 피드백 및 불만 사항 분석

- **중간 신뢰도**: 2주 내 구현 가능
  - **위험**: 복잡도 과소평가로 지연 가능
  - **검증**: 작업 분해 및 일일 진행 상황 모니터링

### 통합 가정

- **높은 신뢰도**: 기존 RAG 파이프라인 구조 유지 가능
  - **증거**: Clean Architecture, 모듈화 잘됨
  - **위험**: 순환 종속성 존재 가능
  - **검증**: 종속성 그래프 분석 및 테스트

## Root Cause Analysis

### 문제 정의

**Surface Problem**: Contextual Recall 32%가 목표 65%의 절반도 안 됨

**Five Whys 분석**:

1. **Why**: 왜 Contextual Recall이 낮은가?
   - **Answer**: 쿼리에 필요한 모든 관련 문서를 검색하지 못함

2. **Why**: 왜 관련 문서를 검색하지 못하는가?
   - **Answer**: 검색 쿼리가 규정 내용과 효과적으로 매칭되지 않음

3. **Why**: 왜 쿼리와 규정 내용이 매칭되지 않는가?
   - **Answer**:
     - 사용자 쿼리가 비격식 언어 사용 (규정의 공식 용어와 다름)
     - 규정 간 참조 (제X조)를 따라가지 않음
     - 관련 개념 검색 부족 (동의어, 상위어 미사용)

4. **Why**: 왜 시스템이 이런 불일치를 처리하지 못하는가?
   - **Answer**:
     - 엔티티 인식이 명시적 키워드만 찾음
     - 쿼리 확장이 미리 정의된 패턴으로 제한됨
     - Top-K 제한으로 충분한 컨텍스트 검색 불가
     - 인용 그래프 순회로 멀티-문서 답변 불가

5. **Why** (Root Cause): 어떤 근본적 문제를 해결해야 하는가?
   - **Answer**: 검색 시스템이 직접 키워드 매칭만으로 충분하다고 가정하지만, 규정은 다음을 필요로 함:
     - 의미적 이해 (비격식 vs 공식 언어)
     - 개념적 관계 (절차, 요건, 혜택)
     - 상호 참조 추적 (규정 간 인용)
     - 포괄적 컨텍스트 (충분한 문서 검색)

### 증상 분석

| 쿼리 예시 | 현재 동작 | 문제 |
|-----------|-----------|------|
| "장학금 신청 방법" | 등록/입학 정보만 반환 | 절차 엔티티 인식 부족 |
| "연구년 자격 요건" | 연구년 정보 누락 | 요건 엔티티 인식 부족 |
| "조교 근무 시간" | 혜택 정보 누락 | 혜택 엔티티 인식 부족 |

## Requirements

### Priority 1: 엔티티 인식 강화 (Week 1, Days 1-2)

#### Ubiquitous Requirements

**REQ-ER-001**: 시스템은 규정 섹션(조, 항, 호)을 엔티티로 인식해야 한다.
**REQ-ER-002**: 시스템은 절차 관련 엔티티(신청, 절차, 방법, 발급, 제출)를 인식해야 한다.
**REQ-ER-003**: 시스템은 요건 관련 엔티티(자격, 요건, 조건, 기준, 제한)를 인식해야 한다.
**REQ-ER-004**: 시스템은 혜택 관련 엔티티(혜택, 지급, 지원, 급여)를 인식해야 한다.
**REQ-ER-005**: 시스템은 마감 관련 엔티티(기한, 마감, 날짜, 기간, ~까지)를 인식해야 한다.
**REQ-ER-006**: 시스템은 상위어 확장(등록금→학사→행정)을 지원해야 한다.

#### Event-Driven Requirements

**REQ-ER-007**: WHEN 섹션 패턴 감지 시, 시스템 SHALL 섹션 번호를 추출해야 한다.
**REQ-ER-008**: WHEN 절차 키워드 감지 시, 시스템 SHALL 관련 절차 동의어를 추가해야 한다.
**REQ-ER-009**: WHEN 요건 키워드 감지 시, 시스템 SHALL 자격/조건 관련 용어를 확장해야 한다.
**REQ-ER-010**: WHEN 혜택 키워드 감지 시, 시스템 SHALL 지급/지원 관련 용어를 포함해야 한다.
**REQ-ER-011**: WHEN 마감 키워드 감지 시, 시스템 SHALL 날짜/기간 관련 패턴을 찾아야 한다.
**REQ-ER-012**: WHEN 상위어 확장 가능 시, 시스템 SHALL 계층적 용어를 추가해야 한다.

#### State-Driven Requirements

**REQ-ER-013**: IF 엔티티 매칭 실패 시, 시스템 SHALL 원본 쿼리로 대체해야 한다.
**REQ-ER-014**: IF 여러 엔티티 감지 시, 시스템 SHALL 모든 엔티티를 결합해야 한다.
**REQ-ER-015**: IF 엔티티 신뢰도 낮음(< 0.7) 시, 시스템 SHALL 확장을 건너뛰어야 한다.

#### Unwanted Behavior Requirements

**REQ-ER-016**: 시스템은 NOT 엔티티 인식으로 인해 쿼리 의미를 변경해서는 안 된다.
**REQ-ER-017**: 시스템은 NOT 너무 일반적인 상위어로 확장해서는 안 된다.

---

### Priority 2: 다단계 쿼리 확장 (Week 1, Days 3-4)

#### Ubiquitous Requirements

**REQ-QE-001**: 시스템은 동의어 확장을 지원해야 한다 (장학금 ↔ 장학금 지원 ↔ 재정 지원).
**REQ-QE-002**: 시스템은 상위어 확장을 지원해야 한다 (등록금 → 학사 → 행정).
**REQ-QE-003**: 시스템은 절차 확장을 지원해야 한다 (신청 → 절차 → 서류 → 제출).

#### Event-Driven Requirements

**REQ-QE-004**: WHEN 쿼리 수신 시, 시스템 SHALL 3단계 확장을 순차적으로 실행해야 한다.
**REQ-QE-005**: WHEN 동의어 확장 시, 시스템 SHALL 최대 3개 관련 용어를 추가해야 한다.
**REQ-QE-006**: WHEN 상위어 확장 시, 시스템 SHALL 최대 2단계 상위 개념을 추가해야 한다.
**REQ-QE-007**: WHEN 절차 확장 시, 시스템 SHALL 절차 체인을 따라 관련 용어를 추가해야 한다.

#### State-Driven Requirements

**REQ-QE-008**: IF 확장 결과가 너무 길음(>10개 키워드) 시, 시스템 SHALL 상위 7개만 유지해야 한다.
**REQ-QE-009**: IF 확장 신뢰도 낮음(< 0.6) 시, 시스템 SHALL 확장을 적용하지 않아야 한다.
**REQ-QE-010**: IF 원본 쿼리가 이미 규정 용어임 시, 시스템 SHALL 확장을 건너뛰어야 한다.

#### Optional Requirements

**REQ-QE-011**: 가능한 경우, 시스템 MAY LLM을 사용한 동적 확장을 제공해야 한다.
**REQ-QE-012**: 가능한 경우, 시스템 MAY 사용자 피드백으로 확장 품질을 개선해야 한다.

---

### Priority 3: 적응형 Top-K (Week 2, Days 1-2)

#### Ubiquitous Requirements

**REQ-AT-001**: 시스템은 쿼리 복잡도에 따라 Top-K를 동적으로 조정해야 한다.
**REQ-AT-002**: 시스템은 단순 쿼리(단일 키워드)에 Top-5를 사용해야 한다.
**REQ-AT-003**: 시스템은 중간 쿼리(자연어 질문)에 Top-10을 사용해야 한다.
**REQ-AT-004**: 시스템은 복잡한 쿼리(다중 조건)에 Top-15를 사용해야 한다.
**REQ-AT-005**: 시스템은 다중 파트 쿼리에 Top-20을 사용해야 한다.

#### Event-Driven Requirements

**REQ-AT-006**: WHEN 쿼리 분류 완료 시, 시스템 SHALL 해당 복잡도 수준의 Top-K를 설정해야 한다.
**REQ-AT-007**: WHEN 쿼리 복잡도 변경 감지 시, 시스템 SHALL Top-K를 재계산해야 한다.
**REQ-AT-008**: WHEN Top-K 증가 시, 시스템 SHALL 응답 시간을 모니터링해야 한다.

#### State-Driven Requirements

**REQ-AT-009**: IF 응답 시간 > 500ms 시, 시스템 SHALL Top-K를 줄여야 한다.
**REQ-AT-010**: IF 쿼리 복잡도 분류 실패 시, 시스템 SHALL 기본 Top-10을 사용해야 한다.
**REQ-AT-011**: IF 검색 결과 수 < Top-K 시, 시스템 SHALL 사용 가능한 결과만 반환해야 한다.

---

### Priority 4: 기본 멀티-홉 검색 (Week 2, Days 3-4)

#### Ubiquitous Requirements

**REQ-MH-001**: 시스템은 "제X조" 참조를 실제 규정 텍스트로 따라가야 한다.
**REQ-MH-002**: 시스템은 "관련 규정" 링크를 따라야 한다.
**REQ-MH-003**: 시스템은 최대 2홉 제한을 적용해야 한다.

#### Event-Driven Requirements

**REQ-MH-004**: WHEN 인용 패턴 감지 시, 시스템 SHALL 인용된 규정을 검색해야 한다.
**REQ-MH-005**: WHEN 인용된 규정 찾음 시, 시스템 SHALL 원본 결과와 결합해야 한다.
**REQ-MH-006**: WHEN 2홉 도달 시, 시스템 SHALL 추적을 중단해야 한다.
**REQ-MH-007**: WHEN 인용된 규정을 찾지 못함 시, 시스템 SHALL 원본 결과만 반환해야 한다.

#### State-Driven Requirements

**REQ-MH-008**: IF 인용 관련성 낮음(< 0.5) 시, 시스템 SHALL 인용을 무시해야 한다.
**REQ-MH-009**: IF 인용 그래프 순환 감지 시, 시스템 SHALL 순환을 중단해야 한다.
**REQ-MH-010**: IF 인용된 규정이 이미 결과에 있음 시, 시스템 SHALL 중복을 제거해야 한다.

#### Unwanted Behavior Requirements

**REQ-MH-011**: 시스템은 NOT 무한 인용 추적으로 인해 타임아웃해서는 안 된다.
**REQ-MH-012**: 시스템은 NOT 관련 없는 인용을 따라가서는 안 된다.

---

### Priority 5: 성능 및 품질 유지 (지속적)

#### Ubiquitous Requirements

**REQ-PQ-001**: 시스템은 응답 시간 500ms 미만을 유지해야 한다.
**REQ-PQ-002**: 시스템은 Faithfulness 점수 51.7% 이상을 유지해야 한다.
**REQ-PQ-003**: 시스템은 Answer Relevancy 점수 73.3% 이상을 유지해야 한다.
**REQ-PQ-004**: 시스템은 기존 기능에 대한 회귀를 방지해야 한다.

#### Event-Driven Requirements

**REQ-PQ-005**: WHEN 응답 시간 > 500ms 시, 시스템 SHALL 경고를 로그해야 한다.
**REQ-PQ-006**: WHEN Faithfulness 저하 감지 시, 시스템 SHALL 롤백을 고려해야 한다.
**REQ-PQ-007**: WHEN Answer Relevancy 저하 감지 시, 시스템 SHALL 원인을 분석해야 한다.
**REQ-PQ-008**: WHEN 회귀 감지 시, 시스템 SHALL 즉시 수정을 우선시해야 한다.

#### State-Driven Requirements

**REQ-PQ-009**: IF 성능 저하 심각함(>1000ms) 시, 시스템 SHALL 새로운 기능을 비활성화해야 한다.
**REQ-PQ-010**: IF 품질 지표 저하 시, 시스템 SHALL A/B 테스트를 실행해야 한다.

---

## Specifications

### Architecture Design

#### Component 1: Enhanced Entity Recognition (REQ-ER-001 ~ REQ-ER-017)

**File**: `src/rag/domain/entity/entity_recognizer.py` (NEW)

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re

class EntityType(Enum):
    """Entity types for regulation search."""
    SECTION = "section"  # 조, 항, 호
    PROCEDURE = "procedure"  # 신청, 절차, 방법
    REQUIREMENT = "requirement"  # 자격, 요건, 조건
    BENEFIT = "benefit"  # 혜택, 지급, 지원
    DEADLINE = "deadline"  # 기한, 마감, 날짜
    HYPERNYM = "hypernym"  # 상위어

@dataclass
class EntityMatch:
    """Entity match result."""
    entity_type: EntityType
    text: str
    start: int
    end: int
    confidence: float
    expanded_terms: List[str]

class RegulationEntityRecognizer:
    """
    Enhanced entity recognizer for regulation queries.

    Recognizes 6 new entity types:
    1. Regulation sections (조, 항, 호)
    2. Procedures (신청, 절차, 방법, 발급, 제출)
    3. Requirements (자격, 요건, 조건, 기준, 제한)
    4. Benefits (혜택, 지급, 지원, 급여)
    5. Deadlines (기한, 마감, 날짜, 기간, ~까지)
    6. Hypernyms (hierarchical expansion: 등록금→학사→행정)
    """

    # Section patterns
    SECTION_PATTERNS = [
        re.compile(r'제(\d+)조'),
        re.compile(r'제(\d+)항'),
        re.compile(r'제(\d+)호'),
    ]

    # Procedure keywords
    PROCEDURE_KEYWORDS = [
        '신청', '절차', '방법', '발급', '제출', '신고',
        '등록', '신청서', '서류', '구비서류', '처리'
    ]

    # Requirement keywords
    REQUIREMENT_KEYWORDS = [
        '자격', '요건', '조건', '기준', '제한', '대상',
        '충족', '요구', '필수', '선택', '우선'
    ]

    # Benefit keywords
    BENEFIT_KEYWORDS = [
        '혜택', '지급', '지원', '급여', '보조', '장학',
        '수당', '비용', '경비', '지원금', '장학금'
    ]

    # Deadline keywords
    DEADLINE_KEYWORDS = [
        '기한', '마감', '날짜', '기간', '까지', '이내',
        '이전', '이후', '부터', '당일', '매월', '매년'
    ]

    # Hypernym mappings
    HYPERNYM_MAPPINGS = {
        '등록금': ['등록금', '학사', '행정'],
        '장학금': ['장학금', '재정', '지원'],
        '휴학': ['휴학', '학적', '행정'],
        '교수': ['교수', '교원', '임용'],
        '조교': ['조교', '교육', '지원'],
    }

    def recognize(self, query: str) -> List[EntityMatch]:
        """
        Recognize entities in query.

        Returns list of entity matches with expanded terms.
        """
        matches = []

        # Section recognition
        for pattern in self.SECTION_PATTERNS:
            for match in pattern.finditer(query):
                matches.append(EntityMatch(
                    entity_type=EntityType.SECTION,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    expanded_terms=[]
                ))

        # Procedure recognition
        for keyword in self.PROCEDURE_KEYWORDS:
            if keyword in query:
                matches.append(EntityMatch(
                    entity_type=EntityType.PROCEDURE,
                    text=keyword,
                    start=query.index(keyword),
                    end=query.index(keyword) + len(keyword),
                    confidence=0.85,
                    expanded_terms=self._expand_procedure(keyword)
                ))

        # Similar for REQUIREMENT, BENEFIT, DEADLINE, HYPERNYM

        return matches

    def _expand_procedure(self, keyword: str) -> List[str]:
        """Expand procedure keyword to related terms."""
        expansions = {
            '신청': ['신청', '신청서', '제출', '등록'],
            '절차': ['절차', '방법', '과정', '단계'],
            '방법': ['방법', '절차', '방식', '요령'],
            # ... more expansions
        }
        return expansions.get(keyword, [])
```

#### Component 2: Multi-Stage Query Expansion (REQ-QE-001 ~ REQ-QE-012)

**File**: `src/rag/infrastructure/query_expander_v2.py` (MODIFIED)

```python
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ExpansionResult:
    """Result of multi-stage expansion."""
    original_query: str
    stage1_synonyms: List[str]
    stage2_hypernyms: List[str]
    stage3_procedures: List[str]
    final_expanded: str
    confidence: float

class MultiStageQueryExpander:
    """
    Multi-stage query expansion pipeline.

    Stage 1: Synonym expansion (장학금 ↔ 장학금 지원 ↔ 재정 지원)
    Stage 2: Hypernym expansion (등록금 → 학사 → 행정)
    Stage 3: Procedure expansion (신청 → 절차 → 서류 → 제출)
    """

    def __init__(self, entity_recognizer: RegulationEntityRecognizer):
        self._entity_recognizer = entity_recognizer

    def expand(self, query: str) -> ExpansionResult:
        """Apply 3-stage expansion."""
        # Skip if query is already formal
        if self._is_formal_query(query):
            return ExpansionResult(
                original_query=query,
                stage1_synonyms=[],
                stage2_hypernyms=[],
                stage3_procedures=[],
                final_expanded=query,
                confidence=1.0
            )

        # Stage 1: Synonym expansion
        synonyms = self._expand_synonyms(query)

        # Stage 2: Hypernym expansion
        hypernyms = self._expand_hypernyms(query)

        # Stage 3: Procedure expansion
        procedures = self._expand_procedures(query)

        # Combine all expansions (max 10 keywords)
        all_terms = [query] + synonyms + hypernyms + procedures
        final_terms = all_terms[:10]  # Limit to prevent noise

        return ExpansionResult(
            original_query=query,
            stage1_synonyms=synonyms,
            stage2_hypernyms=hypernyms,
            stage3_procedures=procedures,
            final_expanded=' '.join(final_terms),
            confidence=self._calculate_confidence(final_terms)
        )

    def _is_formal_query(self, query: str) -> bool:
        """Check if query is already formal regulation language."""
        formal_indicators = ['규정', '조', '항', '호', '세칙', '지침']
        return any(ind in query for ind in formal_indicators)

    def _expand_synonyms(self, query: str) -> List[str]:
        """Stage 1: Synonym expansion."""
        synonym_map = {
            '장학금': ['장학금', '장학금 지급', '장학금 지원', '재정 지원'],
            '연구년': ['연구년', '안식년', '교원연구년'],
            # ... more mappings
        }

        synonyms = []
        for key, values in synonym_map.items():
            if key in query:
                synonyms.extend(values[1:])  # Exclude original

        return synonyms[:3]  # Max 3 synonyms

    def _expand_hypernyms(self, query: str) -> List[str]:
        """Stage 2: Hypernym expansion."""
        hypernyms = []
        entities = self._entity_recognizer.recognize(query)

        for entity in entities:
            if entity.entity_type == EntityType.HYPERNYM:
                hypernyms.extend(entity.expanded_terms[1:])  # Exclude original

        return hypernyms[:2]  # Max 2 hypernyms

    def _expand_procedures(self, query: str) -> List[str]:
        """Stage 3: Procedure expansion."""
        procedure_map = {
            '신청': ['신청', '신청서', '제출', '서류'],
            '절차': ['절차', '방법', '과정'],
            # ... more mappings
        }

        procedures = []
        for key, values in procedure_map.items():
            if key in query:
                procedures.extend(values[1:])  # Exclude original

        return procedures[:2]  # Max 2 procedure terms

    def _calculate_confidence(self, terms: List[str]) -> float:
        """Calculate expansion confidence."""
        # More relevant terms = higher confidence
        # TODO: Implement using relevance scoring
        return 0.7
```

#### Component 3: Adaptive Top-K (REQ-AT-001 ~ REQ-AT-011)

**File**: `src/rag/application/search_config.py` (NEW)

```python
from enum import Enum
from dataclasses import dataclass

class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"       # Single keyword, regulation name
    MEDIUM = "medium"       # Natural questions
    COMPLEX = "complex"     # Multiple conditions
    MULTI_PART = "multi_part"  # Multiple queries

@dataclass
class TopKConfig:
    """Top-K configuration for each complexity level."""
    simple: int = 5
    medium: int = 10
    complex: int = 15
    multi_part: int = 20
    max_limit: int = 25  # Absolute maximum

class AdaptiveTopKSelector:
    """
    Adaptive Top-K selection based on query complexity.

    Simple queries: Top-5 (single keyword, regulation name)
    Medium queries: Top-10 (natural questions)
    Complex queries: Top-15 (multi-part, procedure queries)
    Multi-part queries: Top-20 (multiple conditions)
    """

    def __init__(self, config: TopKConfig = None):
        self._config = config or TopKConfig()

    def select_top_k(self, query: str) -> int:
        """Select Top-K based on query complexity."""
        complexity = self._classify_complexity(query)

        if complexity == QueryComplexity.SIMPLE:
            return self._config.simple
        elif complexity == QueryComplexity.MEDIUM:
            return self._config.medium
        elif complexity == QueryComplexity.COMPLEX:
            return self._config.complex
        else:  # MULTI_PART
            return self._config.multi_part

    def _classify_complexity(self, query: str) -> QueryComplexity:
        """Classify query complexity."""
        # Simple: single keyword or regulation name
        if self._is_simple(query):
            return QueryComplexity.SIMPLE

        # Multi-part: multiple distinct queries
        if self._is_multi_part(query):
            return QueryComplexity.MULTI_PART

        # Complex: procedure or requirement query
        if self._is_complex(query):
            return QueryComplexity.COMPLEX

        # Default: medium
        return QueryComplexity.MEDIUM

    def _is_simple(self, query: str) -> bool:
        """Check if query is simple (single keyword or regulation name)."""
        # Single word without question markers
        words = query.split()
        if len(words) <= 2:
            return True

        # Regulation name only (e.g., "교원인사규정")
        if '규정' in query or '학칙' in query:
            return not any(m in query for m in ['어떻게', '방법', '절차', '신청'])

        return False

    def _is_multi_part(self, query: str) -> bool:
        """Check if query contains multiple parts."""
        multi_part_indicators = [
            ' 그리고 ', ' 또는 ', ' 그리고\n', ' 또는\n',
            ', ', '、', '; '
        ]
        return any(ind in query for ind in multi_part_indicators)

    def _is_complex(self, query: str) -> bool:
        """Check if query is complex (procedure or requirement)."""
        complex_indicators = [
            '방법', '절차', '신청', '자격', '요건', '조건',
            '기준', '제한', '혜택', '지급', '지원'
        ]
        return any(ind in query for ind in complex_indicators)
```

#### Component 4: Basic Multi-Hop Retrieval (REQ-MH-001 ~ REQ-MH-012)

**File**: `src/rag/infrastructure/multi_hop_retriever.py` (NEW)

```python
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Citation reference."""
    source_id: str
    target_id: str
    citation_type: str  # "section_ref", "regulation_ref"
    text: str

@dataclass
class HopResult:
    """Result of multi-hop retrieval."""
    original_results: List[str]  # Original document IDs
    hop_results: List[str]  # Additional documents from citations
    all_results: List[str]  # Combined and deduplicated
    hops_performed: int

class MultiHopRetriever:
    """
    Basic multi-hop retrieval for citation following.

    Follows "제X조" references to actual regulation text.
    Follows "관련 규정" links.
    Max 2 hops to prevent infinite loops.
    """

    MAX_HOPS = 2
    CITATION_PATTERN = re.compile(r'제(\d+)조')
    REGULATION_REF_PATTERN = re.compile(r'(?:관련|상세|참고).*?규정')

    def __init__(self, vector_store, chunk_store):
        self._vector_store = vector_store
        self._chunk_store = chunk_store
        self._citation_graph: Dict[str, List[Citation]] = {}

    def retrieve(
        self,
        query: str,
        initial_results: List[str],
        top_k: int = 10
    ) -> HopResult:
        """
        Perform multi-hop retrieval.

        Args:
            query: Original search query
            initial_results: Initial document IDs from first search
            top_k: Maximum total results to return

        Returns:
            HopResult with original + hop results
        """
        hop_results = []
        visited = set(initial_results)
        current_hop = 0

        while current_hop < self.MAX_HOPS:
            # Find citations in current results
            new_documents = []

            for doc_id in (initial_results if current_hop == 0 else hop_results):
                if doc_id in visited:
                    continue

                citations = self._extract_citations(doc_id)
                for citation in citations:
                    if citation.target_id not in visited:
                        # Check relevance
                        if self._is_citation_relevant(query, citation):
                            new_documents.append(citation.target_id)
                            visited.add(citation.target_id)

            if not new_documents:
                # No more citations to follow
                break

            hop_results.extend(new_documents)
            current_hop += 1

        # Combine and deduplicate
        all_results = list(set(initial_results + hop_results))

        return HopResult(
            original_results=initial_results,
            hop_results=hop_results,
            all_results=all_results[:top_k],
            hops_performed=current_hop
        )

    def _extract_citations(self, doc_id: str) -> List[Citation]:
        """Extract citations from document."""
        # Get document content
        chunk = self._chunk_store.get(doc_id)
        if not chunk:
            return []

        citations = []

        # Extract section references (제X조)
        for match in self.CITATION_PATTERN.finditer(chunk.text):
            section_num = match.group(1)
            # Find target document
            target_id = self._find_section_document(section_num)
            if target_id:
                citations.append(Citation(
                    source_id=doc_id,
                    target_id=target_id,
                    citation_type="section_ref",
                    text=match.group()
                ))

        # Extract regulation references
        for match in self.REGULATION_REF_PATTERN.finditer(chunk.text):
            target_id = self._find_regulation_document(match.group())
            if target_id:
                citations.append(Citation(
                    source_id=doc_id,
                    target_id=target_id,
                    citation_type="regulation_ref",
                    text=match.group()
                ))

        return citations

    def _find_section_document(self, section_num: str) -> Optional[str]:
        """Find document by section number."""
        # Search vector store for section
        results = self._vector_store.query(f"제{section_num}조", top_k=1)
        if results:
            return results[0].id
        return None

    def _find_regulation_document(self, ref_text: str) -> Optional[str]:
        """Find document by regulation reference."""
        # Search vector store for regulation
        results = self._vector_store.query(ref_text, top_k=1)
        if results:
            return results[0].id
        return None

    def _is_citation_relevant(self, query: str, citation: Citation) -> bool:
        """Check if citation is relevant to query."""
        # Simple relevance check based on citation type
        # TODO: Implement more sophisticated relevance scoring

        # For now, assume all section references are relevant
        if citation.citation_type == "section_ref":
            return True

        # For regulation references, check keyword overlap
        if citation.citation_type == "regulation_ref":
            query_terms = set(query.split())
            citation_terms = set(citation.text.split())
            overlap = len(query_terms & citation_terms)
            return overlap >= 1

        return False
```

### File Structure

```
src/rag/
├── domain/
│   ├── entity/
│   │   ├── entity_recognizer.py       # NEW: Enhanced entity recognition
│   │   └── entity_types.py            # NEW: Entity type definitions
│   └── citation/
│       ├── citation_extractor.py      # NEW: Citation extraction
│       └── citation_graph.py          # NEW: Citation graph management
├── infrastructure/
│   ├── query_expander_v2.py           # MODIFIED: Multi-stage expansion
│   ├── multi_hop_retriever.py         # NEW: Basic multi-hop retrieval
│   └── adaptive_top_k.py              # NEW: Adaptive Top-K selector
├── application/
│   ├── search_config.py               # NEW: Search configuration
│   └── enhanced_search_usecase.py     # MODIFIED: Enhanced search orchestration
└── tests/
    ├── unit/
    │   ├── test_entity_recognizer.py
    │   ├── test_query_expander_v2.py
    │   ├── test_adaptive_top_k.py
    │   └── test_multi_hop_retriever.py
    └── integration/
        └── test_enhanced_search.py
```

### Dependencies

**New Dependencies**:
```toml
# No new external dependencies required
# All enhancements use existing infrastructure
```

**Internal Dependencies**:
- `domain/entity/entity_recognizer.py` → Query expansion
- `infrastructure/query_expander_v2.py` → Search orchestration
- `infrastructure/multi_hop_retriever.py` → Vector store, chunk store
- `application/search_config.py` → Search use case

## Traceability

### Requirements to Components Mapping

| Requirement ID | Component | File |
|---------------|-----------|------|
| REQ-ER-001 ~ REQ-ER-017 | EnhancedEntityRecognizer | domain/entity/entity_recognizer.py |
| REQ-QE-001 ~ REQ-QE-012 | MultiStageQueryExpander | infrastructure/query_expander_v2.py |
| REQ-AT-001 ~ REQ-AT-011 | AdaptiveTopKSelector | application/search_config.py |
| REQ-MH-001 ~ REQ-MH-012 | MultiHopRetriever | infrastructure/multi_hop_retriever.py |
| REQ-PQ-001 ~ REQ-PQ-010 | PerformanceMonitoring | application/metrics.py |

### Components to Test Cases Mapping

| Component | Test File | Coverage Target |
|-----------|-----------|-----------------|
| EntityRecognizer | tests/unit/test_entity_recognizer.py | 90% |
| QueryExpanderV2 | tests/unit/test_query_expander_v2.py | 85% |
| AdaptiveTopKSelector | tests/unit/test_adaptive_top_k.py | 85% |
| MultiHopRetriever | tests/unit/test_multi_hop_retriever.py | 80% |
| EnhancedSearch | tests/integration/test_enhanced_search.py | 75% |

## Appendix

### Glossary

- **Contextual Recall**: 검색된 컨텍스트가 ground truth에서 모든 관련 정보를 포함하는지 측정
- **Entity Recognition**: 쿼리에서 의미 있는 단위(엔티티)를 식별하는 과정
- **Query Expansion**: 원본 쿼리에 관련 용어를 추가하여 검색 범위 확장
- **Adaptive Top-K**: 쿼리 복잡도에 따라 검색 결과 수를 동적으로 조정
- **Multi-Hop Retrieval**: 문서 간 인용 관계를 따라 관련 문서를 추가 검색
- **Hypernym**: 상위 개념 (예: 등록금 → 학사 → 행정)

### References

- RAGAS Evaluation Metrics: https://docs.ragas.io/
- Information Retrieval: Manning, Raghavan, Schütze
- Entity Recognition: NER techniques and patterns

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-07 | manager-strategy | Initial SPEC creation for Contextual Recall improvement |

---

**SPEC Status**: Planned
**Next Phase**: /moai:2-run SPEC-RAG-SEARCH-001 (DDD로 구현)
**Estimated Completion**: 2026-02-21 (2주)
