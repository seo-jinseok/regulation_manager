# SPEC-RAG-003 Implementation Plan

## Task Decomposition

### Phase 1: Self-RAG 이중언어 수정 (Critical)

#### Task 1.1: 영어 규정 키워드 추가
- **File**: `src/rag/infrastructure/self_rag.py` (Lines 35-51)
- **Action**: `REGULATION_KEYWORDS` 리스트에 영어 키워드 추가
- **Keywords**:
  ```
  "regulation", "rule", "policy", "guideline", "procedure",
  "tuition", "fee", "scholarship", "grant",
  "leave", "absence", "withdrawal", "return",
  "dormitory", "housing", "residence",
  "registration", "course", "credit", "grade", "GPA",
  "graduation", "degree", "diploma", "thesis",
  "professor", "faculty", "teacher", "instructor",
  "visa", "international", "foreign",
  "how", "when", "what", "where", "who", "apply", "submit",
  "requirement", "qualification", "deadline", "period"
  ```
- **Risk**: LOW (additive, no breaking change)
- **Test**: 영어 쿼리 5개가 keyword pre-filter 통과 확인

#### Task 1.2: Self-RAG LLM 프롬프트 이중언어화
- **File**: `src/rag/infrastructure/self_rag.py` (Lines 75-94)
- **Action**: `RETRIEVAL_NEEDED_PROMPT` 수정
- **Changes**:
  - 영어 예시 추가: "How do I...", "What are the requirements for..."
  - 영어 주제 예시: "tuition, scholarship, leave, dormitory, registration"
  - 주요 지침: "This system handles queries in BOTH Korean and English"
  - 기본 원칙: "If the query is about any university topic in ANY language → [RETRIEVE_YES]"
- **Risk**: LOW (prompt improvement)
- **Test**: 영어 쿼리 5개가 LLM 평가에서 [RETRIEVE_YES] 반환 확인

#### Task 1.3: 대학 주제 오버라이드 메커니즘
- **File**: `src/rag/infrastructure/self_rag.py` (needs_retrieval method, ~Line 155)
- **Action**: LLM이 [RETRIEVE_NO] 반환 시 추가 검증 로직
- **Logic**:
  ```python
  UNIVERSITY_TOPIC_WORDS_EN = {
      "tuition", "fee", "scholarship", "leave", "absence",
      "dormitory", "housing", "course", "registration",
      "grade", "graduation", "thesis", "professor", "visa",
      "international", "student", "university", "campus"
  }
  # If LLM says NO but query has 2+ university topic words → override to YES
  ```
- **Risk**: MEDIUM (could cause over-retrieval, but over-retrieval is safer than missed answers)
- **Test**: 영어 쿼리에서 LLM NO + override 동작 확인

#### Task 1.4: 거부 메시지 이중언어화
- **File**: `src/rag/application/search_usecase.py` (Line 2531)
- **Action**: 쿼리 언어 감지 후 적절한 언어로 거부 메시지 반환
- **Implementation**:
  ```python
  def _detect_language(query: str) -> str:
      # Simple heuristic: if mostly ASCII → English, else Korean
      ascii_ratio = sum(1 for c in query if ord(c) < 128) / len(query)
      return "en" if ascii_ratio > 0.7 else "ko"
  ```
- **Risk**: LOW
- **Test**: 영어 쿼리 거부 시 영어 메시지 반환 확인

---

### Phase 2: 답변 생성 품질 개선 (High)

#### Task 2.1: CoT 출력 억제 프롬프트 추가
- **File**: `src/rag/application/search_usecase.py` (Lines 364-520, fallback prompt)
- **Action**: 답변 생성 프롬프트에 CoT 억제 지침 추가
- **Addition**:
  ```
  ## ⚠️ 출력 형식 (CRITICAL)
  - 사용자에게 직접적인 답변만 제공하세요
  - 절대 출력하지 말 것: 분석 과정, 페르소나 분석, 제약 조건 체크리스트, "Analyze the User's Request" 등
  - 답변은 한국어로 시작하세요 (영어 분석 과정 없이)
  - 번호 매기기 분석 단계(1. Analyze, 2. Check constraints...) 절대 금지
  ```
- **Risk**: LOW (prompt addition)
- **Test**: 응답에 "Analyze", "Constraint", "User Persona" 패턴 없음 확인

#### Task 2.2: 답변 후처리 CoT 제거
- **File**: `src/rag/application/search_usecase.py` (답변 생성 후)
- **Action**: LLM 응답에서 CoT 패턴을 제거하는 후처리 함수 추가
- **Patterns to strip**:
  - `^\d+\.\s+\*\*Analyze.*?\*\*.*?(?=\n[^0-9\s*]|\Z)` (분석 단계)
  - `\*\*User Persona:\*\*.*?\n` (페르소나 분석)
  - `\*\*Constraint.*?\*\*.*?\n` (제약 조건)
- **Risk**: MEDIUM (정규식이 유효한 답변 내용을 잘못 제거할 수 있음)
- **Mitigation**: 현재 통과 쿼리 5개에 대한 characterization test 먼저 작성
- **Test**: 통과 쿼리 답변 보존 + CoT 제거 확인

#### Task 2.3: 답변 완전성 프롬프트 강화
- **File**: `src/rag/application/search_usecase.py` (answer prompt)
- **Action**: 구체적 정보 추출 지침 추가
- **Addition**:
  ```
  ## 📋 답변 완전성 요구사항
  - 절차가 있으면 단계별로 나열하세요
  - 기한/기간 정보가 있으면 반드시 포함하세요
  - 자격 요건이 있으면 모두 나열하세요
  - 필요 서류가 있으면 목록으로 제공하세요
  - 정보가 문맥에 없으면 "해당 정보는 규정에서 확인되지 않습니다"라고 명시하세요
  ```
- **Risk**: LOW
- **Test**: 완전성 점수(Completeness) 0.413 → 0.650+ 목표

---

### Phase 3: 검색 품질 + 환각 방지 (Medium)

#### Task 3.1: 검색 후 관련성 필터링
- **File**: `src/rag/application/search_usecase.py` (search → generation 사이)
- **Action**: 검색 결과에서 낮은 관련성 문서 필터링
- **Implementation**:
  - 문서 relevance score < 0.25 → 제외
  - 필터 후 문서 0개 → "정보 없음" 응답 생성 (환각 방지)
- **Risk**: MEDIUM (과도한 필터링 시 유효 정보 손실)
- **Mitigation**: 임계값을 설정 가능하게 구현
- **Test**: Q11(휴직→비품), Q10(등록금→비품) 같은 환각 쿼리에서 무관문서 제거 확인

#### Task 3.2: 영어 쿼리 Dense 가중치 조정
- **File**: `src/rag/infrastructure/hybrid_search.py`
- **Action**: 영어 쿼리 감지 시 Dense search 가중치 증가
- **Logic**:
  - Korean query: dense_weight=0.5, sparse_weight=0.5 (현재 기본값)
  - English query: dense_weight=0.8, sparse_weight=0.2 (BGE-M3 다국어 활용)
- **Risk**: LOW (BGE-M3가 이미 다국어 지원)
- **Test**: 영어 쿼리에 대한 context_relevance 점수 향상 확인

#### Task 3.3: Corrective RAG 임계값 조정
- **File**: `src/rag/infrastructure/retrieval_evaluator.py`
- **Action**: 동적 임계값 리뷰 및 조정
- **Current**: simple=0.3, medium=0.4, complex=0.5
- **Proposed**: simple=0.35, medium=0.45, complex=0.55 (약간 상향)
- **Risk**: LOW
- **Test**: 임계값 변경 후 Corrective RAG 트리거 빈도 확인

#### Task 3.4: 환각 방지 가드레일
- **File**: `src/rag/application/search_usecase.py`
- **Action**: 관련 정보 없을 때 명확한 "정보 없음" 응답 생성
- **Template**:
  ```
  죄송합니다. "{질문 키워드}"에 대한 규정 정보를 찾을 수 없습니다.
  
  ▶ 다음을 시도해 보세요:
  - 다른 키워드로 검색: (관련 키워드 제안)
  - 관련 부서 문의: (해당 부서명, 가능한 경우)
  ```
- **Risk**: LOW
- **Test**: 정보 없는 쿼리에서 가이드 메시지 제공 확인

---

## Dependency Graph

```
Phase 1 (Independent tasks, can parallel):
  Task 1.1 ─┐
  Task 1.2 ─┤→ Phase 1 Complete
  Task 1.3 ─┤   (Self-RAG bilingual)
  Task 1.4 ─┘

Phase 2 (Depends on Phase 1 for testing):
  Task 2.1 ─┐
  Task 2.2 ─┤→ Phase 2 Complete
  Task 2.3 ─┘   (Answer quality)

Phase 3 (Independent of Phase 2):
  Task 3.1 ─┐
  Task 3.2 ─┤→ Phase 3 Complete
  Task 3.3 ─┤   (Retrieval + hallucination)
  Task 3.4 ─┘
```

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| LLM | GLM-4.7-flash (LM Studio) | Latest |
| Embedding | BGE-M3 | Latest |
| Vector DB | ChromaDB | Latest |
| BM25 | KoNLPy | Latest |
| Testing | pytest | Latest |
| Package | uv | Latest |

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Self-RAG override causing over-retrieval | Medium | Low | Conservative override (2+ topic words) |
| CoT stripping removing valid content | Low | High | Characterization tests for passing queries |
| Relevance filter removing useful docs | Medium | Medium | Configurable threshold |
| Performance degradation | Low | Medium | No heavy computation added |
| Regression in passing queries | Low | High | Run all 30 queries before/after each phase |

---

## Effort Estimation

| Phase | Tasks | Files Modified | Tests Added |
|-------|-------|---------------|-------------|
| Phase 1 | 4 | 2 | 10+ |
| Phase 2 | 3 | 1 | 5+ |
| Phase 3 | 4 | 3 | 8+ |
| **Total** | **11** | **4** | **23+** |
