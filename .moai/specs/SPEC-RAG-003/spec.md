---
id: SPEC-RAG-003
version: "1.0.0"
status: draft
created: "2026-02-26"
updated: "2026-02-26"
author: MoAI
priority: critical
---

# HISTORY

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0 | 2026-02-26 | MoAI | Initial SPEC creation from evaluation report analysis |

---

# SPEC-RAG-003: RAG 품질 개선 - Pass Rate 16.7% → 70%+ 달성

## 1. 개요

### 배경

2026-02-26 RAG 품질 평가 결과 (30개 쿼리, 6개 페르소나):
- **Pass Rate**: 16.7% (5/30) — 목표 대비 심각하게 미달
- **Average Score**: 0.534 — 목표 0.800 대비 33% 부족
- **페르소나별 격차**: 학부생 0.868 vs 유학생 0.122 (7배 차이)

### 근본 원인 요약

| 카테고리 | 영향 쿼리 수 | 핵심 원인 | 코드 위치 |
|----------|-------------|-----------|-----------|
| 영어 쿼리 전부 실패 | 5 | Self-RAG 키워드/프롬프트 한국어 전용 | self_rag.py:35-94 |
| 한국어 오분류 | 3 | Self-RAG LLM 과도한 거부 | self_rag.py:119-170 |
| 내부 추론 노출 | 3+ | 답변 프롬프트 CoT 미억제 | search_usecase.py:364-520 |
| 문서 관련성 낮음 | 4 | 검색 품질 + BM25 한국어 전용 | hybrid_search.py |
| 환각 발생 | 3 | 무관문서 필터링 부재 | search_usecase.py:2754+ |
| 완전성 부족 | 3 | 답변 상세도 부족 | search_usecase.py |

### 목표

- **Primary**: Pass Rate 16.7% → **70%+** (21/30)
- **Secondary**: Average Score 0.534 → **0.750+**
- **유학생 페르소나**: 0.122 → **0.600+**
- **교수 페르소나**: 0.195 → **0.600+**
- **회귀 방지**: 현재 통과 5개 쿼리 유지

---

## 2. 스코프

### In-Scope
- Self-RAG 이중언어 지원 (한국어 + 영어)
- Self-RAG 분류 정확도 개선
- 답변 생성 품질 개선 (CoT 제거, 완전성 향상)
- 검색 후 관련성 필터링
- 영어 쿼리 크로스언어 검색 지원

### Out-of-Scope
- 새로운 규정 문서 추가/변환
- 임베딩 모델 교체 (BGE-M3 유지)
- LLM 모델 교체 (GLM-4.7-flash 유지)
- Web UI 변경
- MCP 서버 인터페이스 변경

---

## 3. EARS 요구사항

### Module 1: Self-RAG 이중언어 강화 (Critical)

**EARS-U-001** (Ubiquitous)
The system shall recognize English regulation-related queries with the same accuracy as Korean queries in the Self-RAG evaluation step.

**EARS-E-001** (Event-Driven)
When a query contains English keywords related to university regulations (e.g., "leave of absence", "tuition", "scholarship", "dormitory", "registration", "course", "grade", "visa"), the system shall classify it as requiring regulation search and proceed with retrieval.

**EARS-E-002** (Event-Driven)
When the Self-RAG evaluator invokes the RETRIEVAL_NEEDED_PROMPT for LLM evaluation, the system shall use a bilingual prompt containing examples and markers in both Korean and English.

**EARS-I-001** (Unwanted Behavior)
If the Self-RAG LLM evaluator returns RETRIEVE_NO but the query contains at least 2 words semantically related to university topics (in any language), the system shall override the NO decision and proceed with retrieval.

### Module 2: 거부 플로우 개선 (High)

**EARS-E-003** (Event-Driven)
When the system determines a query does not require regulation search, the system shall return a rejection message in the same language as the input query (Korean for Korean queries, English for English queries).

**EARS-I-002** (Unwanted Behavior)
If a query is rejected by the Self-RAG evaluator, the system shall provide at least one specific suggestion for how to rephrase the query to get better results.

### Module 3: 답변 생성 품질 (High)

**EARS-U-002** (Ubiquitous)
The system shall not expose internal reasoning processes (chain-of-thought analysis, user persona analysis, constraint checklists, step-by-step analysis plans) in the final answer presented to the user.

**EARS-U-003** (Ubiquitous)
The system shall generate citation references using the standard format "규정명 + 제X조" (e.g., "학칙 제15조") for all regulation references in the answer.

**EARS-E-004** (Event-Driven)
When the retrieved context contains information relevant to the user's question, the system shall extract and present specific details (procedures, deadlines, requirements, qualifications) rather than providing generic summaries.

### Module 4: 검색 품질 강화 (Medium)

**EARS-I-003** (Unwanted Behavior)
If all retrieved documents have context relevance score below 0.3, the system shall trigger Corrective RAG to expand the query using synonym expansion and re-retrieve documents.

**EARS-E-005** (Event-Driven)
When the hybrid search returns documents for answer generation, the system shall filter out individual documents with relevance score below a configurable minimum threshold (default: 0.25) before passing them to the LLM.

**EARS-W-001** (State-Driven)
Where the query language is detected as English, the system shall apply Dense search as the primary retrieval method (leveraging BGE-M3's multilingual capability) and optionally translate the query to Korean for the BM25 component.

### Module 5: 환각 방지 강화 (Medium)

**EARS-I-004** (Unwanted Behavior)
If the retrieved context does not contain information relevant to the user's question (all documents below relevance threshold after filtering), the system shall clearly state that the specific regulation information was not found and provide guidance for alternative resources (e.g., 관련 부서명 안내).

**EARS-U-004** (Ubiquitous)
The system shall never present retrieved documents that are semantically irrelevant to the user's question as if they contain relevant information.

---

## 4. 기술 접근 방식

### 아키텍처 변경 범위

```
src/rag/infrastructure/self_rag.py     ← Module 1, 2 (핵심 변경)
src/rag/application/search_usecase.py  ← Module 2, 3, 4, 5 (프롬프트 + 필터링)
src/rag/infrastructure/hybrid_search.py ← Module 4 (영어 쿼리 처리)
src/rag/infrastructure/retrieval_evaluator.py ← Module 4 (임계값 조정)
```

### 구현 전략

**3-Phase 접근법:**

| Phase | 모듈 | 우선순위 | 예상 개선 |
|-------|------|---------|----------|
| Phase 1 | Module 1 + 2 (Self-RAG) | Critical | +8 쿼리 통과 (→ ~43%) |
| Phase 2 | Module 3 (답변 생성) | High | +4 쿼리 통과 (→ ~57%) |
| Phase 3 | Module 4 + 5 (검색 + 환각 방지) | Medium | +4 쿼리 통과 (→ ~70%) |

### 핵심 기술 결정

1. **BGE-M3 활용**: 이미 다국어 지원 → 영어 Dense 검색은 추가 작업 불필요
2. **BM25 Korean-only 유지**: KoNLPy 토크나이저 변경 대신, 영어 쿼리는 Dense 가중치 증가
3. **GLM-4.7-flash 유지**: 프롬프트 최적화로 개선 (모델 교체 없이)
4. **Clean Architecture 준수**: 변경은 infrastructure/application 레이어에 한정

### 제약사항

- LLM: GLM-4.7-flash (로컬, max_tokens=2048)
- 임베딩: BGE-M3 (1024d, multilingual)
- BM25: KoNLPy 한국어 형태소 분석
- 패키지 관리: uv
- 테스트: pytest
- 응답 시간: 현재 대비 20% 이내 증가

---

## 5. 의존성

- Self-RAG 모듈 (`self_rag.py`) — 직접 수정
- 검색 유스케이스 (`search_usecase.py`) — 직접 수정
- 하이브리드 검색기 (`hybrid_search.py`) — 선택적 수정
- 검색 평가기 (`retrieval_evaluator.py`) — 임계값 조정
- 기존 테스트 (`tests/`) — 확장

외부 의존성 없음 (기존 라이브러리로 구현 가능)

---

## 6. 참조

- 평가 리포트: `data/evaluations/rag_quality_eval_20260226_155729_report.md`
- 평가 데이터: `data/evaluations/rag_quality_eval_20260226_155729.json`
- 연구 문서: `.moai/specs/SPEC-RAG-003/research.md`
- 이전 SPEC: SPEC-RAG-001 (7개 컴포넌트), SPEC-RAG-002 (품질 개선, 87.3% 커버리지)
