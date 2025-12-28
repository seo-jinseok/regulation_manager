# Actionable Tickets by Module

## Scope
Derived from `USAGE_SCENARIOS.md`, `PRODUCT_BACKLOG.md`, `UX_FLOWS.md`.
Focus on Web + MCP, CLI for testing only.

## Ticket Format
ID | Module | Description | AC | Priority | Depends

---

### SearchUseCase / Hybrid Search
T-001 | SearchUseCase | Auto-detect query mode (Search vs Ask) based on query type | 
- 질문형: Ask
- 규정명/조항: Search
- explicit override allowed
| P0 | -

T-002 | SearchUseCase | Audience detection should influence ranking and filters consistently |
- 교수/학생/직원 추정 결과가 상위 결과에 반영됨
- 대상 불일치 시 안내 문구 반환
| P0 | T-001

T-003 | SearchUseCase | Add "full_view" mode when query intent is "전문/전체" |
- 규정 단위 결과가 반환됨
- "전문" 요청은 요약 대신 전문 모드로 전환
| P0 | -

T-004 | HybridSearcher | Support "clarification required" when multiple regulations match |
- 규정 후보 리스트 반환
- 선택 후 재검색 지원
| P1 | T-003

### Interface: Web (Gradio)
T-101 | Gradio UI | Add "전문 보기" entry UI and view |
- 규정 선택기
- 목차 트리 + 본문 패널
- 부칙/별표 분리 탭
| P0 | T-003

T-102 | Gradio UI | Add target selector for ambiguous queries |
- 교수/학생/직원 선택 UI
- 선택 후 재검색
| P0 | T-002

T-103 | Gradio UI | Add "규정 내 검색" for full-view |
- 조항 번호 이동
- 키워드 하이라이트
| P1 | T-101

T-104 | Gradio UI | Show "last updated" info in results |
- 규정집 갱신 일자 표기
| P1 | -

### Interface: MCP
T-201 | MCP Server | Clarification response schema for ambiguous queries |
- type="clarification"
- options array 제공
| P0 | T-004

T-202 | MCP Server | Full-view response schema |
- type="full_view"
- toc + content 제공
| P0 | T-003

T-203 | MCP Server | Normalize path output (no duplicates) |
- path 정규화 적용
| P0 | -

### Infrastructure / Data
T-301 | JSON Loader | Support regulation-level load for full-view |
- 규정 단위 콘텐츠 반환 가능
| P0 | T-003

T-302 | RAG Enhancement | Store/update "last_updated" metadata |
- 월 1회 업데이트 시 갱신
| P1 | -

### Testing
T-401 | Tests | Add tests for full-view and target selection flows |
- full_view trigger detection
- clarification response path
| P1 | T-101, T-201

T-402 | Tests | Add tests for audience mismatches |
- 대상 불일치 시 페널티/경고
| P1 | T-002
