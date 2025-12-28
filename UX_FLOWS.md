# UX Flows (Web + MCP)

## Web Flow A: "전문 보기"

### Entry Triggers
- Query contains: "전문", "전체", "全文", "규정 전체"
- User clicks "전문 보기" on a regulation result

### Step-by-step
1) 사용자 입력
   - 예: "교원인사규정 전문"
2) 규정명 정합성 확인
   - 여러 규정이 매칭되면 후보 리스트 제시
3) 전문 뷰 진입
   - 좌측: 목차 트리 (편/장/절/조)
   - 우측: 본문 패널
4) 부칙/별표 분리
   - 부칙 탭 또는 섹션 분리
5) 조항 이동
   - 조항 번호 클릭 시 해당 본문으로 스크롤

### UI Components
- Regulation Picker (ambiguous case)
- TOC Tree
- Content Panel
- "부칙" / "별표" 분리 탭
- "규정 내 검색" (키워드/조항 번호)

### Success Criteria
- 규정 전체 문맥 확인 가능
- 경로 중복(부칙/부 칙) 표시 없음
- 전문 요청 시 요약 대신 전체 뷰 제공

### Edge Cases
- 규정명 오타 → 유사 규정 제안
- 폐지 규정 → 상태 표시 및 주의 문구

## Web Flow B: "대상 선택/후속 질문"

### Entry Triggers
- 질의에 대상 정보 부족 (예: "징계 기준")
- 교수/학생/직원 모두에서 의미가 가능한 경우

### Step-by-step
1) 질문 입력
2) 시스템이 대상 불명 판단
3) 대상 선택 UI 표시
   - 선택지: 교수 / 학생 / 직원
4) 선택 후 재검색/재질문
5) 선택 결과로 근거와 답변 제공

### UI Components
- Target Selector (segmented control or chips)
- "왜 물었나요?" (선택 이유 안내, optional)
- "선택 후 다시 검색" 버튼

### Success Criteria
- 잘못된 대상 규정 노출 최소화
- 재질문 횟수 감소

## MCP Flow C: "전문 보기"

### Request
```
search_regulations(query="교원인사규정 전문")
```

### Response (Ambiguous)
```
{
  "success": true,
  "type": "clarification",
  "options": ["교원인사규정", "교원복무규정"]
}
```

### Response (Single Match)
```
{
  "success": true,
  "type": "full_view",
  "regulation_name": "교원인사규정",
  "toc": [...],
  "content": [...]
}
```

## MCP Flow D: "대상 선택/후속 질문"

### Response (Needs clarification)
```
{
  "success": true,
  "type": "clarification",
  "reason": "audience_ambiguous",
  "options": ["교수", "학생", "직원"]
}
```

## Instrumentation (Web/MCP)
- ambiguous_query_count
- target_selected_rate
- full_view_open_rate
- doc_search_within_rate

## Implementation Notes
- 대상 판별 기준: QueryAnalyzer 기반 + 규정명 키워드
- 전문 보기 요청 시 결과 리스트 대신 전문 뷰 우선
- MCP는 clarification 응답 스키마 유지
