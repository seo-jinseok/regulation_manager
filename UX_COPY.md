# UX Copy and Component Specs (Web UI)

## Global Principles
- Always show regulation name + article reference when available.
- Avoid hard error tones; use guidance.
- If ambiguous, ask for clarification instead of guessing.

## Components

### 1) Search/Ask Input
**Label**
- "검색어 또는 질문"

**Placeholder**
- "예: 교원 연구년 자격은? / 교원인사규정 전문(원문) / 제15조"

**Helper Text**
- "질문은 자동으로 답변 모드로 전환됩니다."

---

### 2) Mode Auto Banner
**Display Rule**
- Show after query classification.

**Copy**
- "실행 모드: {Search|Ask|Full View}"

---

### 3) Target Selector (Ambiguous Query)
**Title**
- "대상 선택이 필요합니다"

**Body**
- "질문이 교수/학생/직원 중 누구를 대상으로 하는지 명확하지 않습니다."

**Options**
- "교수"
- "학생"
- "직원"

**Primary CTA**
- "선택 후 다시 검색"

**Secondary CTA**
- "대상 없이 계속하기"

---

### 4) Full View Entry
**Trigger**
- Query contains "전문", "원문", "전체", "全文"
- Or "전문 보기" button click

**Entry Banner**
- "전문 보기 모드로 전환했습니다"

---

### 5) Full View Layout
**Left Panel**
- "목차"

**Right Panel**
- "본문"

**Tabs**
- "본문"
- "부칙"
- "별표"

**Search Within**
- Label: "규정 내 검색"
- Placeholder: "조항 번호 또는 키워드 입력"

---

### 6) Answer Output (Ask)
**Header**
- "답변"

**Evidence Section**
- "근거 규정"

**No Evidence**
- "직접적인 근거를 찾지 못했습니다. 관련 규정 원문을 확인하세요."

---

### 7) Search Results (Search)
**Columns**
- 규정명 / 조항 / 점수

**Row CTA**
- "전문 보기"
- "원문 보기"

---

### 8) Update Info
**Copy**
- "최종 업데이트: {YYYY-MM-DD}"

---

### 9) Error States
**No Results**
- "검색 결과가 없습니다. 다른 표현으로 다시 시도해보세요."

**DB Empty**
- "데이터베이스가 비어 있습니다. 관리자에게 동기화를 요청하세요."

---

## MCP Response Copy

### Clarification
```
type: "clarification"
message: "대상이 모호합니다. 교수/학생/직원 중 하나를 선택해주세요."
options: ["교수", "학생", "직원"]
```

### Full View
```
type: "full_view"
message: "요청하신 규정 전문입니다."
```
