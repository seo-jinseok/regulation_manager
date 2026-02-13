# HWPX Direct Parser Chunk 분할 구현 계획

## Context

사용자가 "이전 HWP 파일은 RAG 최적화가 잘 되었는데, 왜 지금은 그렇지 못한가?"라고 질문했습니다. 분석 결과 RAG 최적화는 이미 적용되어 있으나, **chunk가 너무 커서** 검색 정확도가 떨어지는 것이 문제였습니다. 사용자는 **chunk 분할 추가**를 선택했습니다.

## 문제 요약

| 구분 | 기존 파이프라인 | HWPX Direct Parser |
|------|---------------|-------------------|
| chunk 수 | 20,314개 | 3,147개 |
| chunk당 토큰 | ~10개 | ~364개 |
| 커버리지 | 62% | 100% |

## 구현 계획

### 접근 방식

HWPX Direct Parser의 `articles` 구조를 기존 파이프라인의 children 중첩 구조로 변환합니다.

### 기존 구조 (children 중첩)

```
content: [
  {
    type: "chapter",
    display_no: "제1장",
    children: [
      {
        type: "article",
        display_no: "제1조",
        text: "...",
        children: [
          { type: "paragraph", display_no: "①", text: "..." },
          { type: "item", display_no: "1.", text: "..." }
        ]
      }
    ]
  }
]
```

### HWPX Direct Parser 현재 구조

```
content: [
  {
    type: "article",
    display_no: "제1조",
    text: "전체 텍스트 (큼)",
    level: "article"
  }
]
```

### 목표 구조

```
content: [
  {
    type: "article",
    display_no: "제1조",
    text: "조 제목",
    children: [
      { type: "paragraph", display_no: "①", text: "항 내용" },
      { type: "item", display_no: "1.", text: "호 내용" },
      { type: "subitem", display_no: "가.", text: "목 내용" }
    ]
  }
]
```

## 구현 단계

### Step 1: `src/enhance_for_rag.py`에 chunk 분할 함수 추가

새로운 함수 `split_article_into_chunks(article: Dict) -> Dict` 추가:
- article의 `content` 필드에서 항(①②③), 호(1.2.3.), 목(가나다) 패턴 추출
- children 배열로 변환
- 각 children에 RAG 필드 추가

### Step 2: `enhance_document_for_hwpx()` 함수 추가

HWPX Direct Parser 출력을 위한 전용 함수:
- `articles` 배열을 children 중첩 구조로 변환
- `content` 배열 재구성

### Step 3: `enhance_json()` 함수 수정

```python
def enhance_json(data: Dict[str, Any]) -> Dict[str, Any]:
    docs = data.get("docs", [])

    # HWPX Direct Parser 감지 (parsing_method 필드)
    is_hwpx_direct = data.get("parsing_method") == "hwpx_direct"

    for doc in docs:
        if is_hwpx_direct:
            enhance_document_for_hwpx(doc)  # 새로운 함수
        else:
            enhance_document(doc)  # 기존 함수

    return data
```

## 수정 파일

| 파일 | 수정 내용 |
|------|----------|
| `src/enhance_for_rag.py` | chunk 분할 로직 추가 |

## 검증 방법

1. `규정집9-343(20250909).json`으로 테스트
2. chunk 수가 3,147개 → ~20,000개로 증가 확인
3. RAG 필드 유지 확인
4. 검색 정확도 테스트

## 예상 결과

- chunk 수: ~20,000개 (기존 파이프라인과 유사)
- 커버리지: 100% 유지
- 검색 정확도: 향상
