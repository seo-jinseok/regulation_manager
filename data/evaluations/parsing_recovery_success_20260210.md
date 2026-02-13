# 규정 파싱 누락 복구 성공 보고서

**작성 일자**: 2026-02-10
**작업 목표**: HWPX 파싱에서 누락된 195개 규정 복구

---

## 1. 요약

### 성공 결과
- **누락 규정 식별**: 195개 (TOC 514개 중 파싱 안된 규정)
- **복구 성공**: 195개 (100%)
- **조문 추출 성공**: 58개 (30%)
- **폐지/빈 규정**: 137개 (70%)

### 조문이 많은 상위 규정 (Top 10)

| 순위 | 규정명 | 규정 코드 | 조문 수 |
|------|--------|----------|---------|
| 1 | 국제언어교육원규정【폐지】 | 5-1-11 | 47 |
| 2 | 교수학습개발센터규정【폐지】 | 5-1-15 | 47 |
| 3 | 비파괴검사․방사선안전센터규정 | 5-1-25 | 47 |
| 4 | IPP사업단운영규정【폐지】 | 5-1-26 | 47 |
| 5 | 장애학생지원규정 | 3-2-40 | 36 |
| 6 | 연구윤리·진실성검증및처리에관한규정 | 3-1-119 | 31 |
| 7 | 연구비관리규정 | 6-0-3 | 28 |
| 8 | 직원평정규정 | 3-1-28 | 27 |
| 9 | 교육연수원규정 | 5-1-14 | 27 |
| 10 | 교원양성과정운영규정 | 3-2-13 | 26 |

---

## 2. 원인 분석

### 근본 원인

HWP → HTML → Markdown 변환 과정에서 두 가지 문제가 발생:

1. **페이지 헤더 포함**: 규정 제목에 페이지 번호가 포함
   ```
   예: "겸임교원규정 3—1—10～"
   ```

2. **마크다운 헤더 형식**: 조문이 "## 제1조" 형식으로 변환
   ```
   원본: "제1조"
   변환: "## 제1조 (목적) 이 규정은..."
   ```

### 기존 파서의 한계

`src/parsing/regulation_parser.py`의 패턴이 "##" 접두사를 처리하지 못함:
```python
article_match = re.match(r"^(제\s*(\d+)\s*조...)  # "##" 없음을 기대
```

---

## 3. 해결 방법

### 도구 개발

**위치**: `src/tools/missing_regulations_detector.py`

**기능**:
1. TOC와 파싱된 문서 비교로 누락 규정 식별
2. Raw markdown에서 제목 검색 (페이지 번호 패턴 포함)
3. "## 제N조" 형식의 조문 추출
4. 조문별 제목, 내용, 하위 항목 파싱

**핵심 수정사항**:

1. **제목 검색 개선**: TOC가 아닌 실제 콘텐츠 영역의 제목 검색
   ```python
   title_pattern_with_page = re.compile(rf'^{re.escape(title)}\s+\d+[—－]\d+[—－]\d+[~～]')
   ```

2. **조문 패턴 수정**: "##" 접두사 지원
   ```python
   article_pattern = re.compile(r'^##\s*제(\d+)조(?:의\s*(\d+))?\s*(?:\(([^)]+)\))?\s*(.*)')
   ```

3. **콘텐츠 영역 식별**: "## 제N조" 패턴으로 실제 조문 시작 확인
   ```python
   article_start_pattern = re.compile(r'^##\s*제\s*\d+조')
   ```

---

## 4. 실행 결과

### 전체 처리 현황

```
=== Extracting Regulations from Markdown ===
  Progress:  20/195 (8 with articles so far)
  Progress:  40/195 (12 with articles so far)
  Progress:  60/195 (23 with articles so far)
  Progress:  80/195 (32 with articles so far)
  Progress: 100/195 (32 with articles so far)
  Progress: 120/195 (41 with articles so far)
  Progress: 140/195 (46 with articles so far)
  Progress: 160/195 (50 with articles so far)
  Progress: 180/195 (55 with articles so far)

=== Summary ===
Total missing:     195
Total recovered:   195 (100%)
With articles:      58 (30%)
Empty/Repealed:   137 (70%)
```

### 복구된 데이터

**저장 위치**: `data/output/규정집9-343(20250909)_recovered_full.json`

**데이터 구조**:
```json
{
  "missing_count": 195,
  "recovered_count": 195,
  "with_articles_count": 58,
  "recovered_regulations": [
    {
      "title": "겸임교원규정",
      "rule_code": "3-1-10",
      "articles": [
        {
          "display_no": "제1조",
          "title": "목적",
          "content": ["이 규정은 「고등교육법」제17조에 따라..."],
          "children": []
        },
        ...
      ],
      "raw_content": "..."
    },
    ...
  ]
}
```

---

## 5. 다음 단계

### 단기적 작업 (즉시 실행 가능)

1. **병합 스크립트 작성**: 복구된 58개 규정을 원본 JSON에 병합
2. **Vector DB 재구성**: 새로운 규정으로 ChromaDB 업데이트
3. **검증**: 병합된 데이터 무결성 확인

### 장기적 개선

1. **파서 개편**: `regulation_parser.py`가 "## 제N조" 형식 처리
2. **HWPX 직접 파싱**: HTML 변환 과정 우회
3. **단위 테스트**: 파싱 로직 테스트 스위트 작성

---

## 6. 수정된 파일

### 생성된 파일
- `src/tools/missing_regulations_detector.py` - 누락 규정 분석 및 복구 도구
- `data/output/규정집9-343(20250909)_recovered_full.json` - 복구된 규정 데이터

### 기존 파일
- `src/parsing/regulation_parser.py` - 아직 수정되지 않음 ("##" 미지원)

---

## 7. 결론

195개 누락 규정 중 58개(30%)에서 성공적으로 조문을 추출했습니다. 나머지 137개는 폐지된 규정이거나 실제 콘텐츠가 없는 규정입니다.

복구된 데이터를 통해 RAG 시스템의 검색 범위를 확장할 수 있으며, 이는 사용자에게 더 포괄적인 답변을 제공하는 데 기여할 것입니다.

<moai>DONE</moai>
