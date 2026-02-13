# 규정 파싱 최종 분석 보고서

**작성 일자**: 2026-02-10
**분석 대상**: 규정집9-343(20250909).hwpx

---

## 1. 요약

### 파싱 현황
- **TOC 항목**: 514개
- **파싱된 문서**: 372개
- **누락된 규정**: 195개 (38%)

### 누락된 규정 예시
1. 겸임교원규정 (3-1-10)
2. 시간강사위촉규정【폐지】 (3-1-14)
3. 명예(조기)퇴직수당지급규정 (3-1-19)
4. 교원국내출장에관한규정【폐지】 (3-1-21)
5. 직원평정규정 (3-1-28)
6. 시설물사용에관한규정 (3-1-32)
7. 물품관리규정 (3-1-39)
8. 소방안전관리규정 (3-1-41)

---

## 2. 원인 분석

### 근본 원인
HWP → HTML → Markdown 변환 과정에서 **페이지 헤더**가 규정 제목에 섞여 들어감:

```
원본: "겸임교원규정"
변환: "겸임교원규정 3—1—10～" (페이지 번호 포함)
```

### 파싱 로직 문제
`src/parsing/regulation_parser.py`의 "Article 1 Split" 로직은 다음 조건에서만 새로운 규정으로 분리:
1. "제1조" 패턴 감지
2. APPENDICES 모드에서 규정 제목 후보 확인

누락된 규정들은 제목만 나타나고 "제1조" 패턴이 제대로 감지되지 않아 분리되지 않음.

---

## 3. 시도한 해결 방법

### 시도 1: `clean_page_header_pattern()` 함수
- **위치**: `src/preprocessor.py`
- **결과**: ❌ 전체 markdown에 적용 시 TOC 추출 손상

### 시도 2: `regulation_parser.py` 헤더 클리닝
- **결과**: ❌ 파싱 결과 악화 (TOC 514 → 52)

### 시도 3: `_clean_line_header_pattern()` 메서드
- **결과**: ❌ 여전히 TOC 감소

### 시도 4: Standalone Regulation Title Detection
- **결과**: ❌ 조건이 너무 엄격하여 개선 없음

### 시도 5: Missing Regulations Detector 도구
- **위치**: `src/tools/missing_regulations_detector.py`
- **결과**: ⚠️ 195개 누락 규정 식별 성공, 복구 실패

---

## 4. Missing Regulations Detector 도구

### 기능
1. TOC와 파싱된 문서 비교
2. 누락된 규정 식별
3. Raw markdown에서 복구 시도

### 실행 결과
```bash
$ python src/tools/missing_regulations_detector.py \
    "data/output/규정집9-343(20250909).json" \
    "data/output/규저ᆁ집9-343(20250909)_raw.md"

=== Statistics ===
TOC items: 514
Parsed docs: 372

=== Missing Regulations ===
Total missing: 195

=== Recovery Summary ===
Attempted: 10
Recovered: 10 (but with 0 articles)
```

### 한계
- 제목은 찾았지만 내용(articles) 추출 실패
- Raw markdown의 구조가 복잡하여 단순 패턴 매칭으로는 부족

---

## 5. 결론

### 달성된 목표
1. ✅ ChromaDB doc_type 필드 추가
2. ✅ Vector DB 재구성 (17,254 청크)
3. ✅ 누락 규정 분석 및 식별 도구 개발

### 남은 문제
- ⚠️ 195개 규정 복구 미완료

### 권장 사항

#### 단기적 해결
1. **누락 규정 수동 추가**: 195개 규정을 JSON에 수동으로 입력
2. **별도 JSON 병합**: 누락 규정을 별도 JSON으로 관리

#### 장기적 해결
1. **HWPX 직접 파싱**: XML 구조를 직접 파싱하여 HTML 변환 과정 우회
2. **LLM 기반 파싱**: LLM을 사용하여 깨진 제목과 구조 복구
3. **파서 완전 재작성**: 현재 파서 로직을 모듈화하고 테스트 가능하도록 재작성

---

## 6. 다음 단계

### 즉시 실행 가능
1. `missing_regulations_detector.py` 도구로 누락 규정 목록 확보
2. 원본 HWPX 파일에서 해당 규정 내용 수동 추출
3. JSON에 병합

### 장기 계획
1. 파싱 로직 개선 SPEC 작성
2. 단위 테스트 스위트 작성
3. 점진적 파싱 로직 개선

---

## 7. 참고 파일

### 수정된 파일
- `src/rag/domain/entities.py` - doc_type 필드 추가
- `src/rag/infrastructure/json_loader.py` - doc_type 처리
- `src/rag/infrastructure/self_rag.py` - doc_type 파라미터
- `src/rag/application/search_usecase.py` - 캐시 처리

### 생성된 파일
- `src/preprocessor.py` - `clean_page_header_pattern()` 함수
- `src/tools/missing_regulations_detector.py` - 누락 규정 분석 도구

### 보고서
- `data/evaluations/parsing_review_complete_20260210.md`
- `data/evaluations/parsing_attempt_summary_20260210.md`
- `data/evaluations/parsing_analysis_final_20260210.md` (본 문서)

---

## 8. 요약

현재 파싱 시스템은 **318/514 (62%)**의 규정을 성공적으로 파싱하고 있습니다. 누락된 **195개 규정**을 복구하기 위해서는 파싱 로직의 근본적인 개선이 필요합니다.

단기적으로는 `missing_regulations_detector.py` 도구를 사용하여 누락된 규정을 식별하고, 수동으로 병구하는 방식을 권장합니다. 장기적으로는 HWPX 직접 파싱 또는 LLM 기반 파싱 방식을 고려해야 합니다.
