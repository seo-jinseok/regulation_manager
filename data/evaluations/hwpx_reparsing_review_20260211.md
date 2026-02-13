# HWPX 재파싱 품질 검토 보고서

**날짜**: 2026-02-11
**파일**: 규정집9-343(20250909).hwpx
**파서 버전**: 3.0.0

---

## 1. 수행 내용

### 1.1 수정 사항

1. **Import 오류 수정** (`src/parsing/validators/completeness_checker.py`)
   - `Tuple` 타입 임포트 누락 수정

2. **CLI 옵션 추가** (`src/rag/interface/unified_cli.py`)
   - `ConvertArgs` 클래스에 `hwpx` 속성 추가
   - `--hwpx` 명령행 옵션 추가

3. **파서 개선** (`src/parsing/hwpx_direct_parser.py`)
   - articles를 content 필드로 변환하는 로직 추가
   - RAG 시스템 호환성 개선

---

## 2. 파싱 결과 분석

### 2.1 전체 통계

| 항목 | 값 |
|------|-----|
| 전체 문서 | 853개 |
| Docs with content | 164개 (19.2%) |
| Docs without content | 689개 (80.8%) |
| 전체 content nodes | 2,939개 |
| 전체 articles | 2,939개 |
| Article detection rate | 100% |

### 2.2 Articles 분포

```
Articles 0개:  689개 규정 (80.8%)
Articles 1개:  11개 규정
Articles 2개:   2개 규정
Articles 7개:  19개 규정
Articles 8개:  19개 규정
...
```

### 2.3 패턴 분석

**규정이 있는 문서 (상위)**:
- 학교법인동의학원정관 (137 articles)
- 직제규정 (140 articles)
- 동의대학교학칙 (89 articles)
- 동의대학교대학원학칙 (56 articles)
- 사무분장규정 (80 articles)

**규정이 없는 문서 (상위)**:
- 대학혁신지원사업운영규정
- 교육성과관리센터규정
- 콜라보교육센터규정
- 겸임교원임용등에관한시행세칙
- 노사협의회운영규정

---

## 3. 원인 분석

### 3.1 규정이 없는 문서의 특징

1. **유형**: "센터규정", "위원회규정" 등 세부 규정
2. **구조**: 조항(article) 형식이 아닌 목록, 지침 형식일 가능성
3. **위치**: HWPX 파일 내에서 별도 구조로 존재할 수 있음

### 3.2 파서 동작 분석

현재 파서는:
1. section0.xml만 파싱
2. "제N조" 패턴으로 article 감지
3. article이 없는 규정은 content가 빔

---

## 4. 개선 제안

### 4.1 단기 개선 (P1)

1. **목차 기반 내용 추출**
   - section1.xml(TOC)에서 규정 위치 정보 추출
   - TOC의 페이지 번호로 실제 내용 위치 찾기
   - 예상 복잡도: 중간

2. **대체 콘텐츠 감지**
   - article이 없는 규정에 텍스트 내용을 content로 추가
   - "규정 전문"을 하나의 content node로 처리
   - 예상 복잡도: 낮음

### 4.2 중기 개선 (P2)

1. **다중 섹션 파싱**
   - section0.xml 외 다른 섹션도 확인
   - 모든 섹션에서 규정 내용 수집
   - 예상 복잡도: 중간

2. **다양한 조항 패턴 지원**
   - "제N조" 외 "제N항", "별표 N" 등 패턴 추가
   - 예상 복잡도: 낮음

### 4.3 장기 개선 (P3)

1. **LLM 기반 구조 분석**
   - 규정 구조를 LLM으로 자동 분석
   - 비정형 규정도 처리 가능
   - 예상 복잡도: 높음

---

## 5. 권장 사항

### 5.1 즉시 조치

1. 대체 콘텐츠 감지 기능 추가 (P1-2)
2. 다양한 조항 패턴 지원 (P2-2)

### 5.2 다음 단계

1. section1.xml TOC 분석으로 규정 위치 파악
2. 위치 기반 내용 추출 구현

---

## 6. 첨부

### 6.1 생성 파일

- `data/output/규정집9-343(20250909).json` - 재파싱 결과
- `data/output/reparsed_quality_report.json` - 품질 리포트

### 6.2 수정 파일

- `src/parsing/validators/completeness_checker.py` - Tuple 임포트 추가
- `src/rag/interface/unified_cli.py` --hwpx 옵션 추가
- `src/parsing/hwpx_direct_parser.py` - articles→content 변환 추가

---

**보고서 작성자**: MoAI Orchestrator
**검토 상태**: PARTIAL (추가 개선 필요)
