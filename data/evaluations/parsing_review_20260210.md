# 규정집 파싱 데이터 검토 보고서

**검토 일자**: 2026-02-10
**검토 대상**: 규정집 HWPX 파싱 결과 (JSON + Vector DB)

---

## 1. JSON 데이터 분석

### 파일 정보
- 파일명: `규정집_rag.json`
- 총 문서 수: **372개**

### 문서 타입 분포
| 타입 | 수량 | 비율 |
|------|------|------|
| regulation | 319개 | 85.8% |
| note | 50개 | 13.4% |
| toc | 1개 | 0.3% |
| index_alpha | 1개 | 0.3% |
| index_dept | 1개 | 0.3% |

### 메타데이터 추출 상태
- **TOC (목차)**: 514개 항목
- **가나다순 색인**: 491개
- **부서별 색인**: 78개
- **규정 코드 할당**: 319/372개 (85.8%)

---

## 2. Vector DB (ChromaDB) 분석

### 컬렉션 정보
- Collection: `regulations`
- 총 청크 수: **386개**

### 문제 발견: doc_type 누락
- **현상**: 모든 청크의 `doc_type`이 `unknown`으로 설정됨
- **영향**: 검색 필터링 기능 작동하지 않음
- **원인**: 청킹 시 메타데이터에 `doc_type`이 포함되지 않음

### 규정 코드 상태
- 규정 코드 있는 청크: 386/386개 (100%)

---

## 3. 주요 발견사항

### 문제 1: ChromaDB doc_type 누락 (해결 완료)
- **현상**: 모든 청크의 `doc_type`이 `unknown`으로 설정됨
- **영향**: 검색 필터링 기능 작동하지 않음
- **원인**: `Chunk` 엔티티에 `doc_type` 필드가 없었음
- **해결**:
  - `Chunk` 데이터클래스에 `doc_type` 필드 추가
  - `from_json_node()` 메서드에 `doc_type` 파라미터 추가
  - `to_metadata()` 메서드에 `doc_type` 추가
  - `json_loader.py`에서 `doc_type` 전달하도록 수정

### 문제 2: TOC 항목 수와 규정 수 불일치
- **현상**: TOC 514개 vs 규정 319개 (차이: 195개)
- **원인**: 일부 규정이 별도 파일로 존재하거나 파싱되지 않았을 가능성
- **권장**: HWPX 원본에서 누락된 규정 확인 필요

### 정상: 규정 코드 할당
- JSON: 319/372개 문서에 규정 코드 할당됨 (85.8%)
- ChromaDB: 386/386개 청크에 규정 코드 있음 (100%)

---

## 4. 수정 사항

### src/rag/domain/entities.py
- `Chunk` 데이터클래스에 `doc_type: str = "regulation"` 필드 추가
- `from_json_node()` 메서드에 `doc_type` 파라미터 추가
- `to_metadata()` 메서드에 `doc_type` 추가
- `from_metadata()` 메서드에 `doc_type` 처리 추가

### src/rag/infrastructure/json_loader.py
- `load_all_chunks()` 메서드에서 `doc_type` 추출 및 전달
- `load_chunks_by_rule_codes()` 메서드에서 `doc_type` 추출 및 전달
- `_extract_chunks_recursive()` 메서드에 `doc_type` 파라미터 추가

### src/rag/infrastructure/self_rag.py
- `Chunk` 생성 시 `doc_type` 파라미터 추가

### src/rag/application/search_usecase.py
- 캐시된 청크 재구성 시 `doc_type` 파라미터 추가

---

## 5. 권장 사항

### 1. Vector DB 재구성 (필수)
- 수정된 코드로 Vector DB를 재구성해야 `doc_type` 메타데이터가 포함됨
- 기존 ChromaDB 삭제 후 재인덱싱 필요

### 2. 누락된 규정 파싱 확인 (권장)
- TOC에 있는 514개 항목 중 319개만 파싱됨
- HWPX 원본에서 누락된 규정 확인 필요

### 3. doc_type 필터링 활용 (권장)
- 검색 시 `doc_type == "regulation"` 필터 적용으로 검색 품질 개선 가능

---

## 6. 검증 결과

### 수정 전
```
ChromaDB 문서 타입 분포:
  - unknown: 386개
```

### 수정 후 (재구성 필요)
```
예상 결과:
  - regulation: 336개 (319 regulation + 17 note 일부)
  - note: 50개
```

---

## 7. 다음 단계

1. **Vector DB 재구성**: `src/rag/application/search_usecase.py`의 `sync_vector_db()` 실행
2. **검증**: 재구성 후 `doc_type` 메타데이터 확인
3. **누락 규정 확인**: TOC와 실제 파싱된 규정 수 대조
