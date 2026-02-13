# Acceptance Criteria: SPEC-CHUNK-002

## Overview

이 문서는 HWPX Full Reparse and Quality Analysis 기능의 인수 기준을 정의합니다. 모든 기준은 Gherkin 형식(Given-When-Then)으로 작성되었습니다.

---

## Functional Acceptance Criteria

### AC-001: 파일 발견 (File Discovery)

**Given** data/input/ 디렉터리에 HWPX 파일들이 존재함
**When** reparse-hwpx 명령어가 실행됨
**Then** 시스템은 모든 .hwpx 파일을 발견하고 목록화함

**Test Scenarios**:

```gherkin
Scenario: HWPX 파일 발견
  Given data/input/ 디렉터리에 10개의 .hwpx 파일이 존재함
  And data/input/ 디렉터리에 3개의 .txt 파일이 존재함
  When discover_hwpx_files() 함수가 호출됨
  Then 반환된 목록에는 10개의 파일만 포함됨
  And 모든 파일은 .hwpx 확장자를 가짐
  And 파일 목록은 알파벳 순으로 정렬됨

Scenario: 빈 입력 디렉터리
  Given data/input/ 디렉터리가 비어있음
  When reparse-hwpx 명령어가 실행됨
  Then "발견된 HWPX 파일이 없습니다" 메시지가 표시됨
  And 명령어는 정상 종료됨 (exit code 0)

Scenario: 존재하지 않는 입력 디렉터리
  Given 지정된 입력 디렉터리가 존재하지 않음
  When reparse-hwpx 명령어가 실행됨
  Then 에러 메시지가 표시됨
  And 명령어는 에러로 종료됨 (exit code 1)
```

---

### AC-002: 전체 재파싱 with 백업 (Full Reparse with Backup)

**Given** 기존 hwpx_direct.json 파일이 존재함
**When** 해당 HWPX 파일이 재처리됨
**Then** 시스템은 기존 파일을 백업하고 새로운 출력을 생성함

**Test Scenarios**:

```gherkin
Scenario: 기존 출력 파일 백업
  Given data/output/규정집9-349_hwpx_direct.json 파일이 존재함
  And 파일 크기는 1.5MB임
  When 해당 HWPX 파일이 재처리됨
  Then 백업 파일이 생성됨
  And 백업 파일명은 "규정집9-349_hwpx_direct_YYYYMMDD_HHMMSS.json.bak" 형식임
  And 백업 파일 크기는 1.5MB임
  And 새로운 hwpx_direct.json 파일이 생성됨

Scenario: 기존 출력 없이 신규 생성
  Given data/output/규정집9-349_hwpx_direct.json 파일이 존재하지 않음
  When 해당 HWPX 파일이 처리됨
  Then 백업 파일이 생성되지 않음
  And 새로운 hwpx_direct.json 파일이 직접 생성됨

Scenario: RAG 최적화 출력 생성
  Given HWPX 파일이 성공적으로 파싱됨
  When 출력 파일이 생성됨
  Then {filename}_hwpx_direct.json 파일이 생성됨
  And {filename}_hwpx_direct_rag.json 파일이 생성됨
  And 두 파일 모두 유효한 JSON 형식임
```

---

### AC-003: 누락 출력 신규 파싱 (Fresh Parse for Missing Output)

**Given** HWPX 파일에 대한 출력 파일이 존재하지 않음
**When** 해당 HWPX 파일이 처리됨
**Then** 시스템은 백업 없이 새로운 출력을 직접 생성함

**Test Scenarios**:

```gherkin
Scenario: 신규 파일 처리
  Given data/input/새규정.hwpx 파일이 존재함
  And data/output/새규정_hwpx_direct.json 파일이 존재하지 않음
  When 해당 HWPX 파일이 처리됨
  Then 백업 파일이 생성되지 않음
  And 새규정_hwpx_direct.json 파일이 생성됨
  And 새규정_hwpx_direct_rag.json 파일이 생성됨

Scenario: 부분 출력 파일 존재
  Given data/input/규정.hwpx 파일이 존재함
  And data/output/규정_hwpx_direct.json 파일만 존재함
  And data/output/규정_hwpx_direct_rag.json 파일은 없음
  When 해당 HWPX 파일이 처리됨
  Then 규정_hwpx_direct.json은 백업됨
  And 두 출력 파일 모두 새로 생성됨
```

---

### AC-004: 품질 분석 리포트 (Quality Analysis Report)

**Given** 모든 HWPX 파일 처리가 완료됨
**When** 리포트 생성이 요청됨
**Then** 시스템은 종합 품질 분석 리포트를 생성함

**Test Scenarios**:

```gherkin
Scenario: 품질 리포트 생성
  Given 10개의 HWPX 파일이 성공적으로 처리됨
  When 리포트 생성이 완료됨
  Then data/output/quality_analysis_report.json 파일이 생성됨
  And 리포트에는 다음 필드가 포함됨:
    | 필드명                | 설명                    |
    | generated_at          | 생성 시간                |
    | total_files           | 처리된 파일 수           |
    | successful_files      | 성공한 파일 수           |
    | failed_files          | 실패한 파일 수           |
    | total_chunks          | 총 청크 수               |
    | chunk_statistics      | 타입별 분포              |
    | avg_hierarchy_depth   | 평균 계층 깊이           |
    | max_hierarchy_depth   | 최대 계층 깊이           |

Scenario: 청크 타입별 통계
  Given 처리된 파일들의 청크 데이터가 존재함
  When 리포트가 생성됨
  Then chunk_statistics.by_type에 다음 타입별 개수가 포함됨:
    | 타입       | 예상 포함 여부 |
    | chapter    | Yes           |
    | section    | Yes           |
    | article    | Yes           |
    | paragraph  | Yes           |
    | item       | Yes           |
    | subitem    | Yes           |

Scenario: 파일별 상세 정보
  Given 10개의 파일이 처리됨
  When 리포트가 생성됨
  Then file_details 배열에 10개의 항목이 포함됨
  And 각 항목에는 filename, chunks, hierarchy_depth, processing_time_ms가 포함됨
```

---

### AC-005: 에러 처리 및 복구

**Given** 처리 중 에러가 발생할 수 있음
**When** 특정 파일 처리에 실패함
**Then** 시스템은 에러를 로그하고 다음 파일로 계속 진행함

**Test Scenarios**:

```gherkin
Scenario: 개별 파일 에러 격리
  Given data/input/에 5개의 파일이 존재함
  And 그 중 1개 파일이 손상됨
  When reparse-hwpx가 실행됨
  Then 4개의 파일은 정상 처리됨
  And 1개의 파일은 에러로 로그됨
  And 전체 프로세스는 완료됨
  And 리포트에 실패한 파일이 기록됨

Scenario: 파일 무결성 검증 실패
  Given 크기가 0바이트인 HWPX 파일이 존재함
  When 해당 파일이 처리됨
  Then "파일 크기가 0바이트입니다" 에러가 로그됨
  And 해당 파일은 건너뛰어짐
  And 다음 파일 처리가 계속됨

Scenario: 읽기 권한 없음
  Given 읽기 권한이 없는 HWPX 파일이 존재함
  When 해당 파일이 처리됨
  Then "파일 읽기 권한이 없습니다" 에러가 로그됨
  And 해당 파일은 건너뛰어짐
  And 다음 파일 처리가 계속됨
```

---

## Non-Functional Acceptance Criteria

### AC-006: 성능 기준

**Given** 표준 크기의 HWPX 파일
**When** 파일이 처리됨
**Then** 처리 시간과 메모리 사용량이 기준 이내임

**Test Scenarios**:

```gherkin
Scenario: 파일당 처리 시간
  Given 크기가 5MB 이하인 HWPX 파일
  When 해당 파일이 처리됨
  Then 처리 시간은 60초 이내임

Scenario: 메모리 사용량
  Given 10개의 HWPX 파일이 처리됨
  When 전체 처리가 진행 중임
  Then 피크 메모리 사용량은 500MB 이하임

Scenario: 디스크 쓰기 최적화
  Given 10개의 HWPX 파일이 처리됨
  When 각 파일 처리가 완료됨
  Then 각 파일당 최대 3개의 디스크 쓰기 발생:
    | 쓰기 작업              | 최대 횟수 |
    | 백업 생성              | 1        |
    | 표준 JSON 쓰기         | 1        |
    | RAG JSON 쓰기          | 1        |
```

---

### AC-007: 명령어 인터페이스

**Given** 사용자가 CLI를 통해 명령어 실행
**When** 다양한 옵션으로 명령어가 실행됨
**Then** 적절한 동작과 피드백이 제공됨

**Test Scenarios**:

```gherkin
Scenario: 기본 실행
  Given data/input/에 HWPX 파일들이 존재함
  When "uv run reparse-hwpx" 명령어가 실행됨
  Then 기본 입력 디렉터리(data/input/)가 사용됨
  And 기본 출력 디렉터리(data/output/)가 사용됨
  And 진행률이 표시됨
  And 완료 메시지가 표시됨

Scenario: 커스텀 디렉터리 지정
  Given 사용자가 커스텀 경로를 지정함
  When "uv run reparse-hwpx --input-dir /custom/input --output-dir /custom/output" 실행됨
  Then 지정된 입력 디렉터리가 사용됨
  And 지정된 출력 디렉터리가 사용됨

Scenario: 상세 모드
  Given verbose 모드가 활성화됨
  When "uv run reparse-hwpx --verbose" 실행됨
  Then 각 파일 처리 시 상세 로그가 출력됨
  And 청킹 세부 정보가 표시됨

Scenario: 드라이 런
  Given dry-run 모드가 활성화됨
  When "uv run reparse-hwpx --dry-run" 실행됨
  Then 발견된 파일 목록이 표시됨
  And 예상 처리 항목이 표시됨
  And 실제 파일 처리는 수행되지 않음
  And 출력 파일이 생성되지 않음
```

---

### AC-008: 출력 파일 검증

**Given** 파일 처리가 완료됨
**When** 출력 파일이 생성됨
**Then** 모든 출력 파일은 유효한 JSON 형식이어야 함

**Test Scenarios**:

```gherkin
Scenario: JSON 유효성 검증
  Given HWPX 파일 처리가 완료됨
  When 출력 파일이 생성됨
  Then {filename}_hwpx_direct.json은 유효한 JSON임
  And {filename}_hwpx_direct_rag.json은 유효한 JSON임
  And JSON 파싱 에러가 없음

Scenario: 필수 필드 검증
  Given 출력 JSON 파일이 생성됨
  When JSON이 로드됨
  Then 다음 필수 필드가 존재함:
    | 필드            | 위치          |
    | rag_enhanced    | 루트          |
    | rag_chunk_splitting | 루트      |
    | children        | 루트          |
    | type            | 각 노드       |
    | display_no      | 각 노드       |
    | text            | 각 노드       |
    | chunk_level     | 각 노드       |

Scenario: RAG 최적화 검증
  Given RAG 최적화 JSON이 생성됨
  When JSON이 로드됨
  Then 각 노드에 embedding_text 필드가 존재함
  And embedding_text는 비어있지 않음
  And token_count 필드가 존재함
```

---

## Integration Acceptance Criteria

### AC-009: CHUNK-001 통합

**Given** SPEC-CHUNK-001 파서가 구현됨
**When** reparse-hwpx가 실행됨
**Then** CHUNK-001의 청킹 로직이 정상적으로 활용됨

**Test Scenarios**:

```gherkin
Scenario: 계층 구조 유지
  Given CHUNK-001에서 chapter, section 감지가 구현됨
  When HWPX 파일이 처리됨
  Then 출력에 chapter 타입 청크가 포함됨
  And 출력에 section 타입 청크가 포함됨
  And 계층 구조가 올바르게 중첩됨

Scenario: 최대 깊이 지원
  Given CHUNK-001에서 최대 6단계 계층이 지원됨
  When 6단계 계층을 가진 문서가 처리됨
  Then 모든 6단계가 출력에 반영됨
  And 깊이 계산이 정확함
```

---

## Edge Cases

### EC-001: 특수 파일명 처리

```gherkin
Scenario: 한글 파일명
  Given "규정집9-349(최종).hwpx" 파일이 존재함
  When 해당 파일이 처리됨
  Then 출력 파일명은 "규정집9-349(최종)_hwpx_direct.json"임
  And 파일 처리가 성공함

Scenario: 공백 포함 파일명
  Given "학사 규정.hwpx" 파일이 존재함
  When 해당 파일이 처리됨
  Then 출력 파일명이 올바르게 생성됨
  And 파일 처리가 성공함

Scenario: 특수문자 포함 파일명
  Given "규정_v1.0.hwpx" 파일이 존재함
  When 해당 파일이 처리됨
  Then 출력 파일명이 올바르게 생성됨
  And 특수문자가 적절히 처리됨
```

---

## Quality Gate Checklist

### Pre-Merge Requirements

- [ ] 모든 기능 테스트 통과 (AC-001 ~ AC-005)
- [ ] 모든 비기능 테스트 통과 (AC-006 ~ AC-008)
- [ ] 통합 테스트 통과 (AC-009)
- [ ] 에지 케이스 테스트 통과 (EC-001)
- [ ] 코드 리뷰 완료
- [ ] 문서화 업데이트
- [ ] 기존 기능에 회귀 없음

### Test Execution Commands

```bash
# 단위 테스트
pytest tests/test_reparse_hwpx.py -v

# 커버리지 리포트
pytest tests/test_reparse_hwpx.py --cov=src/regulation_manager/commands --cov=src/regulation_manager/analysis --cov-report=term-missing

# 린팅
ruff check src/regulation_manager/commands/reparse_hwpx.py
ruff check src/regulation_manager/analysis/

# 통합 테스트
pytest tests/ -k "reparse or quality" -v

# 전체 테스트 스위트
pytest tests/ -v
```

---

## Definition of Done

요구사항은 다음 조건이 모두 충족될 때 **DONE**으로 간주됨:

1. **Implemented**: 코드 변경이 완료되고 코딩 표준을 준수함
2. **Tested**: 단위 테스트와 통합 테스트가 85%+ 커버리지로 통과함
3. **Documented**: 함수 docstring과 사용자 가이드가 완료됨
4. **Reviewed**: 최소 1명 이상의 리뷰어가 코드 리뷰를 승인함
5. **Integrated**: 변경사항이 main 브랜치에 병합됨
6. **Verified**: 모든 인수 기준이 충족되고 검증됨

---

## Sign-off

| Role            | Name | Date       | Status |
| --------------- | ---- | ---------- | ------ |
| Developer       |      |            |        |
| Code Reviewer   |      |            |        |
| QA              |      |            |        |
| Product Owner   |      |            |        |
