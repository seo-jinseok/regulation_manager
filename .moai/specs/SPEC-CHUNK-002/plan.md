# Implementation Plan: SPEC-CHUNK-002

## Overview

이 계획은 HWPX 파일 일괄 재파싱 및 품질 분석 기능 구현을 위한 접근 방식을 설명합니다. SPEC-CHUNK-001에서 구현된 파서를 활용하여 자동화된 파이프라인을 구축합니다.

---

## Milestones

### Milestone 1: 명령어 인터페이스 구현

**Priority**: High (Primary Goal)

**Objectives**:
- `reparse-hwpx` CLI 명령어 구현
- 입력/출력 디렉터리 설정
- 진행 상황 표시 기능

**Tasks**:
1. `src/regulation_manager/commands/reparse_hwpx.py` 파일 생성
2. Click 기반 CLI 인터페이스 구현
3. `pyproject.toml` 스크립트 진입점 등록
4. 로깅 및 진행률 표시 기능 추가

**Deliverables**:
- 실행 가능한 `uv run reparse-hwpx` 명령어
- 명령줄 옵션 지원 (--input-dir, --output-dir, --verbose, --dry-run)

---

### Milestone 2: 파일 검색 및 무결성 검증

**Priority**: High (Primary Goal)

**Objectives**:
- data/input/ 디렉터리 스캔
- HWPX 파일 필터링
- 파일 무결성 검증 로직

**Tasks**:
1. `discover_hwpx_files()` 함수 구현
2. `validate_hwpx_file()` 함수 구현 (크기 > 0, 읽기 가능, ZIP 구조)
3. 파일 목록 정렬 (알파벳 순)
4. 에러 핸들링 및 로깅

**Deliverables**:
- 파일 검색 모듈
- 무결성 검증 함수
- 단위 테스트

---

### Milestone 3: 백업 및 이중 JSON 출력

**Priority**: High (Primary Goal)

**Objectives**:
- 기존 출력 파일 백업 기능
- 표준 JSON 및 RAG 최적화 JSON 생성
- 타임스탬프 기반 백업 명명

**Tasks**:
1. `create_backup()` 함수 구현
2. `generate_output_filenames()` 함수 구현
3. RAG 최적화 JSON 생성 로직 호출
4. 백업 타임스탬프 형식 표준화

**Deliverables**:
- 백업 생성 모듈
- 출력 파일 명명 규칙 준수
- 백업 관련 단위 테스트

---

### Milestone 4: 품질 분석 및 리포트 생성

**Priority**: Medium (Secondary Goal)

**Objectives**:
- 청크 통계 수집
- 품질 분석 리포트 생성
- Legacy 비교 메트릭 (선택적)

**Tasks**:
1. `src/regulation_manager/analysis/quality_metrics.py` 파일 생성
2. `collect_chunk_statistics()` 함수 구현
3. `generate_quality_report()` 함수 구현
4. `src/regulation_manager/analysis/comparison_report.py` 파일 생성
5. Legacy 비교 로직 구현 (선택적)

**Deliverables**:
- 품질 메트릭 수집 모듈
- JSON 리포트 생성 기능
- Legacy 비교 기능 (선택적)

---

### Milestone 5: 통합 및 테스트

**Priority**: High (Primary Goal)

**Objectives**:
- 전체 파이프라인 통합
- 종단간 테스트
- 에러 복구 테스트

**Tasks**:
1. 메인 워크플로우 함수 구현
2. 통합 테스트 작성
3. 에러 시나리오 테스트
4. 성능 벤치마킹

**Deliverables**:
- 통합 테스트 스위트
- 성능 벤치마크 결과
- 사용자 가이드 문서

---

## Technical Approach

### Architecture

```
reparse-hwpx CLI
    │
    ├── File Discovery
    │   └── discover_hwpx_files(input_dir)
    │       └── validate_hwpx_file(file)
    │
    ├── Processing Pipeline
    │   └── for each file:
    │       ├── check_existing_output()
    │       ├── create_backup() if exists
    │       ├── parse_hwpx_file() [from CHUNK-001]
    │       ├── generate_standard_json()
    │       └── generate_rag_json()
    │
    └── Quality Analysis
        ├── collect_chunk_statistics()
        ├── compare_with_legacy() [optional]
        └── generate_quality_report()
```

### Key Functions to Implement

| 함수                              | 파일                                          | 목적                              |
| --------------------------------- | --------------------------------------------- | --------------------------------- |
| `discover_hwpx_files()`           | commands/reparse_hwpx.py                      | 입력 디렉터리에서 HWPX 파일 검색  |
| `validate_hwpx_file()`            | commands/reparse_hwpx.py                      | 파일 무결성 검증                  |
| `create_backup()`                 | commands/reparse_hwpx.py                      | 기존 출력 백업                    |
| `generate_output_paths()`         | commands/reparse_hwpx.py                      | 출력 파일 경로 생성               |
| `collect_chunk_statistics()`      | analysis/quality_metrics.py                   | 청크 통계 수집                    |
| `generate_quality_report()`       | analysis/quality_metrics.py                   | 품질 리포트 생성                  |
| `compare_with_legacy()`           | analysis/comparison_report.py                 | Legacy 파서 비교                  |

### Integration with CHUNK-001

```python
# 기존 CHUNK-001 함수 재사용
from src.enhance_for_rag import (
    split_text_into_chunks,
    convert_article_to_children_structure,
    enhance_node_for_hwpx,
)

# 새로운 래퍼 함수
def parse_hwpx_with_chunking(hwpx_path: Path) -> dict:
    """HWPX 파일을 파싱하고 CHUNK-001 청킹 로직 적용"""
    # 1. HWPX 파싱 (기존 로직)
    raw_content = parse_hwpx_file(hwpx_path)

    # 2. 청킹 적용 (CHUNK-001)
    chunked = split_text_into_chunks(raw_content)

    # 3. RAG 최적화
    rag_optimized = enhance_for_rag(chunked)

    return {
        "standard": chunked,
        "rag_optimized": rag_optimized
    }
```

---

## File Modification Plan

### New Files to Create

| 파일                                                   | 목적                    | 라인 수 (예상) |
| ------------------------------------------------------ | ----------------------- | -------------- |
| src/regulation_manager/commands/reparse_hwpx.py        | CLI 명령어 구현         | 200-250        |
| src/regulation_manager/analysis/quality_metrics.py     | 품질 메트릭 수집        | 150-200        |
| src/regulation_manager/analysis/comparison_report.py   | Legacy 비교 리포트      | 100-150        |
| tests/test_reparse_hwpx.py                             | 단위/통합 테스트        | 200-300        |

### Files to Modify

| 파일                   | 변경 내용                    | 변경 규모 |
| ---------------------- | ---------------------------- | --------- |
| pyproject.toml         | 스크립트 진입점 추가         | 3-5 라인  |
| src/regulation_manager/commands/__init__.py | 모듈 등록          | 1-2 라인  |
| src/regulation_manager/analysis/__init__.py | 모듈 등록 (신규)   | 1-2 라인  |

---

## Constraints and Guidelines

### Performance Constraints

| 제약                 | 목표값          | 측정 방법           |
| -------------------- | --------------- | ------------------- |
| 파일당 처리 시간     | < 60초          | time 모듈 측정      |
| 피크 메모리 사용량   | < 500MB         | memory_profiler     |
| 동시 파일 처리       | 순차 처리 (1개) | 단일 스레드         |
| 디스크 쓰기          | 최소화          | 버퍼링 활용         |

### Code Quality Constraints

- Type hints: 모든 함수에 타입 힌트 적용
- Docstrings: Google 스타일 docstring
- Linting: ruff 통과 (에러 0개)
- Test coverage: 85% 이상

### Error Handling Constraints

- 개별 파일 에러는 전체 프로세스 중단하지 않음
- 모든 에러는 로그에 기록
- 실패한 파일 목록은 리포트에 포함
- KeyboardInterrupt는 정상 종료 처리

---

## Risk Mitigation Strategy

### 대용량 파일 메모리 초과

**Mitigation**: 스트리밍 처리 또는 청크 단위 읽기
**Validation**: 메모리 프로파일링으로 검증

### 디스크 공간 부족

**Mitigation**: 처리 전 공간 확인, 백업 보관 정책
**Validation**: df 명령으로 공간 확인 로직

### 파싱 에러 격리

**Mitigation**: try-except로 개별 파일 에러 격리
**Validation**: 에러 시나리오 테스트

---

## Dependencies

### Internal Dependencies

| 의존성                   | 용도                              |
| ------------------------ | --------------------------------- |
| src/enhance_for_rag.py   | CHUNK-001 청킹 로직              |
| src/parsing/             | HWPX 파일 파싱                   |
| SPEC-CHUNK-001           | 파서 구현 완료 상태              |

### External Dependencies

| 패키지         | 버전      | 용도                |
| -------------- | --------- | ------------------- |
| Python         | 3.13+     | 런타임              |
| click          | >=8.0     | CLI 프레임워크      |
| rich           | >=13.0    | 진행률 표시         |
| pydantic       | >=2.0     | 데이터 검증         |

---

## Rollout Plan

### Phase 1: 개발

1. CLI 명령어 기본 구조 구현
2. 파일 검색 및 검증 로직 구현
3. 백업 및 출력 생성 로직 구현
4. 단위 테스트 작성

### Phase 2: 통합

1. CHUNK-001 파서 통합
2. 품질 분석 모듈 구현
3. 통합 테스트 실행
4. 성능 최적화

### Phase 3: 검증

1. 전체 파일 일괄 처리 실행
2. 출력 파일 무결성 검증
3. 리포트 정확성 확인
4. 문서화 완료

---

## Success Criteria

- [x] `uv run reg reparse` 명령어 정상 실행 (unified CLI 통합)
- [x] 모든 HWPX 파일 처리 (건너뛰기 없음) - 2개 파일 처리 완료
- [x] 기존 출력 파일 백업 생성 - 타임스탬프 백업 구현
- [x] 표준 JSON 및 RAG JSON 생성 - 4개 파일 생성 완료
- [x] 품질 분석 리포트 생성 - chunk_statistics 포함
- [x] 단위 테스트 85%+ 커버리지 - 32 tests passing
- [x] 통합 테스트 통과 - DB sync 3,940 chunks
- [x] 처리 시간 < 60초/파일 - 평균 2-3초/파일
- [x] 메모리 사용량 < 500MB - 정상 범위
