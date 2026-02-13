# SPEC-CHUNK-002: HWPX Full Reparse and Quality Analysis

---
id: SPEC-CHUNK-002
version: "1.0.0"
status: Implemented
created: 2026-02-13
updated: 2026-02-13
implemented_date: 2026-02-13
commit: adf1316
author: MoAI
priority: High
---

## HISTORY

| Date       | Version | Author | Changes                             |
| ---------- | ------- | ------ | ----------------------------------- |
| 2026-02-13 | 1.0.0   | MoAI   | Initial SPEC creation               |
| 2026-02-13 | 1.0.0   | MoAI   | Implementation complete (commit: adf1316) |

---

## TAG BLOCK

```yaml
tags:
  - CHUNK-002
  - hwpx-reparse
  - quality-analysis
  - backup-strategy
  - batch-processing
dependencies:
  - CHUNK-001
related_spec:
  - SPEC-CHUNK-001
```

---

## Environment

### System Context

이 SPEC은 SPEC-CHUNK-001에서 구현된 HWPX Direct Parser의 청킹 개선 사항을 활용하여 기존 HWPX 파일들을 일괄 재파싱하고 품질 분석 리포트를 생성하는 기능을 정의합니다.

### Current State Analysis

| 항목                 | 현재 상태                          | 목표 상태                           |
| -------------------- | ---------------------------------- | ----------------------------------- |
| 파서 버전            | HWPX Direct Parser (CHUNK-001)     | 동일 (이미 구현됨)                  |
| 출력 파일            | 일부 파일만 hwpx_direct.json 존재  | 모든 HWPX 파일에 대한 출력 생성     |
| 백업 정책            | 없음                               | 타임스탬프 백업 적용                |
| 품질 분석            | 수동 확인                          | 자동화된 통계 리포트                |
| RAG 최적화           | 선택적                             | 기본 제공                           |

### 대상 파일

- **입력 디렉터리**: `data/input/*.hwpx`
- **출력 디렉터리**: `data/output/`
- **출력 파일 형식**:
  - `{filename}_hwpx_direct.json` - 표준 JSON 출력
  - `{filename}_hwpx_direct_rag.json` - RAG 최적화 JSON 출력

---

## Assumptions

### Technical Assumptions

1. SPEC-CHUNK-001의 HWPX Direct Parser가 정상적으로 동작함
2. 모든 HWPX 파일은 파싱 가능한 형식으로 되어 있음
3. 디스크 공간이 백업 및 새 출력 파일 생성에 충분함
4. Python 3.13+ 환경에서 실행됨

### Business Assumptions

1. 기존 출력 파일이 있는 경우 백업 후 덮어쓰는 것이 허용됨
2. RAG 최적화 JSON은 향후 검색 품질 향상에 활용됨
3. 품질 분석 리포트는 정기적으로 생성되어야 함

### Constraint Assumptions

1. 대용량 파일 처리 시 메모리 사용량이 합리적 범위 내에 있어야 함
2. 파일 처리 순서는 알파벳 순으로 일관성 있게 수행됨
3. 처리 중단 시 이미 처리된 파일은 유지됨

---

## Requirements

### REQ-001: 파일 무결성 검증 (Ubiquitous)

시스템은 **항상** 파싱 전에 HWPX 파일 무결성을 검증해야 한다.

**EARS Pattern**: Ubiquitous (Always Active)

**검증 항목**:
- 파일 크기 > 0
- 파일 읽기 가능
- HWPX ZIP 구조 유효성

**Acceptance Criteria**:
- WHEN 파일 크기가 0바이트인 경우 THEN 시스템은 해당 파일을 건너뛰고 로그에 기록
- WHEN 파일이 읽기 불가능한 경우 THEN 시스템은 에러를 로그하고 다음 파일로 진행

---

### REQ-002: 파일 발견 및 파싱 (Event-Driven)

**WHEN** data/input/ 디렉터리에서 HWPX 파일이 발견되면, **THEN** 시스템은 SPEC-CHUNK-001의 HWPX Direct Parser를 사용하여 각 파일을 파싱해야 한다.

**EARS Pattern**: Event-Driven (Trigger-Response)

**처리 순서**:
1. data/input/ 디렉터리 스캔
2. .hwpx 확장자 파일 필터링
3. 알파벳 순 정렬
4. 순차적 파싱 실행

**Acceptance Criteria**:
- WHEN data/input/에 10개의 HWPX 파일이 존재하면 THEN 시스템은 10개 파일 모두 파싱
- WHEN 파싱이 완료되면 THEN 각 파일에 대한 JSON 출력이 생성됨

---

### REQ-003: 이중 JSON 출력 생성 (Event-Driven)

**WHEN** 파일 파싱이 완료되면, **THEN** 시스템은 표준 JSON과 RAG 최적화 JSON 두 가지 형식의 출력을 생성해야 한다.

**EARS Pattern**: Event-Driven (Trigger-Response)

**출력 파일 명명 규칙**:
- 표준 JSON: `{original_filename}_hwpx_direct.json`
- RAG 최적화: `{original_filename}_hwpx_direct_rag.json`

**Acceptance Criteria**:
- WHEN "규정집9-349.hwpx" 파싱이 완료되면 THEN 다음 파일이 생성됨:
  - `규정집9-349_hwpx_direct.json`
  - `규정집9-349_hwpx_direct_rag.json`
- WHEN RAG JSON 생성 시 THEN embedding_text 필드가 최적화됨

---

### REQ-004: 품질 분석 리포트 생성 (Event-Driven)

**WHEN** 모든 파싱 작업이 완료되면, **THEN** 시스템은 청크 통계가 포함된 품질 분석 리포트를 생성해야 한다.

**EARS Pattern**: Event-Driven (Trigger-Response)

**리포트 내용**:
- 처리된 파일 수
- 총 청크 수
- 청크 타입별 분포
- 최대 계층 깊이 평균
- 처리 시간 통계

**Acceptance Criteria**:
- WHEN 모든 파일 처리가 완료되면 THEN `data/output/quality_analysis_report.json` 생성
- WHEN 리포트 생성 시 THEN 모든 필수 메트릭이 포함됨

---

### REQ-005: 기존 출력 백업 (State-Driven)

**IF** 파일에 대한 hwpx_direct.json이 이미 존재하면, **THEN** 시스템은 덮어쓰기 전에 타임스탬프 백업을 생성해야 한다.

**EARS Pattern**: State-Driven (Conditional)

**백업 명명 규칙**:
- `{filename}_hwpx_direct_{YYYYMMDD_HHMMSS}.json.bak`

**Acceptance Criteria**:
- WHEN 기존 `규정집9-349_hwpx_direct.json`이 존재하면 THEN `규정집9-349_hwpx_direct_20260213_143022.json.bak`으로 백업
- WHEN 백업 생성 후 THEN 새로운 출력 파일로 덮어쓰기

---

### REQ-006: 파일 건너뛰기 금지 (Unwanted Behavior)

시스템은 기존 출력 상태와 관계없이 data/input/에서 발견된 **모든** HWPX 파일을 건너뛰지 않아야 한다.

**EARS Pattern**: Unwanted Behavior (Prohibition)

**금지 사항**:
- 기존 출력 존재 여부와 관계없이 모든 파일 처리
- 부분적 처리 금지 (중단 시 제외)
- 에러 발생 시에도 다음 파일로 진행

**Acceptance Criteria**:
- WHEN 10개 파일 중 3개가 이미 출력을 가지고 있어도 THEN 10개 모두 다시 처리됨
- WHEN 특정 파일 처리 중 에러가 발생해도 THEN 나머지 파일 처리 계속

---

### REQ-007: Legacy vs HWPX Direct 비교 (Optional)

**가능하면** Legacy 파서와 HWPX Direct 파서의 비교 메트릭을 제공한다.

**EARS Pattern**: Optional (Nice-to-have)

**비교 항목**:
- 청크 수 차이
- 계층 깊이 차이
- 컨텐츠 커버리지 비교
- 처리 시간 비교

**Acceptance Criteria**:
- WHERE Legacy 출력이 존재하면 THEN 비교 메트릭이 리포트에 포함됨
- WHERE Legacy 출력이 없으면 THEN HWPX Direct 메트릭만 제공

---

## Specifications

### 명령어 인터페이스

```bash
# 기본 실행
uv run reparse-hwpx

# 옵션 포함 실행
uv run reparse-hwpx --input-dir data/input/ --output-dir data/output/ --verbose

# 드라이 런 (파싱만 확인)
uv run reparse-hwpx --dry-run
```

### 출력 파일 구조

```
data/output/
├── {filename}_hwpx_direct.json        # 표준 출력
├── {filename}_hwpx_direct_rag.json    # RAG 최적화
├── {filename}_hwpx_direct_*.json.bak  # 백업 파일 (기존 존재 시)
└── quality_analysis_report.json       # 품질 분석 리포트
```

### 품질 분석 리포트 스키마

```json
{
  "generated_at": "2026-02-13T14:30:22",
  "summary": {
    "total_files": 10,
    "successful_files": 10,
    "failed_files": 0,
    "total_chunks": 125000,
    "processing_time_seconds": 45.2
  },
  "chunk_statistics": {
    "by_type": {
      "chapter": 150,
      "section": 450,
      "article": 15000,
      "paragraph": 45000,
      "item": 40000,
      "subitem": 24400
    },
    "avg_hierarchy_depth": 4.2,
    "max_hierarchy_depth": 6
  },
  "file_details": [
    {
      "filename": "규정집9-349.hwpx",
      "chunks": 12500,
      "hierarchy_depth": 5,
      "processing_time_ms": 4500
    }
  ],
  "comparison": {
    "legacy_total_chunks": 24387,
    "hwpx_direct_total_chunks": 125000,
    "chunk_ratio": 5.12
  }
}
```

---

## Success Metrics

| 메트릭                    | 목표값          | 측정 방법                        |
| ------------------------- | --------------- | -------------------------------- |
| 파일 처리 성공률          | 100%            | 성공_파일수 / 전체_파일수        |
| 백업 생성 정확성          | 100%            | 기존 파일 존재 시 백업 생성 여부 |
| 리포트 완전성             | 100%            | 필수 필드 포함 여부              |
| 처리 시간                 | < 60초/파일     | 평균 처리 시간                   |
| 메모리 사용량             | < 500MB         | 피크 메모리                      |

---

## Risks and Mitigations

| 위험                        | 확률   | 영향 | 완화 방안                              |
| --------------------------- | ------ | ---- | -------------------------------------- |
| 대용량 파일 메모리 초과     | 중간   | 높음 | 스트리밍 처리, 청크 단위 처리          |
| 디스크 공간 부족            | 낮음   | 높음 | 처리 전 공간 확인, 백업 정책           |
| 파싱 에러로 인한 중단       | 중간   | 중간 | 개별 파일 에러 격리, 계속 진행         |
| 백업 파일 충돌              | 낮음   | 낮음 | 타임스탬프 정밀도 (초 단위)            |

---

## References

- SPEC-CHUNK-001: HWPX Direct Parser Chunk Enhancement
- `src/enhance_for_rag.py`: 청킹 로직 구현
- `data/input/`: 입력 HWPX 파일 디렉터리
- `data/output/`: 출력 JSON 파일 디렉터리
