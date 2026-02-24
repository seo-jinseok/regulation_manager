# SPEC-RAG-SKILL-001: rag-quality-local 스킬 프로덕션 완성

## Overview

| Field | Value |
|-------|-------|
| ID | SPEC-RAG-SKILL-001 |
| Status | Completed |
| Priority | P0 |
| Complexity | Medium |
| Estimated Effort | 4-8 hours |
| Dependencies | None |

## Problem Statement

rag-quality-local 스킬은 RAG 시스템 품질 평가를 위한 포괄적인 기능을 제공하지만, **사용자가 실제로 실행하고 결과를 확인할 수 있는 명확한 인터페이스가 부족**합니다.

**Current Pain Points:**
1. 스킬을 "어떻게 호출하는지" 불명확
2. 평가 결과를 "어디서 확인하는지" 불명확
3. 평가 후 "무엇을 해야 하는지" 불명확

## Goals

1. **실행 가능성**: 사용자가 단일 명령으로 평가 실행 가능
2. **결과 확인**: 결과를 명확한 경로로 확인 가능
3. **후속 액션**: 실패 시 개선 가이드 자동 제공
4. **워크플로우 통합**: MoAI 플로우와 자연스럽게 연결

## Requirements (EARS Format)

### REQ-001: 슬래시 커맨드 제공
**WHEN** 사용자가 `/rag-quality` 명령을 입력하면
**THEN** 시스템은 평가 옵션을 표시하고 선택된 평가를 실행한다
**SHALL** 다음 옵션을 지원한다:
- `full`: 전체 평가 (6 페르소나 × 150 시나리오)
- `targeted`: 대상 지정 평가 (--persona, --category 옵션)
- `regression`: 회귀 테스트 (--baseline 비교)
- `status`: 현재 평가 상태 확인

### REQ-002: CLI 진입점 개선
**WHEN** 사용자가 `uv run python run_rag_quality_eval.py`를 실행하면
**THEN** 명확한 CLI 인터페이스가 표시된다
**SHALL** argparse 기반 인터페이스를 제공한다:
```bash
uv run python run_rag_quality_eval.py --help
uv run python run_rag_quality_eval.py --full --output report.md
uv run python run_rag_quality_eval.py --persona freshman,professor --queries 10
```

### REQ-003: Quick Start 개선
**WHEN** 사용자가 SKILL.md를 열면
**THEN** 실행 가능한 명령어가 즉시 보인다
**SHALL** "Copy-Paste" 가능한 명령어를 포함한다:
```bash
# 1분 만에 평가 실행
uv run python run_rag_quality_eval.py --quick

# 결과 확인
cat data/evaluations/latest/report.md

# 대시보드 열기
uv run gradio src.rag.interface.web.quality_dashboard:app
```

### REQ-004: 결과 확인 섹션 추가
**WHEN** 평가가 완료되면
**THEN** 사용자는 결과를 확인할 수 있다
**SHALL** 다음 확인 방법을 제공한다:
- CLI: `--summary` 플래그로 요약 출력
- 파일: `data/evaluations/latest/` 경로 안내
- 대시보드: Gradio UI 링크
- SPEC: 실패 패턴에서 생성된 SPEC 경로

### REQ-005: 후속 액션 가이드
**WHEN** 평가에서 실패가 감지되면
**THEN** 개선 가이드를 자동으로 제공한다
**SHALL** 다음을 포함한다:
- 실패 유형별 권장 조치
- 생성된 SPEC 문서 경로
- `/moai run SPEC-XXX` 실행 제안

## Technical Approach

### Phase 1: 슬래시 커맨드 구현 (Priority: P0)

**File: `.claude/commands/rag-quality.md`**
```markdown
---
name: rag-quality
description: RAG 품질 평가 실행
argument-hint: full|targeted|regression|status
---

# /rag-quality Command

RAG 시스템 품질 평가를 실행합니다.

## Usage

```bash
/rag-quality full                    # 전체 평가
/rag-quality targeted --persona freshman  # 특정 페르소나
/rag-quality regression --baseline eval-001  # 회귀 테스트
/rag-quality status                  # 상태 확인
```

## Execution Flow

1. 옵션 파싱
2. run_rag_quality_eval.py 실행
3. 결과 요약 출력
4. 후속 액션 제안
```

### Phase 2: CLI 개선 (Priority: P0)

**File: `run_rag_quality_eval.py` (수정)**

```python
#!/usr/bin/env python3
"""RAG Quality Evaluation CLI - Production Entry Point."""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="RAG 시스템 품질 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                    # 빠른 평가 (각 페르소나당 5쿼리)
  %(prog)s --full                     # 전체 평가 (150+ 쿼리)
  %(prog)s --persona freshman professor  # 특정 페르소나만
  %(prog)s --category edge_cases      # 특정 카테고리만
  %(prog)s --status                   # 진행 중인 평가 상태
        """
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="빠른 평가 (30쿼리)")
    mode.add_argument("--full", action="store_true", help="전체 평가 (150+쿼리)")
    mode.add_argument("--status", action="store_true", help="평가 상태 확인")

    # Targeting
    parser.add_argument("--persona", nargs="+", choices=PERSONAS, help="대상 페르소나")
    parser.add_argument("--category", nargs="+", choices=CATEGORIES, help="시나리오 카테고리")
    parser.add_argument("--queries", type=int, default=5, help="페르소나당 쿼리 수")

    # Output
    parser.add_argument("--output", "-o", type=Path, help="출력 파일")
    parser.add_argument("--format", choices=["json", "markdown", "both"], default="both")
    parser.add_argument("--summary", action="store_true", help="완료 후 요약 출력")

    # Advanced
    parser.add_argument("--baseline", type=str, help="회귀 테스트용 기준선 ID")
    parser.add_argument("--checkpoint", action="store_true", default=True, help="체크포인트 저장")
    parser.add_argument("--resume", type=str, help="중단된 평가 재개")

    args = parser.parse_args()

    # Execute evaluation...
```

### Phase 3: SKILL.md Quick Start 개선 (Priority: P0)

**수정할 섹션:**

```markdown
## Quick Start

### 1분 만에 평가 실행

```bash
# 빠른 평가 (각 페르소나당 5쿼리, 약 2분 소요)
uv run python run_rag_quality_eval.py --quick --summary

# 결과 확인
cat data/evaluations/latest/report.md

# 웹 대시보드로 확인
uv run gradio src.rag.interface.web.quality_dashboard:app
```

### 상세 평가

```bash
# 전체 평가 (150+ 쿼리, 약 15분 소요)
uv run python run_rag_quality_eval.py --full

# 특정 페르소나만 평가
uv run python run_rag_quality_eval.py --persona freshman professor --queries 10

# 회귀 테스트 (이전 결과와 비교)
uv run python run_rag_quality_eval.py --regression --baseline eval-20260220
```

### Claude Code 내에서 실행

```
/rag-quality full
```
```

### Phase 4: 결과 확인 섹션 추가 (Priority: P0)

**새 섹션:**

```markdown
## 결과 확인

### CLI에서 확인

```bash
# 최신 평가 결과 요약
uv run python run_rag_quality_eval.py --status

# 상세 보고서
cat data/evaluations/latest/report.md

# JSON 데이터
cat data/evaluations/latest/eval_*.json | jq '.overall_metrics'
```

### 대시보드에서 확인

```bash
# 대시보드 실행
uv run gradio src.rag.interface.web.quality_dashboard:app

# 브라우저에서 http://localhost:7860 접속
# "품질 평가" 탭 선택
```

### 결과 파일 위치

| 파일 | 경로 | 설명 |
|------|------|------|
| JSON 데이터 | `data/evaluations/eval_*.json` | 전체 평가 데이터 |
| 마크다운 보고서 | `data/evaluations/report_*.md` | 사람이 읽기 좋은 보고서 |
| SPEC 문서 | `data/evaluations/spec_*.md` | 개선용 SPEC 템플릿 |
| 최신 결과 | `data/evaluations/latest/` | 심볼릭 링크 |
```

### Phase 5: 후속 액션 섹션 추가 (Priority: P1)

```markdown
## 평가 후 후속 조치

### 실패 패턴 분석

평가 완료 후 자동으로 실패 패턴이 분석됩니다:

```bash
# 실패 패턴 보기
cat data/evaluations/latest/failure_analysis.md
```

### 개선 SPEC 생성

실패한 쿼리가 3개 이상 동일한 패턴이면 자동으로 SPEC이 생성됩니다:

```bash
# 생성된 SPEC 확인
ls data/evaluations/latest/spec_*.md

# SPEC을 MoAI 포맷으로 변환
cp data/evaluations/latest/spec_*.md .moai/specs/SPEC-RAG-Q-XXX/spec.md
```

### 개선 실행

```bash
# MoAI로 개선 실행
/moai run SPEC-RAG-Q-XXX

# 또는 자동 수정 루프
/moai loop --spec SPEC-RAG-Q-XXX
```
```

## Acceptance Criteria

### AC-001: 슬래시 커맨드 동작
- [x] `/rag-quality` 입력 시 옵션 안내 표시
- [x] `/rag-quality full` 실행 시 전체 평가 수행
- [x] `/rag-quality status` 실행 시 현재 상태 표시

### AC-002: CLI 동작
- [x] `--help` 실행 시 명확한 사용법 표시
- [x] `--quick` 실행 시 30쿼리 평가 완료
- [x] `--summary` 실행 시 결과 요약 출력

### AC-003: Quick Start 개선
- [x] Copy-Paste 가능한 명령어 포함
- [x] 예상 소요 시간 표시
- [x] 결과 확인 방법 명시

### AC-004: 결과 확인
- [x] 파일 위치 명확히 표시
- [x] 대시보드 링크 포함
- [x] CLI 확인 명령어 포함

### AC-005: 후속 액션
- [x] 실패 시 개선 가이드 표시
- [x] SPEC 파일 경로 표시
- [x] MoAI 실행 제안 포함

## Definition of Done

1. **Documentation**
   - [x] SKILL.md Quick Start 섹션 개선
   - [x] 결과 확인 섹션 추가
   - [x] 후속 액션 섹션 추가

2. **Commands**
   - [x] `.claude/commands/rag-quality.md` 생성
   - [x] 슬래시 커맨드 테스트 통과

3. **CLI**
   - [x] `run_rag_quality_eval.py` argparse 개선
   - [x] `--quick`, `--full`, `--status` 모드 구현
   - [x] `--summary` 출력 구현

4. **Integration**
   - [x] MoAI 워크플로우 연동 섹션 추가
   - [x] 후속 SPEC 실행 가이드 추가

5. **Testing**
   - [x] `--quick` 모드 정상 동작 확인
   - [x] 결과 파일 생성 확인
   - [x] 대시보드 연동 확인

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CLI 변경으로 기존 사용자 혼란 | Medium | Low | --help 명확화, 마이그레이션 가이드 |
| 대시보드 연동 실패 | Low | Medium | fallback to file output |
| SPEC 자동 생성 실패 | Low | Low | 수동 생성 가이드 제공 |

## Out of Scope

- CI/CD 통합 (별도 SPEC)
- 자동 스케줄링 (별도 SPEC)
- 알림 시스템 (별도 SPEC)
- 페르소나 확장 (기존 6개 유지)

## References

- Research Document: `.moai/specs/SPEC-RAG-SKILL-001/research.md`
- Current Skill: `.claude/skills/rag-quality-local/SKILL.md`
- Related Agent: `.claude/agents/rag-quality-evaluator.md`
- Implementation: `src/rag/domain/evaluation/`
- CLI Script: `run_rag_quality_eval.py`

---

**Created:** 2026-02-24
**Author:** MoAI Strategic Orchestrator
**Review Status:** Pending User Approval
