---
name: regulation-quality
description: >
  Regulation Manager 규정 Q&A 시스템의 포괄적인 품질 평가 시스템입니다. 6가지 사용자 페르소나를 시뮬레이션하는 서브에이전트(팀원)들이 병렬로 작동하여 150+개의 테스트 시나리오를 실행하고,
  LLM-as-Judge 방식으로 답변 정확도를 평가하며, 자동으로 개선 SPEC을 생성합니다.

  사용 시나리오:
  - 다양한 사용자 유형(학생, 교수, 교직원, 유학생 등)으로 규정 Q&A 시스템 테스트
  - 단일 턴, 다중 턴, 엣지 케이스, 시간적 쿼리 등 포괄적인 테스트 실행
  - 정답 검증 및 정확도 메트릭 계산 (Precision, Recall, F1, Context Relevance)
  - 실패 패턴 분석 및 개선 권장사항 자동 생성
  - 평가 결과 JSON 저장 및 추세 분석

version: "1.1.0"
category: "domain"
status: "active"
allowed-tools: Read Write Edit Bash Grep Glob Task
metadata:
  modularized: "true"
  tags: "regulation, evaluation, testing, quality, personas, llm-judge, regulation-qa"
  context7-libraries: "ragas,deepeval"
  related-skills: "rag-quality, moai-foundation-quality"
progressive_disclosure:
  enabled: true
  level1_tokens: 120
  level2_tokens: 5000
triggers:
  keywords:
    - "regulation 평가"
    - "규정 Q&A 품질"
    - "RAG 평가"
    - "품질 테스트"
    - "페르소나 시뮬레이션"
    - "LLM-as-Judge"
    - "테스트 시나리오"
    - "자동 평가"
    - "답변 정확도"
    - "edge case"
    - "multi-turn"
  agents: ["rag-quality-evaluator", "rag-student-undergraduate", "rag-student-graduate", "rag-professor", "rag-staff-admin", "rag-parent", "rag-international-student"]
  phases: ["run", "sync"]
  languages: ["python"]
---
# Ragulation Quality - Comprehensive Regulation Q&A Evaluation System

Regulation Manager 규정 Q&A 시스템의 포괄적인 품질 평가를 위한 스킬입니다. 6가지 사용자 페르소나를 시뮬레이션하는 서브에이전트들이 병렬로 작동하여 실제 사용자 패턴을 테스트합니다.

## Quick Reference

### What is Ragulation Quality?

Comprehensive evaluation system that:
- Spawns 6 persona sub-agents (teammates) in parallel for realistic testing
- Executes 150+ test scenarios across 6 categories
- Evaluates using LLM-as-Judge with 4-metric scoring
- Generates automated improvement SPECs
- Tracks quality trends over time

### Key Components

- **6 Sub-Agent Personas**: student-undergraduate, student-graduate, professor, staff-admin, parent, student-international
- **6 Scenario Categories**: simple, complex, multi-turn, edge cases, domain-specific, adversarial
- **4 Evaluation Metrics**: Accuracy, Completeness, Citations, Context Relevance
- **3 Output Formats**: JSON data, Markdown report, SPEC template

### When to Use

- Regulation Q&A system changes require quality validation
- New features need comprehensive testing
- Baseline quality metrics needed
- Improvement tracking required
- Automated evaluation workflow desired

### Metric Thresholds

| Metric | Goal | Description |
|--------|------|-------------|
| Overall | 0.80+ | Pass threshold |
| Accuracy | 0.85+ | No hallucinations |
| Completeness | 0.75+ | All key info present |
| Citations | 0.70+ | Accurate regulation refs |
| Context Relevance | 0.75+ | Relevant sources |

---

## Implementation Guide

### Core Architecture

```
regulation-quality (Skill Coordinator)
    ├── Spawns 6 persona sub-agents in parallel
    ├── Aggregates results
    ├── Runs LLM-as-Judge evaluation
    └── Generates reports and SPECs
```

### Quick Start

**1분 만에 평가 실행 (CLI)**

```bash
# 빠른 평가 (각 페르소나당 5쿼리, 약 2분 소요)
uv run python run_rag_quality_eval.py --quick --summary

# 결과 확인
cat data/evaluations/rag_quality_eval_*/report.md

# 웹 대시보드로 확인
uv run gradio src.rag.interface.web.quality_dashboard:app
```

**Claude Code 내에서 실행**

```
/rag-quality quick
```

**상세 평가**

```bash
# 전체 평가 (150+ 쿼리, 약 15분 소요)
uv run python run_rag_quality_eval.py --full

# 특정 페르소나만 평가
uv run python run_rag_quality_eval.py --persona student-undergraduate professor --queries 10

# 회귀 테스트 (이전 결과와 비교)
uv run python run_rag_quality_eval.py --baseline eval-20260220
```

**평가 모드 옵션**

| Mode | CLI Flag | Queries | Time |
|------|----------|---------|------|
| Quick | `--quick` | 30 | ~2 min |
| Full | `--full` | 150+ | ~15 min |
| Status | `--status` | - | Instant |

---

## 결과 확인

### CLI에서 확인

```bash
# 최신 평가 결과 요약
uv run python run_rag_quality_eval.py --status

# 상세 보고서
cat data/evaluations/rag_quality_eval_*/report.md

# JSON 데이터
cat data/evaluations/rag_quality_eval_*.json | jq '.summary'
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
| JSON 데이터 | `data/evaluations/rag_quality_eval_*.json` | 전체 평가 데이터 |
| 마크다운 보고서 | `data/evaluations/rag_quality_eval_*_report.md` | 사람이 읽기 좋은 보고서 |
| SPEC 문서 | `data/evaluations/spec_*.md` | 개선용 SPEC 템플릿 |

### 메트릭 해석

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Overall Score | >= 0.80 | Pass/Fail 결정 |
| Accuracy | >= 0.85 | 할루시네이션 없음 |
| Completeness | >= 0.75 | 모든 핵심 정보 포함 |
| Citations | >= 0.70 | 정확한 규정 참조 |
| Context Relevance | >= 0.75 | 관련성 높은 소스 |

---

## 평가 후 후속 조치

### 실패 패턴 분석

평가 완료 후 실패 패턴이 자동으로 분석됩니다:

```bash
# 실패 패턴 보기 (보고서 하단)
cat data/evaluations/rag_quality_eval_*_report.md | grep -A 20 "Failure Patterns"
```

### 개선 SPEC 생성

실패한 쿼리가 3개 이상 동일한 패턴이면 개선 가이드를 참조하세요:

```markdown
## Common Failure Patterns and Solutions

| Pattern | Cause | Solution |
|---------|-------|----------|
| Hallucinated contact info | No validation | Add contact verification |
| Missing citations | Low retrieval | Tune reranker weights |
| Incomplete answers | Context limit | Increase top_k |
| Wrong regulation | Ambiguous query | Add disambiguation |
```

### MoAI로 개선 실행

```bash
# SPEC을 MoAI 포맷으로 변환하여 개선 실행
# 1. 실패 패턴 분석 보고서에서 SPEC 생성
# 2. .moai/specs/SPEC-RAG-Q-XXX/ 폴더에 spec.md 파일 생성
# 3. /moai run SPEC-RAG-Q-XXX 실행

# 또는 자동 수정 루프
/moai loop --spec SPEC-RAG-Q-XXX
```

### 대시보드에서 추세 확인

여러 번의 평가 결과를 비교하여 개선 추세를 확인할 수 있습니다:

```bash
# 대시보드 실행
uv run gradio src.rag.interface.web.quality_dashboard:app

# "히스토리" 탭에서 과거 평가 결과 비교
```

---

## Detailed Usage Options

**Option 1: Full Evaluation (All Personas, All Scenarios)**

```bash
uv run python run_rag_quality_eval.py --full
```

**Option 2: Targeted Evaluation (Specific Persona or Scenario)**

```bash
# 특정 페르소나만
uv run python run_rag_quality_eval.py --persona professor staff-admin

# 특정 카테고리만
uv run python run_rag_quality_eval.py --category edge_cases
```

**Option 3: Regression Testing (Compare to Baseline)**

```bash
# 기준선과 비교
uv run python run_rag_quality_eval.py --baseline eval-20260220
```

### Sub-Agent Personas

See @modules/personas.md for detailed persona definitions:

1. **student-undergraduate**: Beginner level, colloquial language, simple queries
2. **student-graduate**: Advanced level, academic language, precise queries
3. **professor**: Expert level, official terminology, comprehensive answers expected
4. **staff-admin**: Intermediate, procedure-focused, administrative queries
5. **parent**: Beginner, everyday language, financial/welfare concerns
6. **student-international**: Language barrier, mixed Korean/English, simple queries

### Test Scenario Categories

See @modules/scenarios.md for complete scenario library:

1. **Simple Queries (30)**: Direct questions, single intent
2. **Complex Queries (25)**: Multi-part questions requiring synthesis
3. **Multi-Turn (20)**: Context-dependent conversations
4. **Edge Cases (40)**: Ambiguous, typos, vague queries
5. **Domain-Specific (25)**: Cross-reference, temporal, procedure, contact
6. **Adversarial (10)**: Hallucination triggers, invalid requests

### Evaluation Pipeline

**Step 1: Spawn Sub-Agents**

```python
personas = [
    "student-undergraduate",
    "student-graduate",
    "professor",
    "staff-admin",
    "parent",
    "student-international"
]

for persona in personas:
    Task(
        subagent_type=persona,
        prompt="Generate test queries for your persona..."
    )
```

**Step 2: Execute Queries via CLI**

```bash
regulation ask "{query}" --format json --output /tmp/rag_response.json
```

**Step 3: LLM-as-Judge Evaluation**

See @modules/evaluation.md for detailed prompts and rubrics.

**Step 4: Calculate Metrics**

See @modules/metrics.md for calculation logic.

**Step 5: Generate Reports**

Create three output formats: JSON, Markdown, SPEC template.

### Output Format

#### JSON Evaluation Record

```json
{
  "evaluation_id": "eval_20250107_140530",
  "timestamp": "2025-01-07T14:05:30Z",
  "rag_config": {"model": "gpt-4o", "use_reranker": true},
  "personas_tested": ["student-undergraduate", "professor", "staff-admin", "parent", "student-international"],
  "scenarios": {
    "simple": {"total": 30, "passed": 27, "avg_score": 0.89},
    "complex": {"total": 25, "passed": 20, "avg_score": 0.82}
  },
  "overall_metrics": {
    "total_queries": 150,
    "passed": 121,
    "pass_rate": 0.807,
    "avg_precision": 0.84,
    "avg_recall": 0.79,
    "avg_f1": 0.81
  }
}
```

### Automated SPEC Generation

For high-priority failure patterns, generate improvement SPECs:

```markdown
# SPEC-RAG-Q-001: Hallucination Fix for Contact Information

## Problem Analysis
Generated from: eval_20250107_140530
Issue: 8 queries contain hallucinated contact information
Current pass rate: 80.7%
Target pass rate: 85%+

## Requirements
- WHEN system detects contact information request
- THEN either provide actual contact OR use generic response
- SHALL NOT generate fake phone numbers
```

---

## Advanced Implementation

### Sub-Agent Task Delegation

Each persona sub-agent receives its own 200K context and operates independently:

```python
# In student-undergraduate sub-agent
Task(
    subagent_type="student-undergraduate",
    prompt="""
    You are a Korean undergraduate student.
    Generate 5 test queries for '휴학' (leave of absence) topic.
    Use colloquial language, simple terms, informal style.

    Return JSON with:
    - query: the question text
    - expected_intent: what information the user seeks
    - difficulty: easy/medium/hard
    """
)
```

### CLI Wrapper

The evaluation wraps the Regulation Manager CLI for query execution:

```python
import subprocess
import json

def execute_rag_query(query: str) -> dict:
    """Execute query through Regulation Manager CLI."""
    result = subprocess.run(
        ["regulation", "ask", query, "--format", "json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)
```

### Result Storage

Evaluation results stored in `data/evaluations/`:

```
data/evaluations/
├── baseline.json
├── eval_20250107_140530.json
├── eval_20250107_153022.json
├── report_20250107_140530.md
└── spec_20250107_140530.md
```

### Report Templates

Markdown reports include:
- Executive summary
- Metrics breakdown by persona and category
- Failure pattern analysis
- Improvement recommendations
- Trend comparison (if baseline exists)

---

## Works Well With

### Related Skills
- **rag-quality**: Existing basic RAG evaluation (complementary)
- **moai-foundation-quality**: TRUST 5 quality framework
- **moai-workflow-spec**: SPEC creation for improvements
- **moai-workflow-testing**: Test strategy and patterns

### Related Agents
- **rag-quality-evaluator**: Expert agent for RAG evaluation
- **rag-persona-student**: Student persona sub-agent
- **rag-persona-professor**: Professor persona sub-agent
- **rag-persona-staff**: Staff admin persona sub-agent
- **rag-persona-parent**: Parent persona sub-agent
- **rag-persona-international**: International student persona sub-agent
- **rag-evaluator**: LLM-as-Judge evaluator sub-agent

### Tools & Integrations
- **RAGAS**: LLM-as-Judge evaluation framework (ragas>=0.4.3)
- **DeepEval**: Alternative evaluation framework (deepeval>=3.8.1)
- **CLI Integration**: `regulation ask` command for query execution

---

## Module Structure

```
.claude/skills/regulation-quality/
├── SKILL.md                          # Main coordinator skill (this file)
├── modules/
│   ├── personas.md                   # Detailed persona definitions
│   ├── scenarios.md                  # Test scenario library
│   ├── evaluation.md                 # LLM-as-Judge prompts
│   └── metrics.md                    # Metrics calculation
└── reference.md                      # External resources
```

---

## Quick Reference Summary

| Command | Description |
|---------|-------------|
| Full evaluation | Run all personas, all scenarios |
| Targeted evaluation | Specific persona or scenario |
| Regression test | Compare to baseline |
| Generate SPEC | Create improvement specs |

| Output | Location | Format |
|--------|----------|--------|
| Evaluation data | `data/evaluations/eval_*.json` | JSON |
| Report | `data/evaluations/report_*.md` | Markdown |
| SPEC template | `data/evaluations/spec_*.md` | SPEC |

---

Last Updated: 2026-02-24
Version: 1.1.0
Status: Active
