---
name: rag-quality-local
description: >
  RAG 시스템의 포괄적인 품질 평가 시스템입니다. 6가지 사용자 페르소나를 시뮬레이션하는 서브에이전트(팀원)들이 병렬로 작동하여 150+개의 테스트 시나리오를 실행하고,
  LLM-as-Judge 방식으로 답변 정확도를 평가하며, 자동으로 개선 SPEC을 생성합니다.

  사용 시나리오:
  - 다양한 사용자 유형(학생, 교수, 교직원, 유학생 등)으로 RAG 시스템 테스트
  - 단일 턴, 다중 턴, 엣지 케이스, 시간적 쿼리 등 포괄적인 테스트 실행
  - 정답 검증 및 정확도 메트릭 계산 (Precision, Recall, F1, Context Relevance)
  - 실패 패턴 분석 및 개선 권장사항 자동 생성
  - 평가 결과 JSON 저장 및 추세 분석

version: "1.0.0"
category: "domain"
status: "active"
allowed-tools: Read Write Edit Bash Grep Glob Task
metadata:
  modularized: "true"
  tags: "rag, evaluation, testing, quality, personas, llm-judge"
  context7-libraries: "ragas,deepeval"
  related-skills: "rag-quality, moai-foundation-quality"
progressive_disclosure:
  enabled: true
  level1_tokens: 120
  level2_tokens: 5000
triggers:
  keywords:
    - "RAG 평가"
    - "품질 테스트"
    - "페르소나 시뮬레이션"
    - "LLM-as-Judge"
    - "테스트 시나리오"
    - "자동 평가"
    - "답변 정확도"
    - "edge case"
    - "multi-turn"
  agents: []
  phases: ["run", "sync"]
  languages: ["python"]
---

# RAG Quality Local - Comprehensive RAG Evaluation System

RAG 시스템의 포괄적인 품질 평가를 위한 로컬 스킬입니다. 6가지 사용자 페르소나를 시뮬레이션하는 서브에이전트들이 병렬로 작동하여 실제 사용자 패턴을 테스트합니다.

## Quick Reference

### What is RAG Quality Local?

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

- RAG system changes require quality validation
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
rag-quality-local (Skill Coordinator)
    ├── Spawns 6 persona sub-agents in parallel
    ├── Aggregates results
    ├── Runs LLM-as-Judge evaluation
    └── Generates reports and SPECs
```

### Quick Start

**Option 1: Full Evaluation (All Personas, All Scenarios)**

```
Use the rag-quality-local skill to run a comprehensive RAG quality evaluation:

1. Spawn all 6 persona sub-agents in parallel
2. Each sub-agent generates test queries based on their persona profile
3. Execute queries through the RAG CLI: regulation ask "{query}"
4. Evaluate responses using LLM-as-Judge
5. Aggregate results and generate reports
6. Create improvement SPECs for failing queries
```

**Option 2: Targeted Evaluation (Specific Persona or Scenario)**

```
Run targeted evaluation:

1. Spawn only specific persona sub-agents (e.g., "professor" and "student-international")
2. Focus on specific scenario categories (e.g., "edge_cases" and "multi_turn")
3. Execute evaluation
4. Generate focused report
```

**Option 3: Regression Testing (Compare to Baseline)**

```
Run regression evaluation against baseline:

1. Load baseline from data/evaluations/baseline.json
2. Execute current evaluation
3. Compare metrics to baseline
4. Identify regressions and improvements
5. Generate trend report
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

The evaluation wraps the RAG CLI for query execution:

```python
import subprocess
import json

def execute_rag_query(query: str) -> dict:
    """Execute query through RAG CLI."""
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
.claude/skills/rag-quality-local/
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

Last Updated: 2025-01-07
Version: 1.0.0
Status: Active
