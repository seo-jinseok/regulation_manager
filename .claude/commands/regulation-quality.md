---
name: rag-quality
description: RAG 시스템 품질 평가 실행
argument-hint: "[quick|full|status] [--persona ...] [--category ...]"
---

# /rag-quality Command

RAG 시스템 품질 평가를 실행합니다.

## Usage

```bash
/rag-quality                           # Default: Quick evaluation (5 queries per persona)
/rag-quality quick                     # Quick evaluation (30 queries total)
/rag-quality full                      # Full evaluation (150+ queries)
/rag-quality status                    # Check evaluation status
/rag-quality --persona freshman        # Target specific persona(s)
/rag-quality --category edge_cases     # Target specific category
```

## Evaluation Modes

| Mode | Queries | Description |
|------|---------|-------------|
| quick | 30 | Fast evaluation (5 queries per persona) |
| full | 150+ | Comprehensive evaluation |
| status | - | Check latest evaluation status |

## Personas

- `student-undergraduate`: Beginner level, colloquial language
- `student-graduate`: Advanced level, academic language
- `professor`: Expert level, official terminology
- `staff-admin`: Intermediate, procedure-focused
- `parent`: Beginner, everyday language
- `student-international`: Language barrier, mixed Korean/English

## Execution Flow

1. Parse command arguments
2. Execute `run_rag_quality_eval.py` with appropriate flags
3. Display evaluation results
4. Provide follow-up actions if failures detected

## Examples

### Quick Evaluation
```
/rag-quality quick
```
Runs 5 queries per persona (~2 minutes).

### Target Specific Personas
```
/rag-quality --persona student-undergraduate professor
```
Evaluates only the specified personas.

### Check Status
```
/rag-quality status
```
Shows the latest evaluation results.

## Output Files

| File | Location |
|------|----------|
| JSON data | `data/evaluations/rag_quality_eval_*.json` |
| Markdown report | `data/evaluations/rag_quality_eval_*_report.md` |

## Follow-up Actions

When evaluation shows failures:

1. Review failure patterns in report
2. Check generated SPEC at `data/evaluations/spec_*.md`
3. Run `/moai run SPEC-RAG-Q-XXX` to fix issues

## Related

- Skill: `ragulation-quality`
- SPEC: `SPEC-RAG-SKILL-001`
- CLI: `uv run python run_rag_quality_eval.py --help`
