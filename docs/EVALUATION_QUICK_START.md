# RAG Quality Evaluation - Quick Start Guide

## Overview

This guide explains how to execute comprehensive RAG quality evaluations using the ParallelPersonaEvaluator system.

## Prerequisites

- Python 3.11+
- Virtual environment activated (`.venv`)
- Required dependencies installed

## Quick Start

### Method 1: Simplified Evaluation (Recommended)

Execute the simplified evaluation script that doesn't require ML dependencies:

```bash
python3 scripts/run_parallel_evaluation_simple.py
```

**Output:**
- JSON results: `data/evaluations/parallel_eval_<timestamp>.json`
- Markdown report: `data/evaluations/comprehensive_report_<timestamp>.md`

### Method 2: Full Evaluation (Requires ML Dependencies)

Execute the full evaluation with actual RAG system and LLM judge:

```bash
.venv/bin/python scripts/run_parallel_evaluation.py
```

**Note:** This requires all ML dependencies including llama_index, mlx, etc.

## Understanding the Results

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Factual correctness without hallucination | ≥ 0.85 |
| **Completeness** | All key information present | ≥ 0.75 |
| **Citations** | Accurate regulation references | ≥ 0.70 |
| **Context Relevance** | Retrieved sources relevance | ≥ 0.75 |
| **Overall Score** | Average of all metrics | ≥ 0.80 |

### Personas

The system evaluates across 6 user personas:

1. **Undergraduate Student** - Simple queries, needs clear explanations
2. **Graduate Student** - Academic queries, detailed citations
3. **Professor** - Advanced queries, comprehensive regulation references
4. **Administrative Staff** - Procedural queries, workflow information
5. **Parent** - Basic queries, parent-friendly explanations
6. **International Student** - Mixed Korean/English queries

## Interpreting Results

### Pass/Fail Criteria

- ✅ **PASS**: Overall score ≥ 0.75
- ❌ **FAIL**: Overall score < 0.75

### Common Issues

| Issue | Description | Solution |
|-------|-------------|----------|
| **정보 불충분** | Insufficient information | Add more comprehensive details |
| **일부 정보 부정확** | Inaccurate information | Verify facts, reduce hallucinations |
| **규정 인용 부족** | Insufficient citations | Include proper regulation references |
| **문서 관련성 낮음** | Low document relevance | Improve retrieval ranking |

## File Structure

```
data/evaluations/
├── parallel_eval_<timestamp>.json          # Raw evaluation data
├── comprehensive_report_<timestamp>.md      # Human-readable report
└── EXECUTION_SUMMARY.md                     # Executive summary
```

## Customization

### Modify Queries Per Persona

Edit the script and change the `queries_per_persona` parameter:

```python
results = evaluator.evaluate_parallel(
    queries_per_persona=10,  # Change from 5 to 10
    personas=personas,
)
```

### Test Specific Personas

```python
results = evaluator.evaluate_parallel(
    queries_per_persona=5,
    personas=["student-undergraduate", "professor"],  # Only test these
)
```

## Troubleshooting

### Issue: ML Library Import Errors

**Solution:** Use the simplified evaluation script (`run_parallel_evaluation_simple.py`) which doesn't require ML dependencies.

### Issue: SSL/Network Errors

**Solution:** Check your internet connection and API key configuration.

### Issue: Empty Results

**Solution:** Verify that:
1. ChromaDB exists at `data/chroma_db`
2. Documents are indexed
3. LLM API credentials are configured

## Best Practices

1. **Run Regularly**: Execute evaluations after significant changes
2. **Track Trends**: Compare results over time
3. **Focus on Weaknesses**: Address top failure patterns first
4. **Persona-Specific**: Optimize for low-performing personas
5. **Threshold Monitoring**: Ensure metrics meet quality targets

## Advanced Usage

### Programmatic Evaluation

```python
from src.rag.domain.evaluation.parallel_evaluator import ParallelPersonaEvaluator

evaluator = ParallelPersonaEvaluator()
results = evaluator.evaluate_parallel(
    queries_per_persona=5,
    personas=["student-undergraduate", "professor"],
)

# Access results
for persona_id, persona_result in results.items():
    print(f"{persona_result.persona}: {persona_result.avg_score:.3f}")

# Save results
evaluator.save_results()

# Generate report
report = evaluator.generate_report()
print(report)
```

### Custom Query Generation

```python
# Generate queries for specific persona
queries = evaluator.generate_persona_queries(
    persona="student-undergraduate",
    count_per_category=3,
    topics=["휴학", "장학금", "성적"],
)

# Evaluate specific query
result = evaluator._evaluate_single_query(
    query=PersonaQuery(
        query="휴학 절차가 어떻게 되나요?",
        persona="student-undergraduate",
        category="simple",
        difficulty="easy",
        expected_intent="leave_of_absence",
        expected_info=["절차", "서류", "기간"],
    )
)
```

## Support

For issues or questions:
1. Check the main documentation: `docs/EXECUTION_SUMMARY.md`
2. Review evaluation results: `data/evaluations/comprehensive_report_*.md`
3. Examine raw data: `data/evaluations/parallel_eval_*.json`

## References

- **ParallelPersonaEvaluator**: `src/rag/domain/evaluation/parallel_evaluator.py`
- **LLM Judge**: `src/rag/domain/evaluation/llm_judge.py`
- **Persona Definitions**: `src/rag/domain/evaluation/personas.py`
- **Quality Thresholds**: `src/rag/domain/evaluation/llm_judge.py` (LLMJudge.THRESHOLDS)

---

**Last Updated:** 2026-02-09
**Version:** 1.0.0
