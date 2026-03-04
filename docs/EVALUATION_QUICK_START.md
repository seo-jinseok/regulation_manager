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

## Extended Evaluation (Perpetual Quality Discovery Engine)

SPEC-RAG-EVAL-002에서 추가된 확장 평가 시스템으로, 정적 30쿼리/4메트릭 한계를 넘어 동적 200+ 쿼리, 10+ 메트릭, 시스템 전체 건강 평가를 제공합니다.

### Extended CLI Flags

| Flag | Description |
|------|-------------|
| `--generate` | 규정 JSON에서 동적 쿼리 생성 (200+, L1-L5 난이도) |
| `--regenerate` | 캐시 무시하고 쿼리 재생성 |
| `--consistency` | 답변 일관성 검사 (3회 반복, cosine similarity ≥0.85) |
| `--health` | 시스템 건강 진단 (코드 품질/커버리지/설정 드리프트) |
| `--tier L1-L5` | 특정 난이도 티어 테스트 |
| `--trend` | 품질 추세 분석 (3+ 평가 이력 필요) |
| `--full-extended` | 모든 확장 기능 활성화 (generate + consistency + health + trend) |

### Quick Examples

```bash
# 확장 전체 평가 (기존 평가 + 5개 모듈 모두 실행)
python run_rag_quality_eval.py --full-extended

# 시스템 건강만 빠르게 확인 (RAG 쿼리 불필요)
python run_rag_quality_eval.py --health

# L3 난이도 (교차규정 합성) 집중 테스트
python run_rag_quality_eval.py --full --tier L3

# 품질 추세 + 개선 로드맵 확인
python run_rag_quality_eval.py --trend
```

### 5 Modules Overview

1. **Dynamic Query Universe** (`query_synthesizer.py`): 규정 JSON을 분석하여 L1(단일 사실) ~ L5(멀티턴) 난이도의 테스트 쿼리를 자동 생성. 교차규정 쿼리, 적대적 쿼리 포함.

2. **Extended Metrics** (`extended_metrics.py`): 기존 4메트릭 외에 응답 지연(p50/p95/p99), 답변 일관성, 인용 실존 검증, 가독성 점수를 측정.

3. **System Health Radar** (`system_health.py`): 코드 품질(bare except, TODO, magic numbers, 긴 함수), 테스트 커버리지 델타, 설정 드리프트를 AST 기반으로 스캔.

4. **Adaptive Difficulty** (`difficulty_manager.py`): L1-L5 티어별 통과율 추적. 95% 이상 2회 연속 달성 시 자동 난이도 상승. "절대 개선할 것이 없다고 말할 수 없는" 구조.

5. **Improvement Roadmap** (`improvement_radar.py`): 실패를 원인별로 클러스터링하고, 영향도×빈도×수정용이성 기반 우선순위 로드맵을 자동 생성. 추세 분석 포함.

### Extended Output Files

| File | Description |
|------|-------------|
| `data/evaluations/generated_queries.json` | 생성된 동적 쿼리 캐시 |
| `data/evaluations/difficulty_state.json` | 티어별 마스터리 상태 |
| `data/evaluations/health_scan.json` | 시스템 건강 진단 결과 |
| `data/evaluations/eval_*_extended.json` | 확장 평가 결과 (전체) |

### Backward Compatibility

기존 `--quick`, `--full`, `--status` 플래그는 변경 없이 동작합니다. 확장 기능은 명시적으로 `--generate`, `--health`, `--trend` 등의 플래그를 지정해야만 활성화됩니다.

## References

- **ParallelPersonaEvaluator**: `src/rag/domain/evaluation/parallel_evaluator.py`
- **LLM Judge**: `src/rag/domain/evaluation/llm_judge.py`
- **Persona Definitions**: `src/rag/domain/evaluation/personas.py`
- **Quality Thresholds**: `src/rag/domain/evaluation/llm_judge.py` (LLMJudge.THRESHOLDS)
- **Query Synthesizer**: `src/rag/domain/evaluation/query_synthesizer.py`
- **Extended Metrics**: `src/rag/domain/evaluation/extended_metrics.py`
- **System Health**: `src/rag/domain/evaluation/system_health.py`
- **Difficulty Manager**: `src/rag/domain/evaluation/difficulty_manager.py`
- **Improvement Radar**: `src/rag/domain/evaluation/improvement_radar.py`

---

**Last Updated:** 2026-03-04
**Version:** 2.0.0
