# SPEC-RAG-QUALITY-013 Acceptance Criteria

## Overall Pass Criteria

All of the following must be met:

### AC-1: Pass Rate
- **Metric**: Pass rate on 30-query evaluation
- **Current**: 16.7% (5/30)
- **Target**: ≥ 80% (24/30)
- **Command**: `uv run python run_rag_quality_eval.py --quick --summary`

### AC-2: Per-Persona Pass Rate
- **Metric**: Minimum pass rate across all 6 persona groups
- **Current**: 0.0% (student-international)
- **Target**: ≥ 60% per group (3/5 minimum)
- **Verification**: Check `persona_results` in evaluation JSON

### AC-3: Context Relevance
- **Metric**: Average context_relevance score
- **Current**: 0.547
- **Target**: ≥ 0.70
- **Verification**: `summary.avg_context_relevance` in evaluation JSON

### AC-4: Completeness
- **Metric**: Average completeness score
- **Current**: 0.512
- **Target**: ≥ 0.70
- **Verification**: `summary.avg_completeness` in evaluation JSON

### AC-5: Citations
- **Metric**: Average citations score
- **Current**: 0.642
- **Target**: ≥ 0.70
- **Verification**: `summary.avg_citations` in evaluation JSON

### AC-6: Test Suite
- **Metric**: All existing + new tests pass
- **Command**: `uv run pytest`
- **Target**: 0 failures

## Per-Requirement Verification

| Requirement | Verification Method | Pass Condition |
|---|---|---|
| REQ-001 | Unit test: no doc < 0.40 in LLM context | Test passes |
| REQ-002 | Unit test: all-low-score → no-info response | Test passes |
| REQ-003 | Unit test: topic mismatch → excluded | Test passes |
| REQ-004 | Unit test: English analysis → Korean extracted | Test passes |
| REQ-005 | Verify compact prompt exists and ≤ 800 tokens | File check |
| REQ-006 | Unit test: analysis-only → retry produces Korean | Test passes |
| REQ-007 | Unit test: glm-4.7 patterns stripped | Test passes |
| REQ-008 | Unit test: conclusion extracted from analysis | Test passes |
| REQ-009 | Unit test: reg-XXXX → 「규정명」 | Test passes |
| REQ-010 | Unit test: article number auto-appended | Test passes |
| REQ-011 | Evaluation: 0 answers with content but no citation | Report check |
| REQ-012 | Evaluation: answers reference ≥ 2 source docs | Report check |
| REQ-013 | Evaluation: procedural queries have all steps | Report check |
| REQ-014 | Evaluation: staff-admin avg ≥ 0.60 | Report check |
| REQ-015 | Evaluation: international pass rate ≥ 40% | Report check |
