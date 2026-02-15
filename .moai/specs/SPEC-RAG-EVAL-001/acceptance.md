# Acceptance Criteria: SPEC-RAG-EVAL-001

## Overview

| Field | Value |
|-------|-------|
| **SPEC ID** | SPEC-RAG-EVAL-001 |
| **Title** | RAG Quality Evaluator Agent Enhancement |
| **Test Framework** | pytest + pytest-asyncio |
| **Target Coverage** | 85%+ |

---

## Test Scenarios

### TS-001: API Budget Optimization

#### TS-001-01: Batch Execution

**Given** a list of 20 test queries
**When** BatchExecutor processes them with batch_size=5
**Then** queries are grouped into 4 batches
**And** each batch is processed sequentially
**And** total API calls equal 4 (one per batch)

```python
def test_batch_executor_groups_queries():
    """BatchExecutor should group queries into batches."""
    executor = BatchExecutor(batch_size=5)
    queries = [PersonaQuery(query=f"test_{i}", ...) for i in range(20)]

    batches = executor._group_into_batches(queries)

    assert len(batches) == 4
    assert all(len(batch) == 5 for batch in batches)
```

#### TS-001-02: Rate Limiting

**Given** a RateLimiter with 10 requests per minute
**When** 15 requests are made within 1 minute
**Then** first 10 requests succeed immediately
**And** remaining 5 requests wait for next minute

```python
@pytest.mark.asyncio
async def test_rate_limiter_enforces_limits():
    """RateLimiter should enforce requests per minute."""
    limiter = RateLimiter(rpm=10)
    start = time.time()

    # Make 15 requests
    tasks = [limiter.acquire() for _ in range(15)]
    await asyncio.gather(*tasks)

    elapsed = time.time() - start
    # Should take at least 60 seconds (rate limit period)
    assert elapsed >= 60
```

#### TS-001-03: Cost Estimation

**Given** a list of queries with estimated token counts
**When** BatchExecutor.estimate_cost() is called
**Then** returns accurate cost estimation
**And** displays warning if cost exceeds threshold

```python
def test_cost_estimation_accuracy():
    """Cost estimation should be accurate within 10%."""
    executor = BatchExecutor()
    queries = create_test_queries(count=10)

    estimate = executor.estimate_cost(queries)
    actual = executor.execute_batch(queries)

    assert abs(estimate - actual.cost) / actual.cost < 0.10
```

---

### TS-002: Persona Coverage

#### TS-002-01: All Personas Tested

**Given** a full evaluation is initiated
**When** evaluation completes
**Then** all 6 personas have been tested
**And** each persona has at least 25 queries

```python
def test_all_personas_covered():
    """All 6 personas should be tested in full evaluation."""
    evaluator = ParallelPersonaEvaluator()
    results = evaluator.evaluate_parallel(queries_per_persona=25)

    assert len(results) == 6
    for persona in PERSONA_AGENTS:
        assert persona in results
        assert results[persona].queries_tested >= 25
```

#### TS-002-02: Professor Persona No Errors

**Given** professor persona test queries
**When** evaluation is executed
**Then** zero `zip()` errors occur
**And** all queries produce valid results

```python
def test_professor_persona_no_zip_errors():
    """Professor persona should not produce zip() errors."""
    evaluator = ParallelPersonaEvaluator()
    queries = evaluator.generate_persona_queries("professor", count_per_category=5)

    results = []
    for query in queries:
        result = evaluator._evaluate_single_query(query)
        results.append(result)

    assert all(r is not None for r in results)
    assert not any(hasattr(r, 'error') and 'zip()' in str(r.error) for r in results)
```

#### TS-002-03: Query Distribution

**Given** generated queries for a persona
**When** difficulty is analyzed
**Then** distribution is approximately 40% easy, 40% medium, 20% hard

```python
def test_query_difficulty_distribution():
    """Query difficulty should follow target distribution."""
    generator = RegulationQueryGenerator()
    queries = generator.generate("freshman", count=100)

    difficulties = [q.difficulty for q in queries]
    easy_pct = difficulties.count("easy") / len(difficulties)
    medium_pct = difficulties.count("medium") / len(difficulties)
    hard_pct = difficulties.count("hard") / len(difficulties)

    assert 0.35 <= easy_pct <= 0.45
    assert 0.35 <= medium_pct <= 0.45
    assert 0.15 <= hard_pct <= 0.25
```

---

### TS-003: Automation Pipeline

#### TS-003-01: Checkpoint Save

**Given** evaluation in progress with 2/4 batches completed
**When** checkpoint is requested
**Then** state is saved with completed and pending queries

```python
def test_checkpoint_saves_state():
    """Checkpoint should save evaluation state correctly."""
    manager = CheckpointManager()
    state = EvaluationState(
        completed_queries=[...],  # 2 batches
        pending_queries=[...],    # 2 batches
        results=[...],
        timestamp=datetime.now()
    )

    manager.save(state)
    loaded = manager.load()

    assert len(loaded.completed_queries) == len(state.completed_queries)
    assert len(loaded.pending_queries) == len(state.pending_queries)
```

#### TS-003-02: Resume from Checkpoint

**Given** a saved checkpoint with 50% completion
**When** evaluation is resumed
**Then** only pending queries are executed
**And** results are merged with checkpoint results

```python
def test_resume_skips_completed():
    """Resume should skip completed queries."""
    # Create checkpoint with 10 completed queries
    manager = CheckpointManager()
    manager.save(EvaluationState(completed_queries=queries[:10]))

    # Resume evaluation
    controller = ResumeController(manager)
    remaining = controller.get_pending_queries()

    assert len(remaining) == len(queries) - 10
```

#### TS-003-03: Graceful Shutdown

**Given** evaluation in progress
**When** SIGINT (Ctrl+C) is received
**Then** current batch completes
**And** checkpoint is saved
**And** partial results are preserved

```python
def test_graceful_shutdown_preserves_state():
    """Graceful shutdown should preserve evaluation state."""
    evaluator = ParallelPersonaEvaluator()
    evaluator.start()

    # Simulate Ctrl+C after 3 seconds
    time.sleep(3)
    evaluator.interrupt()

    # Check state was saved
    manager = CheckpointManager()
    state = manager.load()

    assert state is not None
    assert len(state.results) > 0
```

---

### TS-004: Reporting Enhancement

#### TS-004-01: Failure Classification

**Given** evaluation results with various failure types
**When** FailureClassifier analyzes them
**Then** failures are categorized correctly
**And** patterns are identified

```python
def test_failure_classification_accuracy():
    """Failure classification should identify patterns."""
    classifier = FailureClassifier()

    # Create mock results with known failure types
    results = [
        create_result_with_hallucination(),
        create_result_with_retrieval_failure(),
        create_result_with_citation_error(),
    ]

    patterns = classifier.analyze_batch(results)

    assert any(p.type == "hallucination" for p in patterns)
    assert any(p.type == "retrieval_failure" for p in patterns)
    assert any(p.type == "citation_error" for p in patterns)
```

#### TS-004-02: Recommendation Generation

**Given** classified failure patterns
**When** RecommendationEngine generates recommendations
**Then** each recommendation includes:
  - Root cause description
  - Code change suggestion
  - Before/after example

```python
def test_recommendation_includes_examples():
    """Recommendations should include code examples."""
    engine = RecommendationEngine()
    patterns = [FailurePattern(type="hallucination", count=5)]

    recommendations = engine.generate(patterns)

    assert len(recommendations) > 0
    assert all(r.code_example is not None for r in recommendations)
    assert all(r.before_after is not None for r in recommendations)
```

#### TS-004-03: SPEC Generation

**Given** failure patterns and recommendations
**When** SPECGenerator creates a SPEC
**Then** SPEC includes:
  - Problem Analysis section
  - EARS format requirements
  - Acceptance criteria

```python
def test_spec_generation_ears_format():
    """Generated SPECs should use EARS format."""
    generator = SPECGenerator()
    patterns = create_failure_patterns()

    spec = generator.generate(patterns)

    assert "## Requirements (EARS Format)" in spec.content
    assert "WHEN" in spec.content
    assert "THE SYSTEM SHALL" in spec.content
    assert "## Acceptance Criteria" in spec.content
```

---

### TS-005: Regulation-Specific Queries

#### TS-005-01: Article-Based Queries

**Given** regulation data with articles (제N조)
**When** RegulationQueryGenerator creates queries
**Then** queries reference specific articles

```python
def test_queries_reference_articles():
    """Queries should reference specific regulation articles."""
    generator = RegulationQueryGenerator(regulation_path="data/output/규정집.json")
    queries = generator.generate("professor", count=20)

    article_pattern = re.compile(r"제\d+조")
    matching = sum(1 for q in queries if article_pattern.search(q.query))

    assert matching >= 10  # At least 50% reference articles
```

#### TS-005-02: Cross-Reference Queries

**Given** regulations with cross-references
**When** queries are generated
**Then** some queries reference multiple regulations

```python
def test_cross_reference_queries():
    """Some queries should reference multiple regulations."""
    generator = RegulationQueryGenerator()
    queries = generator.generate("graduate", count=20)

    cross_ref = [q for q in queries if q.cross_references]
    assert len(cross_ref) >= 3  # At least 15% are cross-reference
```

#### TS-005-03: Temporal Queries

**Given** regulations with deadlines and periods
**When** queries are generated
**Then** temporal queries are included

```python
def test_temporal_queries_included():
    """Temporal queries should be generated."""
    generator = RegulationQueryGenerator()
    queries = generator.generate("freshman", count=20)

    temporal_keywords = ["기한", "언제까지", "기간", "마감"]
    temporal = [q for q in queries
                if any(kw in q.query for kw in temporal_keywords)]

    assert len(temporal) >= 3  # At least 15% are temporal
```

---

### TS-006: Interface Integration

#### TS-006-01: CLI Full Evaluation

**Given** CLI is available
**When** `regulation evaluate --full` is executed
**Then** all personas are tested
**And** results are saved to data/evaluations/

```bash
# CLI Test
$ regulation evaluate --full
Starting full evaluation...
[Progress bar]
Evaluation complete. Results saved to data/evaluations/eval_20260215_120000.json
```

#### TS-006-02: CLI Targeted Evaluation

**Given** CLI is available
**When** `regulation evaluate --persona professor` is executed
**Then** only professor persona is tested

```bash
# CLI Test
$ regulation evaluate --persona professor
Testing professor persona only...
[Progress bar]
Professor persona: 23/25 passed (92%)
```

#### TS-006-03: CLI Resume

**Given** a previous interrupted evaluation
**When** `regulation evaluate --resume` is executed
**Then** evaluation resumes from last checkpoint

```bash
# CLI Test
$ regulation evaluate --resume
Resuming from checkpoint...
Skipping 10 completed queries...
Continuing with remaining queries...
```

#### TS-006-04: Gradio Progress Display

**Given** Gradio dashboard is running
**When** evaluation is in progress
**Then** progress bar displays current status
**And** metrics update in real-time

---

## Quality Gates

### Code Quality

- [ ] All new files pass ruff linting (zero errors)
- [ ] All new files pass pyright type checking
- [ ] All modified files pass characterization tests
- [ ] Test coverage >= 85% for new code

### Functional Quality

- [ ] All 6 personas tested with 25+ queries each
- [ ] Zero `zip()` errors in any persona
- [ ] Pass rate >= 80% on all test scenarios
- [ ] Checkpoint/resume works correctly
- [ ] SPEC generation produces valid EARS format

### Performance Quality

- [ ] Batch execution reduces API calls by 40%+
- [ ] Cache hit rate >= 30% for repeated evaluations
- [ ] Progress updates within 100ms
- [ ] Checkpoint save completes within 500ms

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] All tests passing (unit + integration + characterization)
- [ ] Code coverage >= 85%
- [ ] TRUST 5 compliance verified
- [ ] Documentation updated (agent definition, skill file)
- [ ] No LSP errors
- [ ] Code reviewed and approved

---

## References

- SPEC Document: `.moai/specs/SPEC-RAG-EVAL-001/spec.md`
- Implementation Plan: `.moai/specs/SPEC-RAG-EVAL-001/plan.md`
- Test Framework: pytest, pytest-asyncio
- Quality Standard: TRUST 5 Framework
