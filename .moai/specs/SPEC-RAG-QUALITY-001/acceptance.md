# SPEC-RAG-QUALITY-001: Acceptance Criteria

## TAG BLOCK

```yaml
spec_id: SPEC-RAG-QUALITY-001
title: RAG Answer Quality Improvement System
status: Planned
priority: High
created: 2025-01-28
assigned: manager-spec
lifecycle: spec-anchored
estimated_effort: 4 weeks
labels: [rag, quality, evaluation, llm-as-judge, testing]
```

## Quality Gates

### Overall Requirements

- [HARD] All four core metrics (Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall) must meet minimum thresholds
- [HARD] Test coverage must exceed 80% for all evaluation components
- [HARD] All user personas must be tested with minimum 20 query variations each
- [SOFT] Dashboard load time must be under 2 seconds for initial page load
- [SOFT] Evaluation API costs must stay under $30 per batch evaluation

### Metric Thresholds

| Metric | Minimum Threshold | Target Threshold | Critical Threshold |
|--------|-------------------|------------------|-------------------|
| Faithfulness | 0.90 | 0.95 | 0.70 |
| Answer Relevancy | 0.85 | 0.90 | 0.70 |
| Contextual Precision | 0.80 | 0.85 | 0.65 |
| Contextual Recall | 0.80 | 0.85 | 0.65 |

**Failure Criteria**:
- Any metric below minimum threshold: Flag for review
- Any metric below critical threshold: Block deployment, trigger critical alert
- Faithfulness < 0.70: Immediate investigation required (hallucination risk)

---

## Acceptance Criteria by Requirement Category

### Category 1: LLM-as-Judge Evaluation Framework

#### AC-EVAL-001: Single Query Evaluation

**Given** the RAG quality evaluation system is deployed
**And** a user submits a query: "휴학 신청 절차가 어떻게 되나요?"
**And** the RAG pipeline returns an answer with retrieved contexts
**When** the evaluation system processes the query-answer-context triplet
**Then** the system shall return four metric scores:
  - Faithfulness score between 0.0 and 1.0
  - Answer Relevancy score between 0.0 and 1.0
  - Contextual Precision score between 0.0 and 1.0
  - Contextual Recall score between 0.0 and 1.0
**And** all scores must meet minimum thresholds (Faithfulness >= 0.90, Relevancy >= 0.85, Precision >= 0.80, Recall >= 0.80)
**And** the evaluation result must include a pass/fail status based on threshold compliance
**And** the evaluation must complete within 30 seconds

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_single_query_evaluation():
    query = "휴학 신청 절차가 어떻게 되나요?"
    answer, contexts = await rag_pipeline.search(query)

    evaluator = RAGQualityEvaluator()
    result = await evaluator.evaluate(query, answer, contexts)

    assert result.faithfulness >= 0.90
    assert result.answer_relevancy >= 0.85
    assert result.contextual_precision >= 0.80
    assert result.contextual_recall >= 0.80
    assert result.passed is True
    assert result.overall_score >= 0.84  # Average of thresholds
```

---

#### AC-EVAL-002: Batch Evaluation

**Given** the evaluation system contains 100 test queries
**When** the user initiates a batch evaluation
**Then** the system shall evaluate all 100 queries sequentially
**And** calculate aggregate statistics:
  - Mean score for each metric
  - Standard deviation for each metric
  - Pass rate (percentage of queries meeting all thresholds)
  - Failure cases grouped by issue type
**And** the batch evaluation must complete within 30 minutes
**And** results must be persisted to storage for historical analysis

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_batch_evaluation():
    queries = load_test_queries(count=100)
    evaluator = RAGQualityEvaluator()

    results = []
    for query in queries:
        answer, contexts = await rag_pipeline.search(query)
        result = await evaluator.evaluate(query, answer, contexts)
        results.append(result)

    # Calculate aggregates
    avg_faithfulness = mean([r.faithfulness for r in results])
    avg_relevancy = mean([r.answer_relevancy for r in results])
    pass_rate = sum([1 for r in results if r.passed]) / len(results)

    assert avg_faithfulness >= 0.90
    assert avg_relevancy >= 0.85
    assert pass_rate >= 0.80
    assert len(results) == 100
```

---

#### AC-EVAL-003: Quality Threshold Alerting

**Given** the evaluation system has configured thresholds
**When** an evaluation result shows Faithfulness < 0.70 (critical threshold)
**Then** the system shall trigger a critical alert
**And** send notification to administrators within 5 minutes
**And** log the failure with query details and score breakdown
**And** mark the query-answer pair for manual review
**And** pause automatic deployments if critical threshold violation rate exceeds 10%

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_critical_threshold_alerting():
    # Simulate critical failure
    result = EvaluationResult(
        query="critical query",
        answer="hallucinated answer",
        contexts=[],
        faithfulness=0.65,  # Below critical threshold
        answer_relevancy=0.90,
        contextual_precision=0.85,
        contextual_recall=0.85,
        overall_score=0.81,
        passed=False,
        failure_reasons=["Faithfulness below critical threshold"],
        timestamp=datetime.now()
    )

    alert_sent = await alert_manager.check_critical_threshold(result)

    assert alert_sent is True
    assert "Faithfulness" in alert_manager.get_recent_alerts()
```

---

### Category 2: User Persona Simulation

#### AC-PER-001: Freshman Persona Evaluation

**Given** the freshman persona is configured with beginner expertise level and simple vocabulary
**And** the persona has 20 generated query variations covering common topics (휴학, 복학, 성적, 장학금, 수강)
**When** the evaluation system processes all 20 freshman queries
**Then** the system shall calculate persona-specific metrics:
  - Average Faithfulness >= 0.90
  - Average Answer Relevancy >= 0.85
  - Average Contextual Precision >= 0.80
  - Average Contextual Recall >= 0.80
**And** the queries must use simple, non-technical language appropriate for beginners
**And** the answers must prioritize clarity and simplicity over technical detail

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_freshman_persona_evaluation():
    persona_manager = PersonaManager()
    evaluator = RAGQualityEvaluator()

    queries = await persona_manager.generate_queries("freshman", count=20)

    results = []
    for query in queries:
        # Verify language simplicity
        assert len(query.split()) <= 15  # Short sentences
        assert not any(technical_word in query for technical_word in ["조항", "시행세칙", "규정집"])

        answer, contexts = await rag_pipeline.search(query)
        result = await evaluator.evaluate(query, answer, contexts)
        results.append(result)

    # Verify persona-level metrics
    avg_faithfulness = mean([r.faithfulness for r in results])
    avg_relevancy = mean([r.answer_relevancy for r in results])

    assert avg_faithfulness >= 0.90
    assert avg_relevancy >= 0.85
    assert len(results) == 20
```

---

#### AC-PER-002: Professor Persona Evaluation

**Given** the professor persona is configured with advanced expertise level and academic vocabulary
**And** the persona has 20 generated query variations covering professor-specific topics (연구년, 휴직, 승진, 연구비, 교원인사)
**When** the evaluation system processes all 20 professor queries
**Then** the system shall calculate persona-specific metrics:
  - Average Faithfulness >= 0.92 (higher threshold for advanced users)
  - Average Answer Relevancy >= 0.88 (higher expectation for comprehensive answers)
  - Average Contextual Precision >= 0.85 (more precise retrieval needed)
  - Average Contextual Recall >= 0.85 (comprehensive coverage required)
**And** the queries must use academic language appropriate for faculty
**And** the answers must include detailed citations (편/장/절/조)

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_professor_persona_evaluation():
    persona_manager = PersonaManager()
    evaluator = RAGQualityEvaluator()

    queries = await persona_manager.generate_queries("professor", count=20)

    results = []
    for query in queries:
        # Verify academic language
        assert any(academic_word in query for academic_word in ["조항", "적용 기준", "상세히"])

        answer, contexts = await rag_pipeline.search(query)
        result = await evaluator.evaluate(query, answer, contexts)
        results.append(result)

    # Verify higher thresholds for advanced persona
    avg_faithfulness = mean([r.faithfulness for r in results])
    avg_relevancy = mean([r.answer_relevancy for r in results])

    assert avg_faithfulness >= 0.92
    assert avg_relevancy >= 0.88
    assert len(results) == 20
```

---

#### AC-PER-003: International Student Persona Evaluation

**Given** the international student persona is configured with mixed Korean-English vocabulary
**And** the persona has 20 generated query variations including both Korean and English queries
**When** the evaluation system processes all 20 international student queries
**Then** the system shall calculate persona-specific metrics:
  - Average Faithfulness >= 0.90
  - Average Answer Relevancy >= 0.85
  - Average Contextual Precision >= 0.80
  - Average Contextual Recall >= 0.80
**And** at least 30% of queries must be in English or mixed Korean-English
**And** answers to English queries must be in Korean (system language) but clear for non-native speakers

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_international_persona_evaluation():
    persona_manager = PersonaManager()
    evaluator = RAGQualityEvaluator()

    queries = await persona_manager.generate_queries("international", count=20)

    # Verify language mix
    english_queries = [q for q in queries if any(char.isalpha() and char.isascii() for char in q)]
    assert len(english_queries) >= 6  # At least 30%

    results = []
    for query in queries:
        answer, contexts = await rag_pipeline.search(query)
        result = await evaluator.evaluate(query, answer, contexts)
        results.append(result)

    avg_faithfulness = mean([r.faithfulness for r in results])
    assert avg_faithfulness >= 0.90
```

---

#### AC-PER-004: All Personas Coverage

**Given** the system defines six user personas (freshman, graduate, professor, staff, parent, international)
**When** the evaluation system runs batch evaluation across all personas
**Then** the system shall generate minimum 20 queries per persona (120 total queries)
**And** calculate persona-level metrics breakdowns
**And** identify persona-specific performance gaps (metrics < 0.75 for specific persona)
**And** generate persona comparison report showing relative performance

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_all_personas_coverage():
    persona_manager = PersonaManager()
    evaluator = RAGQualityEvaluator()

    personas = ["freshman", "graduate", "professor", "staff", "parent", "international"]
    results_by_persona = {}

    for persona in personas:
        queries = await persona_manager.generate_queries(persona, count=20)
        assert len(queries) == 20

        persona_results = []
        for query in queries:
            answer, contexts = await rag_pipeline.search(query)
            result = await evaluator.evaluate(query, answer, contexts)
            persona_results.append(result)

        results_by_persona[persona] = aggregate_results(persona_results)

    # Verify all personas evaluated
    assert len(results_by_persona) == 6

    # Verify no persona has severe performance gap
    for persona, results in results_by_persona.items():
        assert results.overall_score >= 0.75, f"{persona} has performance gap"
```

---

### Category 3: Synthetic Test Data Generation

#### AC-SYN-001: Flip-the-RAG Procedural Questions

**Given** a regulation section contains procedural steps (numbered list)
**And** the section describes "휴학 신청 절차" with 5 steps
**When** the synthetic data generator processes this section
**Then** the system shall generate minimum 4 procedural question types:
  1. Overall procedure question: "휴학 신청 절차가 어떻게 되나요?"
  2. Step-specific questions: "휴학 신청 1단계는 뭔가요?" (for first 3 steps)
  3. Prerequisite question: "휴학 신청 전에 필요한 준비가 뭐예요?"
  4. Method question: "휴학 어떻게 신청해요?"
**And** extract ground truth answers directly from section content without LLM generation
**And** validate test case quality (answer extractability, query clarity)

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_procedural_question_generation():
    generator = SyntheticDataGenerator()

    section = RegulationSection(
        title="휴학 신청 절차",
        content="""
        1. 학사시스템 로그인
        2. 휴학 신청서 작성
        3. 지도교수 승인
        4. 학과장 승인
        5. 교무처 승인
        """
    )

    test_cases = await generator.generate_from_section(section)

    # Verify question types
    assert len(test_cases) >= 4
    questions = [tc.question for tc in test_cases]

    assert any("절차" in q for q in questions)
    assert any("1단계" in q or "첫 번째" in q for q in questions)
    assert any("준비" in q or "전에" in q for q in questions)
    assert any("어떻게" in q for q in questions)

    # Verify ground truth extraction
    for tc in test_cases:
        assert len(tc.ground_truth) >= 50  # Minimum answer length
        assert tc.ground_truth in section.content or any(
            step in tc.ground_truth for step in section.extract_numbered_steps()
        )
```

---

#### AC-SYN-002: Flip-the-RAG Conditional Questions

**Given** a regulation section contains eligibility criteria
**And** the section describes "연구년 신청 자격" with conditions (6년 재직, 연구 계획서, 추천)
**When** the synthetic data generator processes this section
**Then** the system shall generate minimum 3 conditional question types:
  1. Qualification question: "연구년 신청 자격이 뭐예요?"
  2. Eligibility question: "연구년 누가 할 수 있나요?"
  3. Restriction question: "연구년 제한 사항이 있나요?"
**And** ground truth must extract all eligibility criteria from section
**And** validation must ensure answer covers all conditions

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_conditional_question_generation():
    generator = SyntheticDataGenerator()

    section = RegulationSection(
        title="연구년 신청 자격",
        content="""
        연구년 신청 자격은 다음과 같습니다.
        1. 본 대학교에 6년 이상 재직한 전임교원
        2. 연구 계획서 제출
        3. 학과장 및 학장의 추천
        """
    )

    test_cases = await generator.generate_from_section(section)

    # Verify question types
    assert len(test_cases) >= 3
    questions = [tc.question for tc in test_cases]

    assert any("자격" in q or "조건" in q for q in questions)
    assert any("누구" in q or "할 수 있" in q for q in questions)
    assert any("제한" in q or "불가" in q for q in questions)

    # Verify ground truth covers all criteria
    eligibility_test_case = next(tc for tc in test_cases if "자격" in tc.question)
    assert "6년" in eligibility_test_case.ground_truth
    assert "연구 계획서" in eligibility_test_case.ground_truth
    assert "추천" in eligibility_test_case.ground_truth
```

---

#### AC-SYN-003: Synthetic Dataset Quality Validation

**Given** the synthetic data generator processes 100 regulation sections
**When** the generation completes
**Then** the system shall produce minimum 500 valid test cases
**And** validate each test case for:
  - Question length: 10-200 characters
  - Ground truth length: minimum 50 characters
  - Semantic similarity: cosine similarity >= 0.5 between question and answer
  - Answer extractability: ground truth directly extractable from section
**And** filter out invalid test cases (estimated 10-20% rejection rate)
**And** generate quality report with statistics:
  - Total test cases generated
  - Valid test cases
  - Rejected test cases with rejection reasons
  - Question type distribution (procedural, conditional, factual)

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_synthetic_dataset_quality():
    generator = SyntheticDataGenerator()
    regulations = load_regulations(count=100)

    all_test_cases = []
    for regulation in regulations:
        test_cases = await generator.generate_from_regulation(regulation)
        all_test_cases.extend(test_cases)

    # Validate dataset size
    assert len(all_test_cases) >= 500

    # Validate quality filters applied
    valid_cases = [tc for tc in all_test_cases if tc.valid]
    assert len(valid_cases) >= 400  # At least 80% valid

    # Validate question length constraints
    for tc in valid_cases:
        assert 10 <= len(tc.question) <= 200

    # Validate ground truth length
    for tc in valid_cases:
        assert len(tc.ground_truth) >= 50

    # Validate semantic similarity
    for tc in valid_cases:
        question_emb = embed(tc.question)
        answer_emb = embed(tc.ground_truth)
        similarity = cosine_similarity(question_emb, answer_emb)
        assert similarity >= 0.5
```

---

### Category 4: Automated Feedback Loop

#### AC-FB-001: Hallucination Detection and Suggestion

**Given** evaluation results show 10 queries with Faithfulness < 0.90
**When** the quality analyzer processes these failures
**Then** the system shall categorize failures as "hallucination" type
**And** generate improvement suggestion targeting "prompt_engineering" component
**And** recommendation must include:
  - Specific parameter adjustment: "Reduce LLM temperature to 0.1"
  - Prompt modification: "Add strict context adherence instruction"
  - Expected impact: "+0.15 Faithfulness score"
  - Implementation effort: "Low (1 hour)"
**And** prioritize suggestion by affected count (10 failures = high priority)

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_hallucination_suggestion():
    analyzer = QualityAnalyzer()

    # Create hallucination failures
    failures = [
        EvaluationResult(
            query=f"query_{i}",
            answer="hallucinated answer",
            contexts=[],
            faithfulness=0.85,  # Below threshold
            answer_relevancy=0.90,
            contextual_precision=0.85,
            contextual_recall=0.85,
            overall_score=0.86,
            passed=False,
            failure_reasons=["Faithfulness below threshold"],
            timestamp=datetime.now()
        )
        for i in range(10)
    ]

    suggestions = analyzer.analyze_failures(failures)

    # Verify hallucination suggestion generated
    assert len(suggestions) >= 1

    hallucination_suggestion = next(
        (s for s in suggestions if s.issue_type == "hallucination"),
        None
    )
    assert hallucination_suggestion is not None
    assert hallucination_suggestion.component == "prompt_engineering"
    assert "temperature" in hallucination_suggestion.recommendation.lower()
    assert hallucination_suggestion.affected_count == 10
    assert hallucination_suggestion.expected_impact.startswith("+")
```

---

#### AC-FB-002: Irrelevant Retrieval Detection and Suggestion

**Given** evaluation results show 15 queries with Contextual Precision < 0.80
**When** the quality analyzer processes these failures
**Then** the system shall categorize failures as "irrelevant_retrieval" type
**And** generate improvement suggestion targeting "reranking" component
**And** recommendation must include:
  - Specific threshold adjustment: "Increase BGE reranker threshold from X to Y"
  - Calculated optimal threshold based on failure analysis
  - Expected impact: "+0.20 Contextual Precision score"
  - Implementation effort: "Medium (2 hours)"
**And** prioritize suggestion higher than suggestions with fewer affected queries

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_irrelevant_retrieval_suggestion():
    analyzer = QualityAnalyzer()

    # Create irrelevant retrieval failures
    failures = [
        EvaluationResult(
            query=f"query_{i}",
            answer="relevant answer",
            contexts=["irrelevant context"],
            faithfulness=0.92,
            answer_relevancy=0.88,
            contextual_precision=0.75,  # Below threshold
            contextual_recall=0.82,
            overall_score=0.84,
            passed=False,
            failure_reasons=["Contextual Precision below threshold"],
            timestamp=datetime.now()
        )
        for i in range(15)
    ]

    suggestions = analyzer.analyze_failures(failures)

    # Verify retrieval suggestion generated
    retrieval_suggestion = next(
        (s for s in suggestions if s.issue_type == "irrelevant_retrieval"),
        None
    )
    assert retrieval_suggestion is not None
    assert retrieval_suggestion.component == "reranking"
    assert "threshold" in retrieval_suggestion.recommendation.lower()
    assert retrieval_suggestion.affected_count == 15
```

---

#### AC-FB-003: Improvement Impact Tracking

**Given** a quality improvement suggestion has been implemented
**And** baseline evaluation showed Faithfulness = 0.85
**When** the improvement tracker measures impact after 7 days
**Then** the system shall re-evaluate the same query set
**And** calculate metric deltas:
  - Faithfulness delta: new_score - baseline_score
  - Overall score delta: new_overall - baseline_overall
**And** generate impact report showing:
  - Baseline metrics before change
  - New metrics after change
  - Metric changes with statistical significance (p-value < 0.05)
  - Verdict: Effective / Ineffective based on delta > 0.05

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_improvement_impact_tracking():
    tracker = ImprovementTracker(evaluator)

    # Store baseline
    baseline = EvaluationRun(
        timestamp=datetime.now() - timedelta(days=8),
        faithfulness=0.85,
        answer_relevancy=0.88,
        contextual_precision=0.82,
        contextual_recall=0.83,
        overall_score=0.845
    )

    # Simulate improvement (e.g., prompt change)
    suggestion = ImprovementSuggestion(
        component="prompt_engineering",
        recommendation="Reduce temperature to 0.1"
    )

    # Re-evaluate after 7 days
    new_results = EvaluationRun(
        timestamp=datetime.now(),
        faithfulness=0.92,  # Improved
        answer_relevancy=0.89,
        contextual_precision=0.83,
        contextual_recall=0.84,
        overall_score=0.87
    )

    impact_report = tracker.calculate_impact(baseline, new_results, suggestion)

    # Verify improvement detected
    assert impact_report.faithfulness_delta == 0.07
    assert impact_report.overall_delta == 0.025
    assert impact_report.verdict == "Effective"
    assert impact_report.statistically_significant is True
```

---

### Category 5: Quality Dashboard and Reporting

#### AC-DASH-001: Real-Time Metrics Display

**Given** the quality dashboard is accessible at `http://localhost:7860/quality`
**And** the latest evaluation batch completed 1 hour ago
**When** a user loads the dashboard
**Then** the page must display within 2 seconds
**And** show four metric cards with latest scores:
  - Faithfulness: 0.93 (with trend indicator)
  - Answer Relevancy: 0.88 (with trend indicator)
  - Contextual Precision: 0.84 (with trend indicator)
  - Contextual Recall: 0.86 (with trend indicator)
**And** trend indicators must show:
  - Green arrow up if metric improved vs previous batch
  - Red arrow down if metric regressed vs previous batch
  - Gray dash if no change (delta < 0.01)

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_dashboard_metrics_display():
    # Store latest evaluation
    latest = EvaluationResult(
        timestamp=datetime.now() - timedelta(hours=1),
        faithfulness=0.93,
        answer_relevancy=0.88,
        contextual_precision=0.84,
        contextual_recall=0.86,
        overall_score=0.878
    )
    await eval_store.save_evaluation(latest)

    # Store previous evaluation for trend
    previous = EvaluationResult(
        timestamp=datetime.now() - timedelta(days=1),
        faithfulness=0.91,  # Improved
        answer_relevancy=0.89,  # Regressed
        contextual_precision=0.84,  # No change
        contextual_recall=0.85,  # Improved
        overall_score=0.873
    )
    await eval_store.save_evaluation(previous)

    # Load dashboard
    dashboard_data = await load_dashboard_metrics()

    # Verify metrics
    assert dashboard_data.faithfulness == 0.93
    assert dashboard_data.faithfulness_trend == "up"  # 0.93 > 0.91
    assert dashboard_data.answer_relevancy == 0.88
    assert dashboard_data.answer_relevancy_trend == "down"  # 0.88 < 0.89
    assert dashboard_data.contextual_precision_trend == "stable"  # No change
```

---

#### AC-DASH-002: Time-Series Visualization

**Given** the dashboard has 30 days of historical evaluation data
**And** evaluations were run daily with varying metric scores
**When** a user views the quality trend chart
**Then** the chart must display:
  - X-axis: Dates (last 30 days)
  - Y-axis: Metric scores (0.0 to 1.0)
  - Four lines: One for each metric (Faithfulness, Relevancy, Precision, Recall)
**And** allow user to:
  - Hover over data points to see exact values and dates
  - Toggle metric lines on/off for clarity
  - Zoom into specific date ranges
  - Export chart as PNG image

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_time_series_visualization():
    # Generate 30 days of historical data
    for day in range(30):
        date = datetime.now() - timedelta(days=day)
        result = EvaluationResult(
            timestamp=date,
            faithfulness=0.90 + random.uniform(-0.05, 0.05),
            answer_relevancy=0.87 + random.uniform(-0.03, 0.03),
            contextual_precision=0.82 + random.uniform(-0.04, 0.04),
            contextual_recall=0.84 + random.uniform(-0.03, 0.03),
            overall_score=0.0
        )
        await eval_store.save_evaluation(result)

    # Load chart data
    chart_data = await load_time_series_chart(days=30)

    # Verify data structure
    assert len(chart_data.dates) == 30
    assert len(chart_data.faithfulness_scores) == 30
    assert len(chart_data.relevancy_scores) == 30
    assert len(chart_data.precision_scores) == 30
    assert len(chart_data.recall_scores) == 30

    # Verify date range
    assert chart_data.dates[0] == datetime.now() - timedelta(days=29)
    assert chart_data.dates[-1] == datetime.now()
```

---

#### AC-DASH-003: Persona Comparison View

**Given** the dashboard has evaluation results for all six personas
**When** a user views the persona comparison table
**Then** the table must display rows for each persona:
  | Persona | Faithfulness | Relevancy | Precision | Recall | Queries | Pass Rate |
**And** highlight:
  - Best performing persona in green (highest overall score)
  - Worst performing persona in red (lowest overall score)
**And** allow user to click persona row to drill down to individual query results
**And** show persona-specific query examples

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_persona_comparison_view():
    # Load persona results
    persona_results = {
        "freshman": PersonaMetrics(faithfulness=0.92, relevancy=0.88, pass_rate=0.90),
        "professor": PersonaMetrics(faithfulness=0.95, relevancy=0.91, pass_rate=0.94),  # Best
        "staff": PersonaMetrics(faithfulness=0.89, relevancy=0.85, pass_rate=0.86),  # Worst
        # ... other personas
    }

    # Load dashboard
    table_data = await load_persona_comparison()

    # Verify table structure
    assert len(table_data.rows) == 6

    # Verify best/worst highlighting
    professor_row = next(r for r in table_data.rows if r.persona == "professor")
    assert professor_row.highlight == "green"

    staff_row = next(r for r in table_data.rows if r.persona == "staff")
    assert staff_row.highlight == "red"
```

---

#### AC-DASH-004: PDF Report Generation

**Given** the user clicks "Generate Report" button on dashboard
**When** the report generation completes
**Then** the system must produce a PDF file with:
  - Title: "RAG Quality Evaluation Report"
  - Date: Evaluation date and timestamp
  - Executive Summary:
    - Overall score and pass rate
    - Key findings (top 3 issues)
    - Improvement recommendations
  - Detailed Metrics:
    - Faithfulness: Score, trend, failure count
    - Answer Relevancy: Score, trend, failure count
    - Contextual Precision: Score, trend, failure count
    - Contextual Recall: Score, trend, failure count
  - Persona Breakdown: Table comparing all personas
  - Failure Analysis: Top 10 failed queries with root cause analysis
**And** the PDF must be downloadable via browser download dialog
**And** file name format: `quality_report_YYYYMMDD.pdf`

**Acceptance Test**:
```python
@pytest.mark.asyncio
async def test_pdf_report_generation():
    # Generate report
    report_path = await generate_pdf_report(
        evaluation_date=datetime.now(),
        include_details=True
    )

    # Verify file created
    assert os.path.exists(report_path)
    assert report_path.endswith("quality_report_20250128.pdf")

    # Verify PDF content (basic validation)
    reader = PdfReader(report_path)
    pages = len(reader.pages)
    assert pages >= 3  # Minimum: Summary, Metrics, Breakdown

    # Extract text and verify key sections
    first_page_text = reader.pages[0].extract_text()
    assert "RAG Quality Evaluation Report" in first_page_text
    assert "Executive Summary" in first_page_text
```

---

## Definition of Done

### Core Functionality

- [ ] All four core metrics (Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall) implemented and tested
- [ ] LLM-as-Judge evaluation framework integrated with RAGAS
- [ ] User persona simulation system supports 6 personas with 20+ queries each
- [ ] Synthetic test data generation (Flip-the-RAG) produces 500+ valid test cases
- [ ] Automated feedback loop generates improvement suggestions for 4 issue types
- [ ] Quality dashboard displays real-time metrics with time-series visualization
- [ ] PDF report generation includes all required sections

### Quality Gates

- [ ] All evaluation metrics meet minimum thresholds (Faithfulness >= 0.90, Relevancy >= 0.85, Precision >= 0.80, Recall >= 0.80)
- [ ] Test coverage exceeds 80% for all evaluation components
- [ ] All personas tested with minimum 20 query variations
- [ ] Dashboard load time under 2 seconds
- [ ] Evaluation API costs under $30 per batch

### Documentation

- [ ] API documentation for evaluation endpoints
- [ ] User guide for quality dashboard
- [ ] CLI command reference for evaluation tools
- [ ] Developer guide for extending evaluation metrics

### Deployment

- [ ] Evaluation system deployed to staging environment
- [ ] Initial batch evaluation completed successfully
- [ ] Dashboard accessible and functional
- [ ] Monitoring and alerting configured
- [ ] Rollback plan tested

---

## Summary

This acceptance criteria document provides:

1. **Detailed test scenarios** using Given-When-Then format for all major features
2. **Quantifiable thresholds** for all quality metrics
3. **Persona-specific requirements** reflecting diverse user needs
4. **Automated validation** through pytest acceptance tests
5. **Definition of Done** ensuring production readiness

**Key Success Metrics**:
- Faithfulness >= 0.90 (no hallucinations)
- Answer Relevancy >= 0.85 (addresses user queries)
- Contextual Precision >= 0.80 (relevant documents ranked higher)
- All 6 personas tested with 20+ queries each
- 500+ synthetic test cases generated and validated

**Next Steps**:
1. Review and approve acceptance criteria
2. Begin implementation with Phase 1: Framework Setup
3. Execute acceptance tests at each phase completion
4. Validate all quality gates before production deployment

---

**Acceptance Criteria Status**: Ready for Review
**Next Phase**: /moai:2-run SPEC-RAG-QUALITY-001
**Target Completion**: 2025-02-25
