# SPEC-RAG-QUALITY-001: RAG Answer Quality Improvement System

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

## Environment

### Current System Context

**Project**: University Regulation Manager (대학 규정 관리 시스템)

**Technology Stack**:
- Python 3.11+
- llama-index >= 0.14.10 (RAG framework)
- chromadb >= 1.4.0 (Vector database)
- flagembedding >= 1.3.5 (BGE-M3 embeddings)
- BGE Reranker (BAAI/bge-reranker-v2-m3)
- pytest >= 7.4.0 (Testing framework)

**Current RAG Pipeline**:
- Entry point: `src/rag/interface/unified_cli.py`
- Query processing: QueryHandler → Query Analyzer → Typo Corrector
- Search: HybridSearch (BM25 + Dense) → BGEReranker
- Answer generation: Self-RAG with LLM
- Evaluation: rule-based matching (intent, keyword, rule code)

**Current Test Coverage**: 83.66%

### System Scope

**In Scope**:
- LLM-as-Judge evaluation framework integration
- Quality metrics: Faithfulness, Answer Relevancy, Contextual Precision/Recall
- User persona simulation (freshman, graduate, professor, staff, parent, international)
- Synthetic test data generation using Flip-the-RAG workflow
- Automated quality scoring and reporting
- Continuous improvement feedback loop

**Out of Scope**:
- HWP file processing changes
- New interface development
- Database schema modifications
- Core RAG pipeline architectural changes

## Assumptions

### Technical Assumptions

- **High Confidence**: Current RAG pipeline uses Self-RAG with quality reflection mechanisms
- **High Confidence**: BGE Reranker is already integrated for result re-ranking
- **Medium Confidence**: Existing evaluation dataset contains sufficient query variations
- **Medium Confidence**: LLM API access available for judge model (OpenAI/Gemini)
- **Evidence**: Codebase shows `bge_reranker.py` and `self_rag.py` in domain layer

### Business Assumptions

- **High Confidence**: Answer quality directly impacts user trust and adoption
- **High Confidence**: Different user personas require different answer styles and detail levels
- **Medium Confidence**: Current quality metrics (rule-based) don't capture semantic quality
- **Medium Confidence**: Continuous quality improvement requires automated evaluation
- **Evidence**: Product.md shows diverse user types with varying expertise levels

### Integration Assumptions

- **High Confidence**: RAGAS or DeepEval can be integrated without breaking changes
- **High Confidence**: Existing pytest framework supports parameterized tests
- **Medium Confidence**: Judge LLM API costs are acceptable for evaluation frequency
- **Risk if Wrong**: High API costs may limit evaluation frequency
- **Validation Method**: Calculate API cost projections before implementation

## Requirements

### Priority 1: LLM-as-Judge Evaluation Framework (Weeks 1-2)

#### Ubiquitous Requirements

**REQ-EVAL-001**: The system shall evaluate RAG answer quality using LLM-as-Judge methodology with automated scoring.

**REQ-EVAL-002**: The system shall measure Faithfulness score to detect hallucinations and factual inconsistencies in generated answers.

**REQ-EVAL-003**: The system shall measure Answer Relevancy score to ensure responses address user queries appropriately.

**REQ-EVAL-004**: The system shall measure Contextual Precision score to verify relevant documents are ranked higher than irrelevant ones.

**REQ-EVAL-005**: The system shall measure Contextual Recall score to ensure all relevant information is retrieved from the knowledge base.

**REQ-EVAL-006**: The system shall log all evaluation results with timestamps, query details, and metric scores for historical analysis.

#### Event-Driven Requirements

**REQ-EVAL-007**: WHEN evaluation score falls below threshold (Faithfulness < 0.9), the system SHALL flag the query-answer pair for manual review and trigger quality alert.

**REQ-EVAL-008**: WHEN Answer Relevancy score is below 0.85, the system SHALL log the query pattern for query refinement analysis.

**REQ-EVAL-009**: WHEN Contextual Precision score is below 0.8, the system SHALL trigger retrieval mechanism review and suggest reranking adjustments.

**REQ-EVAL-010**: WHEN evaluation batch completes, the system SHALL generate quality report with metric distributions and failure cases.

**REQ-EVAL-011**: WHEN critical quality issue is detected (Faithfulness < 0.7), the system SHALL immediately notify administrators and pause automatic deployments.

#### State-Driven Requirements

**REQ-EVAL-012**: WHILE evaluation is in progress, the system SHALL display real-time progress with completed/remaining test counts.

**REQ-EVAL-013**: IF evaluation framework selection is not configured, the system SHALL default to RAGAS with configurable fallback to DeepEval.

**REQ-EVAL-014**: IF judge LLM API is unavailable, the system SHALL queue evaluation tasks and retry with exponential backoff (max 3 attempts).

**REQ-EVAL-015**: IF evaluation timeout exceeds threshold (5 minutes per query), the system SHALL mark evaluation as failed and continue with next test case.

#### Optional Requirements

**REQ-EVAL-016**: Where possible, the system MAY support custom evaluation metrics beyond the four core metrics.

**REQ-EVAL-017**: Where possible, the system MAY implement cached evaluation for identical query-answer pairs to reduce API costs.

#### Unwanted Behavior Requirements

**REQ-EVAL-018**: The system shall NOT modify production RAG pipeline based on low sample size evaluations (minimum 50 queries).

**REQ-EVAL-019**: The system shall NOT expose judge LLM prompts or internal evaluation logic in user-facing responses.

---

### Priority 1: User Persona Simulation (Weeks 1-2)

#### Ubiquitous Requirements

**REQ-PER-001**: The system shall simulate diverse user personas including freshman, graduate student, professor, administrative staff, parent, and international student.

**REQ-PER-002**: The system shall generate persona-specific query variations reflecting unique vocabulary, expertise level, and information needs.

**REQ-PER-003**: The system shall maintain persona profile definitions with characteristic query patterns, domain knowledge, and communication style.

**REQ-PER-004**: The system shall evaluate answer quality separately for each persona to identify persona-specific performance gaps.

**REQ-PER-005**: The system shall aggregate evaluation results across personas with persona-level metric breakdowns.

#### Event-Driven Requirements

**REQ-PER-006**: WHEN generating test queries for a persona, the system SHALL create minimum 20 query variations covering common use cases for that user type.

**REQ-PER-007**: WHEN persona evaluation shows consistently low scores (all metrics < 0.75), the system SHALL flag persona-specific RAG optimization needs.

**REQ-PER-008**: WHEN international student persona is evaluated, the system SHALL include queries in mixed Korean-English and pure English.

**REQ-PER-009**: WHEN professor persona queries are evaluated, the system SHALL prioritize detailed citation accuracy and comprehensive coverage metrics.

**REQ-PER-010**: WHEN freshman persona queries are evaluated, the system SHALL prioritize clarity and simplicity metrics over technical detail.

#### State-Driven Requirements

**REQ-PER-011**: IF persona profile is missing query templates, the system SHALL use generic query templates with persona-appropriate vocabulary substitution.

**REQ-PER-012**: IF persona query generation exceeds time limit (30 seconds per persona), the system SHALL log warning and use reduced query set (minimum 10 queries).

**REQ-PER-013**: IF specific persona evaluation fails (e.g., API error), the system SHALL continue evaluation for remaining personas and mark failed persona for retry.

#### Optional Requirements

**REQ-PER-014**: Where possible, the system MAY learn from real user queries to update persona templates automatically.

**REQ-PER-015**: Where possible, the system MAY allow custom persona creation through configuration files.

#### Unwanted Behavior Requirements

**REQ-PER-016**: The system shall NOT use stereotypical or offensive language in persona query generation.

**REQ-PER-017**: The system shall NOT assume all users within a persona have identical query patterns or expertise levels.

---

### Priority 2: Synthetic Test Data Generation (Weeks 2-3)

#### Ubiquitous Requirements

**REQ-SYN-001**: The system shall implement Flip-the-RAG workflow to generate high-quality synthetic test data from existing regulation documents.

**REQ-SYN-002**: The system shall create diverse question types including factual lookup, procedural explanation, comparison, and edge case queries.

**REQ-SYN-003**: The system shall generate ground truth answers by extracting relevant information directly from regulation documents without LLM generation.

**REQ-SYN-004**: The system shall validate synthetic test data for quality before inclusion in evaluation dataset (answer extractability, query clarity).

**REQ-SYN-005**: The system shall maintain synthetic data versioning with regeneration capability for regulation updates.

#### Event-Driven Requirements

**REQ-SYN-006**: WHEN regulation documents are updated, the system SHALL identify affected test cases and flag for regeneration or validation.

**REQ-SYN-007**: WHEN synthetic test generation fails for a document section, the system SHALL log failure reason and skip to next section with partial dataset generation.

**REQ-SYN-008**: WHEN test dataset size exceeds target (500 queries), the system SHALL use stratified sampling to maintain persona and query type distribution.

**REQ-SYN-009**: WHEN ground truth answer extraction is ambiguous (multiple relevant sections), the system SHALL create multiple answer variants and flag for manual review.

**REQ-SYN-010**: WHEN synthetic dataset generation completes, the system SHALL generate statistics report with query type distribution, persona coverage, and quality metrics.

#### State-Driven Requirements

**REQ-SYN-011**: IF document section is too short (< 100 characters), the system SHALL skip query generation and log as insufficient content.

**REQ-SYN-012**: IF document section contains procedural steps (numbered lists), the system SHALL generate procedural questions specifically for that section.

**REQ-SYN-013**: IF document section contains eligibility criteria, the system SHALL generate conditional questions (qualifications, requirements, restrictions).

**REQ-SYN-014**: IF existing evaluation dataset exists, the system SHALL merge with new synthetic data and remove duplicates based on semantic similarity.

#### Optional Requirements

**REQ-SYN-015**: Where possible, the system MAY use LLM to paraphrase generated questions for linguistic diversity while maintaining semantic equivalence.

**REQ-SYN-016**: Where possible, the system MAY generate adversarial test cases (ambiguous queries, multi-intent questions) for robustness testing.

#### Unwanted Behavior Requirements

**REQ-SYN-017**: The system shall NOT generate synthetic test data from document sections marked as confidential or internal-only.

**REQ-SYN-018**: The system shall NOT include personally identifiable information or sensitive content in synthetic test data.

---

### Priority 2: Automated Feedback Loop (Weeks 3-4)

#### Ubiquitous Requirements

**REQ-FB-001**: The system shall implement automated feedback loop that analyzes evaluation results and generates improvement suggestions.

**REQ-FB-002**: The system shall categorize quality issues into types: hallucination, irrelevant retrieval, incomplete answer, unclear explanation.

**REQ-FB-003**: The system shall prioritize improvement suggestions by impact (number of affected queries) and severity (score deficit magnitude).

**REQ-FB-004**: The system shall generate actionable recommendations with specific component targeting (retrieval, reranking, prompt, citation).

**REQ-FB-005**: The system shall track improvement suggestion implementation and measure impact on subsequent evaluations.

#### Event-Driven Requirements

**REQ-FB-006**: WHEN evaluation batch completes, the system SHALL automatically analyze failure patterns and generate top 5 improvement suggestions.

**REQ-FB-007**: WHEN Faithfulness score is consistently low across queries, the system SHALL suggest prompt engineering improvements (temperature, top_p, context window).

**REQ-FB-008**: WHEN Contextual Precision score is low, the system SHALL suggest retrieval mechanism adjustments (reranker threshold, hybrid weight tuning).

**REQ-FB-009**: WHEN specific persona shows low scores, the system SHALL suggest persona-specific prompt adaptation or query rewriting improvements.

**REQ-FB-010**: WHEN improvement suggestion is implemented, the system SHALL schedule re-evaluation after 7 days to measure impact.

#### State-Driven Requirements

**REQ-FB-011**: IF evaluation shows no quality issues (all metrics > 0.95), the system SHALL log congratulations message and reduce evaluation frequency to weekly.

**REQ-FB-012**: IF quality regression is detected (metrics drop by > 0.05 from previous baseline), the system SHALL trigger critical alert and request immediate investigation.

**REQ-FB-013**: IF improvement suggestions are implemented but show no impact after 3 evaluation cycles, the system SHALL mark suggestions as ineffective and suggest alternative approaches.

#### Optional Requirements

**REQ-FB-014**: Where possible, the system MAY implement automated A/B testing for improvement suggestions (e.g., prompt variant comparison).

**REQ-FB-015**: Where possible, the system MAY integrate with issue tracking system to create improvement tickets automatically.

#### Unwanted Behavior Requirements

**REQ-FB-016**: The system shall NOT modify production prompts or retrieval parameters without explicit approval or deployment process.

**REQ-FB-017**: The system shall NOT generate contradictory improvement suggestions (e.g., increase and decrease same parameter).

---

### Priority 3: Quality Dashboard and Reporting (Weeks 3-4)

#### Ubiquitous Requirements

**REQ-DASH-001**: The system shall provide quality dashboard displaying real-time and historical evaluation metrics.

**REQ-DASH-002**: The system shall support metric visualization with time-series trends, distribution histograms, and percentile breakdowns.

**REQ-DASH-003**: The system shall enable persona-based filtering and comparison in dashboard views.

**REQ-DASH-004**: The system shall generate downloadable evaluation reports in PDF and JSON formats.

**REQ-DASH-005**: The system shall implement alerting for quality threshold violations with configurable notification channels (email, Slack).

#### Event-Driven Requirements

**REQ-DASH-006**: WHEN user accesses quality dashboard, the system SHALL load latest evaluation results with caching for rapid page load (< 2 seconds).

**REQ-DASH-007**: WHEN quality threshold violation occurs, the system SHALL send alert notification within 5 minutes of evaluation completion.

**REQ-DASH-008**: WHEN user generates evaluation report, the system SHALL include executive summary, detailed metrics, failure case analysis, and improvement recommendations.

**REQ-DASH-009**: WHEN user compares evaluation results across time periods, the system SHALL highlight significant changes (> 5% metric improvement or regression).

**REQ-DASH-010**: WHEN new evaluation batch completes, the system SHALL automatically update dashboard with latest results.

#### State-Driven Requirements

**REQ-DASH-011**: IF evaluation history exceeds 100 batches, the system SHALL aggregate older data into weekly/monthly summaries to maintain dashboard performance.

**REQ-DASH-012**: IF dashboard user is not authenticated, the system SHALL show read-only public view with limited detailed information.

**REQ-DASH-013**: IF specific metric has no data (e.g., first evaluation run), the system SHALL display "No baseline data available" message in dashboard.

#### Optional Requirements

**REQ-DASH-014**: Where possible, the system MAY provide drill-down capability from aggregate metrics to individual query-level results.

**REQ-DASH-015**: Where possible, the system MAY integrate with existing monitoring systems (Prometheus, Grafana) for unified observability.

#### Unwanted Behavior Requirements

**REQ-DASH-016**: The system shall NOT expose sensitive query content or user PII in public dashboard views.

**REQ-DASH-017**: The system shall NOT cache dashboard data longer than 5 minutes to ensure timely updates.

## Specifications

### Architecture Design

#### LLM-as-Judge Integration (REQ-EVAL-001 to REQ-EVAL-019)

**Component**: `RAGQualityEvaluator` in `src/rag/domain/evaluation/quality_evaluator.py`

**Framework Selection**:
```python
# Primary: RAGAS (https://github.com/explodinggradients/ragas)
# Fallback: DeepEval (https://github.com/confident-ai/deepeval)

class EvaluationFramework:
    RAGAS = "ragas"
    DEEPEVAL = "deepeval"

class RAGQualityEvaluator:
    def __init__(
        self,
        framework: EvaluationFramework = EvaluationFramework.RAGAS,
        judge_llm: str = "gpt-4o",  # Judge model
        metrics: List[Metric] = None
    ):
        self.framework = framework
        self.judge_llm = judge_llm
        self.metrics = metrics or [
            FaithfulnessMetric(threshold=0.9),
            AnswerRelevancyMetric(threshold=0.85),
            ContextualPrecisionMetric(threshold=0.8),
            ContextualRecallMetric(threshold=0.8)
        ]
```

**Evaluation Pipeline**:
```python
@dataclass
class EvaluationResult:
    query: str
    answer: str
    contexts: List[str]
    faithfulness: float
    answer_relevancy: float
    contextual_precision: float
    contextual_recall: float
    overall_score: float
    passed: bool
    failure_reasons: List[str]
    timestamp: datetime

class RAGQualityEvaluator:
    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str]
    ) -> EvaluationResult:
        """Run LLM-as-Judge evaluation"""
        results = {}
        for metric in self.metrics:
            score = await metric.evaluate(
                query=query,
                answer=answer,
                contexts=contexts
            )
            results[metric.name] = score

        overall = self._compute_overall_score(results)
        passed = self._check_thresholds(results)

        return EvaluationResult(...)
```

**Metrics Definition**:

1. **Faithfulness** (Hallucination Detection):
   - Measures factual consistency between answer and retrieved context
   - Score range: 0.0 to 1.0
   - Threshold: 0.9 (strict, no hallucinations allowed)
   - Judge prompt: "Verify each statement in answer is supported by context"

2. **Answer Relevancy** (Query Response):
   - Measures how well answer addresses the original query
   - Score range: 0.0 to 1.0
   - Threshold: 0.85
   - Judge prompt: "Rate how completely and directly answer responds to query"

3. **Contextual Precision** (Retrieval Ranking):
   - Measures whether relevant documents are ranked higher
   - Score range: 0.0 to 1.0
   - Threshold: 0.8
   - Judge prompt: "Identify which retrieved contexts are relevant to query"

4. **Contextual Recall** (Information Completeness):
   - Measures whether all relevant information was retrieved
   - Score range: 0.0 to 1.0
   - Threshold: 0.8
   - Judge prompt: "Identify information in ground truth missing from retrieved contexts"

#### User Persona Simulation (REQ-PER-001 to REQ-PER-017)

**Component**: `PersonaManager` in `src/rag/domain/evaluation/personas.py`

**Persona Definitions**:
```python
@dataclass
class PersonaProfile:
    name: str  # e.g., "freshman", "professor"
    display_name: str  # e.g., "신입생", "교수"
    expertise_level: str  # "beginner", "intermediate", "advanced"
    vocabulary_style: str  # "simple", "academic", "administrative"
    query_templates: List[str]
    common_topics: List[str]
    answer_preferences: Dict[str, Any]  # detail_level, citation_style

PERSONAS = {
    "freshman": PersonaProfile(
        name="freshman",
        display_name="신입생",
        expertise_level="beginner",
        vocabulary_style="simple",
        query_templates=[
            "{규정주제} 어떻게 해요?",
            "{규정주제} 절차 알려주세요",
            "{규정주제} 자격이 뭐예요?",
        ],
        common_topics=["휴학", "복학", "성적", "장학금", "수강"],
        answer_preferences={
            "detail_level": "simple",
            "citation_style": "minimal",
            "clarity_priority": True
        }
    ),
    "professor": PersonaProfile(
        name="professor",
        display_name="교수",
        expertise_level="advanced",
        vocabulary_style="academic",
        query_templates=[
            "{규정주제} 관련 조항 확인 필요",
            "{규정주제} 적용 기준 상세히",
            "{규정주제} 관련 편/장/조 구체적 근거",
        ],
        common_topics=["연구년", "휴직", "승진", "연구비", "교원인사"],
        answer_preferences={
            "detail_level": "comprehensive",
            "citation_style": "detailed",
            "precision_priority": True
        }
    ),
    "international": PersonaProfile(
        name="international",
        display_name="외국인유학생",
        expertise_level="intermediate",
        vocabulary_style="mixed",
        query_templates=[
            "How do I {규정주제}?",
            "{규정주제} procedure for international students",
            "{규정주제} requirements",
        ],
        common_topics=["비자", "등록금", "수업", "기숙사", "언어"],
        answer_preferences={
            "detail_level": "moderate",
            "citation_style": "standard",
            "language": "korean_english_mixed"
        }
    ),
    # ... additional personas: graduate, staff, parent
}
```

**Query Generation**:
```python
class PersonaManager:
    def __init__(self, regulation_database: RegulationDatabase):
        self.regulation_db = regulation_database
        self.personas = PERSONAS

    async def generate_persona_queries(
        self,
        persona_name: str,
        count: int = 20
    ) -> List[str]:
        """Generate persona-specific test queries"""
        persona = self.personas[persona_name]

        # Get regulation topics relevant to persona
        topics = self._get_relevant_topics(persona)

        queries = []
        for topic in random.sample(topics, min(count, len(topics))):
            template = random.choice(persona.query_templates)
            query = template.format(규정주제=topic)
            queries.append(query)

        return queries

    async def evaluate_all_personas(
        self,
        evaluator: RAGQualityEvaluator,
        queries_per_persona: int = 20
    ) -> Dict[str, EvaluationSummary]:
        """Evaluate RAG quality across all personas"""
        results = {}

        for persona_name, persona in self.personas.items():
            queries = await self.generate_persona_queries(
                persona_name,
                queries_per_persona
            )

            persona_results = []
            for query in queries:
                answer, contexts = await self._rag_pipeline.search(query)
                result = await evaluator.evaluate(query, answer, contexts)
                persona_results.append(result)

            results[persona_name] = self._aggregate_results(persona_results)

        return results
```

#### Synthetic Test Data Generation (REQ-SYN-001 to REQ-SYN-018)

**Component**: `SyntheticDataGenerator` in `src/rag/domain/evaluation/synthetic_data.py`

**Flip-the-RAG Workflow**:
```python
class SyntheticDataGenerator:
    """
    Flip-the-RAG: Generate questions from answers (regulations)
    Instead of Q -> A, generate A -> Q from document content
    """

    def __init__(self, regulation_db: RegulationDatabase):
        self.regulation_db = regulation_db

    async def generate_from_regulation(
        self,
        regulation: Regulation
    ) -> List[TestCase]:
        """Generate test cases from single regulation document"""
        test_cases = []

        for section in regulation.sections:
            # Skip if section too short
            if len(section.content) < 100:
                continue

            # Generate question based on section type
            if section.has_numbered_list():
                questions = self._generate_procedural_questions(section)
            elif section.has_eligibility_criteria():
                questions = self._generate_conditional_questions(section)
            else:
                questions = self._generate_factual_questions(section)

            for question in questions:
                # Extract ground truth answer from section
                ground_truth = self._extract_answer(section, question)

                # Validate quality
                if self._validate_test_case(question, ground_truth):
                    test_case = TestCase(
                        question=question,
                        ground_truth=ground_truth,
                        regulation_id=regulation.id,
                        section_id=section.id,
                        question_type=self._classify_question_type(question),
                        metadata={
                            "generated_by": "Flip-the-RAG",
                            "section_type": section.type
                        }
                    )
                    test_cases.append(test_case)

        return test_cases

    def _generate_procedural_questions(
        self,
        section: RegulationSection
    ) -> List[str]:
        """Generate questions for procedural content (numbered lists)"""
        steps = section.extract_numbered_steps()
        questions = []

        # Overall procedure question
        questions.append(f"{section.title} 절차가 어떻게 되나요?")

        # Step-specific questions
        for i, step in enumerate(steps[:3], 1):  # Max 3 steps
            questions.append(f"{section.title} {i}단계는 뭔가요?")

        # Prerequisite question
        questions.append(f"{section.title} 전에 필요한 준비가 뭐예요?")

        return questions

    def _generate_conditional_questions(
        self,
        section: RegulationSection
    ) -> List[str]:
        """Generate questions for eligibility criteria"""
        criteria = section.extract_eligibility_criteria()
        questions = []

        questions.append(f"{section.title} 자격 요건이 뭐예요?")
        questions.append(f"{section.title} 누가 할 수 있나요?")
        questions.append(f"{section.title} 제한 사항이 있나요?")

        return questions
```

**Test Case Validation**:
```python
def _validate_test_case(
    self,
    question: str,
    ground_truth: str
) -> bool:
    """Validate generated test case quality"""
    # Check question clarity
    if len(question) < 10 or len(question) > 200:
        return False

    # Check answer extractability
    if len(ground_truth) < 50:
        return False

    # Check semantic relevance (embedding similarity)
    question_emb = self.embed(question)
    answer_emb = self.embed(ground_truth)
    similarity = cosine_similarity(question_emb, answer_emb)

    if similarity < 0.5:
        return False  # Question and answer not related

    return True
```

#### Automated Feedback Loop (REQ-FB-001 to REQ-FB-017)

**Component**: `QualityAnalyzer` in `src/rag/domain/evaluation/quality_analyzer.py`

**Issue Categorization**:
```python
class QualityIssue:
    HALLUCINATION = "hallucination"  # Faithfulness < 0.9
    IRRELEVANT_RETRIEVAL = "irrelevant_retrieval"  # Contextual Precision < 0.8
    INCOMPLETE_ANSWER = "incomplete_answer"  # Contextual Recall < 0.8
    IRRELEVANT_ANSWER = "irrelevant_answer"  # Answer Relevancy < 0.85

class QualityAnalyzer:
    def analyze_failures(
        self,
        results: List[EvaluationResult]
    ) -> List[ImprovementSuggestion]:
        """Analyze evaluation results and generate suggestions"""
        suggestions = []

        # Categorize failures
        failures_by_type = self._categorize_failures(results)

        # Generate suggestions for each failure type
        for issue_type, failures in failures_by_type.items():
            if len(failures) > 5:  # Only if pattern detected
                suggestion = self._generate_suggestion(issue_type, failures)
                suggestions.append(suggestion)

        # Prioritize by impact
        suggestions.sort(key=lambda s: s.affected_count, reverse=True)

        return suggestions

    def _generate_suggestion(
        self,
        issue_type: str,
        failures: List[EvaluationResult]
    ) -> ImprovementSuggestion:
        """Generate specific improvement suggestion"""
        if issue_type == QualityIssue.HALLUCINATION:
            return ImprovementSuggestion(
                issue_type=QualityIssue.HALLUCINATION,
                affected_count=len(failures),
                severity=self._calculate_severity(failures),
                component="prompt_engineering",
                recommendation="Reduce LLM temperature to 0.1 and add strict context adherence instruction",
                expected_impact="+0.15 Faithfulness score",
                implementation_effort="Low (1 hour)"
            )

        elif issue_type == QualityIssue.IRRELEVANT_RETRIEVAL:
            return ImprovementSuggestion(
                issue_type=QualityIssue.IRRELEVANT_RETRIEVAL,
                affected_count=len(failures),
                severity=self._calculate_severity(failures),
                component="reranking",
                recommendation=f"Increase BGE reranker threshold from current to {self._calculate_optimal_threshold(failures)}",
                expected_impact="+0.20 Contextual Precision score",
                implementation_effort="Medium (2 hours)"
            )

        # ... additional issue types
```

**Continuous Improvement**:
```python
class ImprovementTracker:
    def __init__(self, evaluator: RAGQualityEvaluator):
        self.evaluator = evaluator
        self.baseline = None
        self.improvements = []

    async def track_improvement(
        self,
        suggestion: ImprovementSuggestion
    ) -> ImpactReport:
        """Measure impact of implemented improvement"""
        # Store baseline before change
        if not self.baseline:
            self.baseline = await self.evaluator.evaluate_batch(
                self.test_queries
            )

        # Re-evaluate after implementation
        new_results = await self.evaluator.evaluate_batch(
            self.test_queries
        )

        # Calculate impact
        impact = self._calculate_impact(self.baseline, new_results)

        return ImpactReport(
            suggestion=suggestion,
            baseline_metrics=self.baseline,
            new_metrics=new_results,
            metric_changes=impact
        )
```

#### Quality Dashboard (REQ-DASH-001 to REQ-DASH-017)

**Component**: `QualityDashboard` in `src/rag/interface/web/quality_dashboard.py`

**Dashboard API Endpoints**:
```python
from fastapi import APIRouter, Query

router = APIRouter(prefix="/quality", tags=["quality"])

@router.get("/metrics/overview")
async def get_quality_overview() -> QualityOverview:
    """Get latest quality metrics overview"""
    latest = await eval_store.get_latest_evaluation()
    baseline = await eval_store.get_baseline()

    return QualityOverview(
        faithfulness=latest.faithfulness,
        answer_relevancy=latest.answer_relevancy,
        contextual_precision=latest.contextual_precision,
        contextual_recall=latest.contextual_recall,
        overall_score=latest.overall_score,
        trend=calculate_trend(baseline, latest),
        evaluation_timestamp=latest.timestamp
    )

@router.get("/metrics/history")
async def get_metrics_history(
    days: int = Query(default=30, ge=1, le=365)
) -> List[QualitySnapshot]:
    """Get historical metrics for time-series visualization"""
    start_date = datetime.now() - timedelta(days=days)
    history = await eval_store.get_history_since(start_date)
    return history

@router.get("/metrics/personas")
async def get_persona_metrics() -> Dict[str, PersonaMetrics]:
    """Get metrics breakdown by user persona"""
    results = await eval_store.get_latest_persona_results()

    return {
        persona_name: PersonaMetrics(
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            query_count=result.query_count,
            pass_rate=result.pass_rate
        )
        for persona_name, result in results.items()
    }

@router.get("/report/generate")
async def generate_report(
    format: str = Query(default="pdf", regex="^(pdf|json)$"),
    include_details: bool = True
) -> FileResponse:
    """Generate downloadable evaluation report"""
    latest = await eval_store.get_latest_evaluation()

    if format == "pdf":
        report_path = await generate_pdf_report(latest, include_details)
    else:
        report_path = await generate_json_report(latest, include_details)

    return FileResponse(
        report_path,
        media_type="application/pdf" if format == "pdf" else "application/json",
        filename=f"quality_report_{latest.timestamp.strftime('%Y%m%d')}.{format}"
    )
```

**Dashboard Frontend** (Gradio Integration):
```python
import gradio as gr

def create_quality_dashboard():
    with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
        gr.Markdown("# RAG Quality Dashboard")

        with gr.Row():
            with gr.Column():
                # Metric cards
                faithfulness = gr.Number(label="Faithfulness", precision=3)
                relevancy = gr.Number(label="Answer Relevancy", precision=3)
                precision = gr.Number(label="Contextual Precision", precision=3)
                recall = gr.Number(label="Contextual Recall", precision=3)

            with gr.Column():
                # Time-series chart
                plot = gr.LinePlot(label="Quality Trend Over Time")

        with gr.Row():
            # Persona comparison
            persona_table = gr.Dataframe(label="Persona Breakdown")

        with gr.Row():
            # Generate report button
            report_btn = gr.Button("Generate Report")
            report_download = gr.File(label="Download Report")

        # Auto-refresh every 5 minutes
        dashboard.load(
            update_dashboard,
            inputs=[],
            outputs=[faithfulness, relevancy, precision, recall, plot, persona_table],
            every=300  # 5 minutes
        )

    return dashboard
```

### File Structure

```
src/
├── rag/
│   ├── domain/
│   │   ├── evaluation/
│   │   │   ├── quality_evaluator.py         # NEW: LLM-as-Judge framework
│   │   │   ├── metrics.py                   # NEW: Custom metric definitions
│   │   │   ├── personas.py                  # NEW: User persona simulation
│   │   │   ├── synthetic_data.py            # NEW: Flip-the-RAG data generation
│   │   │   ├── quality_analyzer.py          # NEW: Feedback loop analysis
│   │   │   └── models.py                    # NEW: Data models for evaluation
│   │   └── retrieval/
│   │       └── reranker.py                  # MODIFIED: Threshold tuning integration
│   ├── application/
│   │   └── evaluation/
│   │       ├── evaluation_service.py        # NEW: Orchestrate evaluation workflow
│   │       └── improvement_service.py       # NEW: Track and apply improvements
│   ├── infrastructure/
│   │   ├── storage/
│   │   │   └── evaluation_store.py          # NEW: Evaluation results persistence
│   │   └── llm/
│   │       └── judge_llm.py                 # NEW: Judge model integration
│   ├── interface/
│   │   ├── web/
│   │   │   ├── quality_dashboard.py         # NEW: Gradio dashboard
│   │   │   └── routes/
│   │   │       └── quality.py               # NEW: FastAPI endpoints
│   │   └── cli/
│   │       └── evaluation_commands.py       # NEW: CLI evaluation commands
│   └── tests/
│       ├── evaluation/
│       │   ├── test_quality_evaluator.py    # NEW: Framework tests
│       │   ├── test_personas.py             # NEW: Persona generation tests
│       │   ├── test_synthetic_data.py       # NEW: Data generation tests
│       │   └── test_quality_analyzer.py     # NEW: Analyzer tests
│       └── data/
│           └── evaluation_dataset.json      # NEW: Synthetic test dataset
```

### Dependencies

**New Dependencies**:
```toml
[tool.poetry.dependencies]
ragas = "^0.1.0"  # LLM-as-Judge evaluation framework
deepeval = "^0.21.0"  # Alternative evaluation framework
plotly = "^5.18.0"  # Dashboard visualization
gradio = "^4.0.0"  # Already in project, update version
fastapi = "^0.104.0"  # Dashboard API endpoints

[tool.poetry.dev-dependencies]
pytest-asyncio = "^0.21.0"  # Async evaluation testing
```

**Existing Dependencies Used**:
- llama-index (RAG pipeline integration)
- chromadb (Regulation database access)
- pytest (Test framework)
- faker (Synthetic data generation)

### API Cost Estimation

**Judge LLM API Costs** (GPT-4o as judge):
- Faithfulness evaluation: ~500 tokens per query
- Answer Relevancy: ~300 tokens per query
- Contextual Precision: ~400 tokens per query
- Contextual Recall: ~400 tokens per query
- **Total per query**: ~1,600 tokens ≈ $0.003 per query
- **Full evaluation (500 queries × 4 personas × 4 metrics)**: ~$24 per batch

**Cost Optimization Strategies**:
1. Cache evaluation results for identical query-answer pairs
2. Use cheaper judge model (GPT-4o-mini) for initial evaluation
3. Batch evaluation during off-peak hours
4. Implement incremental evaluation (new queries only)

## Traceability

### Requirements to Components Mapping

| Requirement ID | Component | File |
|---------------|-----------|------|
| REQ-EVAL-001 ~ REQ-EVAL-019 | RAGQualityEvaluator, Metrics | domain/evaluation/quality_evaluator.py |
| REQ-PER-001 ~ REQ-PER-017 | PersonaManager, PersonaProfile | domain/evaluation/personas.py |
| REQ-SYN-001 ~ REQ-SYN-018 | SyntheticDataGenerator | domain/evaluation/synthetic_data.py |
| REQ-FB-001 ~ REQ-FB-017 | QualityAnalyzer, ImprovementTracker | domain/evaluation/quality_analyzer.py |
| REQ-DASH-001 ~ REQ-DASH-017 | QualityDashboard, ReportGenerator | interface/web/quality_dashboard.py |

### Components to Test Cases Mapping

| Component | Test File | Test Coverage Target |
|-----------|-----------|---------------------|
| RAGQualityEvaluator | tests/evaluation/test_quality_evaluator.py | 90% |
| PersonaManager | tests/evaluation/test_personas.py | 85% |
| SyntheticDataGenerator | tests/evaluation/test_synthetic_data.py | 85% |
| QualityAnalyzer | tests/evaluation/test_quality_analyzer.py | 85% |
| EvaluationService | tests/evaluation/test_evaluation_service.py | 80% |
| QualityDashboard | tests/evaluation/test_quality_dashboard.py | 75% |

### Dependencies

**External Dependencies**:
- RAGAS >= 0.1.0 (Evaluation framework)
- OpenAI API / Gemini API (Judge LLM)
- Existing LlamaIndex, ChromaDB stack

**Internal Dependencies**:
- `domain/retrieval/hybrid_search.py` (retrieval for evaluation)
- `domain/llm/self_rag.py` (answer generation for evaluation)
- `application/rag/pipeline.py` (RAG pipeline integration)

---

## Appendix

### Glossary

- **LLM-as-Judge**: Using Large Language Models to evaluate the quality of other LLM outputs
- **Faithfulness**: Measure of factual consistency between generated answer and retrieved context (hallucination detection)
- **Answer Relevancy**: Measure of how well the answer addresses the original user query
- **Contextual Precision**: Measure of whether relevant documents are ranked higher than irrelevant ones in retrieval
- **Contextual Recall**: Measure of whether all relevant information was retrieved from the knowledge base
- **Flip-the-RAG**: Question generation workflow starting from document content (A → Q instead of Q → A)
- **User Persona**: Simulated user profile with specific characteristics, expertise level, and query patterns
- **Synthetic Test Data**: Artificially generated test cases from existing documents for quality evaluation

### References

- RAGAS Documentation: https://docs.ragas.io/
- DeepEval Documentation: https://docs.confident-ai.com/
- LLM-as-Judge Best Practices: [Arize AI Blog](https://arize.com/blog/)
- Flip-the-RAG Methodology: [LlamaIndex Blog](https://blog.llamaindex.ai/)

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-28 | manager-spec | Initial SPEC creation for RAG quality improvement system |

---

**SPEC Status**: Planned
**Next Phase**: /moai:2-run SPEC-RAG-QUALITY-001 (Implementation with DDD)
**Estimated Completion**: 2025-02-25 (4 weeks from start)
