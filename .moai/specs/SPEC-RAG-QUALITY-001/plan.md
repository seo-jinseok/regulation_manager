# SPEC-RAG-QUALITY-001: Implementation Plan

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

## Implementation Milestones

### Primary Goals (Weeks 1-2)

**Goal 1: LLM-as-Judge Framework Integration**
- Integrate RAGAS evaluation framework with project
- Configure judge LLM (GPT-4o/Gemini) with API authentication
- Implement four core metrics: Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall
- Create evaluation pipeline with automated scoring
- Set up result logging and persistence

**Success Criteria**:
- RAGAS successfully installed and configured
- Judge LLM API authenticated and functional
- Single query evaluation completes with all four metrics
- Evaluation results stored in structured format (JSON)

**Dependencies**:
- OpenAI/Gemini API key access
- RAGAS package installation
- Existing RAG pipeline accessible for testing

---

**Goal 2: User Persona Simulation System**
- Define six user personas (freshman, graduate, professor, staff, parent, international)
- Create persona profile configurations with query templates
- Implement persona-specific query generation (20+ queries per persona)
- Build persona-based evaluation orchestrator
- Generate initial test dataset across all personas

**Success Criteria**:
- Six personas defined with distinct characteristics
- 20+ queries generated per persona (120+ total queries)
- Queries reflect persona-appropriate vocabulary and expertise
- Persona evaluation results aggregated with breakdowns

**Dependencies**:
- Regulation database access for topic extraction
- Query template definitions per persona

---

### Secondary Goals (Weeks 2-3)

**Goal 3: Synthetic Test Data Generation**
- Implement Flip-the-RAG workflow for test case generation
- Create question generators for procedural, conditional, and factual queries
- Build ground truth answer extraction from regulation documents
- Implement test case validation (quality filters)
- Generate initial dataset of 500+ test cases

**Success Criteria**:
- Flip-the-RAG pipeline generates questions from regulation documents
- Ground truth answers extracted without LLM generation
- Test case validation removes low-quality cases
- Dataset contains diverse question types and regulation coverage

**Dependencies**:
- Regulation database with structured document access
- Embedding model for semantic similarity validation

---

**Goal 4: Automated Feedback Loop**
- Implement quality issue categorization (hallucination, retrieval, completeness)
- Build failure pattern analyzer with statistical significance testing
- Create improvement suggestion generator with component targeting
- Implement improvement tracker with baseline measurement
- Set up automated re-evaluation scheduling

**Success Criteria**:
- Issues categorized into four types with specific thresholds
- Failure patterns identified with >5 query minimum
- Suggestions generated with specific component and parameter recommendations
- Improvement impact measured against baseline

**Dependencies**:
- LLM-as-Judge framework operational
- Historical evaluation data available for analysis

---

### Final Goals (Weeks 3-4)

**Goal 5: Quality Dashboard and Reporting**
- Build Gradio-based quality dashboard with real-time metrics
- Create FastAPI endpoints for metric retrieval and filtering
- Implement time-series visualization with trend analysis
- Build persona comparison views with drill-down capability
- Generate downloadable evaluation reports (PDF/JSON)

**Success Criteria**:
- Dashboard displays all four metrics with real-time updates
- Time-series charts show historical trends (30-90 days)
- Persona breakdown table with comparative metrics
- PDF report generation with executive summary and detailed analysis
- Auto-refresh every 5 minutes without manual reload

**Dependencies**:
- Gradio framework (already in project)
- FastAPI for API endpoints
- Plotly for visualization
- Evaluation result storage accessible

---

### Optional Goals (Week 4)

**Goal 6: Advanced Features**
- Implement evaluation result caching for cost optimization
- Add A/B testing framework for prompt/reranker comparison
- Create custom metric definition interface
- Build automated improvement ticket generation (Jira/GitHub integration)
- Implement multi-armed bandit for automatic variant optimization

**Success Criteria**:
- Cache hit rate > 30% for repeated evaluations
- A/B testing supports 2+ variants with statistical significance testing
- Custom metrics definable through configuration files
- Improvement tickets auto-generated with actionable details

**Dependencies**:
- Redis for caching (if not already available)
- Issue tracking system API access

---

## Technical Approach

### Architecture Strategy

**Layered Design Following Clean Architecture**:

1. **Domain Layer** (Business Logic):
   - `RAGQualityEvaluator`: Core evaluation orchestration
   - `PersonaManager`: Persona simulation and query generation
   - `SyntheticDataGenerator`: Flip-the-RAG test data creation
   - `QualityAnalyzer`: Feedback loop analysis
   - Models: `EvaluationResult`, `PersonaProfile`, `TestCase`, `ImprovementSuggestion`

2. **Application Layer** (Use Cases):
   - `EvaluationService`: Coordinate evaluation workflow
   - `ImprovementService`: Track and apply improvements
   - `ReportService`: Generate evaluation reports

3. **Infrastructure Layer** (External Integrations):
   - `EvaluationStore`: Persist evaluation results (JSON file or SQLite)
   - `JudgeLLM`: Wrapper for RAGAS/DeepEval with judge model
   - `RegulationDatabase`: Access to regulation documents for synthetic data

4. **Interface Layer** (User Interaction):
   - `QualityDashboard`: Gradio web interface
   - `CLICommands`: Command-line evaluation tools
   - `FastAPIRoutes`: REST API for dashboard data

### Technology Choices

**RAGAS vs DeepEval Decision Matrix**:

| Factor | RAGAS | DeepEval | Decision |
|--------|-------|----------|----------|
| Installation ease | pip install ragas | pip install deepeval | RAGAS (simpler) |
| Metric variety | 4 core metrics | 10+ metrics | RAGAS (sufficient) |
| Documentation quality | Excellent | Good | RAGAS |
| Community support | Growing | Growing | Neutral |
| Judge LLM flexibility | GPT-4/Claude | GPT-4/Claude | Neutral |
| Cost effectiveness | Standard | Similar | Neutral |
| **Recommendation** | **Primary** | **Fallback** | **RAGAS** |

**Judge LLM Selection**:

| Model | Cost per 1K tokens | Speed | Quality | Recommendation |
|-------|-------------------|-------|---------|----------------|
| GPT-4o | $0.005 | Fast | Excellent | Primary (accuracy critical) |
| GPT-4o-mini | $0.00015 | Very Fast | Good | Fallback (cost optimization) |
| Gemini 1.5 Flash | $0.000075 | Very Fast | Good | Alternative (cost-sensitive) |

### Implementation Phases

#### Phase 1: Framework Setup (Week 1)

**Step 1: Environment Preparation**
```bash
# Install RAGAS and dependencies
uv add ragas
uv add deepeval  # Fallback framework
uv add plotly gradio  # Dashboard dependencies

# Configure judge LLM API key
export OPENAI_API_KEY=sk-...  # For GPT-4o judge
export JUDGE_LLM_MODEL=gpt-4o
```

**Step 2: Core Evaluator Implementation**
```python
# File: src/rag/domain/evaluation/quality_evaluator.py

from dataclasses import dataclass
from typing import List
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    contextual_precision,
    contextual_recall
)

class RAGQualityEvaluator:
    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_model = judge_model
        self.metrics = [
            faithfulness,
            answer_relevancy,
            contextual_precision,
            contextual_recall
        ]

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str]
    ) -> EvaluationResult:
        # RAGAS evaluation logic
        result = await evaluate(
            dataset={
                "question": [query],
                "answer": [answer],
                "contexts": [contexts]
            },
            metrics=self.metrics
        )
        # Process and return result
        return EvaluationResult(...)
```

**Step 3: Initial Testing**
```bash
# Run single query evaluation test
uv run python -m rag.interface.evaluation_test --query "휴학 절차가 어떻게 되나요?"

# Expected output: All four metric scores with pass/fail status
```

#### Phase 2: Persona System (Week 1-2)

**Step 1: Persona Configuration**
```python
# File: data/config/personas.yaml

personas:
  freshman:
    display_name: "신입생"
    expertise_level: "beginner"
    vocabulary_style: "simple"
    query_templates:
      - "{topic} 어떻게 해요?"
      - "{topic} 절차 알려주세요"
    common_topics:
      - "휴학"
      - "복학"
      - "성적"
      - "장학금"

  professor:
    display_name: "교수"
    expertise_level: "advanced"
    vocabulary_style: "academic"
    query_templates:
      - "{topic} 관련 조항 확인 필요"
      - "{topic} 적용 기준 상세히"
    common_topics:
      - "연구년"
      - "휴직"
      - "승진"
```

**Step 2: Query Generation**
```python
# File: src/rag/domain/evaluation/personas.py

class PersonaManager:
    async def generate_queries(
        self,
        persona_name: str,
        count: int = 20
    ) -> List[str]:
        persona = self.load_persona(persona_name)
        topics = self.get_regulation_topics(persona.common_topics)

        queries = []
        for _ in range(count):
            template = random.choice(persona.query_templates)
            topic = random.choice(topics)
            query = template.format(topic=topic)
            queries.append(query)

        return queries
```

**Step 3: Batch Evaluation**
```python
# Evaluate all personas
async def evaluate_all_personas():
    results = {}
    for persona in PERSONAS:
        queries = await generate_queries(persona, 20)
        persona_results = []
        for query in queries:
            answer, contexts = await rag_pipeline.search(query)
            result = await evaluator.evaluate(query, answer, contexts)
            persona_results.append(result)
        results[persona] = aggregate(persona_results)
    return results
```

#### Phase 3: Synthetic Data Generation (Week 2)

**Step 1: Document Analysis**
```python
# File: src/rag/domain/evaluation/synthetic_data.py

class SyntheticDataGenerator:
    async def generate_from_regulations(self):
        for regulation in self.regulation_db.get_all():
            for section in regulation.sections:
                # Skip if too short
                if len(section.content) < 100:
                    continue

                # Determine section type
                if section.has_numbered_list():
                    questions = self.generate_procedural(section)
                elif section.has_eligibility():
                    questions = self.generate_conditional(section)
                else:
                    questions = self.generate_factual(section)

                # Create test cases
                for question in questions:
                    ground_truth = self.extract_answer(section, question)
                    if self.validate(question, ground_truth):
                        test_cases.append(TestCase(
                            question=question,
                            ground_truth=ground_truth,
                            regulation_id=regulation.id
                        ))
```

**Step 2: Question Generation Patterns**
```python
def generate_procedural(self, section):
    """Generate questions for numbered list procedures"""
    steps = section.extract_numbered_steps()
    questions = [
        f"{section.title} 절차가 어떻게 되나요?",
        f"{section.title} 필요한 서류는 뭐예요?",
        f"{section.title} 신청 방법 알려주세요"
    ]
    # Add step-specific questions
    for i in range(min(3, len(steps))):
        questions.append(f"{section.title} {i+1}단계는 뭔가요?")
    return questions

def generate_conditional(self, section):
    """Generate questions for eligibility criteria"""
    return [
        f"{section.title} 자격 요건이 뭐예요?",
        f"{section.title} 누가 신청할 수 있나요?",
        f"{section.title} 제한 사항이 있나요?"
    ]
```

**Step 3: Validation**
```python
def validate_test_case(self, question, ground_truth):
    # Check length constraints
    if len(question) < 10 or len(question) > 200:
        return False
    if len(ground_truth) < 50:
        return False

    # Check semantic relevance
    question_emb = self.embed(question)
    answer_emb = self.embed(ground_truth)
    similarity = cosine_similarity(question_emb, answer_emb)
    return similarity >= 0.5
```

#### Phase 4: Feedback Loop (Week 3)

**Step 1: Issue Categorization**
```python
# File: src/rag/domain/evaluation/quality_analyzer.py

class QualityAnalyzer:
    def categorize_failures(self, results):
        categories = {
            "hallucination": [],  # Faithfulness < 0.9
            "irrelevant_retrieval": [],  # Contextual Precision < 0.8
            "incomplete_answer": [],  # Contextual Recall < 0.8
            "irrelevant_answer": []  # Answer Relevancy < 0.85
        }

        for result in results:
            if result.faithfulness < 0.9:
                categories["hallucination"].append(result)
            if result.contextual_precision < 0.8:
                categories["irrelevant_retrieval"].append(result)
            if result.contextual_recall < 0.8:
                categories["incomplete_answer"].append(result)
            if result.answer_relevancy < 0.85:
                categories["irrelevant_answer"].append(result)

        return categories
```

**Step 2: Suggestion Generation**
```python
def generate_suggestions(self, failures_by_type):
    suggestions = []

    for issue_type, failures in failures_by_type.items():
        if len(failures) > 5:  # Pattern threshold
            suggestion = self.create_suggestion(issue_type, failures)
            suggestions.append(suggestion)

    return suggestions

def create_suggestion(self, issue_type, failures):
    if issue_type == "hallucination":
        return ImprovementSuggestion(
            component="prompt_engineering",
            recommendation="Reduce temperature to 0.1 and add context adherence",
            expected_impact="+0.15 Faithfulness",
            affected_count=len(failures)
        )

    elif issue_type == "irrelevant_retrieval":
        # Analyze failure patterns to find optimal threshold
        current_threshold = self.get_current_reranker_threshold()
        optimal_threshold = self.calculate_optimal_threshold(failures)

        return ImprovementSuggestion(
            component="reranking",
            recommendation=f"Adjust threshold from {current_threshold} to {optimal_threshold}",
            expected_impact="+0.20 Contextual Precision",
            affected_count=len(failures)
        )

    # ... other issue types
```

**Step 3: Impact Tracking**
```python
class ImprovementTracker:
    async def track_improvement(self, suggestion):
        # Store baseline
        baseline = await self.evaluator.evaluate_batch(self.test_queries)

        # Apply improvement (manual or automated)
        await self.apply_improvement(suggestion)

        # Re-evaluate
        new_results = await self.evaluator.evaluate_batch(self.test_queries)

        # Calculate impact
        impact = {
            "faithfulness_delta": new_results.faithfulness - baseline.faithfulness,
            "precision_delta": new_results.precision - baseline.precision,
            "overall_delta": new_results.overall - baseline.overall
        }

        return ImpactReport(suggestion, baseline, new_results, impact)
```

#### Phase 5: Dashboard Development (Week 3-4)

**Step 1: Data Storage**
```python
# File: src/rag/infrastructure/storage/evaluation_store.py

class EvaluationStore:
    def __init__(self, storage_path: str = "data/evaluation_results"):
        self.storage_path = storage_path

    async def save_evaluation(self, result: EvaluationResult):
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.storage_path}/eval_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    async def get_latest_evaluation(self) -> EvaluationResult:
        files = glob.glob(f"{self.storage_path}/eval_*.json")
        latest = max(files, key=os.path.getctime)
        with open(latest) as f:
            data = json.load(f)
        return EvaluationResult.from_dict(data)

    async def get_history_since(self, start_date: datetime) -> List[EvaluationResult]:
        files = glob.glob(f"{self.storage_path}/eval_*.json")
        results = []
        for file in files:
            timestamp = self.extract_timestamp(file)
            if timestamp >= start_date:
                with open(file) as f:
                    results.append(EvaluationResult.from_dict(json.load(f)))
        return results
```

**Step 2: Gradio Dashboard**
```python
# File: src/rag/interface/web/quality_dashboard.py

import gradio as gr
import plotly.graph_objects as go

def create_quality_dashboard():
    with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
        gr.Markdown("# RAG Quality Dashboard")

        with gr.Row():
            with gr.Column():
                faithfulness = gr.Number(label="Faithfulness", precision=3)
                relevancy = gr.Number(label="Answer Relevancy", precision=3)
                precision = gr.Number(label="Contextual Precision", precision=3)
                recall = gr.Number(label="Contextual Recall", precision=3)

        with gr.Row():
            plot = gr.LinePlot(label="Quality Trend (30 Days)")

        with gr.Row():
            persona_table = gr.Dataframe(
                label="Persona Breakdown",
                headers=["Persona", "Faithfulness", "Relevancy", "Queries", "Pass Rate"]
            )

        with gr.Row():
            report_btn = gr.Button("Generate Report")
            report_download = gr.File(label="Download Report (PDF)")

        # Auto-refresh every 5 minutes
        dashboard.load(
            update_dashboard,
            inputs=[],
            outputs=[faithfulness, relevancy, precision, recall, plot, persona_table],
            every=300
        )

        # Report generation
        report_btn.click(
            generate_report,
            inputs=[],
            outputs=[report_download]
        )

    return dashboard

def update_dashboard():
    # Fetch latest metrics
    latest = eval_store.get_latest_evaluation()
    history = eval_store.get_history_since(datetime.now() - timedelta(days=30))
    persona_results = eval_store.get_latest_persona_results()

    # Create time-series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r.timestamp for r in history],
        y=[r.overall_score for r in history],
        mode='lines+markers',
        name='Overall Score'
    ))

    # Create persona table
    table_data = [
        [name, r.faithfulness, r.answer_relevancy, r.query_count, r.pass_rate]
        for name, r in persona_results.items()
    ]

    return (
        latest.faithfulness,
        latest.answer_relevancy,
        latest.contextual_precision,
        latest.contextual_recall,
        fig,
        table_data
    )
```

**Step 3: FastAPI Endpoints**
```python
# File: src/rag/interface/web/routes/quality.py

from fastapi import APIRouter

router = APIRouter(prefix="/quality", tags=["quality"])

@router.get("/metrics/latest")
async def get_latest_metrics() -> QualityOverview:
    latest = await eval_store.get_latest_evaluation()
    return QualityOverview(
        faithfulness=latest.faithfulness,
        answer_relevancy=latest.answer_relevancy,
        contextual_precision=latest.contextual_precision,
        contextual_recall=latest.contextual_recall,
        overall_score=latest.overall_score,
        timestamp=latest.timestamp
    )

@router.get("/metrics/history")
async def get_metrics_history(days: int = 30) -> List[QualitySnapshot]:
    start_date = datetime.now() - timedelta(days=days)
    history = await eval_store.get_history_since(start_date)
    return [
        QualitySnapshot(
            timestamp=r.timestamp,
            overall_score=r.overall_score,
            faithfulness=r.faithfulness
        )
        for r in history
    ]
```

#### Phase 6: CLI Integration (Week 4)

**Step 1: Evaluation Commands**
```bash
# File: src/rag/interface/cli/evaluation_commands.py

@click.command()
@click.argument("query")
@click.option("--persona", default="default", help="User persona for evaluation")
@click.option("--output", default="terminal", help="Output format: terminal, json, pdf")
def evaluate_query(query: str, persona: str, output: str):
    """Evaluate single query quality"""
    answer, contexts = rag_pipeline.search(query)
    result = evaluator.evaluate(query, answer, contexts)

    if output == "terminal":
        print(f"Faithfulness: {result.faithfulness:.3f}")
        print(f"Answer Relevancy: {result.answer_relevancy:.3f}")
        print(f"Contextual Precision: {result.contextual_precision:.3f}")
        print(f"Contextual Recall: {result.contextual_recall:.3f}")
        print(f"Overall: {result.overall_score:.3f}")
        print(f"Passed: {result.passed}")
    elif output == "json":
        print(json.dumps(result.to_dict(), indent=2))
    elif output == "pdf":
        generate_pdf_report([result])

@click.command()
@click.option("--personas", default="all", help="Comma-separated persona list or 'all'")
@click.option("--queries-per-persona", default=20, help="Number of queries per persona")
def evaluate_batch(personas: str, queries_per_persona: int):
    """Run batch evaluation across personas"""
    persona_list = personas.split(",") if personas != "all" else PERSONAS

    results = {}
    for persona in persona_list:
        queries = persona_manager.generate_queries(persona, queries_per_persona)
        persona_results = []

        for query in queries:
            answer, contexts = rag_pipeline.search(query)
            result = evaluator.evaluate(query, answer, contexts)
            persona_results.append(result)

        results[persona] = aggregate_results(persona_results)

    print(json.dumps(results, indent=2))
```

**Step 2: Register Commands**
```python
# File: src/rag/interface/unified_cli.py

from .evaluation_commands import evaluate_query, evaluate_batch

@click.group()
def cli():
    """Regulation Manager CLI"""
    pass

# Add evaluation commands
cli.add_command(evaluate_query, name="evaluate-query")
cli.add_command(evaluate_batch, name="evaluate-batch")
```

### Risk Assessment and Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| Judge LLM API rate limits | Medium | High | Implement rate limiting, use caching, batch requests |
| High API costs | Medium | Medium | Use GPT-4o-mini for initial evaluation, cache results |
| RAGAS framework changes | Low | Medium | Pin specific version, create abstraction layer |
| Insufficient test data | Low | High | Use synthetic data generation as primary source |
| Persona bias | Medium | Medium | Validate personas with real user queries |
| Evaluation timeout | Low | Medium | Implement async evaluation with timeout handling |
| Dashboard performance | Low | Low | Use data aggregation for large datasets, implement caching |

### Testing Strategy

**Unit Tests** (80% coverage target):
- `test_quality_evaluator.py`: Test metric calculation logic
- `test_persona_manager.py`: Test query generation and validation
- `test_synthetic_data.py`: Test question generation patterns
- `test_quality_analyzer.py`: Test issue categorization and suggestion generation

**Integration Tests** (70% coverage target):
- `test_evaluation_service.py`: Test full evaluation workflow
- `test_evaluation_store.py`: Test result persistence and retrieval
- `test_dashboard_api.py`: Test FastAPI endpoints

**End-to-End Tests** (key user journeys):
- Single query evaluation with metric output
- Batch evaluation across all personas
- Synthetic data generation from regulation documents
- Dashboard loading with real-time updates
- Report generation (PDF/JSON)

### Deployment Plan

**Staging Deployment** (Week 3):
- Deploy evaluation framework to staging environment
- Run initial evaluation batch with 100 queries
- Validate all metrics and thresholds
- Test dashboard with staging data
- Review API costs and optimize if needed

**Production Deployment** (Week 4):
- Deploy to production with feature flag
- Run initial batch evaluation during off-peak hours
- Monitor API costs and set budget alerts
- Enable dashboard access for administrators
- Schedule automated weekly evaluations

**Rollback Plan**:
- Feature flag to disable evaluation system
- Fallback to existing rule-based evaluation
- Database rollback for evaluation results
- API key rotation if judge LLM access needs revocation

---

## Summary

This implementation plan follows a systematic approach:

1. **Weeks 1-2**: Framework integration and persona system (Primary Goals)
2. **Weeks 2-3**: Synthetic data generation and feedback loop (Secondary Goals)
3. **Weeks 3-4**: Dashboard development and CLI integration (Final Goals)
4. **Week 4**: Advanced features and optimization (Optional Goals)

**Key Success Factors**:
- Judge LLM API access and cost management
- Synthetic data quality and diversity
- Persona validation with real user queries
- Dashboard usability and performance

**Next Steps**:
1. Obtain OpenAI/Gemini API key for judge LLM
2. Create `data/config/personas.yaml` with persona definitions
3. Begin Phase 1: Framework Setup

---

**Plan Status**: Ready for Implementation
**Next Phase**: /moai:2-run SPEC-RAG-QUALITY-001
**Target Completion**: 2025-02-25
