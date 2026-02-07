"""
Parallel Persona Evaluation System for RAG Quality Assessment.

Implements parallel execution of 6 persona sub-agents as defined in
rag-quality-local skill modules/personas.md.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.rag.domain.evaluation import EvaluationBatch, JudgeResult, LLMJudge
from src.rag.domain.evaluation.personas import PERSONAS, PersonaManager
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter
from src.rag.interface.query_handler import QueryHandler, QueryOptions


@dataclass
class PersonaQuery:
    """A test query from a specific persona."""

    query: str
    persona: str
    category: str
    difficulty: str
    expected_intent: str
    expected_info: List[str] = field(default_factory=list)


@dataclass
class PersonaEvaluationResult:
    """Result of evaluating queries for a specific persona."""

    persona: str
    queries_tested: int
    results: List[JudgeResult] = field(default_factory=list)
    avg_score: float = 0.0
    pass_rate: float = 0.0
    issues: Dict[str, int] = field(default_factory=dict)


class ParallelPersonaEvaluator:
    """Evaluates RAG system using parallel persona sub-agents."""

    # 6 personas defined in rag-quality-local skill
    PERSONA_AGENTS = [
        "student-undergraduate",
        "student-graduate",
        "professor",
        "staff-admin",
        "parent",
        "student-international",
    ]

    # 6 scenario categories
    CATEGORIES = [
        "simple",
        "complex",
        "multi-turn",
        "edge_cases",
        "domain_specific",
        "adversarial",
    ]

    def __init__(
        self,
        db_path: str = "data/chroma_db",
        llm_client: Optional[LLMClientAdapter] = None,
        judge: Optional[LLMJudge] = None,
    ):
        """Initialize parallel persona evaluator.

        Args:
            db_path: Path to ChromaDB
            llm_client: Optional LLM client for query execution
            judge: Optional LLM judge for evaluation
        """
        # Initialize RAG components
        self.store = ChromaVectorStore(persist_directory=db_path)
        self.llm_client = llm_client or LLMClientAdapter(
            provider="openai",
            model="gpt-4o",
        )
        self.query_handler = QueryHandler(
            store=self.store,
            llm_client=self.llm_client,
            use_reranker=True,
        )

        # Initialize judge
        self.judge = judge or LLMJudge(llm_client=self.llm_client)

        # Initialize persona manager
        self.persona_mgr = PersonaManager()

        # Storage for results
        self.batch = EvaluationBatch(judge=self.judge)

    def generate_persona_queries(
        self,
        persona: str,
        count_per_category: int = 3,
        topics: Optional[List[str]] = None,
    ) -> List[PersonaQuery]:
        """Generate test queries for a specific persona.

        Args:
            persona: Persona ID (e.g., "student-undergraduate")
            count_per_category: Number of queries per category
            topics: Optional list of topics to focus on

        Returns:
            List of PersonaQuery objects
        """
        queries = []

        # Get persona profile - convert persona ID to persona name
        persona_map = {
            "student-undergraduate": "freshman",
            "student-graduate": "graduate",
            "professor": "professor",
            "staff-admin": "staff",
            "parent": "parent",
            "student-international": "international",
        }

        persona_name = persona_map.get(persona, persona)
        persona_profile = PERSONAS.get(persona_name)
        if not persona_profile:
            return queries

        # Default topics if not specified
        if topics is None:
            topics = persona_profile.common_topics

        # Generate queries for each category
        for category in self.CATEGORIES:
            for topic in topics[:count_per_category]:
                query_text = self._generate_query_for_category(persona, category, topic)

                queries.append(
                    PersonaQuery(
                        query=query_text,
                        persona=persona,
                        category=category,
                        difficulty=self._get_difficulty_for_category(category),
                        expected_intent=self._infer_intent(topic),
                        expected_info=self._get_expected_info(topic),
                    )
                )

        return queries

    def _generate_query_for_category(
        self, persona: str, category: str, topic: str
    ) -> str:
        """Generate a query for a specific persona, category, and topic."""
        # This would normally use the persona sub-agent
        # For now, return predefined patterns based on category

        queries_by_category = {
            "simple": {
                "student-undergraduate": f"{topic} 방법 알려줘",
                "student-graduate": f"{topic} 절차가 궁금합니다",
                "professor": f"{topic} 관련 규정 안내",
                "staff-admin": f"{topic} 처리 절차",
                "parent": f"{topic} 어떻게 하나요?",
                "student-international": f"How do I {topic}?",
            },
            "complex": {
                "student-undergraduate": f"{topic} 기간과 방법, 그리고 필요한 서류가 뭐야?",
                "student-graduate": f"{topic} 자격과 절차, 그리고 제출 서류가 무엇인가요?",
                "professor": f"{topic} 관련 규정과 예외 사항을 포괄적으로 설명하시오",
                "staff-admin": f"{topic} 진행 절차, 필요 서류, 승인 기준을 안내",
                "parent": f"{topic} 신청하는 법과 마감일, 그리고 필요한 것들 알려주세요",
                "student-international": f"Tell me about {topic} process and requirements",
            },
            "edge_cases": {
                "student-undergraduate": f"{topic}...",  # Very vague
                "student-graduate": f"그게 뭐냐면 {topic}",
                "professor": f"{topic} 관련",
                "staff-admin": f"{topic} 어디서 해요?",
                "parent": f"{topic}...",
                "student-international": f"{topic}?",
            },
        }

        category_queries = queries_by_category.get(category, {})
        return category_queries.get(persona, f"{topic} 어떻게 해요?")

    def _get_difficulty_for_category(self, category: str) -> str:
        """Get difficulty level for a category."""
        difficulty_map = {
            "simple": "easy",
            "complex": "hard",
            "multi-turn": "medium",
            "edge_cases": "hard",
            "domain_specific": "medium",
            "adversarial": "hard",
        }
        return difficulty_map.get(category, "medium")

    def _infer_intent(self, topic: str) -> str:
        """Infer expected intent from topic."""
        intent_map = {
            "휴학": "leave_of_absence",
            "등록금": "tuition",
            "성적": "grades",
            "장학금": "scholarship",
            "수강신청": "course_registration",
            "졸업": "graduation",
        }
        return intent_map.get(topic, "general_inquiry")

    def _get_expected_info(self, topic: str) -> List[str]:
        """Get expected information points for a topic."""
        info_map = {
            "휴학": ["기간", "절차", "서류"],
            "등록금": ["금액", "납부 방법", "마감일"],
            "성적": ["조회 방법", "성적 처리"],
            "장학금": ["신청 방법", "자격", "마감일"],
            "수강신청": ["기간", "방법", "정정"],
            "졸업": ["요건", "신청", "절차"],
        }
        return info_map.get(topic, ["관련 정보"])

    def evaluate_parallel(
        self,
        queries_per_persona: int = 5,
        personas: Optional[List[str]] = None,
        max_workers: int = 6,
    ) -> Dict[str, PersonaEvaluationResult]:
        """Evaluate queries from all personas in parallel.

        Args:
            queries_per_persona: Number of queries to test per persona
            personas: Optional list of personas to test (default: all)
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary mapping persona IDs to their evaluation results
        """
        personas_to_test = personas or self.PERSONA_AGENTS

        # Generate queries for each persona
        all_queries = []
        for persona in personas_to_test:
            queries = self.generate_persona_queries(
                persona,
                count_per_category=queries_per_persona // len(self.CATEGORIES) + 1,
            )
            all_queries.extend(queries)

        # Evaluate queries in parallel
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_query = {
                executor.submit(self._evaluate_single_query, query): query
                for query in all_queries
            }

            # Collect results as they complete
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    judge_result = future.result()
                    self.batch.add_result(judge_result)

                    # Add to persona results
                    if query.persona not in results:
                        results[query.persona] = []
                    results[query.persona].append(judge_result)

                except Exception as e:
                    print(f"Error evaluating query '{query.query}': {e}")

        # Generate persona summaries
        persona_summaries = {}
        for persona, persona_results in results.items():
            avg_score = sum(r.overall_score for r in persona_results) / len(
                persona_results
            )
            pass_count = sum(1 for r in persona_results if r.passed)

            # Count issues
            issues = {}
            for r in persona_results:
                for issue in r.issues:
                    issues[issue] = issues.get(issue, 0) + 1

            persona_summaries[persona] = PersonaEvaluationResult(
                persona=persona,
                queries_tested=len(persona_results),
                results=persona_results,
                avg_score=avg_score,
                pass_rate=pass_count / len(persona_results),
                issues=issues,
            )

        return persona_summaries

    def _evaluate_single_query(self, query: PersonaQuery) -> JudgeResult:
        """Evaluate a single query through the RAG system.

        Args:
            query: PersonaQuery to evaluate

        Returns:
            JudgeResult with 4-metric evaluation
        """
        # Execute query through RAG system
        options = QueryOptions(
            top_k=5,
            use_rerank=True,
            force_mode="ask",
        )

        result = self.query_handler.process_query(
            query=query.query,
            options=options,
        )

        # Extract answer and sources
        answer_text = result.content if result.success else ""
        sources = []

        if result.data:
            # Try to extract sources from result
            if "tool_results" in result.data:
                for tool_result in result.data.get("tool_results", []):
                    if tool_result.get("tool_name") == "search_regulations":
                        result_data = tool_result.get("result")
                        if result_data and isinstance(result_data, dict):
                            search_results = result_data.get("results", [])
                            for r in search_results[:5]:
                                if isinstance(r, dict):
                                    sources.append(
                                        {
                                            "title": r.get("title", "")
                                            or r.get("regulation_title", ""),
                                            "text": (
                                                r.get("text", "")
                                                or r.get("content", "")
                                            )[:200],
                                            "rule_code": r.get("rule_code", ""),
                                            "score": r.get("score", 0.0)
                                            or r.get("similarity", 0.0),
                                        }
                                    )
                            break

        # Use LLM judge to evaluate
        judge_result = self.judge.evaluate(
            query=query.query,
            answer=answer_text,
            sources=sources,
            expected_info=query.expected_info,
        )

        return judge_result

    def save_results(self, filepath: Optional[str] = None) -> str:
        """Save evaluation results to JSON file.

        Args:
            filepath: Optional file path (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"data/evaluations/parallel_eval_{timestamp}.json"

        self.batch.save_to_file(filepath)
        return filepath

    def generate_report(self) -> str:
        """Generate markdown evaluation report.

        Returns:
            Markdown report string
        """
        summary = self.batch.get_summary()

        lines = [
            "# RAG Quality Evaluation Report",
            f"Generated: {summary.timestamp}",
            "",
            "## Summary",
            f"- Total Queries: {summary.total_queries}",
            f"- Passed: {summary.passed}",
            f"- Failed: {summary.failed}",
            f"- Pass Rate: {summary.pass_rate:.1%}",
            "",
            "## Average Scores",
            f"- Overall: {summary.avg_overall_score:.3f}",
            f"- Accuracy: {summary.avg_accuracy:.3f}",
            f"- Completeness: {summary.avg_completeness:.3f}",
            f"- Citations: {summary.avg_citations:.3f}",
            f"- Context Relevance: {summary.avg_context_relevance:.3f}",
            "",
            "## Failure Patterns",
        ]

        for issue, count in sorted(
            summary.failure_patterns.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"- {issue}: {count}x")

        lines.extend(
            [
                "",
                "## Detailed Results",
                "",
            ]
        )

        for i, result in enumerate(self.batch.results, 1):
            lines.extend(
                [
                    f"### {i}. {result.query}",
                    f"**Score:** {result.overall_score:.3f} ({'PASS' if result.passed else 'FAIL'})",
                    f"**Metrics:** Acc={result.accuracy:.3f}, Comp={result.completeness:.3f}, "
                    f"Cit={result.citations:.3f}, Ctx={result.context_relevance:.3f}",
                ]
            )

            if result.issues:
                lines.append(f"**Issues:** {', '.join(result.issues)}")
            if result.strengths:
                lines.append(f"**Strengths:** {', '.join(result.strengths)}")

            lines.extend(
                [
                    f"**Answer:** {result.answer[:200]}...",
                    "",
                ]
            )

        return "\n".join(lines)
