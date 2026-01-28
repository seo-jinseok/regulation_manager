"""
Comprehensive RAG Quality Evaluation Runner

Executes automated quality evaluation with diverse user personas,
query styles, and multi-turn conversations.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.application.search_usecase import SearchUseCase
from src.rag.automation.domain.entities import (
    FactCheck,
    FactCheckStatus,
    MultiTurnScenarioResult,
    QualityTestResult,
)
from src.rag.automation.infrastructure.quality_evaluator import QualityEvaluator
from src.rag.automation.infrastructure.test_report_generator import ReportGenerator
from src.rag.config import get_config
from src.rag.infrastructure.bge_reranker import BGEReranker
from src.rag.infrastructure.cache import RAGQueryCache
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.hybrid_search import HybridSearcher
from src.rag.infrastructure.llm_client import LLMClient
from src.rag.infrastructure.query_analyzer import QueryAnalyzer
from src.rag.infrastructure.synonym_expander import SynonymExpander
from test_scenarios.comprehensive_evaluation import (
    ALL_SINGLE_TURN_QUERIES,
    MULTI_TURN_SCENARIOS,
    TestQuery,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class RAGEvaluationRunner:
    """
    Executes comprehensive RAG quality evaluation.

    Tests across multiple dimensions:
    - Diverse user personas (freshman, graduate, professor, staff, parent)
    - Query style variations (precise, ambiguous, colloquial, etc.)
    - Multi-turn conversation quality
    """

    def __init__(self):
        """Initialize the evaluation runner."""
        self.config = get_config()
        self._setup_components()

    def _setup_components(self):
        """Setup RAG components for evaluation."""
        logger.info("Setting up RAG components...")

        # Initialize core components
        self.vector_store = ChromaVectorStore(
            persist_directory=str(self.config.db_path_resolved)
        )

        # Initialize synonym expander
        self.synonym_expander = None
        if self.config.synonyms_path_resolved:
            self.synonym_expander = SynonymExpander(
                synonyms_path=str(self.config.synonyms_path_resolved)
            )

        # Initialize query analyzer
        self.query_analyzer = None
        if self.config.intents_path_resolved:
            self.query_analyzer = QueryAnalyzer(
                intents_path=str(self.config.intents_path_resolved),
                synonym_expander=self.synonym_expander,
            )

        # Initialize reranker
        self.reranker = None
        if self.config.use_reranker:
            self.reranker = BGEReranker(config=self.config.reranker)

        # Initialize hybrid searcher
        self.hybrid_searcher = HybridSearcher(
            vector_store=self.vector_store,
            query_analyzer=self.query_analyzer,
            reranker=self.reranker,
            config=self.config,
        )

        # Initialize LLM client
        self.llm_client = LLMClient(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            fallback_config=self.config.llm_fallback,
        )

        # Initialize cache
        self.cache = (
            RAGQueryCache(
                cache_dir=str(self.config.cache_dir_resolved),
                ttl_hours=self.config.cache_ttl_hours,
            )
            if self.config.enable_cache
            else None
        )

        # Initialize search use case
        self.search_usecase = SearchUseCase(
            hybrid_searcher=self.hybrid_searcher,
            llm_client=self.llm_client,
            query_analyzer=self.query_analyzer,
            cache=self.cache,
            config=self.config,
        )

        # Initialize quality evaluator
        self.quality_evaluator = QualityEvaluator(llm_client=self.llm_client)

        # Initialize report generator
        self.report_generator = ReportGenerator(output_dir=Path("test_reports"))

        logger.info("RAG components setup complete")

    def run_single_turn_tests(self) -> List[QualityTestResult]:
        """
        Execute all single-turn test queries.

        Returns:
            List of quality test results
        """
        logger.info(f"Running {len(ALL_SINGLE_TURN_QUERIES)} single-turn tests...")

        results = []
        for idx, test_query in enumerate(ALL_SINGLE_TURN_QUERIES, 1):
            logger.info(
                f"[{idx}/{len(ALL_SINGLE_TURN_QUERIES)}] Testing: "
                f"{test_query.persona.value} - {test_query.query[:50]}..."
            )

            try:
                # Execute search
                start_time = datetime.now()
                search_result = self.search_usecase.search(
                    query_text=test_query.query,
                    top_k=5,
                )
                execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Generate answer
                if hasattr(search_result, "answer") and search_result.answer:
                    answer = search_result.answer
                else:
                    answer = self._generate_answer_from_context(
                        test_query.query, search_result
                    )

                # Collect sources
                sources = [r.chunk.rule_code or r.chunk.title for r in search_result]

                # Create test result
                test_result = QualityTestResult(
                    test_case_id=f"{test_query.persona.value}_{idx:03d}",
                    query=test_query.query,
                    answer=answer,
                    sources=sources,
                    confidence=0.8,  # Default confidence
                    execution_time_ms=execution_time_ms,
                    passed=False,  # Will be updated after quality evaluation
                )

                # Perform fact checks
                fact_checks = self._perform_fact_checks(test_query, answer, sources)

                # Evaluate quality
                quality_score = self.quality_evaluator.evaluate(
                    test_result, fact_checks
                )
                test_result.quality_score = quality_score
                test_result.passed = quality_score.is_pass
                test_result.fact_checks = fact_checks

                results.append(test_result)

                # Log result
                status = "✅ PASS" if test_result.passed else "❌ FAIL"
                logger.info(f"  {status} - Score: {quality_score.total_score:.2f}/5.0")

            except Exception as e:
                logger.error(f"  Error: {e}")
                results.append(
                    QualityTestResult(
                        test_case_id=f"{test_query.persona.value}_{idx:03d}",
                        query=test_query.query,
                        answer="",
                        sources=[],
                        confidence=0.0,
                        execution_time_ms=0,
                        passed=False,
                        error_message=str(e),
                    )
                )

        return results

    def run_multi_turn_tests(self) -> List[MultiTurnScenarioResult]:
        """
        Execute all multi-turn conversation scenarios.

        Returns:
            List of multi-turn scenario results
        """
        logger.info(f"Running {len(MULTI_TURN_SCENARIOS)} multi-turn scenarios...")

        scenario_results = []

        for scenario in MULTI_TURN_SCENARIOS:
            logger.info(f"Testing scenario: {scenario.scenario_id}")

            scenario_turns = []
            context_preserved_count = 0

            conversation_history = []

            for turn_idx, turn in enumerate(scenario.turns, 1):
                logger.info(
                    f"  Turn {turn_idx}/{len(scenario.turns)}: {turn.query[:50]}..."
                )

                try:
                    # Execute search with conversation context
                    start_time = datetime.now()
                    search_result = self.search_usecase.search(
                        query_text=turn.query,
                        top_k=5,
                        conversation_history=conversation_history
                        if turn.should_preserve_context
                        else None,
                    )
                    execution_time_ms = (
                        datetime.now() - start_time
                    ).total_seconds() * 1000

                    # Generate answer
                    if hasattr(search_result, "answer") and search_result.answer:
                        answer = search_result.answer
                    else:
                        answer = self._generate_answer_from_context(
                            turn.query, search_result
                        )

                    # Check if context was preserved
                    context_preserved = self._check_context_preservation(
                        turn.query, answer, conversation_history
                    )
                    if context_preserved:
                        context_preserved_count += 1

                    # Create turn result
                    from src.rag.automation.domain.entities import TurnResult

                    turn_result = TurnResult(
                        turn_number=turn_idx,
                        query=turn.query,
                        answer=answer,
                        follow_up_type=turn.expected_follow_up_type,
                        context_preserved=context_preserved,
                        confidence=0.8,
                        execution_time_ms=execution_time_ms,
                    )
                    scenario_turns.append(turn_result)

                    # Update conversation history
                    conversation_history.append({"query": turn.query, "answer": answer})

                except Exception as e:
                    logger.error(f"  Error in turn {turn_idx}: {e}")
                    from src.rag.automation.domain.entities import TurnResult

                    scenario_turns.append(
                        TurnResult(
                            turn_number=turn_idx,
                            query=turn.query,
                            answer="",
                            follow_up_type=turn.expected_follow_up_type,
                            context_preserved=False,
                            confidence=0.0,
                            execution_time_ms=0,
                        )
                    )

            # Calculate context preservation rate
            context_preservation_rate = (
                context_preserved_count / len(scenario.turns) if scenario.turns else 0.0
            )

            # Create scenario result
            from src.rag.automation.domain.entities import MultiTurnScenarioResult

            scenario_result = MultiTurnScenarioResult(
                scenario_id=scenario.scenario_id,
                persona_type=scenario.persona,
                description=scenario.description,
                turns=scenario_turns,
                total_turns=len(scenario.turns),
                context_preservation_rate=context_preservation_rate,
                resolved=context_preservation_rate >= 0.7,
            )

            scenario_results.append(scenario_result)

            status = "✅ RESOLVED" if scenario_result.resolved else "❌ UNRESOLVED"
            logger.info(
                f"  {status} - Context Preservation: {context_preservation_rate:.1%}"
            )

        return scenario_results

    def _generate_answer_from_context(self, query: str, search_result) -> str:
        """Generate answer from search context."""
        try:
            # Collect context from search results
            context_parts = []
            for result in search_result:
                chunk = result.chunk
                source = chunk.rule_code or chunk.title
                context_parts.append(f"[{source}] {chunk.text[:500]}")

            context = "\n\n".join(context_parts)

            # Generate answer using LLM
            from src.rag.infrastructure.tool_executor import ToolExecutor

            tool_executor = ToolExecutor(llm_client=self.llm_client)

            response = tool_executor.execute(
                "generate_answer", {"question": query, "context": context}
            )

            return response.result if response.success else "답변 생성에 실패했습니다."

        except Exception as e:
            logger.warning(f"Failed to generate answer: {e}")
            return "답변 생성에 실패했습니다."

    def _perform_fact_checks(
        self, test_query: TestQuery, answer: str, sources: List[str]
    ) -> List[FactCheck]:
        """Perform fact checks on the answer."""
        # Basic fact checking rules
        fact_checks = []

        # Check for hallucinated phone numbers
        import re

        phone_pattern = r"0\d{1,2}-\d{3,4}-\d{4}"
        if re.search(phone_pattern, answer):
            fact_checks.append(
                FactCheck(
                    claim="전화번호 포함",
                    status=FactCheckStatus.FAIL,
                    source="answer",
                    confidence=1.0,
                    correction="전화번호를 만들어내지 않아야 함",
                    explanation="규정에 없는 전화번호를 생성함",
                )
            )

        # Check for other school references
        other_schools = ["서울대", "연세대", "고려대", "한국외대", "이화여대"]
        for school in other_schools:
            if school in answer:
                fact_checks.append(
                    FactCheck(
                        claim=f"{school} 언급",
                        status=FactCheckStatus.FAIL,
                        source="answer",
                        confidence=1.0,
                        correction=f"{school} 규정 언급 금지",
                        explanation="다른 학교 규정을 언급함",
                    )
                )
                break

        # Check for generic disclaimers
        generic_phrases = [
            "대학마다 다릅니다",
            "일반적으로",
            "학교에 따라",
            "확인이 필요합니다",
        ]
        for phrase in generic_phrases:
            if phrase in answer:
                fact_checks.append(
                    FactCheck(
                        claim="일반론 답변",
                        status=FactCheckStatus.FAIL,
                        source="answer",
                        confidence=1.0,
                        correction="구체적인 규정 조항 인용 필요",
                        explanation="회피성 답변 사용",
                    )
                )
                break

        # Check for source citations
        has_citation = any("제" in s and "조" in s for s in sources)
        if not has_citation and "제" not in answer and "조" not in answer:
            fact_checks.append(
                FactCheck(
                    claim="출처 인용",
                    status=FactCheckStatus.FAIL,
                    source="answer",
                    confidence=0.9,
                    correction="규정명과 조항 번호를 명시해야 함",
                    explanation="구체적인 출처 인용 부족",
                )
            )

        return fact_checks

    def _check_context_preservation(
        self, query: str, answer: str, conversation_history: List[Dict]
    ) -> bool:
        """Check if conversation context was preserved."""
        if not conversation_history:
            return True

        # Simple check: does answer reference previous context?
        context_keywords = []
        for turn in conversation_history:
            for word in turn["query"].split():
                if len(word) > 2:
                    context_keywords.append(word)

        # Check if answer contains relevant context keywords
        matching_keywords = sum(1 for kw in context_keywords if kw in answer)
        return matching_keywords > 0

    def run_full_evaluation(self) -> Dict:
        """
        Run the complete evaluation suite.

        Returns:
            Evaluation results dictionary
        """
        logger.info("=" * 70)
        logger.info("STARTING COMPREHENSIVE RAG QUALITY EVALUATION")
        logger.info("=" * 70)

        session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Phase 1: Single-turn tests
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: SINGLE-TURN QUERY TESTS")
        logger.info("=" * 70 + "\n")

        single_turn_results = self.run_single_turn_tests()

        # Phase 2: Multi-turn scenarios
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: MULTI-TURN CONVERSATION TESTS")
        logger.info("=" * 70 + "\n")

        multi_turn_results = self.run_multi_turn_tests()

        # Compile results
        evaluation_results = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "single_turn_results": single_turn_results,
            "multi_turn_results": multi_turn_results,
        }

        # Generate reports
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING REPORTS")
        logger.info("=" * 70 + "\n")

        # Markdown report
        md_report_path = self.report_generator.generate_report(
            session_id=session_id,
            test_results=single_turn_results,
            multi_turn_scenarios=multi_turn_results,
            metadata={
                "total_queries": len(ALL_SINGLE_TURN_QUERIES),
                "total_scenarios": len(MULTI_TURN_SCENARIOS),
                "total_turns": sum(len(s.turns) for s in MULTI_TURN_SCENARIOS),
            },
        )
        logger.info(f"Markdown report: {md_report_path}")

        # HTML report
        html_report_path = self.report_generator.generate_html_report(
            session_id=session_id,
            test_results=single_turn_results,
            multi_turn_scenarios=multi_turn_results,
            metadata={
                "total_queries": len(ALL_SINGLE_TURN_QUERIES),
                "total_scenarios": len(MULTI_TURN_SCENARIOS),
                "total_turns": sum(len(s.turns) for s in MULTI_TURN_SCENARIOS),
            },
        )
        logger.info(f"HTML report: {html_report_path}")

        # Print summary
        self._print_evaluation_summary(single_turn_results, multi_turn_results)

        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 70)

        return evaluation_results

    def _print_evaluation_summary(
        self, single_turn_results: List[QualityTestResult], multi_turn_results: List
    ):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("RAG QUALITY EVALUATION SUMMARY")
        print("=" * 70)

        # Single-turn stats
        total_single = len(single_turn_results)
        passed_single = sum(1 for r in single_turn_results if r.passed)
        pass_rate_single = (
            (passed_single / total_single * 100) if total_single > 0 else 0
        )

        avg_quality = 0
        if single_turn_results:
            quality_scores = [
                r.quality_score.total_score
                for r in single_turn_results
                if r.quality_score
            ]
            avg_quality = (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0
            )

        print("\n📊 SINGLE-TURN TESTS")
        print("-" * 70)
        print(f"  Total Tests:   {total_single}")
        print(f"  Passed:        {passed_single} ({pass_rate_single:.1f}%)")
        print(
            f"  Failed:        {total_single - passed_single} ({100 - pass_rate_single:.1f}%)"
        )
        print(f"  Avg Quality:   {avg_quality:.2f}/5.0")

        # Multi-turn stats
        total_multi = len(multi_turn_results)
        resolved_multi = sum(1 for r in multi_turn_results if r.resolved)
        resolution_rate = (resolved_multi / total_multi * 100) if total_multi > 0 else 0

        avg_context_preservation = 0
        if multi_turn_results:
            context_rates = [r.context_preservation_rate for r in multi_turn_results]
            avg_context_preservation = sum(context_rates) / len(context_rates)

        print("\n💬 MULTI-TURN CONVERSATIONS")
        print("-" * 70)
        print(f"  Total Scenarios:     {total_multi}")
        print(f"  Resolved:            {resolved_multi} ({resolution_rate:.1f}%)")
        print(f"  Avg Context Keep:    {avg_context_preservation:.1%}")

        # Overall grade
        overall_score = (pass_rate_single + resolution_rate) / 2
        if overall_score >= 90:
            grade = "A (Excellent)"
        elif overall_score >= 80:
            grade = "B (Good)"
        elif overall_score >= 70:
            grade = "C (Fair)"
        elif overall_score >= 60:
            grade = "D (Poor)"
        else:
            grade = "F (Fail)"

        print("\n🎯 OVERALL GRADE: {}".format(grade))
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    try:
        runner = RAGEvaluationRunner()
        results = runner.run_full_evaluation()
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
