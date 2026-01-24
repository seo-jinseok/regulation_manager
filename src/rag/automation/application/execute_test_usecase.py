"""
Execute Test Use Case.

Application layer for executing single query tests.
Integrates with SearchUseCase to run RAG pipeline and capture results.

Clean Architecture: Application layer orchestrates domain and infrastructure.
"""

import logging
import time
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from ...application.search_usecase import SearchUseCase
    from ..domain.entities import TestCase, TestResult

logger = logging.getLogger(__name__)


class ExecuteTestUseCase:
    """
    Use Case for executing a single test query.

    Runs the RAG pipeline for a test query and captures:
    - Answer text
    - Source references
    - Confidence score
    - Execution time
    - RAG pipeline logs
    """

    def __init__(self, search_usecase: "SearchUseCase"):
        """
        Initialize the use case.

        Args:
            search_usecase: SearchUseCase for executing RAG queries.
        """
        self.search = search_usecase

    def execute_query(
        self,
        query: str,
        test_case_id: str,
        enable_answer: bool = True,
        top_k: int = 5,
    ) -> "TestResult":
        """
        Execute a single query test.

        Args:
            query: The test query.
            test_case_id: Unique identifier for this test case.
            enable_answer: Whether to generate LLM answer.
            top_k: Number of chunks to retrieve.

        Returns:
            TestResult with answer, sources, and pipeline logs.
        """
        from ..domain.entities import TestResult

        start_time = time.time()
        rag_pipeline_log: Dict = {}

        try:
            # Step 1: Execute search to get chunks
            logger.info(f"Executing test query: {query[:50]}...")

            search_results = self.search.search(
                query_text=query,
                filter=None,
                top_k=top_k,
                include_abolished=False,
            )

            # Log search results
            rag_pipeline_log["search_results_count"] = len(search_results)
            rag_pipeline_log["top_chunks"] = [
                {
                    "rule_code": r.chunk.rule_code,
                    "title": r.chunk.title,
                    "score": r.score,
                    "rank": r.rank,
                }
                for r in search_results[:5]
            ]

            # Extract sources
            sources = [
                f"{r.chunk.rule_code} - {r.chunk.title or r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.rule_code}"
                for r in search_results
            ]

            # Step 2: Generate answer if enabled
            answer = ""
            confidence = 0.0

            if enable_answer and self.search.llm:
                try:
                    answer_obj = self.search.ask(
                        question=query,
                        filter=None,
                        top_k=top_k,
                        include_abolished=False,
                    )

                    answer = answer_obj.text
                    confidence = answer_obj.confidence

                    rag_pipeline_log["llm_generated"] = True
                    rag_pipeline_log["llm_confidence"] = confidence
                    rag_pipeline_log["answer_length"] = len(answer)

                    logger.info(f"Generated answer (confidence={confidence:.2f})")

                except Exception as e:
                    logger.warning(f"LLM answer generation failed: {e}")
                    rag_pipeline_log["llm_generated"] = False
                    rag_pipeline_log["llm_error"] = str(e)
                    answer = ""
                    confidence = 0.0
            else:
                rag_pipeline_log["llm_generated"] = False

            # Calculate execution time
            execution_time_ms = max(1, int((time.time() - start_time) * 1000))

            rag_pipeline_log["execution_time_ms"] = execution_time_ms
            rag_pipeline_log["query_analyzed"] = True

            # Create TestResult
            result = TestResult(
                test_case_id=test_case_id,
                query=query,
                answer=answer,
                sources=sources,
                confidence=confidence,
                execution_time_ms=execution_time_ms,
                rag_pipeline_log=rag_pipeline_log,
                passed=False,  # Will be updated after fact check and quality evaluation
            )

            logger.info(f"Test execution completed in {execution_time_ms}ms")

            return result

        except Exception as e:
            execution_time_ms = max(1, int((time.time() - start_time) * 1000))

            logger.error(f"Test execution failed: {e}")

            # Return result with error
            return TestResult(
                test_case_id=test_case_id,
                query=query,
                answer="",
                sources=[],
                confidence=0.0,
                execution_time_ms=execution_time_ms,
                rag_pipeline_log={"error": str(e)},
                error_message=str(e),
                passed=False,
            )

    def execute_test_case(self, test_case: "TestCase") -> "TestResult":
        """
        Execute a TestCase entity.

        Args:
            test_case: TestCase entity with query and metadata.

        Returns:
            TestResult with execution results.
        """
        return self.execute_query(
            query=test_case.query,
            test_case_id=f"{test_case.persona_type.value}_{test_case.difficulty.value}",
            enable_answer=True,
            top_k=5,
        )

    def batch_execute(
        self,
        queries: List[str],
        test_case_prefix: str = "batch",
    ) -> List["TestResult"]:
        """
        Execute multiple queries in batch.

        Args:
            queries: List of queries to execute.
            test_case_prefix: Prefix for test case IDs.

        Returns:
            List of TestResult objects.
        """
        results = []

        for idx, query in enumerate(queries):
            test_case_id = f"{test_case_prefix}_{idx:03d}"
            result = self.execute_query(
                query=query,
                test_case_id=test_case_id,
                enable_answer=True,
            )
            results.append(result)

        return results
