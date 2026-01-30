"""
Multi-hop Question Handler for Regulation RAG System.

Enables complex multi-hop question answering by decomposing queries
into sequential sub-queries and synthesizing comprehensive answers.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..domain.entities import SearchResult
from ..domain.repositories import ILLMClient, IVectorStore
from ..domain.value_objects import Query

logger = logging.getLogger(__name__)


class HopStatus(Enum):
    """Status of a hop execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class SubQuery:
    """
    A sub-query extracted from a complex multi-hop question.

    Attributes:
        query_id: Unique identifier for this sub-query
        query_text: The actual query text to execute
        hop_order: Execution order (1, 2, 3, ...)
        depends_on: List of query_ids this sub-query depends on
        context_from: Specific query_id to use as context
        reasoning: Why this sub-query is needed
    """

    query_id: str
    query_text: str
    hop_order: int
    depends_on: List[str] = field(default_factory=list)
    context_from: Optional[str] = None
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "hop_order": self.hop_order,
            "depends_on": self.depends_on,
            "context_from": self.context_from,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubQuery":
        """Create SubQuery from dictionary."""
        return cls(
            query_id=data["query_id"],
            query_text=data["query_text"],
            hop_order=data["hop_order"],
            depends_on=data.get("depends_on", []),
            context_from=data.get("context_from"),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class HopResult:
    """
    Result of executing a single hop.

    Attributes:
        hop_id: Identifier matching SubQuery.query_id
        query: The original SubQuery
        answer: Generated answer for this hop
        sources: Source chunks used for the answer
        execution_time_ms: Time taken to execute this hop
        is_relevant: Whether the result was relevant (Self-RAG)
        status: Execution status
        error_message: Error message if failed
    """

    hop_id: str
    query: SubQuery
    answer: str
    sources: List[SearchResult]
    execution_time_ms: float
    is_relevant: bool = True
    status: HopStatus = HopStatus.COMPLETED
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hop_id": self.hop_id,
            "query": self.query.to_dict(),
            "answer": self.answer,
            "sources": [
                {
                    "chunk_id": s.chunk.id,
                    "score": s.score,
                    "title": s.chunk.title,
                }
                for s in self.sources
            ],
            "execution_time_ms": self.execution_time_ms,
            "is_relevant": self.is_relevant,
            "status": self.status.value,
            "error_message": self.error_message,
        }


@dataclass
class MultiHopResult:
    """
    Final result of multi-hop query processing.

    Attributes:
        original_query: The original complex query
        sub_queries: List of decomposed sub-queries
        hop_results: List of individual hop results
        final_answer: Synthesized comprehensive answer
        total_execution_time_ms: Total time for all hops
        hop_count: Number of hops executed
        success: Whether multi-hop processing succeeded
    """

    original_query: str
    sub_queries: List[SubQuery]
    hop_results: List[HopResult]
    final_answer: str
    total_execution_time_ms: float
    hop_count: int
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_query": self.original_query,
            "sub_queries": [sq.to_dict() for sq in self.sub_queries],
            "hop_results": [hr.to_dict() for hr in self.hop_results],
            "final_answer": self.final_answer,
            "total_execution_time_ms": self.total_execution_time_ms,
            "hop_count": self.hop_count,
            "success": self.success,
        }


class DependencyCycleDetector:
    """
    Detects cyclic dependencies in multi-hop queries.

    Uses DFS to detect cycles and prevents infinite loops.
    """

    def __init__(self, max_hops: int = 5):
        """
        Initialize cycle detector.

        Args:
            max_hops: Maximum allowed hop depth
        """
        self.max_hops = max_hops

    def detect_cycle(self, dependencies: Dict[str, List[str]]) -> Optional[List[str]]:
        """
        Detect cycle in dependency graph.

        Args:
            dependencies: Map of query_id -> list of dependencies

        Returns:
            Cycle path if detected, None otherwise
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, current_path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            current_path.append(node)

            # Check depth limit
            if len(current_path) > self.max_hops:
                logger.warning(f"Max hop depth ({self.max_hops}) exceeded")
                return current_path

            # Check dependencies
            for dep in dependencies.get(node, []):
                if dep not in visited:
                    result = dfs(dep, current_path)
                    if result:
                        return result
                elif dep in rec_stack:
                    # Cycle detected
                    cycle_start = current_path.index(dep)
                    return current_path[cycle_start:] + [dep]

            current_path.pop()
            rec_stack.remove(node)
            return None

        for node in dependencies:
            if node not in visited:
                result = dfs(node, [])
                if result:
                    return result

        return None

    def validate_max_hops(self, sub_queries: List[SubQuery]) -> bool:
        """
        Validate that hop count doesn't exceed maximum.

        Args:
            sub_queries: List of sub-queries to validate

        Returns:
            True if valid, False otherwise
        """
        max_order = max((sq.hop_order for sq in sub_queries), default=0)
        return max_order <= self.max_hops


class MultiHopQueryDecomposer:
    """
    Decomposes complex queries into sequential sub-queries.

    Uses LLM to analyze the query structure and identify
    dependencies between sub-queries.
    """

    def __init__(self, llm_client: ILLMClient):
        """
        Initialize query decomposer.

        Args:
            llm_client: LLM client for query decomposition
        """
        self.llm = llm_client

    async def decompose(self, query: str) -> List[SubQuery]:
        """
        Decompose complex query into sub-queries.

        Args:
            query: The complex multi-hop query

        Returns:
            List of SubQuery objects in execution order
        """
        logger.info(f"Decomposing multi-hop query: {query[:100]}...")

        system_prompt = self._get_decomposition_prompt()
        user_message = self._format_decomposition_request(query)

        try:
            response = self.llm.generate(system_prompt, user_message, temperature=0.0)
            sub_queries = self._parse_decomposition_response(response, query)
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            # Fallback: treat as single query
            return [
                SubQuery(
                    query_id=str(uuid.uuid4()),
                    query_text=query,
                    hop_order=1,
                    reasoning="Fallback to single-hop due to decomposition failure",
                )
            ]

    def _get_decomposition_prompt(self) -> str:
        """Get system prompt for query decomposition."""
        return """You are an expert at breaking down complex questions into sequential sub-questions.

Your task is to analyze a complex question and decompose it into simpler sub-questions that need to be answered in sequence.

## Guidelines

1. Identify the logical steps needed to answer the complex question
2. Each sub-question should build on the results of previous sub-questions
3. Limit to 5 sub-questions maximum
4. Number sub-questions in execution order (1, 2, 3, ...)
5. For each sub-question, explain what information it seeks and why it's needed

## Output Format

Respond ONLY with valid JSON in this exact format:

{
  "sub_queries": [
    {
      "query_id": "unique_id_1",
      "query_text": "First sub-question",
      "hop_order": 1,
      "depends_on": [],
      "reasoning": "Why this sub-question is needed"
    },
    {
      "query_id": "unique_id_2",
      "query_text": "Second sub-question that depends on first",
      "hop_order": 2,
      "depends_on": ["unique_id_1"],
      "context_from": "unique_id_1",
      "reasoning": "Why this sub-question is needed"
    }
  ]
}

## Example

Question: "졸업 요건을 충족하려면 어떤 전공 필수 과목을 이수해야 하고, 그 과목들의 선이수 과목은 무엇인가요?"

Response:
{
  "sub_queries": [
    {
      "query_id": "hop_1",
      "query_text": "졸업 요건에 따른 전공 필수 과목 목록",
      "hop_order": 1,
      "depends_on": [],
      "reasoning": "First identify which courses are required for graduation"
    },
    {
      "query_id": "hop_2",
      "query_text": "전공 필수 과목들의 선이수 과목 조회",
      "hop_order": 2,
      "depends_on": ["hop_1"],
      "context_from": "hop_1",
      "reasoning": "Then find prerequisites for each required course"
    }
  ]
}

IMPORTANT: Respond ONLY with valid JSON. No explanations, no markdown code blocks."""

    def _format_decomposition_request(self, query: str) -> str:
        """Format user message for decomposition."""
        return f"""Decompose the following question into sequential sub-questions:

Question: {query}

Provide your response as valid JSON following the specified format."""

    def _parse_decomposition_response(
        self, response: str, original_query: str
    ) -> List[SubQuery]:
        """
        Parse LLM response into SubQuery objects.

        Args:
            response: LLM response string
            original_query: Original query for fallback

        Returns:
            List of SubQuery objects
        """
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            data = json.loads(response)
            sub_queries_data = data.get("sub_queries", [])

            sub_queries = []
            for sq_data in sub_queries_data:
                # Ensure query_id exists
                if "query_id" not in sq_data:
                    sq_data["query_id"] = (
                        f"hop_{sq_data.get('hop_order', len(sub_queries) + 1)}"
                    )

                sub_queries.append(SubQuery.from_dict(sq_data))

            # Sort by hop_order
            sub_queries.sort(key=lambda sq: sq.hop_order)
            return sub_queries

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse decomposition response as JSON: {e}")
            logger.debug(f"Response content: {response[:500]}")
            # Fallback to single query
            return [
                SubQuery(
                    query_id="hop_1",
                    query_text=original_query,
                    hop_order=1,
                    reasoning="Fallback due to JSON parsing error",
                )
            ]
        except Exception as e:
            logger.error(f"Error parsing decomposition response: {e}")
            return [
                SubQuery(
                    query_id="hop_1",
                    query_text=original_query,
                    hop_order=1,
                    reasoning="Fallback due to parsing error",
                )
            ]


class MultiHopHandler:
    """
    Handles multi-hop query execution and answer synthesis.

    Orchestrate sequential sub-query execution, applies Self-RAG
    for relevance filtering, and synthesizes comprehensive answers.
    """

    def __init__(
        self,
        vector_store: IVectorStore,
        llm_client: ILLMClient,
        max_hops: int = 5,
        hop_timeout_seconds: int = 30,
        enable_self_rag: bool = True,
    ):
        """
        Initialize multi-hop handler.

        Args:
            vector_store: Vector store for retrieval
            llm_client: LLM client for answer generation
            max_hops: Maximum number of hops allowed
            hop_timeout_seconds: Timeout per hop
            enable_self_rag: Enable Self-RAG relevance filtering
        """
        self.vector_store = vector_store
        self.llm = llm_client
        self.max_hops = max_hops
        self.hop_timeout = hop_timeout_seconds
        self.enable_self_rag = enable_self_rag

        self.decomposer = MultiHopQueryDecomposer(llm_client)
        self.cycle_detector = DependencyCycleDetector(max_hops)

    async def execute_multi_hop(self, query: str, top_k: int = 10) -> MultiHopResult:
        """
        Execute multi-hop query processing.

        Args:
            query: The complex multi-hop query
            top_k: Number of results to retrieve per hop

        Returns:
            MultiHopResult with final answer and execution details
        """
        start_time = time.time()
        logger.info(f"Starting multi-hop processing for: {query[:100]}...")

        try:
            # Step 1: Decompose query
            sub_queries = await self.decomposer.decompose(query)

            # Step 2: Validate
            if not self.cycle_detector.validate_max_hops(sub_queries):
                logger.error(f"Hop count exceeds maximum of {self.max_hops}")
                return self._create_failure_result(
                    query, sub_queries, "Hop count exceeds maximum"
                )

            # Check for cycles
            dependencies = {sq.query_id: sq.depends_on for sq in sub_queries}
            cycle = self.cycle_detector.detect_cycle(dependencies)
            if cycle:
                logger.error(f"Cyclic dependency detected: {cycle}")
                return self._create_failure_result(
                    query, sub_queries, f"Cyclic dependency: {' -> '.join(cycle)}"
                )

            # Step 3: Execute hops sequentially
            hop_results: List[HopResult] = []
            context: Dict[str, HopResult] = {}

            for sub_query in sub_queries:
                hop_result = await self._execute_hop_with_timeout(
                    sub_query, context, top_k
                )
                hop_results.append(hop_result)
                context[sub_query.query_id] = hop_result

                # Stop if critical hop failed
                if hop_result.status == HopStatus.FAILED:
                    logger.warning(
                        f"Hop {sub_query.hop_order} failed, stopping execution"
                    )
                    break

            # Step 4: Synthesize final answer
            final_answer = await self._synthesize_final_answer(query, hop_results)

            total_time = (time.time() - start_time) * 1000

            result = MultiHopResult(
                original_query=query,
                sub_queries=sub_queries,
                hop_results=hop_results,
                final_answer=final_answer,
                total_execution_time_ms=total_time,
                hop_count=len(hop_results),
                success=True,
            )

            logger.info(
                f"Multi-hop processing completed in {total_time:.0f}ms "
                f"({len(hop_results)} hops)"
            )
            return result

        except Exception as e:
            logger.error(f"Multi-hop processing failed: {e}", exc_info=True)
            return self._create_failure_result(query, [], str(e))

    async def _execute_hop_with_timeout(
        self, sub_query: SubQuery, context: Dict[str, HopResult], top_k: int
    ) -> HopResult:
        """
        Execute a single hop with timeout.

        Args:
            sub_query: The sub-query to execute
            context: Results from previous hops
            top_k: Number of results to retrieve

        Returns:
            HopResult with execution results
        """
        try:
            result = await asyncio.wait_for(
                self._execute_hop(sub_query, context, top_k),
                timeout=self.hop_timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Hop {sub_query.hop_order} timed out")
            return HopResult(
                hop_id=sub_query.query_id,
                query=sub_query,
                answer="",
                sources=[],
                execution_time_ms=self.hop_timeout * 1000,
                is_relevant=False,
                status=HopStatus.TIMEOUT,
                error_message=f"Timeout after {self.hop_timeout} seconds",
            )

    async def _execute_hop(
        self, sub_query: SubQuery, context: Dict[str, HopResult], top_k: int
    ) -> HopResult:
        """
        Execute a single hop.

        Args:
            sub_query: The sub-query to execute
            context: Results from previous hops
            top_k: Number of results to retrieve

        Returns:
            HopResult with execution results
        """
        start_time = time.time()
        logger.info(
            f"Executing hop {sub_query.hop_order}: {sub_query.query_text[:50]}..."
        )

        try:
            # Build context from previous hops
            context_text = self._build_context_text(sub_query, context)

            # Execute search
            search_query = Query(text=sub_query.query_text)
            search_results = self.vector_store.search(search_query, top_k=top_k)

            # Apply Self-RAG if enabled
            is_relevant = True
            if self.enable_self_rag and search_results:
                is_relevant = await self._evaluate_relevance(
                    sub_query.query_text, search_results
                )

            # Generate answer for this hop
            answer = await self._generate_hop_answer(
                sub_query, search_results, context_text
            )

            execution_time = (time.time() - start_time) * 1000

            logger.info(
                f"Hop {sub_query.hop_order} completed in {execution_time:.0f}ms, "
                f"{len(search_results)} sources, relevant={is_relevant}"
            )

            return HopResult(
                hop_id=sub_query.query_id,
                query=sub_query,
                answer=answer,
                sources=search_results,
                execution_time_ms=execution_time,
                is_relevant=is_relevant,
                status=HopStatus.COMPLETED,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Hop {sub_query.hop_order} failed: {e}")
            return HopResult(
                hop_id=sub_query.query_id,
                query=sub_query,
                answer="",
                sources=[],
                execution_time_ms=execution_time,
                is_relevant=False,
                status=HopStatus.FAILED,
                error_message=str(e),
            )

    def _build_context_text(
        self, sub_query: SubQuery, context: Dict[str, HopResult]
    ) -> str:
        """
        Build context text from previous hop results.

        Args:
            sub_query: Current sub-query
            context: Previous hop results

        Returns:
            Context string
        """
        if not sub_query.context_from or sub_query.context_from not in context:
            return ""

        prev_result = context[sub_query.context_from]
        return f"""Previous Question: {prev_result.query.query_text}
Previous Answer: {prev_result.answer}

"""

    async def _evaluate_relevance(
        self, query: str, search_results: List[SearchResult]
    ) -> bool:
        """
        Evaluate relevance of search results using Self-RAG.

        Args:
            query: The query text
            search_results: Retrieved search results

        Returns:
            True if relevant, False otherwise
        """
        if not search_results:
            return False

        # Simple heuristic: check if top result has decent score
        top_score = search_results[0].score
        return top_score > 0.5

    async def _generate_hop_answer(
        self, sub_query: SubQuery, search_results: List[SearchResult], context: str
    ) -> str:
        """
        Generate answer for a single hop.

        Args:
            sub_query: The sub-query
            search_results: Retrieved search results
            context: Context from previous hops

        Returns:
            Generated answer
        """
        if not search_results:
            return f"No relevant information found for: {sub_query.query_text}"

        system_prompt = """You are a regulation expert. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so explicitly.
Do not make up information or add external knowledge."""

        # Build context from search results
        context_chunks = []
        for i, result in enumerate(search_results[:5], 1):
            chunk = result.chunk
            context_chunks.append(f"[Source {i}] {chunk.title}\n{chunk.text}\n")

        context_text = context + "\n".join(context_chunks)
        user_message = f"Question: {sub_query.query_text}\n\nContext:\n{context_text}"

        try:
            answer = self.llm.generate(system_prompt, user_message, temperature=0.0)
            return answer.strip()
        except Exception as e:
            logger.error(f"Failed to generate hop answer: {e}")
            return f"Error generating answer for: {sub_query.query_text}"

    async def _synthesize_final_answer(
        self, original_query: str, hop_results: List[HopResult]
    ) -> str:
        """
        Synthesize final answer from all hop results.

        Args:
            original_query: The original complex query
            hop_results: Results from all hops

        Returns:
            Synthesized comprehensive answer
        """
        if not hop_results:
            return "Unable to answer the question due to processing errors."

        # If only one hop, return its answer directly
        if len(hop_results) == 1:
            return hop_results[0].answer

        # Build synthesis prompt
        hop_summaries = []
        for i, result in enumerate(hop_results, 1):
            hop_summaries.append(
                f"Step {i} ({result.query.query_text}):\n{result.answer}"
            )

        system_prompt = """You are a regulation expert. Synthesize a comprehensive answer from multiple step-by-step results.
Create a coherent, well-structured answer that addresses the original question completely."""

        user_message = f"""Original Question: {original_query}

Step-by-step Analysis:
{chr(10).join(hop_summaries)}

Please provide a comprehensive, synthesized answer to the original question."""

        try:
            final_answer = self.llm.generate(
                system_prompt, user_message, temperature=0.0
            )
            return final_answer.strip()
        except Exception as e:
            logger.error(f"Failed to synthesize final answer: {e}")
            # Fallback: concatenate all hop answers
            return "\n\n".join([hr.answer for hr in hop_results])

    def _create_failure_result(
        self, query: str, sub_queries: List[SubQuery], error_message: str
    ) -> MultiHopResult:
        """Create a failure result."""
        return MultiHopResult(
            original_query=query,
            sub_queries=sub_queries,
            hop_results=[],
            final_answer=f"Multi-hop processing failed: {error_message}",
            total_execution_time_ms=0,
            hop_count=0,
            success=False,
        )
