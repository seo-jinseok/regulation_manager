"""
Self-RAG Evaluator for Enhanced Retrieval-Augmented Generation.

Implements self-reflection mechanism where the LLM evaluates:
1. Whether retrieval is needed for a query
2. Whether retrieved documents are relevant
3. Whether the generated answer is supported by the context

9 RAG Architectures (#5): Self-RAG adds reflection capabilities to verify
retrieval necessity and answer grounding, improving response quality.
"""

import concurrent.futures
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from ..config import get_config

if TYPE_CHECKING:
    from ..domain.entities import SearchResult
    from ..domain.repositories import ILLMClient


logger = logging.getLogger(__name__)


class SelfRAGEvaluator:
    """
    Self-RAG reflection mechanism.

    Uses LLM to evaluate retrieval necessity and result relevance,
    enabling more accurate and grounded responses.
    """

    # Prompts for self-reflection
    RETRIEVAL_NEEDED_PROMPT = """다음 질문에 답하기 위해 외부 문서 검색이 필요한지 판단하세요.

질문: {query}

검색이 필요한 경우:
- 특정 규정, 절차, 자격 요건 등 사실적 정보가 필요한 경우
- 최신 정보나 특정 조항을 확인해야 하는 경우

검색이 불필요한 경우:
- 일반적인 인사말이나 간단한 설명 요청
- 이전 대화 내용에 대한 후속 질문 (컨텍스트가 이미 있는 경우)

답변 형식: [RETRIEVE_YES] 또는 [RETRIEVE_NO] 중 하나만 출력하세요."""

    RELEVANCE_EVAL_PROMPT = """다음 문서가 질문에 답변하는 데 관련이 있는지 평가하세요.

질문: {query}

문서:
{context}

평가 기준:
- 문서가 질문의 주제와 직접적으로 관련이 있는가?
- 문서에 질문에 답하는 데 필요한 정보가 포함되어 있는가?

답변 형식: [RELEVANT] 또는 [IRRELEVANT] 중 하나만 출력하세요."""

    SUPPORT_EVAL_PROMPT = """다음 답변이 제공된 문서에 의해 충분히 뒷받침되는지 평가하세요.

질문: {query}

문서:
{context}

생성된 답변:
{answer}

평가 기준:
- 답변의 주요 주장이 문서에 근거하고 있는가?
- 문서에 없는 정보를 추측하거나 추가하지 않았는가?

답변 형식: [SUPPORTED], [PARTIALLY_SUPPORTED], 또는 [NOT_SUPPORTED] 중 하나만 출력하세요."""

    def __init__(self, llm_client: Optional["ILLMClient"] = None):
        """
        Initialize Self-RAG evaluator.

        Args:
            llm_client: LLM client for evaluation (can be set later)
        """
        self._llm_client = llm_client

    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        """Set the LLM client for evaluation."""
        self._llm_client = llm_client

    def needs_retrieval(self, query: str) -> bool:
        """
        Evaluate if retrieval is needed for this query.

        For regulation Q&A systems, retrieval is almost always needed.
        Only skip retrieval for very simple greetings/chat.

        Args:
            query: User's question

        Returns:
            True if retrieval is recommended
        """
        if not self._llm_client:
            return True  # Default to retrieval if no LLM

        # Quick heuristic: very short queries without question words
        # are probably not factual questions
        if len(query) < 5 and not any(
            k in query for k in ["?", "뭐", "어떻", "왜", "언제"]
        ):
            return True  # Still default to retrieval to be safe

        prompt = self.RETRIEVAL_NEEDED_PROMPT.format(query=query)

        try:
            response = self._llm_client.generate(
                system_prompt="You are a retrieval necessity evaluator.",
                user_message=prompt,
                temperature=0.0,
            )
            # Default to retrieval unless explicitly told not to
            if "[RETRIEVE_NO]" in response.upper():
                return False
            return True  # Default to retrieval
        except Exception:
            return True  # Default to retrieval on error

    def evaluate_relevance(
        self,
        query: str,
        results: List["SearchResult"],
        max_context_chars: Optional[int] = None,
    ) -> tuple:
        """
        Evaluate relevance of search results.

        Args:
            query: User's question
            results: Search results to evaluate
            max_context_chars: Maximum context length to send to LLM (increased to 4000 for long procedure regulations)

        Returns:
            Tuple of (is_relevant: bool, relevant_results: List[SearchResult])
        """
        if max_context_chars is None:
            max_context_chars = get_config().max_context_chars
        if not self._llm_client or not results:
            return (bool(results), results)

        # Build context from top results
        context_parts = []
        current_len = 0

        for result in results[:5]:  # Evaluate top 5
            text = result.chunk.text[:500]
            context_parts.append(f"[{result.chunk.title}]\n{text}")
            current_len += len(text)

            if current_len >= max_context_chars:
                break

        context = "\n\n---\n\n".join(context_parts)
        prompt = self.RELEVANCE_EVAL_PROMPT.format(query=query, context=context)

        try:
            response = self._llm_client.generate(
                system_prompt="You are a document relevance evaluator.",
                user_message=prompt,
                temperature=0.0,
            )
            is_relevant = "[RELEVANT]" in response.upper()

            # If relevant, return all results; otherwise return empty
            return (is_relevant, results if is_relevant else [])
        except Exception:
            return (True, results)  # Default to relevant on error

    def evaluate_support(self, query: str, context: str, answer: str) -> str:
        """
        Evaluate if the answer is supported by the context.

        Args:
            query: User's question
            context: Retrieved context
            answer: Generated answer

        Returns:
            Support level: "SUPPORTED", "PARTIALLY_SUPPORTED", or "NOT_SUPPORTED"
        """
        if not self._llm_client:
            return "SUPPORTED"  # Default to supported if no LLM

        prompt = self.SUPPORT_EVAL_PROMPT.format(
            query=query, context=context[:2000], answer=answer[:1000]
        )

        try:
            response = self._llm_client.generate(
                system_prompt="You are an answer support evaluator.",
                user_message=prompt,
                temperature=0.0,
            )

            response_upper = response.upper()
            if "[SUPPORTED]" in response_upper:
                return "SUPPORTED"
            elif "[PARTIALLY_SUPPORTED]" in response_upper:
                return "PARTIALLY_SUPPORTED"
            else:
                return "NOT_SUPPORTED"
        except Exception:
            return "SUPPORTED"  # Default on error


class SelfRAGPipeline:
    """
    Complete Self-RAG pipeline integrating evaluation with search and generation.

    Features:
    - Retrieval necessity check: Skip retrieval for simple queries
    - Relevance filtering: Remove irrelevant results before answer generation
    - Support verification: Verify answer is grounded in retrieved context
    - Async support evaluation: Run support check in background for performance
    """

    def __init__(
        self,
        search_usecase=None,
        llm_client: Optional["ILLMClient"] = None,
        enable_retrieval_check: bool = True,
        enable_relevance_check: bool = True,
        enable_support_check: bool = True,  # Now enabled by default
        async_support_check: bool = True,  # Run support check asynchronously
    ):
        """
        Initialize Self-RAG pipeline.

        Args:
            search_usecase: SearchUseCase for retrieval
            llm_client: LLM client for evaluation and generation
            enable_retrieval_check: Whether to check if retrieval is needed
            enable_relevance_check: Whether to check result relevance
            enable_support_check: Whether to check answer support (now default True)
            async_support_check: Whether to run support check asynchronously
        """
        self._search_usecase = search_usecase
        self._llm_client = llm_client
        self._evaluator = SelfRAGEvaluator(llm_client)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_support_check: Optional[concurrent.futures.Future] = None

        self.enable_retrieval_check = enable_retrieval_check
        self.enable_relevance_check = enable_relevance_check
        self.enable_support_check = enable_support_check
        self.async_support_check = async_support_check

    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        """Set LLM client for both evaluation and generation."""
        self._llm_client = llm_client
        self._evaluator.set_llm_client(llm_client)

    def should_retrieve(self, query: str) -> bool:
        """Check if retrieval is needed for this query."""
        if not self.enable_retrieval_check:
            return True
        return self._evaluator.needs_retrieval(query)

    def filter_relevant_results(
        self, query: str, results: List["SearchResult"]
    ) -> List["SearchResult"]:
        """Filter results to only include relevant ones."""
        if not self.enable_relevance_check:
            return results
        _, relevant = self._evaluator.evaluate_relevance(query, results)
        return relevant

    def get_support_level(self, query: str, context: str, answer: str) -> str:
        """Get the support level of an answer."""
        if not self.enable_support_check:
            return "SUPPORTED"
        return self._evaluator.evaluate_support(query, context, answer)

    def start_async_support_check(
        self, query: str, context: str, answer: str
    ) -> Optional[concurrent.futures.Future]:
        """
        Start support check in background thread.

        Returns immediately with a Future that can be checked later.
        This allows the main response to be sent while support verification runs.

        Args:
            query: User's question
            context: Retrieved context
            answer: Generated answer

        Returns:
            Future object or None if async is disabled
        """
        if not self.enable_support_check:
            return None
        if not self.async_support_check:
            # Run synchronously
            return None

        try:
            future = self._executor.submit(
                self._evaluator.evaluate_support, query, context, answer
            )
            self._pending_support_check = future
            logger.debug("Started async support check")
            return future
        except Exception as e:
            logger.warning(f"Failed to start async support check: {e}")
            return None

    def get_async_support_result(self, timeout: float = 5.0) -> Optional[str]:
        """
        Get the result of async support check if available.

        Args:
            timeout: Maximum seconds to wait for result

        Returns:
            Support level or None if not available/timed out
        """
        if self._pending_support_check is None:
            return None

        try:
            result = self._pending_support_check.result(timeout=timeout)
            self._pending_support_check = None
            return result
        except concurrent.futures.TimeoutError:
            logger.debug("Async support check timed out")
            return None
        except Exception as e:
            logger.warning(f"Async support check failed: {e}")
            return None

    def evaluate_results_batch(
        self, query: str, results: List["SearchResult"]
    ) -> Tuple[bool, List["SearchResult"], float]:
        """
        Evaluate search results in batch for efficiency.

        Returns:
            Tuple of (is_relevant, filtered_results, confidence_score)
        """
        if not results:
            return (False, [], 0.0)

        # Defensive type checking: Convert dict to SearchResult if necessary
        processed_results = []
        for r in results:
            if isinstance(r, dict):
                # Convert dict representation to SearchResult
                # This handles cases where data was serialized/deserialized
                try:
                    from ..domain.entities import Chunk
                    from ..domain.entities import SearchResult as SR

                    chunk_data = r.get("chunk")
                    score = r.get("score", 0.0)
                    rank = r.get("rank", 0)

                    # Handle chunk as dict or Chunk object
                    if isinstance(chunk_data, dict):
                        chunk = Chunk(
                            id=chunk_data.get("id", ""),
                            rule_code=chunk_data.get("rule_code", ""),
                            level=chunk_data.get("level", "text"),
                            title=chunk_data.get("title", ""),
                            text=chunk_data.get("text", ""),
                            embedding_text=chunk_data.get("embedding_text", ""),
                            full_text=chunk_data.get("full_text", ""),
                            parent_path=chunk_data.get("parent_path", []),
                            token_count=chunk_data.get("token_count", 0),
                            keywords=chunk_data.get("keywords", []),
                            is_searchable=chunk_data.get("is_searchable", True),
                            doc_type=chunk_data.get("doc_type", "regulation"),
                            effective_date=chunk_data.get("effective_date"),
                            status=chunk_data.get("status", "active"),
                        )
                    else:
                        # chunk is already a Chunk object
                        chunk = chunk_data

                    processed_results.append(SR(chunk=chunk, score=score, rank=rank))
                except Exception as e:
                    logger.warning(f"Failed to convert dict to SearchResult: {e}")
                    continue
            else:
                # Already a SearchResult object
                processed_results.append(r)

        if not processed_results:
            return (False, [], 0.0)

        results = processed_results

        # Quick heuristic check first (no LLM call)
        top_score = results[0].score if results else 0.0

        # If top score is very high, skip LLM evaluation
        if top_score > 0.8:
            return (True, results, top_score)

        # Use LLM for borderline cases
        if self.enable_relevance_check:
            is_relevant, filtered = self._evaluator.evaluate_relevance(query, results)
            confidence = top_score if is_relevant else top_score * 0.5
            return (is_relevant, filtered, confidence)

        return (True, results, top_score)
