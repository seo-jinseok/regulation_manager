"""
Self-RAG Evaluator for Enhanced Retrieval-Augmented Generation.

Implements self-reflection mechanism where the LLM evaluates:
1. Whether retrieval is needed for a query
2. Whether retrieved documents are relevant
3. Whether the generated answer is supported by the context
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..domain.entities import SearchResult
    from ..domain.repositories import ILLMClient


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

        Args:
            query: User's question

        Returns:
            True if retrieval is recommended
        """
        if not self._llm_client:
            return True  # Default to retrieval if no LLM

        prompt = self.RETRIEVAL_NEEDED_PROMPT.format(query=query)

        try:
            response = self._llm_client.generate(
                system_prompt="You are a retrieval necessity evaluator.",
                user_message=prompt,
                temperature=0.0,
            )
            return "[RETRIEVE_YES]" in response.upper()
        except Exception:
            return True  # Default to retrieval on error

    def evaluate_relevance(
        self, query: str, results: List["SearchResult"], max_context_chars: int = 2000
    ) -> tuple:
        """
        Evaluate relevance of search results.

        Args:
            query: User's question
            results: Search results to evaluate
            max_context_chars: Maximum context length to send to LLM

        Returns:
            Tuple of (is_relevant: bool, relevant_results: List[SearchResult])
        """
        if not self._llm_client or not results:
            return (bool(results), results)

        # Build context from top results
        context_parts = []
        current_len = 0
        relevant_results = []

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
    """

    def __init__(
        self,
        search_usecase=None,
        llm_client: Optional["ILLMClient"] = None,
        enable_retrieval_check: bool = True,
        enable_relevance_check: bool = True,
        enable_support_check: bool = False,  # Expensive, disabled by default
    ):
        """
        Initialize Self-RAG pipeline.

        Args:
            search_usecase: SearchUseCase for retrieval
            llm_client: LLM client for evaluation and generation
            enable_retrieval_check: Whether to check if retrieval is needed
            enable_relevance_check: Whether to check result relevance
            enable_support_check: Whether to check answer support
        """
        self._search_usecase = search_usecase
        self._llm_client = llm_client
        self._evaluator = SelfRAGEvaluator(llm_client)

        self.enable_retrieval_check = enable_retrieval_check
        self.enable_relevance_check = enable_relevance_check
        self.enable_support_check = enable_support_check

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
