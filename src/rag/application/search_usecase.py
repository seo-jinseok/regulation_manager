"""
Search Use Case for Regulation RAG System.

Provides search functionality with optional LLM-based Q&A.
"""

from typing import List, Optional

from ..domain.entities import Answer, SearchResult
from ..domain.repositories import ILLMClient, IVectorStore
from ..domain.value_objects import Query, SearchFilter


# System prompt for regulation Q&A
REGULATION_QA_PROMPT = """당신은 대학 규정 전문가입니다. 
주어진 규정 내용을 바탕으로 사용자의 질문에 정확하게 답변하세요.

지침:
1. 규정에 명시된 내용만 답변하세요.
2. 불확실한 경우 "규정에서 명시적으로 확인되지 않습니다"라고 답변하세요.
3. 관련 조항 번호와 규정명을 함께 언급하세요.
4. 답변은 간결하고 명확하게 작성하세요.
5. 폐지된 규정은 현행 규정이 아님을 주의하세요.
"""


class SearchUseCase:
    """
    Use case for searching regulations and generating answers.
    
    Supports:
    - Hybrid search (dense + sparse)
    - Metadata filtering
    - LLM-based Q&A
    """

    def __init__(
        self,
        store: IVectorStore,
        llm_client: Optional[ILLMClient] = None,
    ):
        """
        Initialize search use case.

        Args:
            store: Vector store implementation.
            llm_client: Optional LLM client for generating answers.
        """
        self.store = store
        self.llm = llm_client

    def search(
        self,
        query_text: str,
        filter: Optional[SearchFilter] = None,
        top_k: int = 10,
        include_abolished: bool = False,
    ) -> List[SearchResult]:
        """
        Search for relevant regulation chunks.

        Args:
            query_text: The search query.
            filter: Optional metadata filters.
            top_k: Maximum number of results.
            include_abolished: Whether to include abolished regulations.

        Returns:
            List of SearchResult sorted by relevance.
        """
        query = Query(text=query_text, include_abolished=include_abolished)
        return self.store.search(query, filter, top_k)

    def ask(
        self,
        question: str,
        filter: Optional[SearchFilter] = None,
        top_k: int = 5,
        include_abolished: bool = False,
    ) -> Answer:
        """
        Ask a question and get an LLM-generated answer.

        Args:
            question: The user's question.
            filter: Optional metadata filters.
            top_k: Number of chunks to use as context.
            include_abolished: Whether to include abolished regulations.

        Returns:
            Answer with generated text and sources.

        Raises:
            ValueError: If LLM client is not configured.
        """
        if not self.llm:
            raise ValueError("LLM client not configured. Use search() instead.")

        # Get relevant chunks
        results = self.search(
            question,
            filter=filter,
            top_k=top_k,
            include_abolished=include_abolished,
        )

        if not results:
            return Answer(
                text="관련 규정을 찾을 수 없습니다. 다른 검색어로 시도해주세요.",
                sources=[],
                confidence=0.0,
            )

        # Build context from search results
        context = self._build_context(results)

        # Generate answer
        user_message = f"""질문: {question}

참고 규정:
{context}

위 규정 내용을 바탕으로 질문에 답변해주세요."""

        answer_text = self.llm.generate(
            system_prompt=REGULATION_QA_PROMPT,
            user_message=user_message,
            temperature=0.0,
        )

        # Compute confidence based on search scores
        confidence = self._compute_confidence(results)

        return Answer(
            text=answer_text,
            sources=results,
            confidence=confidence,
        )

    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string from search results."""
        context_parts = []

        for i, result in enumerate(results, 1):
            chunk = result.chunk
            path_str = " > ".join(chunk.parent_path) if chunk.parent_path else ""
            
            context_parts.append(
                f"[{i}] {path_str}\n"
                f"    {chunk.text}\n"
                f"    (출처: {chunk.rule_code})"
            )

        return "\n\n".join(context_parts)

    def _compute_confidence(self, results: List[SearchResult]) -> float:
        """
        Compute confidence score based on search results.

        Higher scores = more confident in the answer.
        """
        if not results:
            return 0.0

        # Average of top 3 scores
        top_scores = [r.score for r in results[:3]]
        avg_score = sum(top_scores) / len(top_scores)

        # Normalize to 0-1 range
        return min(1.0, max(0.0, avg_score))

    def search_by_rule_code(
        self,
        rule_code: str,
        top_k: int = 50,
    ) -> List[SearchResult]:
        """
        Get all chunks for a specific rule code.

        Args:
            rule_code: The rule code to search for.
            top_k: Maximum chunks to return.

        Returns:
            List of SearchResult for the rule code.
        """
        filter = SearchFilter(rule_codes=[rule_code])
        # Use a generic query to get all chunks
        query = Query(text="규정", include_abolished=True)
        return self.store.search(query, filter, top_k)
