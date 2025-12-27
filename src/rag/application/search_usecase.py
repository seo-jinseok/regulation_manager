"""
Search Use Case for Regulation RAG System.

Provides search functionality with optional LLM-based Q&A.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from ..domain.entities import Answer, Chunk, SearchResult
from ..domain.repositories import ILLMClient, IVectorStore
from ..domain.value_objects import Query, SearchFilter

if TYPE_CHECKING:
    from ..infrastructure.hybrid_search import HybridSearcher


# Regex for matching article/paragraph/item numbers (제N조, 제N항, 제N호)
ARTICLE_PATTERN = re.compile(
    r"제\s*\d+\s*조(?:\s*의\s*\d+)?|제\s*\d+\s*항|제\s*\d+\s*호"
)
HEADING_ONLY_PATTERN = re.compile(r"^\([^)]*\)\s*$")


def _normalize_article_token(token: str) -> str:
    return re.sub(r"\s+", "", token)


# System prompt for regulation Q&A
REGULATION_QA_PROMPT = """당신은 대학 규정 전문가입니다. 
주어진 규정 내용을 바탕으로 사용자의 질문에 **상세하고 친절하게** 답변하세요.

## 중요 원칙
- **반드시 제공된 규정 내용에 명시된 사항만 답변하세요.**
- 규정에 없는 내용은 절대 추측하거나 일반적인 관행을 언급하지 마세요.
- 괄호로 부연 설명을 달지 마세요.
- 규정에 절차·결정 주체·승인 단계가 **명시된 경우에만** 그 표현을 사용하세요.

## 답변 지침
1. 규정에 명시된 내용을 바탕으로 **단계별로 상세히** 설명하세요.
2. 관련 조항 번호와 규정명을 함께 언급하세요. **경로/번호 표기는 제공된 텍스트를 그대로 사용**하고, "제N조"로 바꾸지 마세요.
3. 규정에 명시된 정보만 포함하세요. 규정에 없는 서류, 조건, 기한 등을 추가하지 마세요.
4. 업무 범위나 담당 부서 **표제만 있는 경우**, 그 범위에 포함됨을 밝히고 절차·권한은 **규정에서 확인되지 않음**이라고 명시하세요.
5. 제목/표제 수준만 제공된 경우, 구체 절차·조건은 **규정에서 확인되지 않음**이라고 명시하세요.
6. 폐지된 규정은 현행 규정이 아님을 주의하세요.
7. 마크다운 형식(번호 목록, 굵은 글씨 등)을 사용하여 가독성을 높이세요.
"""


@dataclass(frozen=True)
class QueryRewriteInfo:
    """Stores query rewrite details for debugging/verbose output."""

    original: str
    rewritten: str
    used: bool
    method: Optional[str] = None
    from_cache: bool = False
    fallback: bool = False
    used_synonyms: Optional[bool] = None
    used_intent: Optional[bool] = None
    matched_intents: Optional[List[str]] = None


class SearchUseCase:
    """
    Use case for searching regulations and generating answers.
    
    Supports:
    - Hybrid search (dense + sparse)
    - Cross-encoder reranking (BGE)
    - Metadata filtering
    - LLM-based Q&A
    """

    def __init__(
        self,
        store: IVectorStore,
        llm_client: Optional[ILLMClient] = None,
        use_reranker: Optional[bool] = None,
        hybrid_searcher: Optional["HybridSearcher"] = None,
        use_hybrid: Optional[bool] = None,
    ):
        """
        Initialize search use case.

        Args:
            store: Vector store implementation.
            llm_client: Optional LLM client for generating answers.
            use_reranker: Whether to use BGE reranker (default: from config).
            hybrid_searcher: Optional HybridSearcher (auto-created if None and use_hybrid=True).
            use_hybrid: Whether to use hybrid search (default: from config).
        """
        # Use config defaults if not explicitly specified
        from ..config import get_config
        config = get_config()
        
        self.store = store
        self.llm = llm_client
        self.use_reranker = use_reranker if use_reranker is not None else config.use_reranker
        self._hybrid_searcher = hybrid_searcher
        self._use_hybrid = use_hybrid if use_hybrid is not None else config.use_hybrid
        self._hybrid_initialized = hybrid_searcher is not None
        self._last_query_rewrite: Optional[QueryRewriteInfo] = None

    @property
    def hybrid_searcher(self) -> Optional["HybridSearcher"]:
        """Lazy-initialize HybridSearcher on first access."""
        if self._use_hybrid and not self._hybrid_initialized:
            self._ensure_hybrid_searcher()
        return self._hybrid_searcher

    def _ensure_hybrid_searcher(self) -> None:
        """Initialize HybridSearcher with documents from vector store."""
        if self._hybrid_initialized:
            return
        
        from ..infrastructure.hybrid_search import HybridSearcher
        
        # Get all documents from store for BM25 indexing
        documents = self.store.get_all_documents()
        if documents:
            self._hybrid_searcher = HybridSearcher()
            self._hybrid_searcher.add_documents(documents)
            # Set LLM client for query rewriting if available
            if self.llm:
                self._hybrid_searcher.set_llm_client(self.llm)
        
        self._hybrid_initialized = True


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
        
        # Rewrite query using LLM if HybridSearcher is available (with LLM)
        # Falls back to synonym expansion if LLM is not available
        rewritten_query_text = query_text
        rewrite_used = False
        rewrite_method: Optional[str] = None
        rewrite_from_cache = False
        rewrite_fallback = False
        used_synonyms: Optional[bool] = None
        used_intent: Optional[bool] = None
        matched_intents: Optional[List[str]] = None
        if self.hybrid_searcher:
            # Set LLM client if not already set
            if self.llm and not self.hybrid_searcher._query_analyzer._llm_client:
                self.hybrid_searcher.set_llm_client(self.llm)
            # Rewrite query (uses LLM if available, otherwise expands with synonyms)
            rewrite_info = self.hybrid_searcher._query_analyzer.rewrite_query_with_info(query_text)
            rewritten_query_text = rewrite_info.rewritten
            rewrite_used = True
            rewrite_method = rewrite_info.method
            rewrite_from_cache = rewrite_info.from_cache
            rewrite_fallback = rewrite_info.fallback
            used_intent = rewrite_info.used_intent
            used_synonyms = rewrite_info.used_synonyms
            matched_intents = rewrite_info.matched_intents
            analyzer = self.hybrid_searcher._query_analyzer
            if used_synonyms is None:
                used_synonyms = analyzer.has_synonyms(query_text)
            if rewritten_query_text and rewritten_query_text != query_text:
                used_synonyms = used_synonyms or analyzer.has_synonyms(rewritten_query_text)
            # Use rewritten query for dense search too
            query = Query(text=rewritten_query_text, include_abolished=include_abolished)

        self._last_query_rewrite = QueryRewriteInfo(
            original=query_text,
            rewritten=rewritten_query_text,
            used=rewrite_used,
            method=rewrite_method,
            from_cache=rewrite_from_cache,
            fallback=rewrite_fallback,
            used_synonyms=used_synonyms,
            used_intent=used_intent,
            matched_intents=matched_intents,
        )
        scoring_query_text = self._select_scoring_query(query_text, rewritten_query_text)
        
        # Get dense search results from vector store
        dense_results = self.store.search(query, filter, top_k * 3)
        
        # Apply Hybrid Search if HybridSearcher is available
        if self.hybrid_searcher:
            from ..infrastructure.hybrid_search import ScoredDocument
            
            # Get BM25 sparse results (already uses expanded query internally)
            sparse_query_text = rewritten_query_text or query_text
            sparse_results = self.hybrid_searcher.search_sparse(sparse_query_text, top_k * 3)
            # Convert dense results to ScoredDocument format for fusion
            dense_docs = [
                ScoredDocument(
                    doc_id=r.chunk.id,
                    score=r.score,
                    content=r.chunk.text,
                    metadata=r.chunk.to_metadata(),
                )
                for r in dense_results
            ]
            
            # Fuse results using RRF
            fused = self.hybrid_searcher.fuse_results(
                sparse_results=sparse_results,
                dense_results=dense_docs,
                top_k=top_k * 3,
                query_text=query_text,
            )
            
            # Convert back to SearchResult (rebuild chunks from metadata)
            id_to_result = {r.chunk.id: r for r in dense_results}
            results = []
            for i, doc in enumerate(fused):
                if doc.doc_id in id_to_result:
                    # Use existing chunk from dense results
                    original = id_to_result[doc.doc_id]
                    results.append(SearchResult(
                        chunk=original.chunk,
                        score=doc.score,
                        rank=i + 1,
                    ))
                else:
                    # BM25-only result: need to rebuild chunk from metadata
                    from ..domain.entities import Chunk
                    chunk = Chunk.from_metadata(doc.doc_id, doc.content, doc.metadata)
                    results.append(SearchResult(
                        chunk=chunk,
                        score=doc.score,
                        rank=i + 1,
                    ))
        else:
            # No hybrid searcher, use dense results directly
            results = dense_results
        
        # Apply keyword bonus: boost score if query terms appear in text or keywords
        query_terms = scoring_query_text.lower().split()
        query_text_lower = scoring_query_text.lower()
        boosted_results = []
        for r in results:
            text_lower = r.chunk.text.lower()
            # Count matching terms
            matches = sum(1 for term in query_terms if term in text_lower)
            # Bonus: 0.1 per matching term
            bonus = matches * 0.1

            keyword_bonus = 0.0
            if r.chunk.keywords:
                keyword_hits = sum(
                    kw.weight for kw in r.chunk.keywords
                    if kw.term and kw.term.lower() in query_text_lower
                )
                keyword_bonus = min(0.3, keyword_hits * 0.05)

            # Article number exact match bonus (제N조, 제N항, 제N호)
            article_bonus = 0.0
            query_articles = {
                _normalize_article_token(t)
                for t in ARTICLE_PATTERN.findall(query_text)
            }
            if query_articles:
                article_haystack = r.chunk.embedding_text or r.chunk.text
                text_articles = {
                    _normalize_article_token(t)
                    for t in ARTICLE_PATTERN.findall(article_haystack)
                }
                if query_articles & text_articles:
                    article_bonus = 0.2  # Exact article match

            new_score = min(1.0, r.score + bonus + keyword_bonus + article_bonus)
            boosted_results.append(SearchResult(
                chunk=r.chunk,
                score=new_score,
                rank=r.rank,
            ))
        
        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: -x.score)
        
        # Apply reranker if enabled
        if self.use_reranker and boosted_results:
            from ..infrastructure.reranker import rerank
            
            # Prepare documents for reranking (use top candidates)
            candidates = boosted_results[:top_k * 2]
            documents = [
                (r.chunk.id, r.chunk.text, {})
                for r in candidates
            ]
            
            # Rerank using BGE cross-encoder
            reranked = rerank(scoring_query_text, documents, top_k=top_k)
            
            # Map back to SearchResult
            id_to_result = {r.chunk.id: r for r in candidates}
            final_results = []
            for i, rr in enumerate(reranked):
                original = id_to_result.get(rr.doc_id)
                if original:
                    final_results.append(SearchResult(
                        chunk=original.chunk,
                        score=rr.score,
                        rank=i + 1,
                    ))
            return final_results
        
        return boosted_results[:top_k]

    def get_last_query_rewrite(self) -> Optional[QueryRewriteInfo]:
        """Return last query rewrite info (if any)."""
        return self._last_query_rewrite

    def _select_scoring_query(self, original: str, rewritten: str) -> str:
        """Choose query text for scoring/reranking without losing article refs."""
        if not rewritten:
            return original
        if rewritten == original:
            return original
        if ARTICLE_PATTERN.search(original) and not ARTICLE_PATTERN.search(rewritten):
            return f"{original} {rewritten}"
        return rewritten

    def search_unique(
        self,
        query_text: str,
        filter: Optional[SearchFilter] = None,
        top_k: int = 10,
        include_abolished: bool = False,
    ) -> List[SearchResult]:
        """
        Search with deduplication by rule_code.

        Returns only the top-scoring chunk from each regulation.

        Args:
            query_text: The search query.
            filter: Optional metadata filters.
            top_k: Maximum number of unique regulations.
            include_abolished: Whether to include abolished regulations.

        Returns:
            List of SearchResult with one chunk per regulation.
        """
        # Get more results to ensure enough unique regulations
        results = self.search(
            query_text,
            filter=filter,
            top_k=top_k * 5,
            include_abolished=include_abolished,
        )

        # Keep only the best result per rule_code
        seen_codes = set()
        unique_results = []

        for result in results:
            code = result.chunk.rule_code
            if code not in seen_codes:
                seen_codes.add(code)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break

        # Update ranks
        for i, r in enumerate(unique_results):
            unique_results[i] = SearchResult(
                chunk=r.chunk,
                score=r.score,
                rank=i + 1,
            )

        return unique_results

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
            top_k=top_k * 3,
            include_abolished=include_abolished,
        )

        if not results:
            return Answer(
                text="관련 규정을 찾을 수 없습니다. 다른 검색어로 시도해주세요.",
                sources=[],
                confidence=0.0,
            )

        # Filter out low-signal headings when possible
        filtered_results = self._select_answer_sources(results, top_k)
        if not filtered_results:
            filtered_results = results[:top_k]

        # Build context from search results
        context = self._build_context(filtered_results)

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
        confidence = self._compute_confidence(filtered_results)

        return Answer(
            text=answer_text,
            sources=filtered_results,
            confidence=confidence,
        )

    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string from search results."""
        context_parts = []

        for i, result in enumerate(results, 1):
            chunk = result.chunk
            path_str = " > ".join(chunk.parent_path) if chunk.parent_path else ""
            
            context_parts.append(
                f"[{i}] 규정명/경로: {path_str or chunk.rule_code}\n"
                f"    본문: {chunk.text}\n"
                f"    (출처: {chunk.rule_code})"
            )

        return "\n\n".join(context_parts)

    def _select_answer_sources(
        self,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Select best sources for LLM answer, skipping low-signal headings."""
        selected: List[SearchResult] = []
        seen_ids = set()

        for result in results:
            if result.chunk.id in seen_ids:
                continue

            if self._is_low_signal_chunk(result.chunk):
                continue

            seen_ids.add(result.chunk.id)
            selected.append(result)
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for result in results:
                if result.chunk.id in seen_ids:
                    continue
                selected.append(result)
                seen_ids.add(result.chunk.id)
                if len(selected) >= top_k:
                    break

        # Re-rank after filtering
        return [
            SearchResult(chunk=r.chunk, score=r.score, rank=i + 1)
            for i, r in enumerate(selected)
        ]

    def _is_low_signal_chunk(self, chunk: Chunk) -> bool:
        """Heuristic: drop heading-only chunks when richer text exists."""
        text = (chunk.text or "").strip()
        if not text:
            return True

        content = text
        if ":" in text:
            content = text.split(":", 1)[-1].strip()

        if HEADING_ONLY_PATTERN.match(content) and chunk.token_count < 30:
            return True

        return False

    def _compute_confidence(self, results: List[SearchResult]) -> float:
        """
        Compute confidence score based on search results.

        Uses two metrics:
        1. Absolute score: Reranker sigmoid scores (typically 0.001~0.02 range)
        2. Score spread: Difference between top and bottom scores (indicates clear ranking)
        
        Higher scores = more confident in the answer.
        """
        if not results:
            return 0.0

        scores = [r.score for r in results[:5]]
        avg_score = sum(scores) / len(scores)
        
        # Reranker sigmoid scores are typically 0.001 ~ 0.02 range
        # A score of 0.005+ is considered good (50%+ confidence)
        # A score of 0.01+ is considered excellent (100% confidence)
        abs_confidence = min(1.0, avg_score / 0.01)
        
        # Also consider score spread (clear differentiation = higher confidence)
        if len(scores) >= 2:
            spread = max(scores) - min(scores)
            # If top score is significantly higher than bottom, it's a good sign
            spread_confidence = min(1.0, spread / 0.005) if spread > 0 else 0.5
        else:
            spread_confidence = 0.5
        
        # Combine both metrics (weighted average)
        combined = (abs_confidence * 0.7) + (spread_confidence * 0.3)
        
        return max(0.0, min(1.0, combined))

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
