"""
HyDE (Hypothetical Document Embeddings) for Regulation RAG System.

HyDE improves retrieval for vague or ambiguous queries by:
1. Generating a hypothetical answer document using LLM
2. Embedding that hypothetical document
3. Using that embedding to find similar real documents

This is especially useful for queries like "학교에 가기 싫어" where the user's
actual intent (휴학, 휴직, 연구년 등) is not explicitly stated.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..domain.repositories import ILLMClient, IVectorStore
    from ..domain.entities import SearchResult
    from ..domain.value_objects import Query, SearchFilter

logger = logging.getLogger(__name__)


@dataclass
class HyDEResult:
    """Result of HyDE query expansion."""
    
    original_query: str
    hypothetical_doc: str
    from_cache: bool
    cache_key: Optional[str] = None


class HyDEGenerator:
    """
    Generates hypothetical documents for improved retrieval.
    
    HyDE (Hypothetical Document Embeddings) works by:
    1. Taking a vague/ambiguous query
    2. Generating what an ideal answer document might look like
    3. Using that document's embedding for similarity search
    
    This helps bridge the semantic gap between informal queries and formal
    regulation text.
    """
    
    SYSTEM_PROMPT = """당신은 대학 규정 전문가입니다. 
사용자의 질문에 답하는 대학 규정 조문을 작성하세요.

작성 규칙:
1. 실제 대학 규정처럼 형식적인 문체로 작성
2. 관련 키워드와 용어를 포함
3. 100-200자 내외로 간결하게 작성
4. 규정명이나 조항 번호는 포함하지 않음

예시:
질문: "학교 안 가고 싶어"
답변: 교직원의 휴직은 다음 각 호의 사유에 해당하는 경우 신청할 수 있다. 1. 연구년: 교원이 연구에 전념하기 위하여 일정 기간 강의 및 교무를 면제받는 제도. 2. 병가: 질병 또는 부상으로 인하여 직무를 수행할 수 없는 경우."""

    def __init__(
        self,
        llm_client: Optional["ILLMClient"] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize HyDE generator.
        
        Args:
            llm_client: LLM client for generating hypothetical documents.
            cache_dir: Directory to cache generated documents.
            enable_cache: Whether to use caching.
        """
        self._llm_client = llm_client
        self._enable_cache = enable_cache
        
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "cache" / "hyde"
        
        if self._enable_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache: dict = self._load_cache()
        else:
            self._cache = {}
    
    def _load_cache(self) -> dict:
        """Load cache from disk."""
        cache_file = self._cache_dir / "hyde_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load HyDE cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        if not self._enable_cache:
            return
        cache_file = self._cache_dir / "hyde_cache.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save HyDE cache: {e}")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.encode("utf-8")).hexdigest()[:16]
    
    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        """Set LLM client."""
        self._llm_client = llm_client
    
    def generate_hypothetical_doc(self, query: str) -> HyDEResult:
        """
        Generate a hypothetical regulation document for the given query.
        
        Args:
            query: User's search query.
            
        Returns:
            HyDEResult containing the hypothetical document.
        """
        cache_key = self._get_cache_key(query)
        
        # Check cache
        if cache_key in self._cache:
            logger.debug(f"HyDE cache hit for query: {query[:30]}...")
            return HyDEResult(
                original_query=query,
                hypothetical_doc=self._cache[cache_key],
                from_cache=True,
                cache_key=cache_key,
            )
        
        # Generate if LLM client available
        if not self._llm_client:
            logger.debug("No LLM client for HyDE, using original query")
            return HyDEResult(
                original_query=query,
                hypothetical_doc=query,
                from_cache=False,
            )
        
        try:
            # Generate hypothetical document
            # Note: LLMClientAdapter.generate() only accepts system_prompt, user_message, temperature
            hypothetical_doc = self._llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=f"질문: {query}",
                temperature=0.3,  # Low temperature for consistency
            )
            
            # Cache result
            if hypothetical_doc and len(hypothetical_doc) > 20:
                self._cache[cache_key] = hypothetical_doc
                self._save_cache()
                
            logger.debug(f"Generated HyDE doc for: {query[:30]}... -> {hypothetical_doc[:50]}...")
            
            return HyDEResult(
                original_query=query,
                hypothetical_doc=hypothetical_doc,
                from_cache=False,
                cache_key=cache_key,
            )
            
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return HyDEResult(
                original_query=query,
                hypothetical_doc=query,
                from_cache=False,
            )
    
    def should_use_hyde(self, query: str, complexity: str = "medium") -> bool:
        """
        Determine if HyDE should be used for this query.
        
        HyDE is most useful for:
        - Vague/ambiguous queries
        - Natural language queries without technical terms
        - Complex queries that need semantic expansion
        
        HyDE is NOT useful for:
        - Simple structural queries (조문 번호, 규정명)
        - Queries already containing technical terms
        
        Args:
            query: User's search query.
            complexity: Query complexity from Adaptive RAG.
            
        Returns:
            True if HyDE should be applied.
        """
        # Skip for simple queries
        if complexity == "simple":
            return False
        
        # Skip if query already contains regulatory terms
        regulatory_terms = ["규정", "규칙", "조", "항", "호", "세칙", "지침"]
        if any(term in query for term in regulatory_terms):
            return False
        
        # Use for vague/emotional queries
        vague_indicators = [
            "싶어", "싫어", "어떻게", "뭐야", "있어?", "할까",
            "해야", "가능", "수 있", "받고", "하고"
        ]
        if any(indicator in query for indicator in vague_indicators):
            return True
        
        # Use for complex queries
        if complexity == "complex":
            return True
        
        return False


class HyDESearcher:
    """
    Search using HyDE-enhanced queries.
    
    This class combines HyDE generation with the existing search infrastructure
    to improve retrieval for ambiguous queries.
    """
    
    def __init__(
        self,
        hyde_generator: HyDEGenerator,
        store: "IVectorStore",
    ):
        """
        Initialize HyDE searcher.
        
        Args:
            hyde_generator: HyDE document generator.
            store: Vector store for search.
        """
        self._hyde = hyde_generator
        self._store = store
    
    def search_with_hyde(
        self,
        query: str,
        filter: Optional["SearchFilter"] = None,
        top_k: int = 10,
    ) -> List["SearchResult"]:
        """
        Search using hypothetical document embedding.
        
        Args:
            query: Original user query.
            filter: Optional search filter.
            top_k: Number of results to return.
            
        Returns:
            List of search results.
        """
        from ..domain.value_objects import Query
        
        # Generate hypothetical document
        hyde_result = self._hyde.generate_hypothetical_doc(query)
        
        # Search with hypothetical document
        hyde_query = Query(text=hyde_result.hypothetical_doc)
        results = self._store.search(hyde_query, filter, top_k * 2)
        
        # Also search with original query and merge
        original_query = Query(text=query)
        original_results = self._store.search(original_query, filter, top_k * 2)
        
        # Merge results (deduplicate and re-score)
        return self._merge_results(results, original_results, top_k)
    
    def _merge_results(
        self,
        hyde_results: List["SearchResult"],
        original_results: List["SearchResult"],
        top_k: int,
    ) -> List["SearchResult"]:
        """Merge HyDE and original search results."""
        seen_ids = set()
        merged = []
        
        # Interleave results, giving slight preference to HyDE results
        for i in range(max(len(hyde_results), len(original_results))):
            if i < len(hyde_results):
                result = hyde_results[i]
                if result.chunk.id not in seen_ids:
                    seen_ids.add(result.chunk.id)
                    merged.append(result)
            
            if i < len(original_results):
                result = original_results[i]
                if result.chunk.id not in seen_ids:
                    seen_ids.add(result.chunk.id)
                    merged.append(result)
        
        # Re-sort by score and return top_k
        merged.sort(key=lambda x: -x.score)
        return merged[:top_k]
