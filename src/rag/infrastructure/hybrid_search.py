"""
Hybrid Search implementation combining BM25 and Dense retrieval.

Provides weighted fusion of sparse (keyword-based) and dense (embedding-based)
search results for improved retrieval quality.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..domain.repositories import IHybridSearcher
from .query_analyzer import (
    Audience,
    IntentMatch,
    IntentRule,
    QueryAnalyzer,
    QueryRewriteResult,
    QueryType,
)

if TYPE_CHECKING:
    from ..domain.repositories import ILLMClient


@dataclass
class ScoredDocument:
    """A document with its relevance score."""

    doc_id: str
    score: float
    content: str
    metadata: Dict


class BM25:
    """
    Simple BM25 implementation for sparse retrieval.

    BM25 parameters:
    - k1: Term frequency saturation (default: 1.5)
    - b: Document length normalization (default: 0.75)
    - tokenize_mode: "simple" (regex-based) or "morpheme" (Korean morphological analysis)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenize_mode: str = "simple",
    ):
        self.k1 = k1
        self.b = b
        self.tokenize_mode = tokenize_mode
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        self.term_doc_freq: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.documents: Dict[str, str] = {}
        self.doc_metadata: Dict[str, Dict] = {}  # NEW: Store metadata

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms based on tokenize_mode."""
        if self.tokenize_mode == "morpheme":
            return self._tokenize_morpheme(text)
        return self._tokenize_simple(text)

    def _tokenize_simple(self, text: str) -> List[str]:
        """Simple regex-based tokenizer for Korean + English."""
        text = text.lower()
        # Split on whitespace and punctuation, keep Korean characters
        tokens = re.findall(r"[가-힣]+|[a-z0-9]+", text)
        return tokens

    def _tokenize_morpheme(self, text: str) -> List[str]:
        """
        Korean morphological analysis tokenizer.
        
        Splits compound Korean words for better recall.
        Falls back to simple tokenizer if no morpheme analyzer available.
        """
        text = text.lower()
        tokens = []
        
        # 기본 토큰 추출
        base_tokens = re.findall(r"[가-힣]+|[a-z0-9]+", text)
        
        # 한글 복합어 분리 패턴 (간단한 규칙 기반)
        # 실제로는 형태소 분석기(KoNLPy 등)를 사용할 수 있음
        compound_patterns = [
            # 접미사 패턴: ~신청, ~규정, ~절차, ~처리
            (r"(.+?)(신청|규정|절차|처리|관리|운영|지원|위원회)$", [1, 2]),
            # 접두사 패턴: 육아~, 연구~, 교원~
            (r"^(육아|연구|교원|학생|직원|교직원)(.+)$", [1, 2]),
        ]
        
        for token in base_tokens:
            if len(token) <= 2:
                # 짧은 토큰은 그대로 추가
                tokens.append(token)
                continue
                
            # 복합어 분리 시도
            split_found = False
            for pattern, groups in compound_patterns:
                match = re.match(pattern, token)
                if match:
                    for g in groups:
                        part = match.group(g)
                        if part and len(part) >= 2:
                            tokens.append(part)
                    split_found = True
                    break
            
            if not split_found:
                # 분리 실패 시 원본 토큰 추가
                tokens.append(token)
        
        return tokens

    def add_documents(self, documents: List[Tuple[str, str, Dict]]) -> None:
        """
        Add documents to the index.

        Args:
            documents: List of (doc_id, content, metadata) tuples.
        """
        for doc_id, content, metadata in documents:
            tokens = self._tokenize(content)
            self.doc_lengths[doc_id] = len(tokens)
            self.documents[doc_id] = content
            self.doc_metadata[doc_id] = metadata  # NEW: Store metadata

            # Count term frequencies
            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            # Update inverted index
            for term, freq in term_freq.items():
                if doc_id not in self.inverted_index[term]:
                    self.term_doc_freq[term] += 1
                self.inverted_index[term][doc_id] = freq

        # Update statistics
        self.doc_count = len(self.doc_lengths)
        if self.doc_count > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.doc_count

    def search(self, query: str, top_k: int = 10) -> List[ScoredDocument]:
        """
        Search for documents matching the query.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of ScoredDocument sorted by relevance.
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            # Calculate IDF
            doc_freq = self.term_doc_freq[term]
            idf = self._calculate_idf(doc_freq)

            # Score each document containing this term
            for doc_id, term_freq in self.inverted_index[term].items():
                doc_length = self.doc_lengths[doc_id]
                tf_component = self._calculate_tf(term_freq, doc_length)
                scores[doc_id] += idf * tf_component

        # Sort by score and return top_k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            ScoredDocument(
                doc_id=doc_id,
                score=score,
                content=self.documents.get(doc_id, ""),
                metadata=self.doc_metadata.get(doc_id, {}),
            )
            for doc_id, score in sorted_docs
        ]

    def _calculate_idf(self, doc_freq: int) -> float:
        """Calculate inverse document frequency."""
        import math

        if doc_freq == 0:
            return 0.0
        return math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def _calculate_tf(self, term_freq: int, doc_length: int) -> float:
        """Calculate term frequency component."""
        if self.avg_doc_length == 0:
            return 0.0
        length_norm = 1 - self.b + self.b * (doc_length / self.avg_doc_length)
        return (term_freq * (self.k1 + 1)) / (term_freq + self.k1 * length_norm)

    def clear(self) -> None:
        """Clear all indexed documents."""
        self.doc_lengths.clear()
        self.term_doc_freq.clear()
        self.inverted_index.clear()
        self.documents.clear()
        self.doc_metadata.clear()
        self.avg_doc_length = 0.0
        self.doc_count = 0

    def save_index(self, path: str) -> None:
        """
        Save BM25 index to disk using pickle.
        
        Args:
            path: File path to save the index.
        """
        import pickle
        from pathlib import Path
        
        index_data = {
            'inverted_index': dict(self.inverted_index),
            'doc_lengths': self.doc_lengths,
            'term_doc_freq': dict(self.term_doc_freq),
            'documents': self.documents,
            'doc_metadata': self.doc_metadata,
            'avg_doc_length': self.avg_doc_length,
            'doc_count': self.doc_count,
            'k1': self.k1,
            'b': self.b,
            'tokenize_mode': self.tokenize_mode,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self, path: str) -> bool:
        """
        Load BM25 index from disk.
        
        Args:
            path: File path to load the index from.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        import pickle
        from pathlib import Path
        
        if not Path(path).exists():
            return False
        
        try:
            with open(path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.inverted_index = defaultdict(dict, index_data['inverted_index'])
            self.doc_lengths = index_data['doc_lengths']
            self.term_doc_freq = defaultdict(int, index_data['term_doc_freq'])
            self.documents = index_data['documents']
            self.doc_metadata = index_data['doc_metadata']
            self.avg_doc_length = index_data['avg_doc_length']
            self.doc_count = index_data['doc_count']
            self.k1 = index_data['k1']
            self.b = index_data['b']
            self.tokenize_mode = index_data['tokenize_mode']
            
            return True
        except Exception:
            return False


class HybridSearcher(IHybridSearcher):
    """
    Combines BM25 and dense retrieval for hybrid search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from
    sparse and dense retrieval methods.

    Supports dynamic weight adjustment based on query characteristics.
    """

    def __init__(
        self,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        rrf_k: int = 60,
        use_dynamic_weights: bool = True,
        use_dynamic_rrf_k: bool = False,
        synonyms_path: Optional[str] = None,
        intents_path: Optional[str] = None,
        index_cache_path: Optional[str] = None,
    ):
        """
        Initialize hybrid searcher.

        Args:
            bm25_weight: Default weight for BM25 scores (default: 0.3).
            dense_weight: Default weight for dense scores (default: 0.7).
            rrf_k: RRF ranking constant (default: 60).
            use_dynamic_weights: Enable query-based dynamic weighting (default: True).
            use_dynamic_rrf_k: Enable query-based dynamic RRF k value (default: False).
            index_cache_path: Path to cache BM25 index (default: None).
        """
        self.bm25 = BM25()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.use_dynamic_weights = use_dynamic_weights
        self.use_dynamic_rrf_k = use_dynamic_rrf_k
        self.index_cache_path = index_cache_path
        if synonyms_path is None:
            try:
                from ..config import get_config

                synonyms_path = get_config().synonyms_path
            except Exception:
                synonyms_path = None
        if intents_path is None:
            try:
                from ..config import get_config

                intents_path = get_config().intents_path
            except Exception:
                intents_path = None
        self._query_analyzer = QueryAnalyzer(
            synonyms_path=synonyms_path,
            intents_path=intents_path,
        )

    def set_llm_client(self, llm_client: Optional["ILLMClient"]) -> None:
        """
        Set LLM client for query rewriting.

        Args:
            llm_client: LLM client to use for query rewriting.
        """
        self._query_analyzer._llm_client = llm_client

    def add_documents(self, documents: List[Tuple[str, str, Dict]]) -> None:
        """
        Add documents to the BM25 index.

        If index_cache_path is set and cache exists, loads from cache.
        Otherwise builds index and saves to cache.

        Args:
            documents: List of (doc_id, content, metadata) tuples.
        """
        # Try loading from cache
        if self.index_cache_path and self.bm25.load_index(self.index_cache_path):
            return
        
        # Build index
        self.bm25.add_documents(documents)
        
        # Save to cache
        if self.index_cache_path:
            self.bm25.save_index(self.index_cache_path)

    def get_dynamic_rrf_k(self, query_text: str) -> int:
        """
        Get dynamic RRF k value based on query type.

        Lower k values prioritize top-ranked documents more strongly,
        which is useful for precise queries (article references).
        Higher k values give more balanced ranking across positions,
        better for exploratory natural language queries.

        Args:
            query_text: The search query.

        Returns:
            RRF k value (30-80 range).
        """
        if not self.use_dynamic_rrf_k:
            return self.rrf_k

        query_type = self._query_analyzer.analyze(query_text)

        # 조문 참조: 정확한 매칭 중요 → 낮은 k (30-40)
        if query_type == QueryType.ARTICLE_REFERENCE:
            return 35

        # 규정명 검색: 중간 k
        if query_type == QueryType.REGULATION_NAME:
            return 45

        # 자연어 질문 / 의도 기반: 다양한 결과 필요 → 높은 k (60-80)
        if query_type in (QueryType.NATURAL_QUESTION, QueryType.INTENT):
            return 70

        # 기본값 (GENERAL)
        return self.rrf_k

    def fuse_results(
        self,
        sparse_results: List[ScoredDocument],
        dense_results: List[ScoredDocument],
        top_k: int = 10,
        query_text: Optional[str] = None,
    ) -> List[ScoredDocument]:
        """
        Fuse sparse and dense results using weighted RRF.

        Args:
            sparse_results: Results from BM25 search.
            dense_results: Results from dense/embedding search.
            top_k: Maximum number of results.
            query_text: Optional query for dynamic weight and k calculation.
                If provided and use_dynamic_weights is True, weights are
                automatically adjusted based on query characteristics.

        Returns:
            Fused results sorted by combined score.
        """
        # Determine weights: dynamic or static
        if self.use_dynamic_weights and query_text:
            bm25_w, dense_w = self._query_analyzer.get_weights(query_text)
        else:
            bm25_w, dense_w = self.bm25_weight, self.dense_weight

        # Determine RRF k: dynamic or static
        if query_text:
            effective_k = self.get_dynamic_rrf_k(query_text)
        else:
            effective_k = self.rrf_k

        scores: Dict[str, float] = defaultdict(float)
        doc_data: Dict[str, ScoredDocument] = {}

        # Add sparse results with RRF scoring
        for rank, doc in enumerate(sparse_results, start=1):
            rrf_score = 1 / (effective_k + rank)
            scores[doc.doc_id] += bm25_w * rrf_score
            doc_data[doc.doc_id] = doc

        # Add dense results with RRF scoring
        for rank, doc in enumerate(dense_results, start=1):
            rrf_score = 1 / (effective_k + rank)
            scores[doc.doc_id] += dense_w * rrf_score
            if doc.doc_id not in doc_data:
                doc_data[doc.doc_id] = doc

        # Sort by combined score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            ScoredDocument(
                doc_id=doc_id,
                score=score,
                content=doc_data[doc_id].content,
                metadata=doc_data[doc_id].metadata,
            )
            for doc_id, score in sorted_ids
            if doc_id in doc_data
        ]

    def search_sparse(self, query: str, top_k: int = 20) -> List[ScoredDocument]:
        """
        Perform BM25 sparse search with query expansion.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of sparse search results.
        """
        # Expand query with synonyms for better recall
        expanded_query = self._query_analyzer.expand_query(query)
        return self.bm25.search(expanded_query, top_k)

    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms.

        Args:
            query: The original search query.

        Returns:
            Expanded query with synonyms appended.
        """
        return self._query_analyzer.expand_query(query)

    def clear(self) -> None:
        """Clear the BM25 index."""
        self.bm25.clear()
