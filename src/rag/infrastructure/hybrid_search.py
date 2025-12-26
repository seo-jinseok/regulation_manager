"""
Hybrid Search implementation combining BM25 and Dense retrieval.

Provides weighted fusion of sparse (keyword-based) and dense (embedding-based)
search results for improved retrieval quality.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class QueryType(Enum):
    """Types of queries for dynamic weight adjustment."""

    ARTICLE_REFERENCE = "article_reference"  # 제N조, 제N항, 제N호
    REGULATION_NAME = "regulation_name"  # OO규정, OO학칙
    NATURAL_QUESTION = "natural_question"  # 어떻게, 무엇, ?
    GENERAL = "general"  # Default


class QueryAnalyzer:
    """
    Analyzes query text to determine optimal BM25/Dense weights.

    Detects query patterns:
    - Article references: 제N조, 제N항 → favor BM25
    - Regulation names: OO규정, OO학칙 → balanced
    - Academic keywords: 휴학, 복학, 등록 등 → favor BM25
    - Natural questions: 어떻게, 무엇 → favor Dense
    """

    # Pattern for article/paragraph/item numbers
    ARTICLE_PATTERN = re.compile(
        r"제\s*\d+\s*조(?:\s*의\s*\d+)?|제\s*\d+\s*항|제\s*\d+\s*호"
    )

    # Pattern for regulation names
    REGULATION_PATTERN = re.compile(r"[가-힣]*(?:규정|학칙|내규|세칙|지침)")

    # Academic/procedural keywords that benefit from exact match
    ACADEMIC_KEYWORDS = [
        "휴학", "복학", "제적", "자퇴", "전과", "편입", "졸업", "입학",
        "등록", "수강", "장학", "학점", "성적", "시험", "출석", "학위",
        "논문", "석사", "박사", "교원", "교수", "조교", "학생회",
    ]

    # Question markers indicating natural language queries
    QUESTION_MARKERS = ["어떻게", "무엇", "왜", "언제", "어디", "누가", "어떤", "할까", "인가", "?"]

    # Weight presets for each query type: (bm25_weight, dense_weight)
    WEIGHT_PRESETS: Dict[QueryType, Tuple[float, float]] = {
        QueryType.ARTICLE_REFERENCE: (0.6, 0.4),  # Favor exact keyword match
        QueryType.REGULATION_NAME: (0.5, 0.5),  # Balanced (also used for academic keywords)
        QueryType.NATURAL_QUESTION: (0.4, 0.6),  # Slightly favor semantic, but still consider keywords
        QueryType.GENERAL: (0.5, 0.5),  # Balanced default (increased BM25 from 0.3)
    }

    def analyze(self, query: str) -> QueryType:
        """
        Analyze query text and determine its type.

        Args:
            query: The search query text.

        Returns:
            QueryType indicating the detected query pattern.
        """
        # Check for article references (highest priority for exact match)
        if self.ARTICLE_PATTERN.search(query):
            return QueryType.ARTICLE_REFERENCE

        # Check for regulation name patterns
        if self.REGULATION_PATTERN.search(query):
            return QueryType.REGULATION_NAME

        # Check for academic keywords (treat like regulation names for balanced search)
        if any(keyword in query for keyword in self.ACADEMIC_KEYWORDS):
            return QueryType.REGULATION_NAME

        # Check for natural language questions
        if any(marker in query for marker in self.QUESTION_MARKERS):
            return QueryType.NATURAL_QUESTION

        return QueryType.GENERAL

    def get_weights(self, query: str) -> Tuple[float, float]:
        """
        Get optimal BM25/Dense weights for the given query.

        Args:
            query: The search query text.

        Returns:
            Tuple of (bm25_weight, dense_weight).
        """
        query_type = self.analyze(query)
        return self.WEIGHT_PRESETS[query_type]


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
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        self.term_doc_freq: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.documents: Dict[str, str] = {}
        self.doc_metadata: Dict[str, Dict] = {}  # NEW: Store metadata

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Simple Korean + English tokenizer
        text = text.lower()
        # Split on whitespace and punctuation, keep Korean characters
        tokens = re.findall(r"[가-힣]+|[a-z0-9]+", text)
        return tokens

    def add_documents(
        self, documents: List[Tuple[str, str, Dict]]
    ) -> None:
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
        self.avg_doc_length = 0.0
        self.doc_count = 0


class HybridSearcher:
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
    ):
        """
        Initialize hybrid searcher.

        Args:
            bm25_weight: Default weight for BM25 scores (default: 0.3).
            dense_weight: Default weight for dense scores (default: 0.7).
            rrf_k: RRF ranking constant (default: 60).
            use_dynamic_weights: Enable query-based dynamic weighting (default: True).
        """
        self.bm25 = BM25()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.use_dynamic_weights = use_dynamic_weights
        self._query_analyzer = QueryAnalyzer()

    def add_documents(self, documents: List[Tuple[str, str, Dict]]) -> None:
        """
        Add documents to the BM25 index.

        Args:
            documents: List of (doc_id, content, metadata) tuples.
        """
        self.bm25.add_documents(documents)

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
            query_text: Optional query for dynamic weight calculation.
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

        scores: Dict[str, float] = defaultdict(float)
        doc_data: Dict[str, ScoredDocument] = {}

        # Add sparse results with RRF scoring
        for rank, doc in enumerate(sparse_results, start=1):
            rrf_score = 1 / (self.rrf_k + rank)
            scores[doc.doc_id] += bm25_w * rrf_score
            doc_data[doc.doc_id] = doc

        # Add dense results with RRF scoring
        for rank, doc in enumerate(dense_results, start=1):
            rrf_score = 1 / (self.rrf_k + rank)
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
        Perform BM25 sparse search.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of sparse search results.
        """
        return self.bm25.search(query, top_k)

    def clear(self) -> None:
        """Clear the BM25 index."""
        self.bm25.clear()
