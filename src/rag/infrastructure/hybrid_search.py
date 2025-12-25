"""
Hybrid Search implementation combining BM25 and Dense retrieval.

Provides weighted fusion of sparse (keyword-based) and dense (embedding-based)
search results for improved retrieval quality.
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ScoredDocument:
    """A document with its relevance score."""

    doc_id: str
    score: float
    content: str
    metadata: Dict


class BM25:
    """
    BM25 sparse retrieval implementation.

    Standard BM25 algorithm for keyword-based document ranking.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with tuning parameters.

        Args:
            k1: Term frequency saturation parameter (default: 1.5).
            b: Length normalization parameter (default: 0.75).
        """
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        self.term_doc_freq: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.documents: Dict[str, str] = {}

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
                metadata={},
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
    """

    def __init__(
        self,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid searcher.

        Args:
            bm25_weight: Weight for BM25 scores (default: 0.3).
            dense_weight: Weight for dense scores (default: 0.7).
            rrf_k: RRF ranking constant (default: 60).
        """
        self.bm25 = BM25()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

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
    ) -> List[ScoredDocument]:
        """
        Fuse sparse and dense results using weighted RRF.

        Args:
            sparse_results: Results from BM25 search.
            dense_results: Results from dense/embedding search.
            top_k: Maximum number of results.

        Returns:
            Fused results sorted by combined score.
        """
        scores: Dict[str, float] = defaultdict(float)
        doc_data: Dict[str, ScoredDocument] = {}

        # Add sparse results with RRF scoring
        for rank, doc in enumerate(sparse_results, start=1):
            rrf_score = 1 / (self.rrf_k + rank)
            scores[doc.doc_id] += self.bm25_weight * rrf_score
            doc_data[doc.doc_id] = doc

        # Add dense results with RRF scoring
        for rank, doc in enumerate(dense_results, start=1):
            rrf_score = 1 / (self.rrf_k + rank)
            scores[doc.doc_id] += self.dense_weight * rrf_score
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
