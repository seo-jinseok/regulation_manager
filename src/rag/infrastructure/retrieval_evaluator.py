"""
Retrieval Evaluator for Corrective RAG.

Evaluates the relevance of search results and determines
if re-retrieval or query rewriting is needed.
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..domain.entities import SearchResult


class RetrievalEvaluator:
    """
    Evaluate relevance of search results for Corrective RAG.
    
    Uses heuristic scoring based on:
    - Top result score
    - Keyword overlap between query and results
    - Result diversity
    """
    
    # Threshold below which correction is triggered
    RELEVANCE_THRESHOLD = 0.4
    
    # Minimum results needed for reliable evaluation
    MIN_RESULTS_FOR_EVAL = 2
    
    def __init__(self, relevance_threshold: float = None):
        """
        Initialize evaluator.
        
        Args:
            relevance_threshold: Custom threshold (default: 0.4)
        """
        self.threshold = relevance_threshold or self.RELEVANCE_THRESHOLD
    
    def evaluate(self, query: str, results: List["SearchResult"]) -> float:
        """
        Evaluate search result relevance.
        
        Args:
            query: Original search query
            results: List of SearchResult objects
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not results:
            return 0.0
        
        # Component scores
        top_score = self._evaluate_top_score(results)
        keyword_score = self._evaluate_keyword_overlap(query, results)
        diversity_score = self._evaluate_diversity(results)
        
        # Weighted combination
        final_score = (
            top_score * 0.5 +
            keyword_score * 0.3 +
            diversity_score * 0.2
        )
        
        return min(1.0, max(0.0, final_score))
    
    def needs_correction(self, query: str, results: List["SearchResult"]) -> bool:
        """
        Check if search results need correction.
        
        Args:
            query: Original search query
            results: List of SearchResult objects
            
        Returns:
            True if correction (re-retrieval) is recommended
        """
        if not results:
            return True
        
        if len(results) < self.MIN_RESULTS_FOR_EVAL:
            return True
        
        score = self.evaluate(query, results)
        return score < self.threshold
    
    def _evaluate_top_score(self, results: List["SearchResult"]) -> float:
        """Evaluate based on top result's retrieval score."""
        if not results:
            return 0.0
        
        # Top result score (already normalized 0-1 from reranker)
        return results[0].score
    
    def _evaluate_keyword_overlap(
        self, query: str, results: List["SearchResult"]
    ) -> float:
        """Evaluate keyword overlap between query and results."""
        if not results:
            return 0.0
        
        # Tokenize query (simple Korean tokenization)
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return 0.5  # Neutral if no terms
        
        # Count matches in top results
        match_count = 0
        check_results = results[:3]  # Check top 3
        
        for result in check_results:
            result_text = f"{result.chunk.title} {result.chunk.text[:200]}"
            result_terms = set(self._tokenize(result_text))
            
            if query_terms & result_terms:
                match_count += 1
        
        return match_count / len(check_results)
    
    def _evaluate_diversity(self, results: List["SearchResult"]) -> float:
        """Evaluate result diversity (different regulations)."""
        if not results:
            return 0.0
        
        # Count unique regulations
        unique_regs = set()
        for result in results[:5]:
            if result.chunk.rule_code:
                unique_regs.add(result.chunk.rule_code.split("-")[0])
        
        # More diverse is better (up to 3 unique sources)
        return min(1.0, len(unique_regs) / 3)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple Korean tokenization (space + basic filtering)."""
        import re
        
        # Remove punctuation and split
        text = re.sub(r"[^\w\s가-힣]", " ", text)
        tokens = text.split()
        
        # Filter short tokens and stopwords
        stopwords = {"은", "는", "이", "가", "을", "를", "의", "에", "로", "으로", "하다"}
        return [t for t in tokens if len(t) >= 2 and t not in stopwords]


class CorrectionStrategy:
    """Strategy for correcting low-relevance searches."""
    
    def __init__(self, query_analyzer=None):
        """
        Initialize correction strategy.
        
        Args:
            query_analyzer: QueryAnalyzer for query rewriting
        """
        self._query_analyzer = query_analyzer
    
    def get_corrected_query(self, original_query: str) -> Optional[str]:
        """
        Generate a corrected query for re-retrieval.
        
        Args:
            original_query: The original search query
            
        Returns:
            Corrected query or None if no correction possible
        """
        if not self._query_analyzer:
            return None
        
        # Try intent-based expansion
        expanded = self._query_analyzer.expand_query(original_query)
        if expanded != original_query:
            return expanded
        
        # Try LLM-based rewriting if available
        try:
            rewrite_result = self._query_analyzer.rewrite_query_with_info(original_query)
            if rewrite_result.rewritten != original_query:
                return rewrite_result.rewritten
        except Exception:
            pass
        
        return None
