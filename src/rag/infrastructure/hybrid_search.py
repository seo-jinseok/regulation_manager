"""
Hybrid Search implementation combining BM25 and Dense retrieval.

Provides weighted fusion of sparse (keyword-based) and dense (embedding-based)
search results for improved retrieval quality.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..domain.repositories import ILLMClient


class QueryType(Enum):
    """Types of queries for dynamic weight adjustment."""

    ARTICLE_REFERENCE = "article_reference"  # 제N조, 제N항, 제N호
    REGULATION_NAME = "regulation_name"  # OO규정, OO학칙
    NATURAL_QUESTION = "natural_question"  # 어떻게, 무엇, ?
    INTENT = "intent"  # 자연어 의도 표현
    GENERAL = "general"  # Default


class QueryAnalyzer:
    """
    Analyzes query text to determine optimal BM25/Dense weights.

    Detects query patterns:
    - Article references: 제N조, 제N항 → favor BM25
    - Regulation names: OO규정, OO학칙 → balanced
    - Academic keywords: 휴학, 복학, 등록 등 → favor BM25
    - Natural questions: 어떻게, 무엇 → favor Dense
    - Intent expressions: ~하기 싫어, 그만두고 싶어 → favor Dense
    
    Also provides query expansion with synonyms for better recall.
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
        QueryType.INTENT: (0.35, 0.65),  # Intent-heavy queries lean semantic
        QueryType.GENERAL: (0.5, 0.5),  # Balanced default (increased BM25 from 0.3)
    }

    # Synonym dictionary for query expansion (학사 용어)
    SYNONYMS: Dict[str, List[str]] = {
        # 학과/전공 관련
        "폐과": ["학과 폐지", "전공 폐지", "학부 폐지", "과 폐지"],
        "모집정지": ["신입생 모집 정지", "학생 모집 중단", "모집 중단", "입학 정지"],
        "정원": ["입학정원", "모집정원", "학생정원"],
        # 학적 관련
        "휴학": ["휴학원", "휴학 신청", "학업 중단"],
        "복학": ["복학원", "복학 신청", "학업 복귀"],
        "제적": ["학적 상실", "등록금 미납 제적"],
        "자퇴": ["자퇴원", "자퇴 신청", "자진 퇴학"],
        "전과": ["전과 신청", "학과 이동", "전공 변경"],
        "편입": ["편입학", "편입 신청"],
        # 등록/재정 관련
        "등록금": ["수업료", "납입금"],
        "장학금": ["장학", "학비 지원"],
        "분납": ["등록금 분납", "분할 납부"],
        # 학위/졸업 관련
        "졸업": ["학위 수여", "졸업 요건"],
        "학위": ["학사 학위", "석사 학위", "박사 학위"],
        "논문": ["학위 논문", "졸업 논문"],
        # 교원 관련
        "교원": ["교수", "교직원", "전임 교원"],
        "교수": ["교원", "교직원"],
        "임용": ["교원 임용", "교수 채용"],
        "재임용": ["계약 갱신", "임기 연장"],
        "승진": ["교원 승진", "직급 상승"],
    }

    # Intent patterns for natural language queries
    INTENT_PATTERNS = [
        (re.compile(r"(학교|출근|근무).*(가기|출근).*싫"), ["휴직", "휴가", "연구년", "안식년"]),
        (re.compile(r"(월급|급여|보수|연봉).*더.*받.*싶"), ["보수", "수당", "승급", "호봉"]),
        (re.compile(r"(그만두고\s*싶|그만\s*두고\s*싶|퇴직|사직)"), ["퇴직", "사직", "명예퇴직"]),
        (re.compile(r"(수업|강의).*안.*하.*싶"), ["휴강", "보강", "강의", "면제"]),
    ]

    # LLM Query Rewriting prompt
    QUERY_REWRITE_PROMPT = """당신은 대학 규정 검색 시스템의 쿼리 분석기입니다.
사용자의 자연어 질문을 분석하여 규정 검색에 적합한 키워드를 추출하세요.

## 지침
1. 사용자의 **실제 의도**를 파악하세요.
2. 규정집에서 검색할 수 있는 **구체적인 키워드**로 변환하세요.
3. 결과는 **키워드만** 공백으로 구분하여 출력하세요.
4. 설명이나 부연 없이 키워드만 출력하세요.

## 예시
- "학교에 가기 싫어" → "휴직 휴가 연구년 안식년"
- "월급 더 받고 싶어" → "보수 수당 승급 호봉"
- "그만두고 싶어" → "퇴직 사직 명예퇴직"
- "수업 안 하고 싶어" → "휴강 보강 강의 면제"

## 출력 형식
키워드1 키워드2 키워드3"""

    def __init__(self, llm_client: Optional["ILLMClient"] = None):
        """
        Initialize QueryAnalyzer.
        
        Args:
            llm_client: Optional LLM client for query rewriting.
                       If not provided, falls back to synonym-based expansion.
        """
        self._llm_client = llm_client
        self._cache: Dict[str, str] = {}  # Query rewrite cache

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite natural language query to regulation search keywords using LLM.
        
        Args:
            query: The original natural language query.
        
        Returns:
            Rewritten query with search keywords.
            Falls back to expand_query() on LLM failure.
        """
        # Check cache first
        if query in self._cache:
            return self._cache[query]
        
        intent_keywords = self._intent_keywords(query)

        # No LLM client: fall back to synonym expansion
        if not self._llm_client:
            return self.expand_query(query)
        
        # Call LLM for rewriting
        try:
            response = self._llm_client.generate(
                system_prompt=self.QUERY_REWRITE_PROMPT,
                user_message=query,
                temperature=0.0,
            )
            # Clean up response (remove extra whitespace, newlines)
            rewritten = response.strip().replace("\n", " ")
            if intent_keywords:
                rewritten = self._merge_keywords(rewritten, intent_keywords)
            # Cache the result
            self._cache[query] = rewritten
            return rewritten
        except Exception:
            # On any error, fall back to synonym expansion
            return self.expand_query(query)

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

        # Check for intent expressions before academic keywords
        if self._intent_keywords(query):
            return QueryType.INTENT

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

    def has_synonyms(self, query: str) -> bool:
        """Check if query contains any terms with synonyms."""
        cleaned = self.clean_query(query)
        return any(term in cleaned for term in self.SYNONYMS)

    def get_weights(self, query: str) -> Tuple[float, float]:
        """
        Get optimal BM25/Dense weights for the given query.

        Args:
            query: The search query text.

        Returns:
            Tuple of (bm25_weight, dense_weight).
        """
        query_type = self.analyze(query)
        bm25_w, dense_w = self.WEIGHT_PRESETS[query_type]
        
        # If query has synonyms, boost BM25 weight for better keyword matching
        if self.has_synonyms(query):
            # Shift weight towards BM25 (e.g., 0.5/0.5 -> 0.7/0.3)
            bm25_w = min(0.8, bm25_w + 0.2)
            dense_w = max(0.2, dense_w - 0.2)
        
        return bm25_w, dense_w

    # Stopwords to remove from queries (너무 일반적인 단어들)
    STOPWORDS = [
        # 규정/법률 관련 일반어 (조사 포함 형태도)
        "규정", "규정은", "규정이", "규정을", "규정에", "규정의",
        "규칙", "조례", "법", "관련", "내용", "사항", "경우",
        "대해", "대한", "대해서",
        # 질문 관련
        "뭐", "뭔가", "무엇", "어떤", "어떻게", "왜", "언제", "어디",
        "알려줘", "알려주세요", "설명해줘", "설명해주세요",
        # 기타
        "좀", "것", "수", "때", "중", "있나요", "있어요", "있습니까",
    ]

    def clean_query(self, query: str) -> str:
        """
        Clean query by removing stopwords and normalizing.

        Args:
            query: The original search query text.

        Returns:
            Cleaned query with stopwords removed.
        """
        # Remove question mark and other punctuation
        cleaned = query.replace("?", "").replace("!", "").replace(".", "").strip()
        
        # Remove stopwords (exact match)
        words = cleaned.split()
        filtered_words = [w for w in words if w not in self.STOPWORDS]
        
        # If all words were filtered, return original (without punctuation)
        if not filtered_words:
            return cleaned
        
        return " ".join(filtered_words)

    def expand_query(self, query: str) -> str:
        """
        Clean and expand query with synonyms for better recall.

        Args:
            query: The original search query text.

        Returns:
            Cleaned and expanded query with synonyms appended.
        """
        # First clean the query
        cleaned = self.clean_query(query)
        tokens = cleaned.split() if cleaned else []

        intent_keywords = self._intent_keywords(cleaned or query)
        if intent_keywords:
            tokens = self._merge_token_list(tokens, intent_keywords)

        expansions = self._get_synonym_expansions(cleaned)
        if expansions:
            tokens = self._merge_token_list(tokens, expansions)

        return " ".join(tokens) if tokens else cleaned

    def _get_synonym_expansions(self, cleaned_query: str) -> List[str]:
        """Collect synonym expansions for the cleaned query."""
        expansions: List[str] = []
        for term, synonyms in self.SYNONYMS.items():
            if term in cleaned_query:
                # Add first 2 synonyms to avoid over-expansion
                expansions.extend(synonyms[:2])
        return expansions

    def _intent_keywords(self, query: str) -> List[str]:
        """Return intent-based keywords for colloquial queries."""
        for pattern, keywords in self.INTENT_PATTERNS:
            if pattern.search(query):
                return list(keywords)
        return []

    def _merge_token_list(self, base_tokens: List[str], extra_tokens: List[str]) -> List[str]:
        """Merge keyword lists while preserving order and uniqueness."""
        seen = set(base_tokens)
        merged = list(base_tokens)
        for token in extra_tokens:
            if token not in seen:
                merged.append(token)
                seen.add(token)
        return merged

    def _merge_keywords(self, keyword_string: str, extra_tokens: List[str]) -> str:
        """Merge keyword string with extra tokens."""
        base_tokens = keyword_string.split()
        merged = self._merge_token_list(base_tokens, extra_tokens)
        return " ".join(merged)


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
