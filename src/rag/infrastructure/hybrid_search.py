"""
Hybrid Search implementation combining BM25 and Dense retrieval.

Provides weighted fusion of sparse (keyword-based) and dense (embedding-based)
search results for improved retrieval quality.
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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


class Audience(Enum):
    """Target audience for the regulation query."""

    ALL = "all"
    STUDENT = "student"
    FACULTY = "faculty"
    STAFF = "staff"


@dataclass(frozen=True)
class QueryRewriteResult:
    """Detailed result of query rewriting."""

    original: str
    rewritten: str
    method: str  # "llm" or "rules"
    from_cache: bool
    used_llm: bool
    used_intent: bool
    used_synonyms: bool
    fallback: bool
    matched_intents: List[str]


@dataclass(frozen=True)
class IntentRule:
    """Intent rule with triggers/patterns and expansion keywords."""

    intent_id: str
    label: str
    keywords: List[str]
    patterns: List[re.Pattern]
    triggers: List[str]


@dataclass(frozen=True)
class IntentMatch:
    """Matched intent with score and keywords."""

    intent_id: str
    label: str
    keywords: List[str]
    score: int


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
        "휴학",
        "복학",
        "제적",
        "자퇴",
        "전과",
        "편입",
        "졸업",
        "입학",
        "등록",
        "수강",
        "장학",
        "학점",
        "성적",
        "시험",
        "출석",
        "학위",
        "논문",
        "석사",
        "박사",
        "교원",
        "교수",
        "조교",
        "학생회",
    ]

    # Question markers indicating natural language queries
    QUESTION_MARKERS = [
        "어떻게",
        "무엇",
        "왜",
        "언제",
        "어디",
        "누가",
        "어떤",
        "할까",
        "인가",
        "?",
    ]

    # Audience keywords
    FACULTY_KEYWORDS = [
        "교수",
        "교원",
        "강사",
        "전임",
        "안식년",
        "연구년",
        "책임시수",
        "업적평가",
    ]
    STUDENT_KEYWORDS = [
        "학생",
        "학부",
        "대학원",
        "수강",
        "성적",
        "졸업",
        "휴학",
        "복학",
        "장학",
        "등록금",
        "학점",
    ]
    STAFF_KEYWORDS = ["직원", "행정", "사무", "참사", "주사", "승진", "전보"]
    AMBIGUOUS_AUDIENCE_KEYWORDS = ["징계", "처분", "위반", "제재", "윤리", "고충"]

    # Weight presets for each query type: (bm25_weight, dense_weight)
    WEIGHT_PRESETS: Dict[QueryType, Tuple[float, float]] = {
        QueryType.ARTICLE_REFERENCE: (0.6, 0.4),  # Favor exact keyword match
        QueryType.REGULATION_NAME: (
            0.5,
            0.5,
        ),  # Balanced (also used for academic keywords)
        QueryType.NATURAL_QUESTION: (
            0.4,
            0.6,
        ),  # Slightly favor semantic, but still consider keywords
        QueryType.INTENT: (0.35, 0.65),  # Intent-heavy queries lean semantic
        QueryType.GENERAL: (0.5, 0.5),  # Balanced default (increased BM25 from 0.3)
    }

    # Synonym dictionary for query expansion (minimal seed).
    # Prefer external dictionaries via RAG_SYNONYMS_PATH for full coverage.
    SYNONYMS: Dict[str, List[str]] = {
        "폐과": ["학과 폐지", "전공 폐지"],
        "휴학": ["휴학원", "휴학 신청"],
        "복학": ["복학원", "복학 신청"],
        "제적": ["학적 상실"],
        "자퇴": ["자퇴원", "자퇴 신청"],
        "전과": ["전과 신청"],
        "편입": ["편입학"],
        "등록금": ["수업료"],
        "장학금": ["장학"],
        "교수": ["교원", "교직원"],
    }

    # Intent patterns for natural language queries (minimal seed).
    # Prefer external dictionaries via RAG_INTENTS_PATH for full coverage.
    INTENT_PATTERNS = [
        (
            re.compile(r"(학교|출근|근무).*(가기|출근).*싫"),
            ["휴직", "휴가", "연구년", "안식년"],
        ),
        (
            re.compile(r"(그만두고\s*싶|그만\s*두고\s*싶|퇴직|사직)"),
            ["퇴직", "사직", "명예퇴직"],
        ),
        (re.compile(r"(수업|강의).*안.*하.*싶"), ["휴강", "보강", "강의"]),
    ]
    INTENT_MAX_MATCHES = 3

    # LLM Query Rewriting prompt
    QUERY_REWRITE_PROMPT = """당신은 대학 규정 검색 시스템의 쿼리 분석기입니다.
사용자의 자연어 질문을 분석하여 규정 검색에 적합한 키워드를 추출하세요.

## 지침
1. 사용자의 **실제 의도**를 파악하세요.
2. 규정집에서 검색할 수 있는 **구체적인 키워드**로 변환하세요.
3. 결과는 **키워드만** 공백으로 구분하여 출력하세요.
4. 설명이나 부연 없이 키워드만 출력하세요.
5. **감정적 표현**(싫다, 화난다, 짜증 등)은 관련 **규정 절차**로 해석하세요.
6. **불만/고충**은 고충처리, 민원, 신고 절차로 연결하세요.
7. **"화를 낸다"**는 "화재"가 아니라 "분노/감정적 행동"을 의미합니다.
8. 교수/직원의 부적절한 행동 관련 질문은 **고충처리, 윤리, 징계** 키워드로 연결하세요.

## 예시
- "학교에 가기 싫어" → "휴직 휴가 연구년 안식년"
- "월급 더 받고 싶어" → "보수 수당 승급 호봉"
- "그만두고 싶어" → "퇴직 사직 명예퇴직"
- "수업 안 하고 싶어" → "휴강 보강 강의 면제"
- "교수가 정치적 발언을 해" → "교원윤리 고충처리 학생권리 신고"
- "교수가 자주 화를 내" → "고충처리 민원 교원윤리 징계"
- "교수님 태도가 이상해" → "고충처리 민원 교원윤리 학생권리"
- "부당한 대우 받았어" → "고충처리 민원 이의신청 학생권리"
- "신고하고 싶어" → "신고 고충처리 민원 징계"

## 출력 형식
키워드1 키워드2 키워드3"""

    def __init__(
        self,
        llm_client: Optional["ILLMClient"] = None,
        synonyms_path: Optional[str] = None,
        intents_path: Optional[str] = None,
    ):
        """
        Initialize QueryAnalyzer.

        Args:
            llm_client: Optional LLM client for query rewriting.
                       If not provided, falls back to synonym-based expansion.
        """
        self._llm_client = llm_client
        self._cache: Dict[str, QueryRewriteResult] = {}  # Query rewrite cache
        self._synonyms = self._load_synonyms(synonyms_path)
        self._intent_rules = self._load_intents(intents_path)

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite natural language query to regulation search keywords using LLM.

        Args:
            query: The original natural language query.

        Returns:
            Rewritten query with search keywords.
            Falls back to expand_query() on LLM failure.
        """
        return self.rewrite_query_with_info(query).rewritten

    def rewrite_query_with_info(self, query: str) -> QueryRewriteResult:
        """
        Rewrite query and return detailed metadata.

        Args:
            query: The original natural language query.

        Returns:
            QueryRewriteResult containing rewrite metadata.
        """
        # Check cache first
        if query in self._cache:
            cached = self._cache[query]
            return QueryRewriteResult(
                original=cached.original,
                rewritten=cached.rewritten,
                method=cached.method,
                from_cache=True,
                used_llm=cached.used_llm,
                used_intent=cached.used_intent,
                used_synonyms=cached.used_synonyms,
                fallback=cached.fallback,
                matched_intents=list(cached.matched_intents),
            )

        intent_matches = self._match_intents(query)
        intent_keywords = self._intent_keywords_from_matches(intent_matches)
        used_intent = bool(intent_matches)
        matched_intents = [m.label or m.intent_id for m in intent_matches]

        # No LLM client: fall back to synonym expansion
        if not self._llm_client:
            cleaned = self.clean_query(query)
            used_synonyms = bool(self._get_synonym_expansions(cleaned))
            rewritten = self.expand_query(query)
            result = QueryRewriteResult(
                original=query,
                rewritten=rewritten,
                method="rules",
                from_cache=False,
                used_llm=False,
                used_intent=used_intent,
                used_synonyms=used_synonyms,
                fallback=False,
                matched_intents=matched_intents,
            )
            self._cache[query] = result
            return result

        # Call LLM for rewriting
        try:
            response = self._llm_client.generate(
                system_prompt=self.QUERY_REWRITE_PROMPT,
                user_message=query,
                temperature=0.0,
            )
            # Clean up response (remove extra whitespace, newlines)
            rewritten = response.strip().replace("\n", " ")
            if not rewritten:
                raise ValueError("LLM returned empty rewrite")
            if intent_keywords:
                rewritten = self._merge_keywords(rewritten, intent_keywords)
            result = QueryRewriteResult(
                original=query,
                rewritten=rewritten,
                method="llm",
                from_cache=False,
                used_llm=True,
                used_intent=used_intent,
                used_synonyms=False,
                fallback=False,
                matched_intents=matched_intents,
            )
            self._cache[query] = result
            return result
        except Exception:
            # On any error, fall back to synonym expansion
            cleaned = self.clean_query(query)
            used_synonyms = bool(self._get_synonym_expansions(cleaned))
            rewritten = self.expand_query(query)
            result = QueryRewriteResult(
                original=query,
                rewritten=rewritten,
                method="rules",
                from_cache=False,
                used_llm=False,
                used_intent=used_intent,
                used_synonyms=used_synonyms,
                fallback=True,
                matched_intents=matched_intents,
            )
            self._cache[query] = result
            return result

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
        if self.has_intent(query):
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

    def detect_audience(self, query: str) -> Audience:
        """
        Detect the target audience from the query.

        Args:
            query: The search query text.

        Returns:
            Audience: Detected audience (STUDENT, FACULTY, STAFF, or ALL).
        """
        candidates = self.detect_audience_candidates(query)
        if len(candidates) == 1:
            return candidates[0]
        return Audience.ALL

    def detect_audience_candidates(self, query: str) -> List[Audience]:
        """Return matching audiences for the query."""
        query_lower = query.lower()
        matches: List[Audience] = []

        if any(k in query_lower for k in self.FACULTY_KEYWORDS):
            matches.append(Audience.FACULTY)
        if any(k in query_lower for k in self.STUDENT_KEYWORDS):
            matches.append(Audience.STUDENT)
        if any(k in query_lower for k in self.STAFF_KEYWORDS):
            matches.append(Audience.STAFF)

        if matches:
            return matches

        if any(k in query_lower for k in self.AMBIGUOUS_AUDIENCE_KEYWORDS):
            return [Audience.STUDENT, Audience.FACULTY, Audience.STAFF]

        return [Audience.ALL]

    def is_audience_ambiguous(self, query: str) -> bool:
        """Return True if multiple audiences match or intent is ambiguous."""
        candidates = self.detect_audience_candidates(query)
        return len(candidates) > 1

    def has_synonyms(self, query: str) -> bool:
        """Check if query contains any terms with synonyms."""
        cleaned = self.clean_query(query)
        return any(term in cleaned for term in self._synonyms)

    def has_intent(self, query: str) -> bool:
        """Check if query matches intent patterns."""
        return bool(self._match_intents(query))

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
        "규정",
        "규정은",
        "규정이",
        "규정을",
        "규정에",
        "규정의",
        "규칙",
        "조례",
        "법",
        "관련",
        "내용",
        "사항",
        "경우",
        "대해",
        "대한",
        "대해서",
        # 질문 관련
        "뭐",
        "뭔가",
        "무엇",
        "어떤",
        "어떻게",
        "왜",
        "언제",
        "어디",
        "알려줘",
        "알려주세요",
        "설명해줘",
        "설명해주세요",
        # 기타
        "좀",
        "것",
        "수",
        "때",
        "중",
        "있나요",
        "있어요",
        "있습니까",
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
        for term, synonyms in self._synonyms.items():
            if term in cleaned_query:
                # Add first 2 synonyms to avoid over-expansion
                expansions.extend(synonyms[:2])
        return expansions

    def _intent_keywords(self, query: str) -> List[str]:
        """Return intent-based keywords for colloquial queries."""
        matches = self._match_intents(query)
        return self._intent_keywords_from_matches(matches)

    def _intent_keywords_from_matches(self, matches: List[IntentMatch]) -> List[str]:
        """Flatten keywords from matched intents (top-N)."""
        keywords: List[str] = []
        for match in matches[: self.INTENT_MAX_MATCHES]:
            keywords = self._merge_token_list(keywords, match.keywords)
        return keywords

    def _merge_token_list(
        self, base_tokens: List[str], extra_tokens: List[str]
    ) -> List[str]:
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

    def _load_synonyms(self, synonyms_path: Optional[str]) -> Dict[str, List[str]]:
        """Load external synonyms and merge with built-in defaults."""
        merged: Dict[str, List[str]] = {
            term: list(synonyms) for term, synonyms in self.SYNONYMS.items()
        }

        path_value = synonyms_path or os.getenv("RAG_SYNONYMS_PATH")
        if not path_value:
            return merged

        path = Path(path_value)
        if not path.exists():
            return merged

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return merged

        if isinstance(data, dict) and "terms" in data:
            data = data.get("terms")

        if not isinstance(data, dict):
            return merged

        for term, synonyms in data.items():
            if not isinstance(term, str):
                continue
            term = term.strip()
            if not term:
                continue
            if isinstance(synonyms, str):
                synonyms_list = [synonyms]
            elif isinstance(synonyms, list):
                synonyms_list = synonyms
            else:
                continue

            cleaned = [
                s.strip() for s in synonyms_list if isinstance(s, str) and s.strip()
            ]
            if not cleaned:
                continue

            if term not in merged:
                merged[term] = []

            for synonym in cleaned:
                if synonym not in merged[term]:
                    merged[term].append(synonym)

        return merged

    def _load_intents(self, intents_path: Optional[str]) -> List[IntentRule]:
        """Load external intents and merge with built-in rules."""
        rules: List[IntentRule] = []

        # Built-in rules
        for idx, (pattern, keywords) in enumerate(self.INTENT_PATTERNS, start=1):
            label = " / ".join(keywords[:2]) if keywords else f"intent_{idx}"
            rules.append(
                IntentRule(
                    intent_id=f"legacy_{idx}",
                    label=label,
                    keywords=list(keywords),
                    patterns=[pattern],
                    triggers=[],
                )
            )

        path_value = intents_path or os.getenv("RAG_INTENTS_PATH")
        if not path_value:
            return rules

        path = Path(path_value)
        if not path.exists():
            return rules

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return rules

        intents = data.get("intents") if isinstance(data, dict) else data
        if not isinstance(intents, list):
            return rules

        for item in intents:
            if not isinstance(item, dict):
                continue
            intent_id = str(item.get("id") or "").strip()
            label = str(item.get("label") or intent_id).strip()
            if not intent_id:
                continue

            raw_keywords = item.get("keywords") or []
            raw_triggers = item.get("triggers") or item.get("examples") or []
            raw_patterns = item.get("patterns") or []

            keywords = [
                k.strip() for k in raw_keywords if isinstance(k, str) and k.strip()
            ]
            triggers = [
                t.strip() for t in raw_triggers if isinstance(t, str) and t.strip()
            ]

            patterns: List[re.Pattern] = []
            for raw in raw_patterns:
                if not isinstance(raw, str) or not raw.strip():
                    continue
                try:
                    patterns.append(re.compile(raw))
                except re.error:
                    continue

            if not (keywords or triggers or patterns):
                continue

            rules.append(
                IntentRule(
                    intent_id=intent_id,
                    label=label or intent_id,
                    keywords=keywords,
                    patterns=patterns,
                    triggers=triggers,
                )
            )

        return rules

    def _match_intents(self, query: str) -> List[IntentMatch]:
        """Match query against intent rules and return ranked matches."""
        if not query:
            return []

        cleaned = self.clean_query(query)
        haystack = cleaned or query
        matches: List[IntentMatch] = []

        for rule in self._intent_rules:
            score = 0
            for pattern in rule.patterns:
                if pattern.search(haystack):
                    score += 2
            for trigger in rule.triggers:
                if trigger in haystack:
                    score += 1
            for keyword in rule.keywords:
                if keyword and keyword in haystack:
                    score += 1

            if score > 0:
                matches.append(
                    IntentMatch(
                        intent_id=rule.intent_id,
                        label=rule.label or rule.intent_id,
                        keywords=rule.keywords,
                        score=score,
                    )
                )

        matches.sort(key=lambda m: (-m.score, m.intent_id))

        # Refined Logic: If we have strong matches (score >= 2 from pattern/trigger),
        # exclude weak matches (score < 2 from keywords only) to prevent query pollution.
        if matches and matches[0].score >= 2:
            matches = [m for m in matches if m.score >= 2]

        return matches[: self.INTENT_MAX_MATCHES]


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
        synonyms_path: Optional[str] = None,
        intents_path: Optional[str] = None,
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
