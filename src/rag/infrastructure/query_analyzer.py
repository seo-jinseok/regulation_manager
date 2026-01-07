"""
Query Analyzer for Regulation RAG System.

Analyzes query text to determine optimal BM25/Dense weights
and provides query expansion with synonyms for better recall.
"""

import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .patterns import ARTICLE_PATTERN

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

    # Context keywords for better audience detection
    STUDENT_CONTEXT_KEYWORDS = [
        "공부", "수업", "과제", "시험", "학점",
        "F", "학사경고", "휴학", "자퇴", "졸업",
        "장학금", "등록금", "기숙사", "생활관",
    ]
    FACULTY_CONTEXT_KEYWORDS = [
        "강의", "연구", "논문", "연구년", "승진",
        "교수", "책임시수", "업적", "안식년", "학회",
    ]

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
        QueryType.INTENT: (0.4, 0.6),  # Intent queries favor semantic search (adjusted from 0.35)
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
            re.compile(r"(그만두고\s*싶|그만\s*두고\s*싶|퇴직|사직|관두고|나가고)"),
            ["퇴직", "사직", "명예퇴직"],
        ),
        (re.compile(r"(수업|강의).*안.*하.*싶"), ["휴강", "보강", "강의"]),
        (re.compile(r"연구.*(부정|신고).*싶"), ["연구윤리", "부정행위", "신고"]),
        (re.compile(r".*싶어"), ["희망", "원함"]),  # General desire pattern (lowest priority)
        # New patterns added for specific failure cases
        (re.compile(r"장학금.*(받|타|신청).*싶"), ["장학금", "신청", "지급"]),
        (re.compile(r"(학회|출장).*(가|참석).*싶"), ["국외여비", "출장", "학회"]),
        (re.compile(r"(공부|학업).*하기.*싫"), ["휴학", "자퇴"]),
    ]
    INTENT_MAX_MATCHES = 3

    # LLM Query Rewriting prompt
    QUERY_REWRITE_PROMPT = """당신은 사용자의 질문을 검색용 키워드로 변환하는 AI 조교입니다.
특히 한국어 사용자의 의도적 오타, 구어체, 띄어쓰기 오류를 교정하여 검색 품질을 높여야 합니다.

지침:
1. 사용자의 오타나 구어체를 표준어로 교정한 후 핵심 키워드를 추출하세요.
   - 예: "휴학하고 시퍼" -> "휴학 희망"
   - 예: "강의면제 바드려면" -> "강의 면제 신청"
   - 예: "장학금 타고 시퍼요" -> "장학금 수혜 조건"
2. 검색에 불필요한 조사, 어미, 감탄사는 제거하세요.
3. 오직 공백으로 구분된 핵심 단어들만 출력하세요. (설명 금지)

출력 형식:
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
        self._synonym_session_cache: Dict[str, List[str]] = {}  # LLM-generated synonym cache
        self._synonyms = self._load_synonyms(synonyms_path)
        self._intent_rules = self._load_intents(intents_path)

    def _clean_llm_response(self, text: str) -> str:
        """Clean chatty LLM response to extract just the keywords."""
        # 0. Handle loops/hallucinations: if it contains tags like <user> or <assistant>
        # Just take the part before the first tag if it exists.
        if "<user>" in text:
            text = text.split("<user>")[0]
        if "<assistant>" in text:
            text = text.split("<assistant>")[0]
        if "---" in text:
            text = text.split("---")[0]

        # 1. Remove quotes if the entire string is quoted
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        elif text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        # 2. Split into lines and take the first non-empty line
        # LLMs looping often start with the answer then loop.
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return ""
        text = lines[0]

        # 3. Remove known prefixes.
        prefixes = [
            "키워드:",
            "keywords:",
            "answer:",
            "output:",
            "답변:",
            "추출된 키워드:",
            "검색 키워드:",
            "extracted keywords:",
            "here are the keywords:",
            "sure, here are",
            "search keywords:",
        ]
        
        lower_text = text.lower()
        for prefix in prefixes:
            if lower_text.startswith(prefix):
                text = text[len(prefix):].strip()
                lower_text = text.lower()
        
        # 4. If there are still colons, might be "Concept: Keyword", take the part after colon
        if ":" in text:
            parts = text.split(":")
            # Only split if it's a short label followed by keywords
            if len(parts[0]) < 20: 
                text = parts[1].strip()

        return text.strip()

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
        """
        if not query:
             return QueryRewriteResult(original="", rewritten="", method="none", from_cache=False, used_llm=False, used_intent=False, used_synonyms=False, fallback=False, matched_intents=[])

        query = unicodedata.normalize("NFC", query)
        
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
            
            # Post-process response
            rewritten = self._clean_llm_response(response)
            
            # Robust length limit: Search keywords shouldn't be excessively long
            # If the response is more than 200 chars or 30 words, it's likely a hallucination
            if len(rewritten) > 200 or len(rewritten.split()) > 30:
                # Take only the first few words as a safety measure
                rewritten = " ".join(rewritten.split()[:10])

            # Collapse multiple spaces
            rewritten = " ".join(rewritten.split())

            if not rewritten:
                from ..exceptions import HybridSearchError
                raise HybridSearchError("LLM returned empty rewrite")
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
        """
        if not query:
            return QueryType.GENERAL
            
        query = unicodedata.normalize("NFC", query)
        
        # Check for article references (highest priority for exact match)
        if ARTICLE_PATTERN.search(query):
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
        if not query:
            return [Audience.ALL]
            
        query = unicodedata.normalize("NFC", query)
        query_lower = query.lower()
        matches: List[Audience] = []

        # Primary keywords (explicit audience mention)
        if any(k in query_lower for k in self.FACULTY_KEYWORDS):
            matches.append(Audience.FACULTY)
        if any(k in query_lower for k in self.STUDENT_KEYWORDS):
            matches.append(Audience.STUDENT)
        if any(k in query_lower for k in self.STAFF_KEYWORDS):
            matches.append(Audience.STAFF)

        if matches:
            return matches

        # Secondary: Context-based detection (when no explicit keywords)
        context_matches: List[Audience] = []
        if any(k in query_lower for k in self.STUDENT_CONTEXT_KEYWORDS):
            context_matches.append(Audience.STUDENT)
        if any(k in query_lower for k in self.FACULTY_CONTEXT_KEYWORDS):
            context_matches.append(Audience.FACULTY)
        
        if context_matches:
            return context_matches

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
        """Collect synonym expansions, using LLM for unknown terms if available."""
        expansions: List[str] = []
        tokens = cleaned_query.split() if cleaned_query else []

        for token in tokens:
            # Skip short tokens (likely particles or noise)
            if len(token) < 2:
                continue

            # Check if term exists in synonyms.json
            if token in self._synonyms:
                # Add first 2 synonyms to avoid over-expansion
                expansions.extend(self._synonyms[token][:2])
            elif self._llm_client:
                # Generate synonyms via LLM for unknown terms
                llm_synonyms = self._generate_synonyms_cached(token)
                expansions.extend(llm_synonyms[:2])

        return expansions

    def _generate_synonyms_cached(self, term: str) -> List[str]:
        """Generate synonyms via LLM with session cache."""
        # Check session cache first
        if term in self._synonym_session_cache:
            return self._synonym_session_cache[term]

        try:
            from ..application.synonym_generator_service import SynonymGeneratorService

            service = SynonymGeneratorService(llm_client=self._llm_client)
            synonyms = service.generate_synonyms(term, exclude_existing=False)
            self._synonym_session_cache[term] = synonyms
            return synonyms
        except Exception:
            # On any error, return empty list (don't block search)
            self._synonym_session_cache[term] = []
            return []

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

    def _normalize_for_matching(self, text: str) -> str:
        """
        Normalize text for flexible intent matching.
        """
        if not text:
            return ""
        
        text = unicodedata.normalize("NFC", text)
        # Remove all whitespace
        normalized = re.sub(r'\s+', '', text)
        # Remove common polite/formal endings for flexibility
        normalized = re.sub(r'(요|습니다|니다|세요|줘|주세요|할까요|하나요|인가요)$', '', normalized)
        return normalized

    def _match_intents(self, query: str) -> List[IntentMatch]:
        """Match query against intent rules and return ranked matches."""
        if not query:
            return []

        cleaned = self.clean_query(query)
        haystack = cleaned or query
        # Pre-compute normalized version for trigger matching
        normalized_query = self._normalize_for_matching(haystack)
        matches: List[IntentMatch] = []

        for rule in self._intent_rules:
            score = 0
            # Pattern matching (regex - already flexible)
            for pattern in rule.patterns:
                if pattern.search(haystack):
                    score += 2
            # Trigger matching with normalization (handles whitespace/ending variations)
            # Triggers are more specific than patterns, so give them higher score
            for trigger in rule.triggers:
                normalized_trigger = self._normalize_for_matching(trigger)
                if normalized_trigger and normalized_trigger in normalized_query:
                    # Exact match (normalized trigger equals normalized query) gets bonus
                    if normalized_trigger == normalized_query:
                        score += 4  # Exact match: highest priority
                    else:
                        score += 2  # Partial match: same as pattern
            
            # Note: We do NOT match on rule.keywords here. 
            # Keywords are for query expansion (output), not for intent detection (input).
            # Using keywords for detection causes false positives (e.g. "장학금" keyword matching "장학금 규정")

            if score > 0:
                # Prefer external intents (from intents.json) over legacy built-in patterns
                is_legacy = rule.intent_id.startswith("legacy_")
                matches.append(
                    IntentMatch(
                        intent_id=rule.intent_id,
                        label=rule.label or rule.intent_id,
                        keywords=rule.keywords,
                        score=score if not is_legacy else score - 1,  # Penalize legacy patterns
                    )
                )

        matches.sort(key=lambda m: (-m.score, m.intent_id))

        # Refined Logic: If we have strong matches (score >= 2 from pattern/trigger),
        # exclude weak matches (score < 2 from keywords only) to prevent query pollution.
        if matches and matches[0].score >= 2:
            matches = [m for m in matches if m.score >= 2]

        return matches[: self.INTENT_MAX_MATCHES]
