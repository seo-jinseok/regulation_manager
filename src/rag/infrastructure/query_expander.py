"""
Dynamic Query Expander for Regulation RAG System.

LLM-based dynamic query expansion that generates relevant search keywords
without relying on pre-defined patterns in intents.json or synonyms.json.

This addresses the pattern explosion problem by dynamically understanding
user intent and generating appropriate search terms.

Cycle 5: Added cache metrics and batch save optimization.
Cycle 6: Integrated with RAGQueryCache for centralized caching and metrics.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from ..domain.repositories import ILLMClient

logger = logging.getLogger(__name__)


@dataclass
class QueryExpansionResult:
    """Result of dynamic query expansion."""

    original_query: str
    keywords: List[str]
    expanded_query: str
    intent: str
    confidence: float
    from_cache: bool = False
    cache_key: Optional[str] = None
    method: str = "llm"  # "llm", "pattern", "cache"


# System prompt for query expansion
QUERY_EXPANSION_PROMPT = """당신은 대학 규정 검색 시스템의 쿼리 분석 전문가입니다.
사용자의 질문을 분석하여 검색에 사용할 키워드를 추출합니다.

## 분석 규칙
1. 사용자의 진짜 의도를 파악하세요.
2. 구어체/비격식 표현을 공식 규정 용어로 변환하세요.
3. 관련 규정명, 조문, 키워드를 포함하세요.
4. 동의어와 관련 용어를 추가하세요.

## 변환 예시
- "학교 가기 싫어" → 휴학, 휴직, 연구년, 안식년
- "졸업하려면 뭐 필요해?" → 졸업요건, 이수학점, 학점, 졸업인증
- "교수 승진" → 승진, 교원인사규정, 업적평가, 승진임용
- "장학금 받으려면" → 장학금, 성적기준, 지급기준, 장학금지급규정

## 출력 형식 (반드시 JSON)
{
    "intent": "의도 설명 (예: 졸업 요건 문의)",
    "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
    "confidence": 0.9
}

키워드는 3-7개를 추출하고, 가장 중요한 것을 앞에 배치하세요."""


@dataclass
class ExpansionRule:
    """Lightweight pattern-based expansion rule."""

    patterns: List[str]
    keywords: List[str]
    intent: str


# Fallback patterns when LLM is unavailable
FALLBACK_RULES: List[ExpansionRule] = [
    ExpansionRule(
        patterns=["장학금", "장학"],
        keywords=["장학금", "성적기준", "지급기준", "장학금지급"],
        intent="장학금 문의",
    ),
    ExpansionRule(
        patterns=["졸업", "졸업학점", "이수학점"],
        keywords=["졸업", "졸업요건", "이수학점", "졸업인증", "학칙"],
        intent="졸업 요건 문의",
    ),
    ExpansionRule(
        patterns=["승진", "교수 승진", "교원 승진"],
        keywords=["승진", "교원인사규정", "업적평가", "승진임용"],
        intent="교원 승진 문의",
    ),
    ExpansionRule(
        patterns=["평가", "교수 평가", "교원 평가", "업적 평가", "성과 평가"],
        keywords=["업적평가", "교원", "교원인사규정", "연구성과", "교육성과"],
        intent="교원 업적 평가 문의",
    ),
    ExpansionRule(
        patterns=["휴학", "휴직", "가기 싫", "쉬고 싶"],
        keywords=["휴학", "휴직", "연구년", "안식년", "복학"],
        intent="휴학/휴직 문의",
    ),
    ExpansionRule(
        patterns=["등록금", "납부", "환불"],
        keywords=["등록금", "납부", "환불", "분납", "등록금납부"],
        intent="등록금 문의",
    ),
    ExpansionRule(
        patterns=["성적", "학점", "평점"],
        keywords=["성적", "학점", "평점", "성적평가", "학사"],
        intent="성적/학점 문의",
    ),
    ExpansionRule(
        patterns=["영어", "토익", "어학", "외국어"],
        keywords=["어학인증", "토익", "졸업요건", "외국어", "영어"],
        intent="어학 요건 문의",
    ),
]


class DynamicQueryExpander:
    """
    LLM-based dynamic query expansion.

    Unlike static pattern matching in intents.json, this class:
    1. Uses LLM to understand user intent dynamically
    2. Generates relevant keywords without pre-defined mappings
    3. Falls back to lightweight patterns when LLM unavailable
    4. Caches results using centralized RAGQueryCache (Cycle 6)

    This solves the "pattern explosion" problem where every new
    expression requires manual addition to configuration files.
    """

    def __init__(
        self,
        llm_client: Optional["ILLMClient"] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        max_keywords: int = 7,
        expansion_cache: Optional[Any] = None,  # QueryExpansionCache from cache.py
    ):
        """
        Initialize dynamic query expander.

        Args:
            llm_client: LLM client for generating expansions.
            cache_dir: Directory to cache expansions (legacy, for backward compatibility).
            enable_cache: Whether to use caching.
            max_keywords: Maximum number of keywords to generate.
            expansion_cache: Optional QueryExpansionCache instance (Cycle 6).
        """
        self._llm_client = llm_client
        self._enable_cache = enable_cache
        self._max_keywords = max_keywords

        # Cache directory (legacy, for backward compatibility)
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "cache"
                / "query_expansion"
            )

        # Cycle 6: Use centralized cache
        if expansion_cache is not None:
            self._expansion_cache = expansion_cache
        elif enable_cache:
            from .cache import QueryExpansionCache

            self._expansion_cache = QueryExpansionCache(
                enabled=True,
                ttl_hours=168,  # 7 days
            )
        else:
            self._expansion_cache = None

        # Legacy cache for migration (deprecated)
        if self._enable_cache and expansion_cache is None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._legacy_cache: dict = self._load_cache()
        else:
            self._legacy_cache = {}

    def _load_cache(self) -> dict:
        """Load expansion cache from disk."""
        cache_file = self._cache_dir / "expansion_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load expansion cache: {e}")
        return {}

    def _save_cache(self) -> None:
        """Save expansion cache to disk."""
        if not self._enable_cache:
            return
        cache_file = self._cache_dir / "expansion_cache.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self._legacy_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save expansion cache: {e}")

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]

    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        """Set LLM client for expansion."""
        self._llm_client = llm_client

    def expand(self, query: str) -> QueryExpansionResult:
        """
        Expand query with relevant keywords.

        Tries in order:
        1. QueryExpansionCache lookup (Cycle 6)
        2. Legacy cache lookup (backward compatibility)
        3. LLM-based expansion
        4. Pattern-based fallback

        Args:
            query: User's search query.

        Returns:
            QueryExpansionResult with keywords and expanded query.
        """
        # 1. Check new centralized cache (Cycle 6)
        if self._expansion_cache:
            cached = self._expansion_cache.get_expansion(query)
            if cached:
                logger.debug(f"Expansion cache hit (centralized): {query[:30]}...")
                return QueryExpansionResult(
                    original_query=query,
                    keywords=cached["keywords"],
                    expanded_query=cached["expanded_query"],
                    intent=cached["intent"],
                    confidence=cached["confidence"],
                    from_cache=True,
                    cache_key=self._get_cache_key(query),
                    method="cache",
                )

        # 2. Check legacy cache (backward compatibility)
        cache_key = self._get_cache_key(query)
        if cache_key in self._legacy_cache:
            cached = self._legacy_cache[cache_key]
            logger.debug(f"Expansion cache hit (legacy): {query[:30]}...")
            # Migrate to new cache
            if self._expansion_cache:
                self._expansion_cache.set_expansion(
                    query=query,
                    keywords=cached["keywords"],
                    expanded_query=cached["expanded_query"],
                    intent=cached["intent"],
                    confidence=cached["confidence"],
                    method="cache",
                )
            return QueryExpansionResult(
                original_query=query,
                keywords=cached["keywords"],
                expanded_query=cached["expanded_query"],
                intent=cached["intent"],
                confidence=cached["confidence"],
                from_cache=True,
                cache_key=cache_key,
                method="cache",
            )

        # 3. Try LLM expansion
        if self._llm_client:
            result = self._llm_expand(query)
            if result and result.confidence >= 0.5:
                # Cache successful result in both systems
                if self._expansion_cache:
                    self._expansion_cache.set_expansion(
                        query=query,
                        keywords=result.keywords,
                        expanded_query=result.expanded_query,
                        intent=result.intent,
                        confidence=result.confidence,
                        method="llm",
                    )
                # Legacy cache (deprecated)
                self._legacy_cache[cache_key] = {
                    "keywords": result.keywords,
                    "expanded_query": result.expanded_query,
                    "intent": result.intent,
                    "confidence": result.confidence,
                }
                self._save_cache()
                return result

        # 4. Fallback to pattern matching
        return self._pattern_expand(query)

    def _llm_expand(self, query: str) -> Optional[QueryExpansionResult]:
        """
        Use LLM to expand query.

        Args:
            query: User's search query.

        Returns:
            QueryExpansionResult or None if failed.
        """
        start_time = time.time()
        try:
            response = self._llm_client.generate(
                system_prompt=QUERY_EXPANSION_PROMPT,
                user_message=f"질문: {query}",
                temperature=0.3,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # Record LLM call metrics (Cycle 6)
            if self._expansion_cache:
                self._expansion_cache.record_llm_call(elapsed_ms)

            # Parse JSON response
            result = self._parse_llm_response(response)
            if result:
                keywords = result.get("keywords", [])[: self._max_keywords]
                intent = result.get("intent", "unknown")
                confidence = result.get("confidence", 0.7)

                # Build expanded query
                expanded_query = self._build_expanded_query(query, keywords)

                logger.debug(
                    f"LLM expansion: {query[:30]}... -> {keywords[:3]}... ({elapsed_ms:.1f}ms)"
                )

                return QueryExpansionResult(
                    original_query=query,
                    keywords=keywords,
                    expanded_query=expanded_query,
                    intent=intent,
                    confidence=confidence,
                    from_cache=False,
                    method="llm",
                )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(f"LLM expansion failed after {elapsed_ms:.1f}ms: {e}")

        return None

    def _parse_llm_response(self, response: str) -> Optional[dict]:
        """Parse JSON from LLM response."""
        try:
            # Try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from text
        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _pattern_expand(self, query: str) -> QueryExpansionResult:
        """
        Fallback pattern-based expansion.

        Args:
            query: User's search query.

        Returns:
            QueryExpansionResult based on pattern matching.
        """
        # Record pattern fallback (Cycle 6)
        if self._expansion_cache:
            self._expansion_cache.record_pattern_fallback()

        query_lower = query.lower()

        for rule in FALLBACK_RULES:
            for pattern in rule.patterns:
                if pattern in query_lower:
                    expanded_query = self._build_expanded_query(query, rule.keywords)
                    logger.debug(
                        f"Pattern expansion: {query[:30]}... -> {rule.keywords[:3]}..."
                    )
                    return QueryExpansionResult(
                        original_query=query,
                        keywords=rule.keywords,
                        expanded_query=expanded_query,
                        intent=rule.intent,
                        confidence=0.7,
                        from_cache=False,
                        method="pattern",
                    )

        # No pattern matched - return original with basic keywords
        basic_keywords = self._extract_basic_keywords(query)
        return QueryExpansionResult(
            original_query=query,
            keywords=basic_keywords,
            expanded_query=query,
            intent="unknown",
            confidence=0.5,
            from_cache=False,
            method="pattern",
        )

    def _build_expanded_query(self, query: str, keywords: List[str]) -> str:
        """
        Build expanded query by appending keywords.

        Args:
            query: Original query.
            keywords: Keywords to append.

        Returns:
            Expanded query string.
        """
        # Don't duplicate existing keywords
        existing = set(query.lower().split())
        new_keywords = [kw for kw in keywords if kw.lower() not in existing]

        if new_keywords:
            return f"{query} {' '.join(new_keywords[:3])}"
        return query

    def _extract_basic_keywords(self, query: str) -> List[str]:
        """Extract basic keywords from query using simple heuristics."""
        # Remove common particles and short words
        stopwords = {
            "이",
            "가",
            "을",
            "를",
            "은",
            "는",
            "의",
            "에",
            "로",
            "으로",
            "와",
            "과",
            "도",
            "만",
            "부터",
            "까지",
            "에서",
            "처럼",
            "어떻게",
            "뭐",
            "뭘",
            "무엇",
            "언제",
            "어디",
            "누구",
            "해",
            "돼",
            "할",
            "된",
            "하는",
            "되는",
            "있",
            "없",
        }

        words = re.findall(r"[가-힣A-Za-z0-9]+", query)
        keywords = [w for w in words if w not in stopwords and len(w) >= 2]

        return keywords[:5] if keywords else [query]

    def get_metrics(self) -> Optional[dict]:
        """
        Get query expansion cache metrics (Cycle 6).

        Returns:
            Dictionary with cache metrics or None if cache not enabled.
        """
        if self._expansion_cache:
            return self._expansion_cache.stats()
        return None

    def reset_metrics(self) -> None:
        """Reset query expansion cache metrics (Cycle 6)."""
        if self._expansion_cache:
            self._expansion_cache.reset_metrics()

    def should_expand(self, query: str) -> bool:
        """
        Determine if query should be expanded.

        Expansion is useful for:
        - Vague/informal queries
        - Queries without regulatory terms
        - Short queries with ambiguous intent

        Args:
            query: User's search query.

        Returns:
            True if expansion is recommended.
        """
        # Skip if already contains regulatory terms
        regulatory_terms = ["규정", "규칙", "조", "항", "호", "세칙", "지침", "학칙"]
        if any(term in query for term in regulatory_terms):
            return False

        # Expand vague/informal queries
        vague_indicators = [
            "싶어",
            "싫어",
            "어떻게",
            "뭐야",
            "있어?",
            "해야",
            "받으려면",
            "하려면",
            "되려면",
            "가능",
        ]
        if any(ind in query for ind in vague_indicators):
            return True

        # Expand short queries
        if len(query) <= 20:
            return True

        return False


class QueryExpansionPipeline:
    """
    Pipeline that combines DynamicQueryExpander with existing components.

    Integrates with:
    - QueryAnalyzer (for intent classification)
    - HyDE (for hypothetical document generation)
    - HybridSearcher (for search execution)
    """

    def __init__(
        self,
        expander: DynamicQueryExpander,
        enable_hyde: bool = True,
        enable_self_rag: bool = True,
    ):
        """
        Initialize expansion pipeline.

        Args:
            expander: Dynamic query expander instance.
            enable_hyde: Whether to combine with HyDE.
            enable_self_rag: Whether to use Self-RAG evaluation.
        """
        self._expander = expander
        self._enable_hyde = enable_hyde
        self._enable_self_rag = enable_self_rag

    def process_query(self, query: str) -> QueryExpansionResult:
        """
        Process query through expansion pipeline.

        Args:
            query: Original user query.

        Returns:
            Expansion result with keywords and expanded query.
        """
        # Check if expansion is needed
        if not self._expander.should_expand(query):
            logger.debug(f"Skipping expansion for: {query[:30]}...")
            return QueryExpansionResult(
                original_query=query,
                keywords=[],
                expanded_query=query,
                intent="direct",
                confidence=1.0,
                from_cache=False,
                method="skip",
            )

        return self._expander.expand(query)
