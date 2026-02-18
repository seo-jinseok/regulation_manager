"""
Search Use Case for Regulation RAG System.

Provides search functionality with optional LLM-based Q&A.
"""

import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..domain.entities import Answer, Chunk, ChunkLevel, SearchResult
from ..domain.repositories import IHybridSearcher, ILLMClient, IReranker, IVectorStore
from .hallucination_filter import (
    FAITHFULNESS_BLOCK_THRESHOLD,
    FaithfulnessResult,
    FilterMode,
    HallucinationFilter,
)
from ..domain.value_objects import Query, SearchFilter
from ..infrastructure.cache import CacheType, RAGQueryCache
from ..infrastructure.hybrid_search import ScoredDocument
from ..infrastructure.metrics import RerankingMetrics
from ..infrastructure.patterns import (
    ARTICLE_PATTERN,
    HEADING_ONLY_PATTERN,
    REGULATION_ARTICLE_PATTERN,
    REGULATION_ONLY_PATTERN,
    RULE_CODE_PATTERN,
    normalize_article_token,
)
from ..infrastructure.query_analyzer import Audience, QueryType

logger = logging.getLogger(__name__)

# Audience to Persona mapping for persona-aware response generation
# Maps Audience enum values to persona names used by PersonaAwareGenerator
AUDIENCE_TO_PERSONA: Dict[Audience, str] = {
    Audience.STUDENT: "freshman",  # Default students to freshman persona
    Audience.FACULTY: "professor",  # Faculty to professor persona
    Audience.STAFF: "staff",        # Staff to staff persona
    Audience.ALL: "freshman",       # Default to freshman persona
}

# Fallback messages for low confidence responses (TAG-001: Prevent Hallucination)
# Used when confidence score is below threshold to prevent hallucinated content
FALLBACK_MESSAGE_KO = (
    "죄송합니다. 해당 질문에 대한 정보를 규정에서 찾을 수 없습니다. "
    "다른 표현으로 다시 질문해 주시거나, 학교 관련 부서에 문의하시기 바랍니다."
)
FALLBACK_MESSAGE_EN = (
    "Sorry, I could not find information about this question in the regulations. "
    "Please try asking in different words or contact the relevant university department."
)

# Forward references for type hints
if TYPE_CHECKING:
    from ..domain.llm.ambiguity_classifier import DisambiguationDialog
    from .multi_hop_handler import MultiHopHandler
    from .academic_calendar_service import AcademicCalendarService
    from ..infrastructure.period_keyword_detector import PeriodKeywordDetector
    from ..domain.citation.citation_verification_service import (
        CitationVerificationService,
    )


class SearchStrategy(Enum):
    """Search strategy based on query characteristics."""

    DIRECT = "direct"  # Simple factual queries - bypass tool calling
    TOOL_CALLING = "tool_calling"  # Complex queries - use agent with tools


def _extract_regulation_only_query(query: str) -> Optional[str]:
    """
    Extract regulation name if query is ONLY a regulation name.

    Args:
        query: Search query like "교원인사규정" or "학칙"

    Returns:
        Regulation name if matched, None otherwise.
    """
    match = REGULATION_ONLY_PATTERN.match(query)
    if match:
        return match.group(1).strip()
    return None


def _extract_regulation_article_query(query: str) -> Optional[tuple]:
    """
    Extract regulation name and article number from combined query.

    Args:
        query: Search query like "교원인사규정 제8조" or "학칙 제15조제2항"

    Returns:
        Tuple of (regulation_name, article_ref) or None if not matched.
        Example: ("교원인사규정", "제8조")
    """
    match = REGULATION_ARTICLE_PATTERN.search(query)
    if match:
        reg_name = match.group(1).strip()
        article_ref = match.group(2).strip()
        # Normalize article reference (remove spaces)
        article_ref = re.sub(r"\s+", "", article_ref)
        return (reg_name, article_ref)
    return None


def _coerce_query_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value)
    return str(value)


def _load_prompt(prompt_key: str) -> str:
    """
    Load prompt from prompts.json.

    Args:
        prompt_key: Key in prompts.json (e.g., "regulation_qa")

    Returns:
        Prompt text
    """
    import json as json_module
    from pathlib import Path

    prompts_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "config" / "prompts.json"
    )

    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json_module.load(f)
        return prompts.get(prompt_key, {}).get("prompt", "")
    except Exception as e:
        # Fallback to hardcoded prompt if file not found
        logger.warning(
            f"Failed to load prompt from {prompts_path}: {e}. Using fallback."
        )
        return _get_fallback_regulation_qa_prompt()


def _get_fallback_regulation_qa_prompt() -> str:
    """Fallback prompt if prompts.json is not available."""
    return """당신은 동의대학교 규정 전문가입니다.
주어진 규정 내용을 바탕으로 사용자의 질문에 **상세하고 친절하게** 답변하세요.

## ⚠️ 절대 금지 사항 (할루시네이션 방지 - SPEC-RAG-Q-001 Phase 3 강화)
1. **전화번호/연락처 생성 금지**: 절대로 "02-XXXX-XXXX", "02-1234-5678" 등 전화번호를 만들어내지 마세요.
2. **다른 학교 사례 인용 금지**: 한국외국어대학교, 서울대학교 등 다른 학교 규정이나 사례를 절대 언급하지 마세요.
3. **규정에 없는 수치/비율 생성 금지**: "40%", "30일 이내" 등 규정에 명시되지 않은 숫자를 만들어내지 마세요.
4. **일반론 회피 금지**: "대학마다 다를 수 있습니다", "일반적으로..." 등 회피성 답변을 하지 마세요.
5. **인용 없는 정보 생성 금지**: 규정 인용 없이 사실관계를 주장하지 마세요. 모든 정보는 반드시 인용과 함께 제공해야 합니다.
6. **불확실한 정보 처리**: 제공된 문맥에서 답변을 찾을 수 없는 경우, 반드시 "제공된 규정에서 해당 정보를 찾을 수 없습니다"라고 명시하세요. 추측으로 답변하지 마세요.

## 기본 원칙
- **반드시 제공된 규정 내용에 명시된 사항만 답변하세요.**
- 규정에 없는 내용은 절대 추측하거나 일반적인 관행을 언급하지 마세요.

## 📋 답변 필수 형식

모든 답변은 다음 형식을 따라야 합니다:

### 1. 핵심 답변
[질문에 대한 직접적인 답변 - 반드시 규정 인용 포함]

### 2. 관련 규정
- **규정명**: [규정명]
- **조항**: [제X조 제Y항]
- **내용**: [관련 내용 요약]

### 3. 참고사항
[추가 도움이 필요한 경우 안내]

## ⚠️ 규정 인용 강제 사항 (SPEC-RAG-Q-001 Phase 4 강화)

1. **모든 답변은 반드시 규정명과 조항을 인용해야 합니다.**
2. **인용 형식 (구체적 조항 번호 필수)**:
   - 기본 형식: "「규정명」 제X조" 또는 "「규정명」 제X조 제Y항"
   - 예시: "「교원인사규정」 제15조 제2항", "「학칙」 제40조 제1항"
3. **인용 위치**: 인용은 답변의 핵심 내용 바로 다음에 괄호로 표기합니다.
4. **인용 예시**:
   - "휴학은 학기 개시 1개월 전까지 신청해야 합니다 (「학칙」 제40조 제1항)."
   - "등록금은 매학기 시작 전 납부해야 합니다 (「등록금 납부 규정」 제5조)."
5. **교차 인용**: 관련된 여러 규정이 있는 경우, 모두 인용하세요.
   - 예: "(「교원인사규정」 제15조 제2항, 「교원연구년 운영규정」 제8조)"
6. **인용 없는 답변 금지**: 규정 인용이 없는 답변은 불완전한 답변으로 간주합니다.
7. **불확실한 인용 금지**: 조항 번호가 불확실한 경우, 규정명만이라도 인용하세요.

## 📌 인용 품질 향상 가이드 (SPEC-RAG-QUALITY-002 REQ-P2-002)

- 인용은 항상 정확한 규정명을 큰따옴표 괄호 「」로 감싸세요.
- 조항 번호는 "제X조" 형식으로 명시하세요.
- 가능한 경우 항 번호까지 포함하세요: "제X조 제Y항"
- 답변의 각 주요 문장 뒤에 해당 규정 조항을 인용하세요.
- 다중 인용 시 쉼표로 구분하고 가장 관련성 높은 조항을 먼저 배치하세요."""


# System prompt for regulation Q&A (loaded from prompts.json)
# System prompt for regulation Q&A (loaded from prompts.json)
REGULATION_QA_PROMPT = (
    _load_prompt("regulation_qa") or _get_fallback_regulation_qa_prompt()
)


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


@dataclass
class MultiHopResult:
    """Result of multi-hop query execution."""

    answer: Answer
    hops: List[Dict[str, Any]]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


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
        hybrid_searcher: Optional[IHybridSearcher] = None,
        use_hybrid: Optional[bool] = None,
        reranker: Optional[IReranker] = None,
        enable_warmup: Optional[bool] = None,
        hallucination_filter: Optional[HallucinationFilter] = None,
        period_keyword_detector: Optional["PeriodKeywordDetector"] = None,
        academic_calendar_service: Optional["AcademicCalendarService"] = None,
    ):
        """
        Initialize search use case.

        Args:
            store: Vector store implementation.
            llm_client: Optional LLM client for generating answers.
            use_reranker: Whether to use BGE reranker (default: from config).
            hybrid_searcher: Optional HybridSearcher (auto-created if None and use_hybrid=True).
            use_hybrid: Whether to use hybrid search (default: from config).
            reranker: Optional reranker implementation (auto-created if None and use_reranker=True).
            enable_warmup: Whether to warmup in background (default: WARMUP_ON_INIT env).
            hallucination_filter: Optional HallucinationFilter (auto-created from config if None).
            period_keyword_detector: Optional PeriodKeywordDetector for deadline/date queries.
            academic_calendar_service: Optional AcademicCalendarService for academic calendar info.
        """
        # Use config defaults if not explicitly specified
        from ..config import get_config
        from ..infrastructure.llm_adapter import LLMClientAdapter

        config = get_config()

        self.store = store

        # Auto-wrap LLMClient with LLMClientAdapter for generate() method
        if llm_client is not None:
            # Check if it's already an LLMClientAdapter (has generate method)
            if hasattr(llm_client, "generate"):
                self.llm = llm_client
            else:
                # Wrap raw LLMClient with LLMClientAdapter
                logger.info(
                    "Wrapping LLMClient with LLMClientAdapter for generate() method"
                )
                # Extract provider, model, base_url from LLMClient if possible
                provider = getattr(llm_client, "provider", "ollama")
                model = getattr(llm_client, "model", None)
                base_url = getattr(llm_client, "base_url", None)
                self.llm = LLMClientAdapter(
                    provider=provider, model=model, base_url=base_url
                )
        else:
            self.llm = None

        self.use_reranker = (
            use_reranker if use_reranker is not None else config.use_reranker
        )
        self._hybrid_searcher = hybrid_searcher
        self._use_hybrid = use_hybrid if use_hybrid is not None else config.use_hybrid
        self._hybrid_initialized = hybrid_searcher is not None
        self._last_query_rewrite: Optional[QueryRewriteInfo] = None
        self._reranker = reranker
        self._reranker_initialized = reranker is not None

        # Corrective RAG components
        self._retrieval_evaluator = None
        self._corrective_rag_enabled = True
        self._crag_retriever = None  # Lazy initialized in _apply_corrective_rag

        # HyDE components (from config)
        self._enable_hyde = config.enable_hyde
        self._hyde_generator = None  # Lazy initialized
        self._hyde_searcher = None  # Lazy initialized

        # Self-RAG components (from config)
        self._enable_self_rag = config.enable_self_rag
        self._self_rag_pipeline = None  # Lazy initialized

        # Fact check components (from config)
        self._enable_fact_check = config.enable_fact_check
        self._fact_check_max_retries = config.fact_check_max_retries
        self._fact_checker = None  # Lazy initialized

        # Hallucination filter components (SPEC-RAG-Q-002)
        if hallucination_filter is not None:
            self.hallucination_filter = hallucination_filter
        elif config.enable_hallucination_filter:
            try:
                mode = FilterMode(config.hallucination_filter_mode)
                self.hallucination_filter = HallucinationFilter(mode=mode)
                logger.info(f"Hallucination filter enabled (mode={mode.value})")
            except ValueError as e:
                logger.warning(
                    f"Invalid hallucination filter mode '{config.hallucination_filter_mode}': {e}. "
                    f"Using SANITIZE mode."
                )
                self.hallucination_filter = HallucinationFilter(
                    mode=FilterMode.SANITIZE
                )
        else:
            self.hallucination_filter = None

        # Confidence threshold for fallback response (TAG-001: Prevent Hallucination)
        # When confidence score is below this threshold, return fallback message
        # instead of generating potentially hallucinated content
        self._confidence_threshold = config.confidence_threshold
        logger.info(f"Confidence threshold set to {self._confidence_threshold}")

        # Period Keyword Detection (SPEC-RAG-Q-003)
        # Initialize PeriodKeywordDetector for deadline/date queries
        if period_keyword_detector is not None:
            self._period_keyword_detector = period_keyword_detector
        else:
            # Lazy import to avoid circular dependency
            from ..infrastructure.period_keyword_detector import PeriodKeywordDetector

            self._period_keyword_detector = PeriodKeywordDetector()
            logger.debug("PeriodKeywordDetector initialized")

        # Academic Calendar Service (SPEC-RAG-Q-003)
        # Optional service - will be None until implemented
        self._academic_calendar_service = academic_calendar_service
        if academic_calendar_service is not None:
            logger.info("AcademicCalendarService enabled for period queries")

        # Dynamic Query Expansion components (Phase 3)
        self._enable_query_expansion = config.enable_query_expansion
        self._query_expander = None  # Lazy initialized

        # Phase 1 Integration: Query Expansion Service
        self._query_expansion_service = None  # Will be initialized if enabled

        # Persona-Aware Response Generation (TAG-005: REQ-004)
        # Lazy initialized PersonaAwareGenerator for persona-based prompt enhancement
        self._persona_generator = None

        # Multi-hop Question Answering components
        self._enable_multi_hop = getattr(config, "enable_multi_hop", True)
        self._multi_hop_handler = None  # Lazy initialized
        self._max_hops = getattr(config, "max_hops", 5)
        self._hop_timeout_seconds = getattr(config, "hop_timeout_seconds", 30)

        # Ambiguity Classifier (SPEC-RAG-001 Component 2)
        from ..domain.llm.ambiguity_classifier import AmbiguityClassifier

        self.ambiguity_classifier = AmbiguityClassifier()

        # Reranking metrics (Cycle 3)
        self._reranking_metrics = RerankingMetrics()

        # Citation Verification Service (SPEC-RAG-Q-004)
        # Lazy initialized to avoid circular imports
        self._citation_verification_service: Optional["CitationVerificationService"] = (
            None
        )

        # Cache initialization
        self._enable_cache = config.enable_cache
        self._query_cache = None
        if self._enable_cache:
            self._query_cache = RAGQueryCache(
                enabled=config.enable_cache,
                ttl_hours=config.cache_ttl_hours,
                redis_host=config.redis_host,
                redis_port=config.redis_port,
                redis_password=config.redis_password,
                cache_dir=str(config.cache_dir_resolved),
            )
            logger.info(f"RAG query cache enabled (TTL={config.cache_ttl_hours}h)")

        # Long-term Memory (conversation context) initialization
        self._enable_conversation_memory = getattr(
            config, "enable_conversation_memory", True
        )
        self._memory_manager = None
        if self._enable_conversation_memory:
            from .conversation_memory import create_memory_manager

            enable_mcp = getattr(config, "enable_memory_mcp", False)
            self._memory_manager = create_memory_manager(
                llm_client=self.llm, enable_mcp=enable_mcp
            )
            logger.info("Conversation memory enabled with MCP=%s", enable_mcp)

        # Background warmup
        if enable_warmup is None:
            enable_warmup = os.environ.get("WARMUP_ON_INIT", "").lower() == "true"
        if enable_warmup:
            threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self) -> None:
        """Pre-initialize components in background for faster first query."""
        try:
            logger.info("Starting background warmup...")
            self._ensure_hybrid_searcher()
            if self.use_reranker:
                self._ensure_reranker()
            logger.info("Background warmup completed.")
        except Exception as e:
            logger.warning(f"Background warmup failed: {e}")

    @property
    def hybrid_searcher(self) -> Optional[IHybridSearcher]:
        """Lazy-initialize HybridSearcher on first access."""
        if self._use_hybrid and not self._hybrid_initialized:
            self._ensure_hybrid_searcher()
        return self._hybrid_searcher

    def _ensure_hybrid_searcher(self) -> None:
        """Initialize HybridSearcher with documents from vector store."""
        if self._hybrid_initialized:
            return

        from ..config import get_config
        from ..infrastructure.hybrid_search import HybridSearcher

        config = get_config()
        index_cache_path = None
        if config.bm25_index_cache_path_resolved:
            index_cache_path = str(config.bm25_index_cache_path_resolved)

        # Get all documents from store for BM25 indexing
        documents = self.store.get_all_documents()
        if documents:
            self._hybrid_searcher = HybridSearcher(index_cache_path=index_cache_path)
            self._hybrid_searcher.add_documents(documents)
            # Set LLM client for query rewriting if available
            if self.llm:
                self._hybrid_searcher.set_llm_client(self.llm)

        self._hybrid_initialized = True

    # --- Phase 2: Search Strategy Branching ---

    # Patterns for simple factual queries
    SIMPLE_FACTUAL_PATTERNS = [
        re.compile(r"^.{2,15}(이|가)\s*(몇|얼마|언제)", re.UNICODE),  # "졸업학점이 몇"
        re.compile(
            r"^.{2,15}\s*(어떻게|뭐야|뭔가요|몇이야)", re.UNICODE
        ),  # "승진 기준이 어떻게"
        re.compile(
            r"^.{2,20}(필요해\??|있어\??|돼\??)$", re.UNICODE
        ),  # "영어 점수도 필요해?"
        re.compile(r"^.{2,15}\s*기준", re.UNICODE),  # "장학금 성적 기준"
    ]

    def _determine_search_strategy(self, query: str) -> SearchStrategy:
        """
        Determine search strategy based on query characteristics.

        Simple factual queries can bypass tool calling for faster response.
        Complex queries benefit from agent-based multi-step reasoning.

        Args:
            query: User query text.

        Returns:
            SearchStrategy.DIRECT for simple queries, TOOL_CALLING otherwise.
        """
        query = query.strip()

        # Short queries (≤15 chars) are likely simple factual
        if len(query) <= 15:
            logger.debug(f"Short query detected ({len(query)} chars) - DIRECT strategy")
            return SearchStrategy.DIRECT

        # Check against simple factual patterns
        if self._is_simple_factual(query):
            logger.debug("Simple factual query detected - DIRECT strategy")
            return SearchStrategy.DIRECT

        # Default to tool calling for complex queries
        return SearchStrategy.TOOL_CALLING

    def _is_simple_factual(self, query: str) -> bool:
        """
        Detect simple factual queries that can be answered directly.

        Examples:
            - "졸업학점이 몇 학점이야?"
            - "교수 승진 기준이 어떻게 됩니까?"
            - "장학금 받으려면 성적이 몇 점이어야 해?"

        Args:
            query: User query text.

        Returns:
            True if query matches simple factual patterns.
        """
        for pattern in self.SIMPLE_FACTUAL_PATTERNS:
            if pattern.search(query):
                return True
        return False

    def get_recommended_strategy(self, query: str) -> SearchStrategy:
        """
        Public method to get recommended search strategy for a query.

        This can be used by QueryHandler to decide whether to use
        tool calling or direct search.

        Args:
            query: User query text.

        Returns:
            Recommended SearchStrategy.
        """
        return self._determine_search_strategy(query)

    def check_ambiguity(
        self, query_text: str
    ) -> tuple[bool, Optional["DisambiguationDialog"]]:
        """
        Check query for ambiguity and return disambiguation dialog if needed.

        This method implements REQ-AMB-001 through REQ-AMB-015 from SPEC-RAG-001.

        Args:
            query_text: User's search query text.

        Returns:
            Tuple of (needs_clarification, disambiguation_dialog).
            - needs_clarification: True if query is AMBIGUOUS or HIGHLY_AMBIGUOUS
            - disambiguation_dialog: Dialog with clarification options, or None if CLEAR
        """
        from ..domain.llm.ambiguity_classifier import AmbiguityLevel

        classification = self.ambiguity_classifier.classify(query_text)

        if classification.level == AmbiguityLevel.CLEAR:
            return False, None

        # Generate disambiguation dialog for ambiguous queries
        dialog = self.ambiguity_classifier.generate_disambiguation_dialog(
            classification
        )
        return True, dialog

    def apply_disambiguation(
        self, original_query: str, selected_option_index: int
    ) -> str:
        """
        Apply user's disambiguation selection to clarify query.

        Args:
            original_query: Original user query.
            selected_option_index: Index of selected option from disambiguation dialog.

        Returns:
            Clarified query text.
        """

        classification = self.ambiguity_classifier.classify(original_query)
        dialog = self.ambiguity_classifier.generate_disambiguation_dialog(
            classification
        )

        if dialog and 0 <= selected_option_index < len(dialog.options):
            selected_option = dialog.options[selected_option_index]
            return self.ambiguity_classifier.apply_user_selection(
                original_query, selected_option
            )

        # If invalid selection, return original
        return original_query

    def search(
        self,
        query_text: str,
        filter: Optional[SearchFilter] = None,
        top_k: int = 10,
        include_abolished: bool = False,
        audience_override: Optional["Audience"] = None,
    ) -> List[SearchResult]:
        """
        Search for relevant regulation chunks.

        Args:
            query_text: The search query.
            filter: Optional metadata filters.
            top_k: Maximum number of results.
            include_abolished: Whether to include abolished regulations.
            audience_override: Optional audience override for ranking penalties.

        Returns:
            List of SearchResult sorted by relevance.
        """
        query_text = _coerce_query_text(query_text).strip()
        if not query_text:
            return []

        query_text = unicodedata.normalize("NFC", query_text)

        # 1. Rule code pattern (e.g., "3-1-24")
        if RULE_CODE_PATTERN.match(query_text):
            return self._search_by_rule_code_pattern(
                query_text, filter, top_k, include_abolished
            )

        # 2. Regulation name only (e.g., "교원인사규정")
        reg_only = _extract_regulation_only_query(query_text)
        if reg_only:
            result = self._search_by_regulation_only(
                query_text, reg_only, filter, top_k, include_abolished
            )
            if result is not None:
                return result

        # 3. Regulation + article (e.g., "교원인사규정 제8조")
        reg_article = _extract_regulation_article_query(query_text)
        if reg_article:
            return self._search_by_regulation_article(
                query_text, reg_article, filter, top_k, include_abolished
            )

        # 4. General search with hybrid/reranking
        return self._search_general(
            query_text, filter, top_k, include_abolished, audience_override
        )

    def _search_by_rule_code_pattern(
        self,
        query_text: str,
        filter: Optional[SearchFilter],
        top_k: int,
        include_abolished: bool,
    ) -> List[SearchResult]:
        """Handle rule code pattern search (e.g., '3-1-24')."""
        self._last_query_rewrite = QueryRewriteInfo(
            original=query_text,
            rewritten=query_text,
            used=False,
            method=None,
            from_cache=False,
            fallback=False,
            used_synonyms=None,
            used_intent=None,
            matched_intents=None,
        )
        rule_filter = self._build_rule_code_filter(filter, query_text)
        query = Query(text="규정", include_abolished=include_abolished)
        results = self.store.search(query, rule_filter, top_k * 5)
        return self._deduplicate_by_article(results, top_k)

    def _search_by_regulation_only(
        self,
        query_text: str,
        reg_only: str,
        filter: Optional[SearchFilter],
        top_k: int,
        include_abolished: bool,
    ) -> Optional[List[SearchResult]]:
        """Handle regulation name only search (e.g., '교원인사규정')."""
        self._last_query_rewrite = QueryRewriteInfo(
            original=query_text,
            rewritten=reg_only,
            used=True,
            method="regulation_only",
            from_cache=False,
            fallback=False,
            used_synonyms=False,
            used_intent=False,
            matched_intents=None,
        )

        # Find the regulation's rule_code
        target_rule_code = self._find_regulation_rule_code(
            reg_only, filter, include_abolished
        )

        if not target_rule_code:
            return None  # Fall through to general search

        # Get all articles from this regulation
        rule_filter = self._build_rule_code_filter(filter, target_rule_code)
        all_chunks = self.store.search(
            Query(text="규정 조항", include_abolished=include_abolished),
            rule_filter,
            200,
        )

        raw_results = [
            SearchResult(chunk=r.chunk, score=r.score, rank=i + 1)
            for i, r in enumerate(all_chunks)
        ]
        return self._deduplicate_by_article(raw_results, top_k)

    def _search_by_regulation_article(
        self,
        query_text: str,
        reg_article: tuple,
        filter: Optional[SearchFilter],
        top_k: int,
        include_abolished: bool,
    ) -> List[SearchResult]:
        """Handle regulation + article search (e.g., '교원인사규정 제8조')."""
        reg_name, article_ref = reg_article
        self._last_query_rewrite = QueryRewriteInfo(
            original=query_text,
            rewritten=f"{reg_name} {article_ref}",
            used=True,
            method="regulation_article",
            from_cache=False,
            fallback=False,
            used_synonyms=False,
            used_intent=False,
            matched_intents=None,
        )

        # Find regulation's rule_code
        target_rule_code = self._find_regulation_rule_code(
            reg_name, filter, include_abolished, exact_match_priority=True
        )

        if not target_rule_code:
            return []

        # Get chunks from this regulation
        rule_filter = self._build_rule_code_filter(filter, target_rule_code)
        all_chunks = self.store.search(
            Query(
                text=f"{reg_name} {article_ref}", include_abolished=include_abolished
            ),
            rule_filter,
            500,
        )

        # Filter by article number match
        normalized_article = normalize_article_token(article_ref)
        filtered_results = []
        for r in all_chunks:
            article_haystack = r.chunk.embedding_text or r.chunk.text
            text_articles = {
                normalize_article_token(t)
                for t in ARTICLE_PATTERN.findall(article_haystack)
            }

            if normalized_article in text_articles:
                filtered_results.append(
                    SearchResult(
                        chunk=r.chunk, score=1.0, rank=len(filtered_results) + 1
                    )
                )
            elif any(normalized_article in ta for ta in text_articles):
                filtered_results.append(
                    SearchResult(
                        chunk=r.chunk, score=0.8, rank=len(filtered_results) + 1
                    )
                )

        filtered_results.sort(key=lambda x: -x.score)
        filtered_results.sort(key=lambda x: -x.score)
        return self._deduplicate_by_article(filtered_results, top_k)

    def _find_regulation_rule_code(
        self,
        reg_name: str,
        filter: Optional[SearchFilter],
        include_abolished: bool,
        exact_match_priority: bool = False,
    ) -> Optional[str]:
        """Find rule_code for a regulation name."""
        reg_query = Query(text=reg_name, include_abolished=include_abolished)
        reg_results = self.store.search(reg_query, filter, 50)

        target_rule_code = None
        best_score = 0.0
        for r in reg_results:
            chunk_reg_name = (
                r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            ) or ""
            if reg_name in chunk_reg_name or chunk_reg_name in reg_name:
                if chunk_reg_name == reg_name:
                    match_score = 1.0
                elif chunk_reg_name.endswith(reg_name):
                    match_score = 0.9
                elif exact_match_priority:
                    match_score = 0.5
                else:
                    match_score = 0.8
                if match_score > best_score:
                    best_score = match_score
                    target_rule_code = r.chunk.rule_code
                    if match_score == 1.0:
                        break
        return target_rule_code

    def _search_composite(
        self,
        sub_queries: List[str],
        original_query: str,
        filter: Optional[SearchFilter],
        top_k: int,
        include_abolished: bool,
        audience_override: Optional["Audience"],
    ) -> List[SearchResult]:
        """
        Search with composite query decomposition and RRF merge.

        Searches each sub-query separately and merges results using
        Reciprocal Rank Fusion (RRF) for better coverage.

        Args:
            sub_queries: List of decomposed sub-queries.
            original_query: The original composite query.
            filter: Optional metadata filters.
            top_k: Maximum number of results.
            include_abolished: Whether to include abolished regulations.
            audience_override: Optional audience override.

        Returns:
            Merged and deduplicated results.
        """
        logger.info(
            "Composite search: decomposed '%s' into %d sub-queries: %s",
            original_query,
            len(sub_queries),
            sub_queries,
        )

        # Collect results from each sub-query
        all_results: Dict[str, List[tuple]] = {}  # chunk_id -> [(rank, result), ...]

        for _sq_idx, sub_query in enumerate(sub_queries):
            # Search each sub-query (avoid recursive decomposition)
            query = Query(text=sub_query, include_abolished=include_abolished)

            # Get dense results
            dense_results = self.store.search(query, filter, top_k * 3)

            # Apply hybrid search if available
            if self.hybrid_searcher:
                from ..infrastructure.hybrid_search import ScoredDocument

                sparse_results = self.hybrid_searcher.search_sparse(
                    sub_query, top_k * 2
                )
                sparse_results = self._filter_sparse_results(
                    sparse_results, filter=filter, include_abolished=include_abolished
                )
                dense_docs = [
                    ScoredDocument(
                        doc_id=r.chunk.id,
                        score=r.score,
                        content=r.chunk.text,
                        metadata=r.chunk.to_metadata(),
                    )
                    for r in dense_results
                ]
                fused = self.hybrid_searcher.fuse_results(
                    sparse_results=sparse_results,
                    dense_results=dense_docs,
                    query_text=sub_query,
                )
                # Convert back to SearchResult
                sub_results = []
                for doc in fused:
                    for orig in dense_results:
                        if orig.chunk.id == doc.doc_id:
                            sub_results.append(
                                SearchResult(chunk=orig.chunk, score=doc.score, rank=0)
                            )
                            break
            else:
                sub_results = dense_results

            # Record rank for RRF
            for rank, result in enumerate(sub_results, start=1):
                chunk_id = result.chunk.id
                if chunk_id not in all_results:
                    all_results[chunk_id] = []
                all_results[chunk_id].append((rank, result))

        # RRF fusion across sub-queries
        rrf_k = 60  # Standard RRF constant
        rrf_scores: Dict[str, float] = {}
        best_results: Dict[str, SearchResult] = {}

        for chunk_id, rank_result_list in all_results.items():
            rrf_score = sum(1.0 / (rrf_k + rank) for rank, _ in rank_result_list)
            rrf_scores[chunk_id] = rrf_score
            # Keep the result with best original score
            best_result = max(rank_result_list, key=lambda x: x[1].score)[1]
            best_results[chunk_id] = SearchResult(
                chunk=best_result.chunk, score=rrf_score, rank=0
            )

        # Sort by RRF score
        merged = sorted(best_results.values(), key=lambda x: -x.score)

        # Assign ranks
        for i, r in enumerate(merged):
            r.rank = i + 1

        # Store query rewrite info
        self._last_query_rewrite = QueryRewriteInfo(
            original=original_query,
            rewritten=" | ".join(sub_queries),
            used=True,
            method="composite_decomposition",
            from_cache=False,
            fallback=False,
            used_synonyms=True,
            used_intent=True,
            matched_intents=[f"sub:{sq}" for sq in sub_queries],
        )

        logger.info(
            "Composite search merged %d unique results from %d sub-queries",
            len(merged),
            len(sub_queries),
        )

        # Deduplicate by article
        return self._deduplicate_by_article(merged, top_k)

    def _deduplicate_by_article(
        self, results: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """
        Deduplicate results to ensure only one chunk per article is returned.

        Key logic:
        - Identify the 'Article' context for each chunk (Regulation + Article Number).
        - Keep only the highest-scoring chunk for each Article context.
        - Chunks not belonging to an article (e.g. regulation metadata) are always kept (unless duplicates by ID).
        """
        seen_keys = set()
        unique_results = []

        for result in results:
            # 1. Generate deduplication key
            # Key format: (rule_code, article_identifier)
            # If no article identifier found, use (rule_code, chunk_id) to allow unique non-article chunks.

            chunk = result.chunk
            article_key = None

            # Check title first
            if chunk.level == ChunkLevel.ARTICLE:
                article_key = chunk.title

            # Check parent path if not found in title (or if level is paragraph)
            # We look for the "Article" node in the parent path
            if not article_key and chunk.parent_path:
                for path_item in reversed(chunk.parent_path):
                    # Check if path item matches "Article N" pattern
                    # We use simple string check or regex
                    if ARTICLE_PATTERN.match(path_item):
                        article_key = path_item
                        break

            if article_key:
                # Normalize key to handle slight variations if needed,
                # strictly we use the string as is assuming consistent naming in same reg
                key = (chunk.rule_code, article_key)
            else:
                # Not an article chunk (or preamble, etc), treat as unique by ID
                key = (chunk.rule_code, chunk.id)

            if key not in seen_keys:
                seen_keys.add(key)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break

        return unique_results

    def _classify_query_complexity(
        self, query_text: str, matched_intents: Optional[List[str]] = None
    ) -> str:
        """
        Classify query complexity for Adaptive RAG strategy selection.

        Returns:
            - "simple": Structural queries (Article/Regulation name) - Fast retrieval
            - "medium": Standard natural language queries - Hybrid + Reranker
            - "complex": Multi-intent or comparative queries - Full pipeline, no reranker

        The reranker may have adverse effects on intent-expanded queries because
        the reranker scores based on original query, not the expanded keywords.
        """
        # Check for structural patterns (simple)
        if RULE_CODE_PATTERN.match(query_text):
            return "simple"
        if _extract_regulation_only_query(query_text):
            return "simple"
        if _extract_regulation_article_query(query_text):
            return "simple"
        if HEADING_ONLY_PATTERN.match(query_text):
            return "simple"

        # Check for complex queries
        # 1. Multiple intents matched
        if matched_intents and len(matched_intents) >= 2:
            return "complex"

        # 2. Comparative or analytical keywords
        complex_markers = ["비교", "차이", "어떤 게 나아", "뭐가 다르", "vs", "또는"]
        if any(marker in query_text for marker in complex_markers):
            return "complex"

        # 3. Very long queries (likely complex multi-part questions)
        if len(query_text) > 80:
            return "complex"

        # Default: medium complexity
        return "medium"

    def _should_skip_reranker(
        self,
        complexity: str,
        matched_intents: Optional[List[str]] = None,
        query_significantly_expanded: bool = False,
        query_type: Optional["QueryType"] = None,
        query_text: str = "",
    ) -> tuple[bool, str]:
        """
        Determine if reranker should be skipped for this query.

        Reranking is skipped for:
        - Simple, precise queries (article references, regulation names)
        - Very short queries (< 15 characters) where keyword matching is sufficient
        - Queries already well-matched by BM25+Dense hybrid search

        Reranking is applied for:
        - Natural language questions requiring semantic understanding
        - Intent-based queries where LLM expansion was used
        - Complex queries needing cross-encoder refinement

        Args:
            complexity: Query complexity classification (simple, medium, complex)
            matched_intents: List of matched intent IDs
            query_significantly_expanded: Whether query was heavily expanded
            query_type: QueryType enum (ARTICLE_REFERENCE, REGULATION_NAME, etc.)
            query_text: Original query text for length check

        Returns:
            Tuple of (should_skip: bool, reason: str).
            reason is one of: article_reference, regulation_name, short_simple, no_intent, apply.
        """
        # Skip reranker for article references (already precise with BM25)
        if query_type == QueryType.ARTICLE_REFERENCE:
            logger.debug(
                f"Skipping reranker for article reference query: {query_text[:30]}..."
            )
            return True, "article_reference"

        # Skip reranker for regulation name searches (exact string match is sufficient)
        if query_type == QueryType.REGULATION_NAME:
            logger.debug(
                f"Skipping reranker for regulation name query: {query_text[:30]}..."
            )
            return True, "regulation_name"

        # Skip reranker for very short simple queries
        if complexity == "simple" and len(query_text.strip()) < 15:
            logger.debug(
                f"Skipping reranker for short simple query: {query_text[:30]}..."
            )
            return True, "short_simple"

        # Skip reranker for simple queries without intent matching
        if complexity == "simple" and not matched_intents:
            logger.debug(
                f"Skipping reranker for simple query without intent: {query_text[:30]}..."
            )
            return True, "no_intent"

        # Apply reranker for all other cases
        return False, "apply"

    def _search_general(
        self,
        query_text: str,
        filter: Optional[SearchFilter],
        top_k: int,
        include_abolished: bool,
        audience_override: Optional["Audience"],
    ) -> List[SearchResult]:
        """Perform general search with hybrid search, scoring, and reranking."""
        # Check cache first
        cached_results = self._check_retrieval_cache(
            query_text, filter, top_k, include_abolished
        )
        if cached_results is not None:
            return cached_results

        # Step 4/5: Composite query decomposition and merging
        if self.hybrid_searcher:
            sub_queries = self.hybrid_searcher._query_analyzer.decompose_query(
                query_text
            )
            if len(sub_queries) >= 2:
                return self._search_composite(
                    sub_queries,
                    query_text,
                    filter,
                    top_k,
                    include_abolished,
                    audience_override,
                )

        # Query rewriting
        query, rewritten_query_text = self._perform_query_rewriting(
            query_text, include_abolished
        )
        scoring_query_text = self._select_scoring_query(
            query_text, rewritten_query_text
        )

        # Detect audience
        audience = self._detect_audience(query_text, audience_override)

        # Determine recall multiplier based on query type
        is_intent = False
        if self._last_query_rewrite and (
            self._last_query_rewrite.used_intent
            or self._last_query_rewrite.method == "llm"
        ):
            is_intent = True

        # Increase recall for intent/llm queries to ensure correct candidates are found
        fetch_k = top_k * 6 if is_intent else top_k * 3

        # Adaptive RAG: Classify query complexity early for HyDE decision
        matched_intents = (
            self._last_query_rewrite.matched_intents
            if self._last_query_rewrite
            else None
        )
        complexity = self._classify_query_complexity(query_text, matched_intents)

        # Get query type for conditional reranking
        query_type = None
        # Note: classify_query method not available, using classify_intent if needed
        # if self.hybrid_searcher:
        #     query_type = self.hybrid_searcher._query_analyzer.classify_intent(query_text)

        # Phase 4 (Hybrid): Apply LLM-based dynamic expansion FIRST (primary)
        # Then merge with intent/synonym matching (secondary)
        expanded_query_text = query_text
        combined_keywords = []

        # 1. Always try LLM expansion first (with cache for speed)
        llm_expanded, llm_keywords = self._apply_dynamic_expansion(query_text)
        if llm_keywords:
            combined_keywords.extend(llm_keywords)
            logger.debug(
                f"LLM expansion (primary): {query_text[:30]}... -> "
                f"keywords={llm_keywords[:3]}"
            )

        # 2. Add intent-matched keywords if available (secondary)
        if matched_intents and self._last_query_rewrite:
            # Intent keywords are already in rewritten_query_text from _perform_query_rewriting
            # We add any unique keywords from the rewrite that aren't in LLM results
            intent_words = set(rewritten_query_text.split()) - set(query_text.split())
            for word in intent_words:
                if word not in combined_keywords and len(word) >= 2:
                    combined_keywords.append(word)

        # 3. Build final expanded query with combined keywords
        if combined_keywords:
            # Deduplicate and limit keywords
            unique_keywords = list(dict.fromkeys(combined_keywords))[:7]
            existing_words = set(query_text.lower().split())
            new_keywords = [
                kw for kw in unique_keywords if kw.lower() not in existing_words
            ]
            if new_keywords:
                expanded_query_text = f"{query_text} {' '.join(new_keywords[:4])}"
                logger.debug(
                    f"Hybrid expansion: {query_text[:20]}... -> "
                    f"combined={new_keywords[:4]} (LLM+Intent)"
                )

        # Get dense results (use expanded query if expansion was applied)
        search_query = Query(text=expanded_query_text) if combined_keywords else query
        dense_results = self.store.search(search_query, filter, fetch_k)

        # Apply HyDE for vague/complex queries (merge with dense results)
        if self._should_use_hyde(query_text, complexity):
            hyde_results = self._apply_hyde(query_text, filter, fetch_k // 2)
            if hyde_results:
                # Merge HyDE results with dense results
                seen_ids = {r.chunk.id for r in dense_results}
                for r in hyde_results:
                    if r.chunk.id not in seen_ids:
                        seen_ids.add(r.chunk.id)
                        dense_results.append(r)

        # Apply hybrid search if available
        results = self._apply_hybrid_search(
            dense_results,
            query_text,
            rewritten_query_text,
            filter,
            include_abolished,
            fetch_k // 2,
        )

        # Apply score bonuses
        boosted_results = self._apply_score_bonuses(
            results, query_text, scoring_query_text, audience
        )

        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: -x.score)

        # Determine if reranker should be skipped (Cycle 2: Conditional Reranking)
        skip_reranker, skip_reason = self._should_skip_reranker(
            complexity,
            matched_intents,
            query_significantly_expanded=False,
            query_type=query_type,
            query_text=query_text,
        )

        # Record query and apply/skip decision (Cycle 3: Metrics)
        self._reranking_metrics.record_query()
        if skip_reranker:
            self._reranking_metrics.record_skip(
                query_type=query_type.name if query_type else None,
                reason=skip_reason,
            )
        # Apply reranking if enabled and not skipped by Adaptive RAG
        if self.use_reranker and boosted_results and not skip_reranker:
            import time

            start_time = time.time()
            rerank_k = top_k * 5 if is_intent else top_k * 2
            boosted_results = self._apply_reranking(
                boosted_results, scoring_query_text, top_k, candidate_k=rerank_k
            )
            reranker_time_ms = (time.time() - start_time) * 1000
            self._reranking_metrics.record_apply(
                query_type=query_type.name if query_type else None,
                reranker_time_ms=reranker_time_ms,
            )
        elif skip_reranker and boosted_results:
            logger.debug(
                f"Adaptive RAG: Skipping reranker for complexity={complexity}, "
                f"matched_intents={matched_intents}, reason={skip_reason}"
            )

        # Corrective RAG: Check if results need correction (with dynamic threshold)
        if self._corrective_rag_enabled and boosted_results:
            boosted_results = self._apply_corrective_rag(
                query_text,
                boosted_results,
                filter,
                top_k,
                include_abolished,
                audience_override,
                complexity=complexity,  # Pass complexity for dynamic threshold
            )

        # Deduplicate by article (One Chunk per Article)
        # Store results in cache
        self._store_retrieval_cache(
            query_text, boosted_results, filter, top_k, include_abolished
        )

        final_results = self._deduplicate_by_article(boosted_results, top_k)
        logger.info(
            f"DEBUG: _search_general returning {len(final_results)} results (boosted: {len(boosted_results)})"
        )
        return final_results

    def _perform_query_rewriting(
        self, query_text: str, include_abolished: bool
    ) -> tuple:
        """Perform query rewriting using hybrid searcher if available."""
        rewritten_query_text = query_text
        rewrite_used = False
        rewrite_method: Optional[str] = None
        rewrite_from_cache = False
        rewrite_fallback = False
        used_synonyms: Optional[bool] = None
        used_intent: Optional[bool] = None
        matched_intents: Optional[List[str]] = None

        if self.hybrid_searcher:
            if self.llm and not self.hybrid_searcher._query_analyzer._llm_client:
                self.hybrid_searcher.set_llm_client(self.llm)
            rewrite_info = self.hybrid_searcher._query_analyzer.rewrite_query_with_info(
                query_text
            )
            rewritten_query_text = _coerce_query_text(rewrite_info.rewritten).strip()
            if not rewritten_query_text:
                rewritten_query_text = query_text
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
                used_synonyms = used_synonyms or analyzer.has_synonyms(
                    rewritten_query_text
                )

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

        query = Query(text=rewritten_query_text, include_abolished=include_abolished)
        return query, rewritten_query_text

    def _detect_audience(
        self, query_text: str, audience_override: Optional["Audience"]
    ) -> Optional["Audience"]:
        """Detect audience from query if not overridden."""
        if audience_override is not None:
            return audience_override
        if self.hybrid_searcher:
            return self.hybrid_searcher._query_analyzer.detect_audience(query_text)
        return None

    def _apply_hybrid_search(
        self,
        dense_results: List[SearchResult],
        query_text: str,
        rewritten_query_text: str,
        filter: Optional[SearchFilter],
        include_abolished: bool,
        top_k: int,
    ) -> List[SearchResult]:
        """Apply hybrid search (BM25 + dense) with RRF fusion."""
        if not self.hybrid_searcher:
            return dense_results

        from ..infrastructure.hybrid_search import ScoredDocument

        sparse_query_text = rewritten_query_text or query_text
        sparse_results = self.hybrid_searcher.search_sparse(
            sparse_query_text, top_k * 3
        )
        sparse_results = self._filter_sparse_results(
            sparse_results, filter=filter, include_abolished=include_abolished
        )

        # Filter dense_results to ensure they meet filter criteria
        filtered_dense_results = self._filter_search_results(
            dense_results, filter=filter, include_abolished=include_abolished
        )

        dense_docs = [
            ScoredDocument(
                doc_id=r.chunk.id,
                score=r.score,
                content=r.chunk.text,
                metadata=r.chunk.to_metadata(),
            )
            for r in filtered_dense_results
        ]

        fused = self.hybrid_searcher.fuse_results(
            sparse_results=sparse_results,
            dense_results=dense_docs,
            top_k=top_k * 3,
            query_text=query_text,
        )

        id_to_result = {r.chunk.id: r for r in filtered_dense_results}
        results = []
        for i, doc in enumerate(fused):
            if doc.doc_id in id_to_result:
                original = id_to_result[doc.doc_id]
                results.append(
                    SearchResult(chunk=original.chunk, score=doc.score, rank=i + 1)
                )
            else:
                # Re-validate filter conditions for sparse documents
                if self._chunk_matches_filter(
                    doc.metadata, filter=filter, include_abolished=include_abolished
                ):
                    from ..domain.entities import Chunk

                    chunk = Chunk.from_metadata(doc.doc_id, doc.content, doc.metadata)
                    results.append(
                        SearchResult(chunk=chunk, score=doc.score, rank=i + 1)
                    )

        return results

    def _apply_score_bonuses(
        self,
        results: List[SearchResult],
        query_text: str,
        scoring_query_text: str,
        audience: Optional["Audience"],
    ) -> List[SearchResult]:
        """Apply keyword, article, and audience-based score bonuses/penalties."""
        query_terms = scoring_query_text.lower().split()
        query_text_lower = scoring_query_text.lower()
        boosted_results = []

        fundamental_codes = {"2-1-1", "3-1-5", "3-1-26", "1-0-1"}

        for r in results:
            text_lower = r.chunk.text.lower()
            matches = sum(1 for term in query_terms if term in text_lower)
            bonus = matches * 0.1

            # Fundamental regulation priority (Increased to 0.3 to meet evaluation thresholds)
            if r.chunk.rule_code in fundamental_codes:
                bonus += 0.3

            # Keyword bonus
            keyword_bonus = 0.0
            if r.chunk.keywords:
                keyword_hits = sum(
                    kw.weight
                    for kw in r.chunk.keywords
                    if kw.term and kw.term.lower() in query_text_lower
                )
                keyword_bonus = min(0.3, keyword_hits * 0.05)

            # Article match bonus
            article_bonus = 0.0
            query_articles = {
                normalize_article_token(t) for t in ARTICLE_PATTERN.findall(query_text)
            }
            if query_articles:
                article_haystack = r.chunk.embedding_text or r.chunk.text
                text_articles = {
                    normalize_article_token(t)
                    for t in ARTICLE_PATTERN.findall(article_haystack)
                }
                if query_articles & text_articles:
                    article_bonus = 0.2

            new_score = min(1.0, r.score + bonus + keyword_bonus + article_bonus)

            # Audience penalty
            new_score = self._apply_audience_penalty(r.chunk, audience, new_score)

            boosted_results.append(
                SearchResult(chunk=r.chunk, score=new_score, rank=r.rank)
            )

        return boosted_results

    def _apply_audience_penalty(
        self,
        chunk: Chunk,
        audience: Optional["Audience"],
        score: float,
    ) -> float:
        """Apply audience mismatch penalty to score."""
        if not audience:
            return score

        from ..infrastructure.query_analyzer import Audience

        reg_name = chunk.parent_path[0] if chunk.parent_path else chunk.title
        reg_name_lower = reg_name.lower()

        if audience == Audience.FACULTY:
            # Penalize student-specific regulations
            is_student_reg = any(
                k in reg_name_lower
                for k in ["학생", "학사", "장학", "수강", "졸업", "동아리"]
            )
            # But don't penalize if it explicitly mentions faculty/staff
            is_student_reg = is_student_reg and not any(
                k in reg_name_lower for k in ["교원", "직원", "교수", "인사"]
            )

            if is_student_reg:
                return score * 0.4  # Strong penalty

        elif audience == Audience.STUDENT:
            # Penalize faculty/staff-specific regulations
            is_faculty_reg = any(
                k in reg_name_lower
                for k in [
                    "교원",
                    "직원",
                    "인사",
                    "복무",
                    "업적",
                    "채용",
                    "연구년",
                    "조교",
                ]
            )
            # But don't penalize if it explicitly mentions students
            is_faculty_reg = is_faculty_reg and "학생" not in reg_name_lower

            if is_faculty_reg:
                return score * 0.4  # Strong penalty

        return score

    def _apply_reranking(
        self,
        results: List[SearchResult],
        scoring_query_text: str,
        top_k: int,
        candidate_k: Optional[int] = None,
        use_hybrid_scoring: bool = True,
        alpha: float = 0.7,
    ) -> List[SearchResult]:
        """
        Apply cross-encoder reranking with optional hybrid scoring.

        Hybrid Scoring combines reranker scores with keyword-boosted scores
        to preserve important keyword matches that reranker might miss.

        Formula: final_score = α * reranker_score + (1 - α) * boosted_score

        Args:
            results: Search results with boosted scores
            scoring_query_text: Query text for reranking
            top_k: Number of results to return
            candidate_k: Number of candidates to rerank
            use_hybrid_scoring: Whether to combine reranker + boost scores
            alpha: Weight for reranker score (0.7 = 70% reranker, 30% boost)
        """
        self._ensure_reranker()

        if candidate_k is None:
            # SPEC-RAG-QUALITY-004 Phase 2: Increase candidate pool for better recall
            # Previous: top_k * 2 (only 20 candidates for top_k=10)
            # Updated: top_k * 5 (50 candidates for top_k=10)
            candidate_k = top_k * 5

        candidates = results[:candidate_k]

        # Store original boosted scores for hybrid scoring
        id_to_boosted_score = {r.chunk.id: r.score for r in candidates}

        documents = [(r.chunk.id, r.chunk.text, {}) for r in candidates]
        reranked = self._reranker.rerank(scoring_query_text, documents, top_k=top_k)

        id_to_result = {r.chunk.id: r for r in candidates}
        final_results = []

        for i, rr in enumerate(reranked):
            doc_id, content, reranker_score, metadata = rr
            original = id_to_result.get(doc_id)
            if original:
                if use_hybrid_scoring:
                    # Hybrid scoring: combine reranker and boosted scores
                    boosted_score = id_to_boosted_score.get(doc_id, 0.0)
                    # Normalize reranker score to similar range as boosted score
                    # Reranker typically returns 0-1, boosted score is similar
                    final_score = alpha * reranker_score + (1 - alpha) * boosted_score
                    logger.debug(
                        f"Hybrid score: {final_score:.3f} = {alpha}*{reranker_score:.3f} + "
                        f"{1 - alpha}*{boosted_score:.3f} for {doc_id[:30]}"
                    )
                else:
                    final_score = reranker_score

                final_results.append(
                    SearchResult(chunk=original.chunk, score=final_score, rank=i + 1)
                )

        # Re-sort by final score (hybrid scoring may change order)
        if use_hybrid_scoring:
            final_results.sort(key=lambda x: -x.score)
            # Update ranks
            for i, r in enumerate(final_results):
                final_results[i] = SearchResult(
                    chunk=r.chunk, score=r.score, rank=i + 1
                )

        return final_results

    def _ensure_reranker(self) -> None:
        """Initialize reranker if not already initialized."""
        if not self._reranker_initialized:
            from ..infrastructure.reranker import BGEReranker, warmup_reranker

            self._reranker = BGEReranker()
            # Pre-load the actual FlagEmbedding model to avoid cold start
            warmup_reranker()
            self._reranker_initialized = True

    def _ensure_hyde(self) -> None:
        """Initialize HyDE generator if not already initialized."""
        if self._hyde_generator is None and self._enable_hyde:
            from ..config import get_config
            from ..infrastructure.hyde import HyDEGenerator

            config = get_config()
            self._hyde_generator = HyDEGenerator(
                llm_client=self.llm,
                cache_dir=config.hyde_cache_dir,
                enable_cache=config.hyde_cache_enabled,
            )

    def _should_use_hyde(self, query_text: str, complexity: str) -> bool:
        """Determine if HyDE should be used for this query."""
        if not self._enable_hyde:
            return False

        self._ensure_hyde()
        if self._hyde_generator is None:
            return False

        return self._hyde_generator.should_use_hyde(query_text, complexity)

    def _apply_hyde(
        self,
        query_text: str,
        filter: Optional[SearchFilter],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Apply HyDE: generate hypothetical document and search with it.

        Returns additional results from HyDE-enhanced search.
        """
        self._ensure_hyde()
        if self._hyde_generator is None:
            return []

        try:
            # Generate hypothetical document
            hyde_result = self._hyde_generator.generate_hypothetical_doc(query_text)

            # Defensive check: Skip HyDE if hypothetical doc is empty
            if (
                not hyde_result.hypothetical_doc
                or not hyde_result.hypothetical_doc.strip()
            ):
                logger.info(
                    f"HyDE generated empty doc for '{query_text}', skipping HyDE search"
                )
                # Fall back to normal search
                from ..domain.value_objects import Query

                normal_query = Query(text=query_text)
                return self.store.search(normal_query, filter, top_k)

            # Search with hypothetical document
            from ..domain.value_objects import Query

            hyde_query = Query(text=hyde_result.hypothetical_doc)
            hyde_results = self.store.search(hyde_query, filter, top_k)

            logger.debug(
                f"HyDE: Generated doc (from_cache={hyde_result.from_cache}), "
                f"found {len(hyde_results)} results"
            )

            return hyde_results
        except Exception as e:
            logger.warning(f"HyDE search failed: {e}")
            return []

    def _ensure_query_expander(self) -> None:
        """Initialize DynamicQueryExpander if not already initialized."""
        if self._query_expander is None and self._enable_query_expansion:
            from ..config import get_config
            from ..infrastructure.query_expander import DynamicQueryExpander

            config = get_config()
            self._query_expander = DynamicQueryExpander(
                llm_client=self.llm,
                cache_dir=str(config.query_expansion_cache_dir_resolved),
                enable_cache=True,
            )

    def _ensure_query_expansion_service(self) -> None:
        """Initialize QueryExpansionService if not already initialized (Phase 1)."""
        if self._query_expansion_service is None and self._enable_query_expansion:
            try:
                from ..application.query_expansion import QueryExpansionService

                # Initialize QueryExpansionService with synonym-based expansion
                self._query_expansion_service = QueryExpansionService(
                    store=self.store,
                    synonym_service=None,  # Will use built-in academic synonyms
                    llm_client=None,  # No LLM needed for synonym-based expansion
                )
                logger.debug(
                    "QueryExpansionService initialized for synonym-based expansion"
                )
            except ImportError as e:
                logger.warning(f"Failed to import QueryExpansionService: {e}")
                self._query_expansion_service = None
            except Exception as e:
                logger.warning(f"Failed to initialize QueryExpansionService: {e}")
                self._query_expansion_service = None

    def _apply_dynamic_expansion(self, query_text: str) -> tuple[str, list[str]]:
        """
        Apply dynamic query expansion using LLM and QueryExpansionService.

        Phase 1 Integration: Uses QueryExpansionService for synonym-based expansion.

        Args:
            query_text: Original query text.

        Returns:
            Tuple of (expanded_query, keywords).
        """
        if not self._enable_query_expansion:
            return query_text, []

        # Phase 1: Try QueryExpansionService first (synonym-based expansion)
        self._ensure_query_expansion_service()
        if self._query_expansion_service is not None:
            try:
                # Use synonym-based expansion (fast, no LLM required)
                expanded_queries = self._query_expansion_service.expand_query(
                    query_text,
                    max_variants=3,
                    method="synonym",  # Use synonym-based expansion
                )

                if expanded_queries and len(expanded_queries) > 1:
                    # Extract keywords from expanded queries
                    keywords = []
                    for exp in expanded_queries[1:]:  # Skip original query
                        # Extract key terms from expanded query
                        exp_lower = exp.expanded_text.lower()
                        query_lower = query_text.lower()

                        # Find new words not in original query
                        new_words = [
                            word
                            for word in exp_lower.split()
                            if word not in query_lower and len(word) > 1
                        ]
                        keywords.extend(
                            new_words[:3]
                        )  # Limit to 3 keywords per expansion

                    if keywords:
                        logger.debug(
                            f"QueryExpansionService: {query_text[:30]}... -> keywords={keywords[:5]}"
                        )
                        return query_text, keywords[:7]  # Limit total keywords

            except Exception as e:
                logger.warning(f"QueryExpansionService failed: {e}")

        # Fallback to existing LLM-based expansion
        self._ensure_query_expander()
        if self._query_expander is None:
            return query_text, []

        try:
            # Check if expansion is needed
            if not self._query_expander.should_expand(query_text):
                logger.debug(f"Skipping dynamic expansion for: {query_text[:30]}...")
                return query_text, []

            # Perform expansion
            result = self._query_expander.expand(query_text)
            logger.debug(
                f"Dynamic expansion: {query_text[:30]}... -> {result.keywords[:3]}... "
                f"(method={result.method}, confidence={result.confidence:.2f})"
            )

            return result.expanded_query, result.keywords
        except Exception as e:
            logger.warning(f"Dynamic query expansion failed: {e}")
            return query_text, []

    def _ensure_self_rag(self) -> None:
        """Initialize Self-RAG pipeline if not already initialized."""
        if self._self_rag_pipeline is None and self._enable_self_rag:
            from ..infrastructure.self_rag import SelfRAGPipeline

            self._self_rag_pipeline = SelfRAGPipeline(
                search_usecase=self,
                llm_client=self.llm,
                enable_retrieval_check=True,
                enable_relevance_check=True,
                enable_support_check=True,
                async_support_check=True,
            )

    def _ensure_fact_checker(self) -> None:
        """Initialize FactChecker if not already initialized."""
        if self._fact_checker is None and self._enable_fact_check:
            from ..infrastructure.fact_checker import FactChecker

            self._fact_checker = FactChecker(store=self.store)

    def _ensure_persona_generator(self) -> None:
        """Initialize PersonaAwareGenerator if not already initialized."""
        if self._persona_generator is None:
            from ..domain.personas.persona_generator import PersonaAwareGenerator

            self._persona_generator = PersonaAwareGenerator()
            logger.debug("PersonaAwareGenerator initialized")

    def _enhance_prompt_with_persona(
        self, base_prompt: str, persona: str, query: Optional[str] = None
    ) -> str:
        """
        Enhance base prompt with persona-specific instructions.

        Args:
            base_prompt: Original system prompt.
            persona: Persona ID or persona name.
            query: Optional query for additional context.

        Returns:
            Enhanced prompt with persona-specific instructions.
        """
        self._ensure_persona_generator()
        if not self._persona_generator:
            return base_prompt

        try:
            enhanced = self._persona_generator.enhance_prompt(
                base_prompt, persona, query
            )
            logger.debug(f"Prompt enhanced for persona: {persona}")
            return enhanced
        except Exception as e:
            logger.warning(f"Failed to enhance prompt for persona {persona}: {e}")
            return base_prompt

    def _get_persona_from_audience(self, audience: Optional["Audience"]) -> Optional[str]:
        """
        Map Audience enum to persona name for prompt enhancement.

        Args:
            audience: Audience enum value from query analysis.

        Returns:
            Persona name string or None if no mapping found.
        """
        if audience is None:
            return None
        return AUDIENCE_TO_PERSONA.get(audience)

    def _ensure_citation_verification_service(self) -> "CitationVerificationService":
        """
        Initialize CitationVerificationService if not already initialized.

        Returns:
            CitationVerificationService instance.
        """
        if self._citation_verification_service is None:
            from ..domain.citation.citation_verification_service import (
                CitationVerificationService,
            )

            self._citation_verification_service = CitationVerificationService()
            logger.debug("CitationVerificationService initialized")
        return self._citation_verification_service

    def _apply_self_rag_relevance_filter(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Apply Self-RAG relevance filtering to results."""
        if not self._enable_self_rag or not results:
            return results

        self._ensure_self_rag()
        if self._self_rag_pipeline is None:
            return results

        try:
            is_relevant, filtered, confidence = (
                self._self_rag_pipeline.evaluate_results_batch(query, results)
            )
            if filtered:
                logger.debug(
                    f"Self-RAG: Filtered {len(results)} -> {len(filtered)} results "
                    f"(relevant={is_relevant}, confidence={confidence:.2f})"
                )
                return filtered
        except Exception as e:
            logger.warning(f"Self-RAG relevance filter failed: {e}")

        return results

    def _apply_corrective_rag(
        self,
        query_text: str,
        results: List[SearchResult],
        filter: Optional[SearchFilter],
        top_k: int,
        include_abolished: bool,
        audience_override: Optional["Audience"],
        complexity: str = "medium",
    ) -> List[SearchResult]:
        """
        Apply Enhanced Corrective RAG (Cycle 9).

        Uses CRAGRetriever for:
        1. Optimized relevance scoring with multiple signals
        2. T-Fix re-retrieval with adaptive thresholds
        3. Enhanced document re-ranking
        4. Comprehensive performance metrics
        """
        # Initialize CRAGRetriever on first use
        if self._crag_retriever is None:
            from ..infrastructure.crag_retriever import CRAGRetriever

            query_analyzer = (
                self.hybrid_searcher._query_analyzer if self.hybrid_searcher else None
            )

            self._crag_retriever = CRAGRetriever(
                hybrid_searcher=self.hybrid_searcher,
                query_analyzer=query_analyzer,
                llm_client=self.llm,
                enable_tfix=True,
                enable_rerank=True,
                max_tfix_attempts=2,
            )

        # Step 1: Evaluate retrieval quality
        from ..infrastructure.crag_retriever import RetrievalQuality

        quality, score = self._crag_retriever.evaluate_retrieval_quality(
            query_text, results, complexity
        )

        # Step 2: Apply T-Fix if triggered
        current_results = results
        if self._crag_retriever.should_trigger_tfix(quality, score):
            logger.info(
                f"CRAG T-Fix triggered: quality={quality.value}, score={score:.3f}"
            )

            # Synchronous T-Fix implementation
            for attempt in range(self._crag_retriever._max_tfix_attempts):
                corrected_query = self._crag_retriever._generate_corrected_query(
                    query_text, current_results, attempt
                )

                if not corrected_query or corrected_query == query_text:
                    break

                # Re-retrieve with corrected query
                try:
                    # Disable CRAG to avoid recursion
                    self._corrective_rag_enabled = False
                    try:
                        new_results = self._search_general(
                            corrected_query,
                            filter,
                            top_k * 2,
                            include_abolished,
                            audience_override,
                        )
                    finally:
                        self._corrective_rag_enabled = True

                    # Evaluate improvement
                    _, new_score = self._crag_retriever.evaluate_retrieval_quality(
                        query_text, new_results, complexity
                    )

                    if new_score > score + 0.05:
                        logger.info(
                            f"CRAG T-Fix improved: {score:.3f} -> {new_score:.3f} "
                            f"(attempt {attempt + 1})"
                        )
                        current_results = new_results
                        score = new_score
                        quality = self._crag_retriever.evaluate_retrieval_quality(
                            query_text, current_results, complexity
                        )[0]

                        if quality in [
                            RetrievalQuality.EXCELLENT,
                            RetrievalQuality.GOOD,
                        ]:
                            break
                    else:
                        break

                except Exception as e:
                    logger.warning(f"CRAG T-Fix attempt {attempt + 1} failed: {e}")
                    break

        # Step 3: Apply re-ranking for ADEQUATE/GOOD quality
        if quality in [RetrievalQuality.ADEQUATE, RetrievalQuality.GOOD]:
            current_results = self._crag_retriever.apply_rerank(
                query_text, current_results
            )

        return current_results

    def get_crag_metrics(self) -> Optional[dict]:
        """Get CRAG performance metrics (Cycle 9)."""
        if self._crag_retriever is None:
            return None
        return self._crag_retriever.metrics.to_dict()

    def get_crag_metrics_summary(self) -> Optional[str]:
        """Get human-readable CRAG metrics summary (Cycle 9)."""
        if self._crag_retriever is None:
            return None
        return self._crag_retriever.metrics.get_summary()

    def _search_single_pass(
        self,
        query_text: str,
        filter: Optional[SearchFilter],
        top_k: int,
        include_abolished: bool,
        audience_override: Optional["Audience"],
    ) -> List[SearchResult]:
        """Perform a single search pass (used by CRAG T-Fix)."""
        query_text = _coerce_query_text(query_text).strip()
        if not query_text:
            return []

        query_text = unicodedata.normalize("NFC", query_text)

        query = Query(text=query_text, include_abolished=include_abolished)
        fetch_k = top_k * 3
        results = self.store.search(query, filter, fetch_k)

        if self.hybrid_searcher:
            results = self._apply_hybrid_search(
                results,
                query_text,
                query_text,
                filter,
                include_abolished,
                fetch_k // 2,
            )

        audience = self._detect_audience(query_text, audience_override)
        boosted_results = self._apply_score_bonuses(
            results, query_text, query_text, audience
        )
        boosted_results.sort(key=lambda x: -x.score)

        return boosted_results

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

    def _filter_sparse_results(
        self,
        results: List["ScoredDocument"],
        filter: Optional[SearchFilter],
        include_abolished: bool,
    ) -> List["ScoredDocument"]:
        """Filter BM25 results to match metadata filters/abolished policy."""
        if not results:
            return results

        where_clauses = filter.to_metadata_filter() if filter else {}
        if not include_abolished and "status" not in where_clauses:
            where_clauses["status"] = "active"

        if not where_clauses:
            return results

        return [r for r in results if self._metadata_matches(where_clauses, r.metadata)]

    def _filter_search_results(
        self,
        results: List[SearchResult],
        filter: Optional[SearchFilter],
        include_abolished: bool,
    ) -> List[SearchResult]:
        """Filter dense search results to match metadata filters/abolished policy."""
        if not results:
            return results

        where_clauses = filter.to_metadata_filter() if filter else {}
        if not include_abolished and "status" not in where_clauses:
            where_clauses["status"] = "active"

        if not where_clauses:
            return results

        return [
            r
            for r in results
            if self._metadata_matches(where_clauses, r.chunk.to_metadata())
        ]

    def _chunk_matches_filter(
        self, metadata: dict, filter: Optional[SearchFilter], include_abolished: bool
    ) -> bool:
        """Check if chunk metadata satisfies filter conditions."""
        where_clauses = filter.to_metadata_filter() if filter else {}
        if not include_abolished and "status" not in where_clauses:
            where_clauses["status"] = "active"

        if not where_clauses:
            return True

        return self._metadata_matches(where_clauses, metadata)

    def _metadata_matches(self, filters: dict, metadata: dict) -> bool:
        """Check if metadata satisfies simple filter clauses."""
        for key, condition in filters.items():
            value = metadata.get(key)
            if isinstance(condition, dict) and "$in" in condition:
                # For $in conditions, value must be in the list
                if value is None or value not in condition["$in"]:
                    return False
            else:
                # For equality conditions, value must match exactly
                if value != condition:
                    return False
        return True

    def search_unique(
        self,
        query_text: str,
        filter: Optional[SearchFilter] = None,
        top_k: int = 10,
        include_abolished: bool = False,
        audience_override: Optional["Audience"] = None,
    ) -> List[SearchResult]:
        """
        Search with deduplication by rule_code.

        Returns only the top-scoring chunk from each regulation.
        Exception: If query is a regulation name only, skip deduplication
        and return all articles from that regulation.

        Args:
            query_text: The search query.
            filter: Optional metadata filters.
            top_k: Maximum number of unique regulations.
            include_abolished: Whether to include abolished regulations.
            audience_override: Optional audience override for ranking penalties.

        Returns:
            List of SearchResult with one chunk per regulation.
        """
        # Check if query is "regulation name only" pattern
        # If so, return all articles without deduplication
        reg_only = _extract_regulation_only_query(query_text)
        if reg_only:
            # Return search results directly (no deduplication)
            results = self.search(
                query_text,
                filter=filter,
                top_k=top_k,
                include_abolished=include_abolished,
                audience_override=audience_override,
            )
            return results

        # Get more results to ensure enough unique regulations
        results = self.search(
            query_text,
            filter=filter,
            top_k=top_k * 5,
            include_abolished=include_abolished,
            audience_override=audience_override,
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
        audience_override: Optional["Audience"] = None,
        history_text: Optional[str] = None,
        search_query: Optional[str] = None,
        debug: bool = False,
        custom_prompt: Optional[str] = None,
    ) -> Answer:
        """
        Ask a question and get an LLM-generated answer.

        Args:
            question: The user's question.
            filter: Optional metadata filters.
            top_k: Number of chunks to use as context.
            include_abolished: Whether to include abolished regulations.
            audience_override: Optional audience override for ranking penalties.
            history_text: Optional conversation context for the LLM.
            search_query: Optional override for retrieval query.
            debug: Whether to print debug info (prompt).
            custom_prompt: Optional custom system prompt (e.g., for persona-specific responses).

        Returns:
            Answer with generated text and sources.

        Raises:
            ConfigurationError: If LLM client is not configured.
        """
        if not self.llm:
            from ..exceptions import ConfigurationError

            raise ConfigurationError("LLM client not configured. Use search() instead.")

        # Self-RAG: Check if retrieval is even needed
        retrieval_query = search_query or question
        if self._enable_self_rag:
            self._ensure_self_rag()
            if self._self_rag_pipeline and not self._self_rag_pipeline.should_retrieve(
                question
            ):
                # Simple query that doesn't need retrieval (rare for regulation Q&A)
                logger.debug("Self-RAG: Skipping retrieval for simple query")
                return Answer(
                    text="이 질문은 규정 검색이 필요하지 않습니다. 구체적인 규정 관련 질문을 해주세요.",
                    sources=[],
                    confidence=0.5,
                )

        # Multi-hop: Check if query requires multi-hop processing
        # Only apply if no explicit search_query override (user wants normal search)
        if self._should_use_multi_hop(question) and search_query is None:
            logger.info("Multi-hop query detected, routing to multi-hop handler")
            try:
                multi_hop_result = self.ask_multi_hop_sync(
                    question=question,
                    filter=filter,
                    top_k=top_k,
                    include_abolished=include_abolished,
                )

                # Convert MultiHopResult to Answer
                # Collect all sources from all hops
                all_sources = []
                for hop_result in multi_hop_result.hop_results:
                    all_sources.extend(hop_result.sources)

                # Deduplicate sources by chunk ID
                seen_ids = set()
                unique_sources = []
                for source in all_sources:
                    if source.chunk.id not in seen_ids:
                        seen_ids.add(source.chunk.id)
                        unique_sources.append(source)

                # Update ranks
                for i, source in enumerate(unique_sources[:top_k]):
                    unique_sources[i] = SearchResult(
                        chunk=source.chunk,
                        score=source.score,
                        rank=i + 1,
                    )

                return Answer(
                    text=multi_hop_result.final_answer,
                    sources=unique_sources[:top_k],
                    confidence=0.8 if multi_hop_result.success else 0.3,
                )
            except Exception as e:
                logger.warning(
                    f"Multi-hop processing failed: {e}, falling back to single-hop"
                )
                # Fall through to normal single-hop processing

        # Get relevant chunks
        # Phase 1 Integration: Apply query expansion before search
        expanded_query, expansion_keywords = self._apply_dynamic_expansion(
            retrieval_query
        )
        if expansion_keywords:
            logger.debug(
                f"Query expansion applied: {retrieval_query[:30]}... -> keywords={expansion_keywords[:5]}"
            )

        results = self.search(
            expanded_query,  # Use expanded query for search
            filter=filter,
            top_k=top_k * 3,
            include_abolished=include_abolished,
            audience_override=audience_override,
        )

        if not results:
            return Answer(
                text="관련 규정을 찾을 수 없습니다. 다른 검색어로 시도해주세요.",
                sources=[],
                confidence=0.0,
            )

        # Self-RAG: Apply relevance filtering
        results = self._apply_self_rag_relevance_filter(question, results)

        # Filter out low-signal headings when possible
        filtered_results = self._select_answer_sources(results, top_k)
        if not filtered_results:
            # Normalize results before using as fallback to handle cached dicts
            filtered_results = self._normalize_search_results(results[:top_k])

        # TAG-001: Check confidence threshold before generating answer
        # If confidence is too low, return fallback message to prevent hallucination
        confidence = self._compute_confidence(filtered_results)
        if confidence < self._confidence_threshold:
            logger.warning(
                f"Low confidence ({confidence:.3f} < {self._confidence_threshold:.3f}), "
                f"returning fallback message for query: {question[:50]}..."
            )
            return Answer(
                text=FALLBACK_MESSAGE_KO,
                sources=[],
                confidence=confidence,
            )

        # Build context from search results
        context = self._build_context(filtered_results)

        # SPEC-RAG-Q-003: Enhance context with period information if query is period-related
        if self._is_period_related_query(question):
            context = self._enhance_context_with_period_info(question, context)

        # Generate answer
        user_message = self._build_user_message(question, context, history_text)

        if debug:
            logger.debug("=" * 40 + " PROMPT " + "=" * 40)
            logger.debug(f"[System]\n{custom_prompt or REGULATION_QA_PROMPT}\n")
            logger.debug(f"[User]\n{user_message}")
            logger.debug("=" * 80)

        # Generate answer with fact-check loop
        # TAG-005: Detect persona for persona-aware response generation
        detected_audience = self._detect_audience(question, audience_override)
        persona = self._get_persona_from_audience(detected_audience)

        answer_text = self._generate_with_fact_check(
            question=question,
            context=context,
            history_text=history_text,
            debug=debug,
            custom_prompt=custom_prompt,
            persona=persona,
        )

        # Self-RAG: Start async support verification
        if self._enable_self_rag and self._self_rag_pipeline:
            self._self_rag_pipeline.start_async_support_check(
                question, context, answer_text
            )

        # Apply hallucination filter (SPEC-RAG-Q-002)
        if self.hallucination_filter:
            context_texts = [
                r.chunk.text for r in filtered_results if hasattr(r, "chunk")
            ]
            filter_result = self.hallucination_filter.filter_response(
                response=answer_text, context=context_texts
            )
            answer_text = filter_result.sanitized_response
            if filter_result.issues:
                logger.info(
                    f"Hallucination filter applied: {len(filter_result.issues)} issues fixed"
                )
            if filter_result.blocked:
                logger.warning(
                    f"Hallucination filter blocked response: {filter_result.block_reason}"
                )

        # SPEC-RAG-QUALITY-004: Faithfulness check
        # Block answers with low faithfulness score (< 0.3)
        faithfulness_result: Optional[FaithfulnessResult] = None
        if self.hallucination_filter:
            context_texts = [
                r.chunk.text for r in filtered_results if hasattr(r, "chunk")
            ]
            faithfulness_result = self.hallucination_filter.calculate_faithfulness(
                response=answer_text, context=context_texts
            )
            logger.info(
                f"Faithfulness score: {faithfulness_result.score:.3f} "
                f"(block_threshold={FAITHFULNESS_BLOCK_THRESHOLD})"
            )

            if faithfulness_result.should_block:
                logger.warning(
                    f"Low faithfulness detected ({faithfulness_result.score:.3f} < "
                    f"{FAITHFULNESS_BLOCK_THRESHOLD}): {faithfulness_result.reason}"
                )
                # Generate safe response instead
                safe_response = self._generate_safe_response(
                    question=question,
                    sources=filtered_results,
                    faithfulness_score=faithfulness_result.score,
                )
                return Answer(
                    text=safe_response,
                    sources=filtered_results,
                    confidence=faithfulness_result.score,  # Use faithfulness as confidence
                )

        # SPEC-RAG-Q-004: Verify citations against source chunks
        answer_text = self._verify_citations(answer_text, filtered_results)

        # Compute confidence based on search scores
        confidence = self._compute_confidence(filtered_results)

        # Phase 1 Integration: Enhance citations in answer
        enhanced_answer_text = self._enhance_answer_citations(
            answer_text, filtered_results
        )

        # SPEC-RAG-QUALITY-006: Validate and enrich citations
        # Ensure all citations have proper format (규정명 제X조)
        validated_answer_text = self._validate_and_enrich_citations(
            enhanced_answer_text, filtered_results
        )

        return Answer(
            text=validated_answer_text,
            sources=filtered_results,
            confidence=confidence,
        )

    def _generate_with_fact_check(
        self,
        question: str,
        context: str,
        history_text: Optional[str] = None,
        debug: bool = False,
        custom_prompt: Optional[str] = None,
        custom_user_message: Optional[str] = None,
        persona: Optional[str] = None,
    ) -> str:
        """
        Generate answer with iterative fact-checking and correction.

        If fact check fails, regenerate with feedback until all citations
        are verified or max retries reached.

        Args:
            question: User's question.
            context: Search result context.
            history_text: Optional conversation history.
            debug: Whether to print debug info.
            custom_prompt: Optional custom system prompt (e.g., for English).
            custom_user_message: Optional pre-built user message.
            persona: Optional persona name for persona-aware response generation.

        Returns:
            Verified answer text.
        """
        # Use custom user message if provided, otherwise build default
        if custom_user_message:
            user_message = custom_user_message
        else:
            user_message = self._build_user_message(question, context, history_text)

        # Use custom prompt if provided, otherwise use default
        system_prompt = custom_prompt or REGULATION_QA_PROMPT

        # TAG-005: Enhance prompt with persona-specific instructions
        if persona and not custom_prompt:
            system_prompt = self._enhance_prompt_with_persona(
                system_prompt, persona, question
            )

        # Initial generation
        answer_text = self.llm.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.0,
        )

        # Skip fact check if disabled
        if not self._enable_fact_check:
            return answer_text

        self._ensure_fact_checker()
        if not self._fact_checker:
            return answer_text

        # Fact check loop
        for attempt in range(self._fact_check_max_retries + 1):
            fact_result = self._fact_checker.check(answer_text)

            if debug:
                logger.debug(
                    f"Fact check attempt {attempt + 1}: "
                    f"{fact_result.verified_count}/{fact_result.total_count} verified"
                )

            # All citations verified - done!
            if fact_result.all_verified:
                if attempt > 0:
                    logger.info(
                        f"Fact check passed after {attempt + 1} attempts "
                        f"({fact_result.verified_count} citations verified)"
                    )
                return answer_text

            # Max retries reached - return best effort with warning
            if attempt >= self._fact_check_max_retries:
                logger.warning(
                    f"Fact check failed after {attempt + 1} attempts. "
                    f"Unverified citations: {[c.original_text for c in fact_result.failed_citations]}"
                )
                # Add warning to answer
                failed_refs = ", ".join(
                    c.original_text for c in fact_result.failed_citations
                )
                answer_text += (
                    f"\n\n---\n⚠️ **주의**: 일부 인용({failed_refs})은 "
                    f"데이터베이스에서 확인되지 않았습니다. 담당 부서에 확인하시기 바랍니다."
                )
                return answer_text

            # Build correction feedback and regenerate
            feedback = self._fact_checker.build_correction_feedback(fact_result)
            corrected_user_message = (
                f"{user_message}\n\n---\n{feedback}\n\n"
                f"위 피드백을 반영하여 다시 답변해주세요. "
                f"검증되지 않은 조항은 언급하지 마세요."
            )

            if debug:
                logger.debug(f"Regenerating with feedback:\n{feedback}")

            answer_text = self.llm.generate(
                system_prompt=REGULATION_QA_PROMPT,
                user_message=corrected_user_message,
                temperature=0.0,
            )

        return answer_text

    def ask_stream(
        self,
        question: str,
        filter: Optional[SearchFilter] = None,
        top_k: int = 5,
        include_abolished: bool = False,
        audience_override: Optional["Audience"] = None,
        history_text: Optional[str] = None,
        search_query: Optional[str] = None,
    ):
        """
        Ask a question and stream the LLM-generated answer token by token.

        Args:
            question: The user's question.
            filter: Optional metadata filters.
            top_k: Number of chunks to use as context.
            include_abolished: Whether to include abolished regulations.
            audience_override: Optional audience override for ranking penalties.
            history_text: Optional conversation context for the LLM.
            search_query: Optional override for retrieval query.

        Yields:
            dict: First yield contains metadata (sources, confidence).
                  Subsequent yields contain answer tokens.

        Raises:
            ConfigurationError: If LLM client is not configured.
        """
        if not self.llm:
            from ..exceptions import ConfigurationError

            raise ConfigurationError("LLM client not configured. Use search() instead.")

        # Check if llm_client supports streaming
        if not hasattr(self.llm, "stream_generate"):
            # Fallback to non-streaming
            answer = self.ask(
                question=question,
                filter=filter,
                top_k=top_k,
                include_abolished=include_abolished,
                audience_override=audience_override,
                history_text=history_text,
                search_query=search_query,
            )
            yield {
                "type": "metadata",
                "sources": answer.sources,
                "confidence": answer.confidence,
            }
            yield {"type": "token", "content": answer.text}
            return

        # Get relevant chunks (same as ask)
        retrieval_query = search_query or question

        # Phase 1 Integration: Apply query expansion before search
        expanded_query, expansion_keywords = self._apply_dynamic_expansion(
            retrieval_query
        )
        if expansion_keywords:
            logger.debug(
                f"Query expansion applied: {retrieval_query[:30]}... -> keywords={expansion_keywords[:5]}"
            )

        results = self.search(
            expanded_query,  # Use expanded query for search
            filter=filter,
            top_k=top_k * 3,
            include_abolished=include_abolished,
            audience_override=audience_override,
        )

        if not results:
            yield {"type": "metadata", "sources": [], "confidence": 0.0}
            yield {
                "type": "token",
                "content": "관련 규정을 찾을 수 없습니다. 다른 검색어로 시도해주세요.",
            }
            return

        # Filter out low-signal headings
        filtered_results = self._select_answer_sources(results, top_k)
        if not filtered_results:
            # Normalize results before using as fallback to handle cached dicts
            filtered_results = self._normalize_search_results(results[:top_k])

        # Build context
        context = self._build_context(filtered_results)
        user_message = self._build_user_message(question, context, history_text)
        confidence = self._compute_confidence(filtered_results)

        # First yield: metadata (sources and confidence)
        yield {
            "type": "metadata",
            "sources": filtered_results,
            "confidence": confidence,
        }

        # Stream LLM response token by token
        answer_tokens = []
        for token in self.llm.stream_generate(
            system_prompt=REGULATION_QA_PROMPT,
            user_message=user_message,
            temperature=0.0,
        ):
            answer_tokens.append(token)
            yield {"type": "token", "content": token}

        # Combine tokens into answer text
        answer_text = "".join(answer_tokens)

        # Apply hallucination filter (SPEC-RAG-Q-002)
        if self.hallucination_filter:
            context_texts = [
                r.chunk.text for r in filtered_results if hasattr(r, "chunk")
            ]
            filter_result = self.hallucination_filter.filter_response(
                response=answer_text, context=context_texts
            )
            answer_text = filter_result.sanitized_response
            if filter_result.issues:
                logger.info(
                    f"Hallucination filter applied: {len(filter_result.issues)} issues fixed"
                )

        # SPEC-RAG-QUALITY-004: Faithfulness check for streaming response
        faithfulness_result: Optional[FaithfulnessResult] = None
        if self.hallucination_filter:
            context_texts = [
                r.chunk.text for r in filtered_results if hasattr(r, "chunk")
            ]
            faithfulness_result = self.hallucination_filter.calculate_faithfulness(
                response=answer_text, context=context_texts
            )
            logger.info(
                f"Faithfulness score: {faithfulness_result.score:.3f} "
                f"(block_threshold={FAITHFULNESS_BLOCK_THRESHOLD})"
            )

            if faithfulness_result.should_block:
                logger.warning(
                    f"Low faithfulness in streaming response "
                    f"({faithfulness_result.score:.3f} < {FAITHFULNESS_BLOCK_THRESHOLD}): "
                    f"{faithfulness_result.reason}"
                )
                # Generate safe response instead
                safe_response = self._generate_safe_response(
                    question=question,
                    sources=filtered_results,
                    faithfulness_score=faithfulness_result.score,
                )
                # Yield safe response and return early
                yield {"type": "safe_response", "content": safe_response}
                return

        # SPEC-RAG-Q-004: Verify citations against source chunks
        answer_text = self._verify_citations(answer_text, filtered_results)

        # Phase 1 Integration: Apply citation enhancement after answer generation
        enhanced_answer = self._enhance_answer_citations(
            answer_text=answer_text,
            sources=filtered_results,
        )

        # If enhancement modified the answer, yield the enhanced version
        if enhanced_answer != answer_text:
            yield {"type": "enhancement", "content": enhanced_answer}

    def _build_user_message(
        self,
        question: str,
        context: str,
        history_text: Optional[str],
    ) -> str:
        if history_text:
            return f"""대화 기록:
{history_text}

현재 질문: {question}

참고 규정:
{context}

위 규정 내용을 바탕으로 질문에 답변해주세요."""

        return f"""질문: {question}

참고 규정:
{context}

위 규정 내용을 바탕으로 질문에 답변해주세요."""

    def _ensure_multi_hop_handler(self) -> None:
        """Initialize MultiHopHandler if not already initialized."""
        if self._multi_hop_handler is None and self._enable_multi_hop:
            if self.llm is None:
                logger.warning(
                    "Multi-hop handler requires LLM client, skipping initialization"
                )
                return

            self._multi_hop_handler = MultiHopHandler(
                vector_store=self.store,
                llm_client=self.llm,
                max_hops=self._max_hops,
                hop_timeout_seconds=self._hop_timeout_seconds,
                enable_self_rag=self._enable_self_rag,
            )
            logger.info(f"Multi-hop handler initialized (max_hops={self._max_hops})")

    def _should_use_multi_hop(self, query: str) -> bool:
        """
        Determine if a query requires multi-hop processing.

        Multi-hop queries typically contain:
        - Multiple questions (using "and", "or", "then")
        - Nested dependencies ("prerequisites for X which is required for Y")
        - Complex reasoning ("what if", "how does X affect Y")

        Args:
            query: The query text to analyze

        Returns:
            True if multi-hop processing should be used
        """
        if not self._enable_multi_hop:
            return False

        query_lower = query.lower()

        # Multi-hop indicators
        multi_hop_indicators = [
            # Dependency indicators
            ("선이수", "prerequisite"),
            ("선행", "preceding"),
            ("요건", "requirement"),
            ("조건", "condition"),
            # Sequential indicators
            ("그리고", "and then"),
            ("다음으로", "next"),
            ("이후", "after"),
            # Nested questions
            (" 어떻게", "how"),
            (" 왜", "why"),
            # Multiple items
            ("각각", "each"),
            ("모든", "all"),
            ("모든", "every"),
        ]

        # Check for multi-hop indicators
        for ko_term, en_term in multi_hop_indicators:
            if ko_term in query_lower or en_term in query_lower:
                return True

        # Check query complexity indicators
        # Long queries with multiple clauses often require multi-hop
        if len(query) > 100 and (" 그리고 " in query or " 및 " in query):
            return True

        # Queries with "what about" patterns
        if " 어떤 " in query and " 과목" in query:
            return True

        return False

    async def ask_multi_hop(
        self,
        question: str,
        filter: Optional[SearchFilter] = None,
        top_k: int = 10,
        include_abolished: bool = False,
    ) -> MultiHopResult:
        """
        Ask a multi-hop question and get a comprehensive answer.

        This method decomposes complex questions into sequential sub-queries,
        executes each hop with context from previous hops, and synthesizes
        a comprehensive final answer.

        Args:
            question: The complex multi-hop question
            filter: Optional metadata filters
            top_k: Number of results to retrieve per hop
            include_abolished: Whether to include abolished regulations

        Returns:
            MultiHopResult with final answer and execution details

        Raises:
            ConfigurationError: If LLM client is not configured
        """
        if not self.llm:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                "LLM client not configured for multi-hop processing"
            )

        self._ensure_multi_hop_handler()
        if self._multi_hop_handler is None:
            raise ConfigurationError("Multi-hop handler initialization failed")

        logger.info(f"Multi-hop query: {question[:100]}...")

        # Execute multi-hop processing
        result = await self._multi_hop_handler.execute_multi_hop(question, top_k)

        logger.info(
            f"Multi-hop completed: {result.hop_count} hops, "
            f"{result.total_execution_time_ms:.0f}ms, success={result.success}"
        )

        return result

    def ask_multi_hop_sync(
        self,
        question: str,
        filter: Optional[SearchFilter] = None,
        top_k: int = 10,
        include_abolished: bool = False,
    ) -> MultiHopResult:
        """
        Synchronous wrapper for ask_multi_hop.

        Args:
            question: The complex multi-hop question
            filter: Optional metadata filters
            top_k: Number of results to retrieve per hop
            include_abolished: Whether to include abolished regulations

        Returns:
            MultiHopResult with final answer and execution details
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.ask_multi_hop(question, filter, top_k, include_abolished)
        )

    def _normalize_search_results(
        self, results: List["SearchResult"]
    ) -> List["SearchResult"]:
        """
        Normalize search results by converting dicts to SearchResult objects.

        This handles cases where cached data was serialized/deserialized.
        """
        normalized = []
        for r in results:
            if isinstance(r, dict):
                # Convert dict representation to SearchResult
                try:
                    from ..domain.entities import Chunk as ChunkEntity
                    from ..domain.entities import ChunkLevel

                    chunk_data = r.get("chunk")
                    score = r.get("score", 0.0)
                    rank = r.get("rank", 0)

                    # Skip if chunk data is missing
                    if chunk_data is None:
                        logger.warning(
                            "Skipping SearchResult with None chunk during normalization"
                        )
                        continue

                    # Handle chunk as dict or Chunk object
                    if isinstance(chunk_data, dict):
                        # Convert string level to ChunkLevel enum
                        level_value = chunk_data.get("level", "text")
                        if isinstance(level_value, str):
                            level = ChunkLevel.from_string(level_value)
                        else:
                            level = level_value

                        # Convert string status to RegulationStatus enum
                        status_value = chunk_data.get("status", "active")
                        if isinstance(status_value, str):
                            from ..domain.entities import RegulationStatus

                            try:
                                status = RegulationStatus(status_value)
                            except ValueError:
                                status = RegulationStatus.ACTIVE
                        else:
                            status = status_value

                        chunk = ChunkEntity(
                            id=chunk_data.get("id", ""),
                            rule_code=chunk_data.get("rule_code", ""),
                            level=level,
                            title=chunk_data.get("title", ""),
                            text=chunk_data.get("text", ""),
                            embedding_text=chunk_data.get("embedding_text", ""),
                            full_text=chunk_data.get("full_text", ""),
                            parent_path=chunk_data.get("parent_path", []),
                            token_count=chunk_data.get("token_count", 0),
                            keywords=chunk_data.get("keywords", []),
                            is_searchable=chunk_data.get("is_searchable", True),
                            effective_date=chunk_data.get("effective_date"),
                            status=status,
                        )
                    else:
                        # chunk is already a Chunk object
                        chunk = chunk_data

                    normalized.append(SearchResult(chunk=chunk, score=score, rank=rank))
                except Exception as e:
                    logger.warning(
                        f"Failed to convert dict to SearchResult during normalization: {e}"
                    )
                    continue
            else:
                # Already a SearchResult object
                normalized.append(r)

        return normalized

    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string from search results."""
        context_parts = []

        for i, result in enumerate(results, 1):
            # Defensive check for None chunk
            if result.chunk is None:
                logger.warning(
                    f"Skipping result at position {i} with None chunk in _build_context"
                )
                continue

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

        # Defensive type checking: Convert dict to SearchResult if necessary
        # This handles cases where cached data was serialized/deserialized
        processed_results = []
        for r in results:
            if isinstance(r, dict):
                # Convert dict representation to SearchResult
                try:
                    from ..domain.entities import Chunk as ChunkEntity
                    from ..domain.entities import ChunkLevel

                    chunk_data = r.get("chunk")
                    score = r.get("score", 0.0)
                    rank = r.get("rank", 0)

                    # Skip if chunk data is missing
                    if chunk_data is None:
                        logger.warning("Skipping SearchResult with None chunk")
                        continue

                    # Handle chunk as dict or Chunk object
                    if isinstance(chunk_data, dict):
                        # Convert string level to ChunkLevel enum
                        level_value = chunk_data.get("level", "text")
                        if isinstance(level_value, str):
                            level = ChunkLevel.from_string(level_value)
                        else:
                            level = level_value

                        # Convert string status to RegulationStatus enum
                        status_value = chunk_data.get("status", "active")
                        if isinstance(status_value, str):
                            from ..domain.entities import RegulationStatus

                            try:
                                status = RegulationStatus(status_value)
                            except ValueError:
                                status = RegulationStatus.ACTIVE
                        else:
                            status = status_value

                        chunk = ChunkEntity(
                            id=chunk_data.get("id", ""),
                            rule_code=chunk_data.get("rule_code", ""),
                            level=level,
                            title=chunk_data.get("title", ""),
                            text=chunk_data.get("text", ""),
                            embedding_text=chunk_data.get("embedding_text", ""),
                            full_text=chunk_data.get("full_text", ""),
                            parent_path=chunk_data.get("parent_path", []),
                            token_count=chunk_data.get("token_count", 0),
                            keywords=chunk_data.get("keywords", []),
                            is_searchable=chunk_data.get("is_searchable", True),
                            effective_date=chunk_data.get("effective_date"),
                            status=status,
                        )
                    else:
                        # chunk is already a Chunk object
                        chunk = chunk_data

                    processed_results.append(
                        SearchResult(chunk=chunk, score=score, rank=rank)
                    )
                except Exception as e:
                    logger.warning(f"Failed to convert dict to SearchResult: {e}")
                    continue
            else:
                # Already a SearchResult object
                processed_results.append(r)

        for result in processed_results:
            # Skip if chunk is None (defensive check)
            if result.chunk is None:
                logger.warning("Skipping SearchResult with None chunk")
                continue

            if result.chunk.id in seen_ids:
                continue

            if self._is_low_signal_chunk(result.chunk):
                continue

            seen_ids.add(result.chunk.id)
            selected.append(result)
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for result in processed_results:
                # Skip if chunk is None (defensive check)
                if result.chunk is None:
                    continue

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
        1. Absolute score: score magnitude (supports 0..1 and small-score regimes)
        2. Score spread: Difference between top and bottom scores (indicates clear ranking)

        Higher scores = more confident in the answer.
        """
        if not results:
            return 0.0

        scores = [r.score for r in results[:5]]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)

        # Normalize absolute confidence with a small-score fallback.
        if max_score < 0.1:
            abs_scale = 0.05
            spread_scale = 0.01
        else:
            abs_scale = 1.0
            spread_scale = 0.2

        abs_confidence = min(1.0, avg_score / abs_scale)

        # Also consider score spread (clear differentiation = higher confidence)
        if len(scores) >= 2:
            spread = max_score - min_score
            spread_confidence = min(1.0, spread / spread_scale) if spread > 0 else 0.5
        else:
            spread_confidence = 0.5

        # Combine both metrics (weighted average)
        combined = (abs_confidence * 0.7) + (spread_confidence * 0.3)

        return max(0.0, min(1.0, combined))

    def _generate_safe_response(
        self,
        question: str,
        sources: List[SearchResult],
        faithfulness_score: float = 0.0,
    ) -> str:
        """
        Generate a safe response when faithfulness is too low.

        SPEC-RAG-QUALITY-004: Faithfulness Improvement - Safe Response Generator

        When the answer cannot be trusted (faithfulness < 0.3), this method
        generates a context-based safe response that:
        1. Acknowledges the limitation
        2. Suggests related regulations that might be helpful
        3. Guides user to appropriate resources

        Args:
            question: Original user question
            sources: Available search results
            faithfulness_score: The faithfulness score that triggered the block

        Returns:
            Safe response string in Korean
        """
        # Base safe response message
        safe_response = (
            "죄송합니다. 해당 질문에 대한 정보를 제공된 규정에서 찾을 수 없습니다.\n\n"
        )

        # Extract regulation names from sources to suggest related content
        regulation_names = set()
        article_refs = []

        for source in sources[:5]:  # Top 5 sources
            if hasattr(source, "chunk") and source.chunk:
                chunk = source.chunk
                # Extract regulation name from rule_code (e.g., "학칙" from rule_code)
                if hasattr(chunk, "rule_code") and chunk.rule_code:
                    regulation_names.add(chunk.rule_code)
                # Also check for regulation_name attribute (alternative)
                if hasattr(chunk, "regulation_name") and chunk.regulation_name:
                    regulation_names.add(chunk.regulation_name)
                # Extract article number if available
                if hasattr(chunk, "article_number") and chunk.article_number:
                    article_refs.append(f"제{chunk.article_number}조")
                # Also check for article_reference attribute (alternative)
                if hasattr(chunk, "article_reference") and chunk.article_reference:
                    article_refs.append(chunk.article_reference)

        # Add helpful suggestions if we found related content
        if regulation_names:
            reg_list = ", ".join(sorted(regulation_names)[:3])
            safe_response += f"**관련 규정:** {reg_list}\n\n"

        if article_refs:
            unique_articles = list(dict.fromkeys(article_refs[:3]))
            article_list = ", ".join(unique_articles)
            safe_response += f"**참고할 수 있는 조항:** {article_list}\n\n"

        # Add guidance for further assistance
        safe_response += (
            "**도움을 받을 수 있는 방법:**\n"
            "1. 다른 표현으로 질문을 다시 해보세요\n"
            "2. 구체적인 규정명이나 조항을 포함해서 질문해 보세요\n"
            "3. 학교 관련 부서(학적팀, 교무처 등)에 직접 문의하시기 바랍니다\n\n"
            f"* 신뢰도 점수: {faithfulness_score:.2f} (기준: 0.30 이상 필요)"
        )

        return safe_response

    def _enhance_answer_citations(
        self, answer_text: str, sources: List[SearchResult]
    ) -> str:
        """
        Enhance citations in the answer text using CitationEnhancer (Phase 1).

        Args:
            answer_text: Original answer text
            sources: Search results used for the answer

        Returns:
            Answer text with enhanced citations
        """
        try:
            from ..domain.citation.citation_enhancer import CitationEnhancer

            enhancer = CitationEnhancer()

            # Extract chunks from sources
            chunks = [source.chunk for source in sources]

            # Enhance citations
            enhanced_citations = enhancer.enhance_citations(chunks)

            if not enhanced_citations:
                # No valid citations found, return original answer
                return answer_text

            # Build citation string
            citation_str = enhancer.format_citations(enhanced_citations)

            # Check if answer already has citations
            has_citations = any(
                marker in answer_text for marker in ["「", "제", "조", "규정"]
            )

            # If answer doesn't have proper citations, append them
            if not has_citations and citation_str:
                # Add citations at the end
                enhanced_answer = f"{answer_text}\n\n**참고 규정:** {citation_str}"
                logger.debug(f"Enhanced answer with citations: {citation_str}")
                return enhanced_answer

            return answer_text

        except ImportError:
            logger.warning(
                "CitationEnhancer not available, skipping citation enhancement"
            )
            return answer_text
        except Exception as e:
            logger.warning(f"Citation enhancement failed: {e}")
            return answer_text

    def _verify_citations(self, answer: str, source_chunks: List[SearchResult]) -> str:
        """
        Verify citations in answer against source chunks (SPEC-RAG-Q-004).

        Post-generation hook that extracts citations from the answer,
        verifies each against the source chunks used for generation,
        and sanitizes unverifiable citations.

        Metrics tracked (TASK-014):
        - Total citations found
        - Verified citations (found in sources)
        - Sanitized citations (not found in sources)
        - Verification rate

        Args:
            answer: LLM-generated answer text.
            source_chunks: Search results used as context for the answer.

        Returns:
            Answer with sanitized citations (unverifiable ones replaced).
        """
        if not answer or not source_chunks:
            return answer

        try:
            service = self._ensure_citation_verification_service()

            # Extract citations from answer
            citations = service.extract_citations(answer)

            if not citations:
                logger.debug("Citation verification: No citations found in answer")
                return answer

            # Convert SearchResult list to dict format expected by service
            chunks_for_verification = []
            for result in source_chunks:
                chunk = result.chunk
                chunk_dict = {
                    "text": chunk.text,
                    "metadata": {
                        "regulation_name": getattr(chunk, "regulation_name", None),
                        "article": getattr(chunk, "article", None),
                        "paragraph": getattr(chunk, "paragraph", None),
                    },
                }
                chunks_for_verification.append(chunk_dict)

            # Track metrics for logging
            total_citations = len(citations)
            verified_citations = []
            sanitized_citations = []
            sanitized_answer = answer

            # Verify each citation
            for citation in citations:
                is_verified = service.verify_grounding(
                    citation, chunks_for_verification
                )

                if is_verified:
                    verified_citations.append(citation.to_standard_format())
                    logger.debug(
                        f"Citation verified: {citation.to_standard_format()} "
                        f"(regulation={citation.regulation_name}, article={citation.article})"
                    )
                else:
                    # Sanitize unverifiable citation
                    replacement = service.sanitize_unverifiable(citation)
                    original_format = citation.to_standard_format()

                    # Replace in answer (case-insensitive to handle variations)
                    replaced = False
                    if original_format in sanitized_answer:
                        sanitized_answer = sanitized_answer.replace(
                            original_format, replacement
                        )
                        replaced = True
                    # Also try to replace the original text if different
                    elif citation.original_text in sanitized_answer:
                        sanitized_answer = sanitized_answer.replace(
                            citation.original_text, replacement
                        )
                        replaced = True

                    if replaced:
                        sanitized_citations.append(original_format)
                        logger.debug(
                            f"Citation sanitized: {original_format} -> {replacement} "
                            f"(regulation={citation.regulation_name}, article={citation.article})"
                        )

            # Log comprehensive metrics
            verification_rate = (
                len(verified_citations) / total_citations * 100
                if total_citations > 0
                else 0
            )
            logger.info(
                f"Citation verification complete: "
                f"total={total_citations}, "
                f"verified={len(verified_citations)}, "
                f"sanitized={len(sanitized_citations)}, "
                f"rate={verification_rate:.1f}%"
            )

            # Log detailed info for debugging
            if verified_citations:
                logger.debug(f"Verified citations: {verified_citations}")
            if sanitized_citations:
                logger.debug(f"Sanitized citations: {sanitized_citations}")

            return sanitized_answer

        except ImportError:
            logger.warning("CitationVerificationService not available")
            return answer
        except Exception as e:
            logger.warning(f"Citation verification failed: {e}")
            return answer

    def _validate_and_enrich_citations(
        self, answer: str, sources: List[SearchResult]
    ) -> str:
        """
        Validate citation format and enrich with regulation names (SPEC-RAG-QUALITY-006).

        Post-processing step that:
        1. Validates citation format (규정명 제X조)
        2. Checks citation density
        3. Enriches missing regulation names from context sources
        4. Forces citation generation if LLM response has no citations

        Args:
            answer: LLM-generated answer text.
            sources: Search results used as context for the answer.

        Returns:
            Answer with validated and enriched citations.
        """
        if not answer:
            return answer

        try:
            from ..infrastructure.citation_validator import CitationValidator

            validator = CitationValidator(strict_mode=False)

            # Extract regulation names from source chunks
            context_sources = []
            # Extract (regulation_name, article_number) for forced citation
            citations_data: List[tuple] = []
            for result in sources:
                if hasattr(result, "chunk"):
                    reg_name = None
                    if result.chunk.parent_path:
                        reg_name = result.chunk.parent_path[0]
                        if reg_name and reg_name not in context_sources:
                            context_sources.append(reg_name)
                    # Collect article number for forced citation generation
                    if reg_name and result.chunk.article_number:
                        citations_data.append(
                            (reg_name, result.chunk.article_number)
                        )

            # Validate citation format
            validation_result = validator.validate_citation(answer, context_sources)

            logger.info(
                f"Citation validation: "
                f"count={validation_result.citation_count}, "
                f"density={validation_result.citation_density:.3f}, "
                f"valid={validation_result.is_valid}"
            )

            # Log issues if any
            if validation_result.issues:
                for issue in validation_result.issues:
                    logger.debug(f"Citation issue: {issue}")

            # SPEC-RAG-QUALITY-006: Force citation generation if no citations
            if validation_result.citation_count == 0 and citations_data:
                # Deduplicate citations
                unique_citations = list(set(citations_data))
                # Format citations: "「규정명」 제X조"
                formatted = [
                    f"「{reg}」 {art}"
                    for reg, art in unique_citations[:5]  # Max 5 citations
                ]
                citation_section = "\n\n**출처**: " + ", ".join(formatted)
                logger.info(
                    f"Forced citation generation: {len(formatted)} citations added"
                )
                return answer + citation_section

            # If citations are missing or invalid, try to enrich
            has_issues = (
                not validation_result.is_valid
                or validation_result.missing_regulation_names
            )
            if has_issues:
                enrichment = validator.enrich_citation(answer, context_sources)

                if enrichment.added_citations:
                    logger.info(
                        f"Citation enrichment: {len(enrichment.added_citations)} "
                        "citations enriched"
                    )
                    return enrichment.enriched_answer

            return answer

        except ImportError:
            logger.debug("CitationValidator not available, skipping validation")
            return answer
        except Exception as e:
            logger.warning(f"Citation validation failed: {e}")
            return answer

    def _is_period_related_query(self, query: str) -> bool:
        """
        Check if the query is related to periods, deadlines, or dates.

        Uses PeriodKeywordDetector to identify queries that ask about
        deadlines, dates, schedules, or time periods.

        Args:
            query: User's question text.

        Returns:
            True if the query contains period-related keywords.
        """
        if not self._period_keyword_detector:
            return False

        return self._period_keyword_detector.is_period_related(query)

    def _enhance_context_with_period_info(self, query: str, context: str) -> str:
        """
        Enhance context with academic calendar information for period-related queries.

        SPEC-RAG-Q-003: When a query is period-related, this method enhances
        the context with academic calendar information if available.

        Args:
            query: User's question text.
            context: Current context built from search results.

        Returns:
            Enhanced context with academic calendar information if available,
            otherwise the original context.
        """
        # Detect period keywords for logging
        detected_keywords = []
        if self._period_keyword_detector:
            detected_keywords = self._period_keyword_detector.detect_period_keywords(
                query
            )
            if detected_keywords:
                logger.debug(f"Period keywords detected in query: {detected_keywords}")

        # If AcademicCalendarService is available, enhance context
        if self._academic_calendar_service:
            try:
                # Get relevant calendar events for the query
                events = self._academic_calendar_service.get_relevant_events(query)

                if events:
                    # Format events for context
                    calendar_info_lines = []
                    for event in events:
                        if hasattr(event, "start_date") and hasattr(event, "name"):
                            if hasattr(event, "end_date") and event.end_date:
                                date_str = f"{event.start_date} ~ {event.end_date}"
                            else:
                                date_str = event.start_date
                            description = (
                                f" ({event.description})"
                                if hasattr(event, "description") and event.description
                                else ""
                            )
                            calendar_info_lines.append(
                                f"- {event.name}: {date_str}{description}"
                            )

                    if calendar_info_lines:
                        calendar_info = "\n".join(calendar_info_lines)
                        # Prepend calendar information to context
                        enhanced_context = (
                            f"[학사일정 정보]\n{calendar_info}\n\n"
                            f"[관련 규정]\n{context}"
                        )
                        logger.info(
                            "Context enhanced with academic calendar information"
                        )
                        return enhanced_context
            except Exception as e:
                logger.warning(f"Failed to enhance context with calendar info: {e}")

        # If no calendar service or no info available, return original context
        # The LLM prompt already contains period guidelines from prompts.json
        return context

    def _get_cache_key(
        self,
        query: str,
        filter: Optional["SearchFilter"] = None,
        top_k: int = 10,
        include_abolished: bool = False,
    ) -> str:
        """Generate cache key for query parameters."""
        filter_options = None
        if filter and hasattr(filter, "to_metadata_filter"):
            filter_options = filter.to_metadata_filter()
        parts = [query.strip().lower()]
        if filter_options:
            parts.append(str(sorted(filter_options.items())))
        parts.extend([str(top_k), str(include_abolished)])
        return "::".join(parts)

    def _check_retrieval_cache(
        self,
        query: str,
        filter: Optional["SearchFilter"] = None,
        top_k: int = 10,
        include_abolished: bool = False,
    ) -> Optional[List["SearchResult"]]:
        """Check cache for retrieval results."""
        if not self._query_cache:
            return None
        filter_options = None
        if filter and hasattr(filter, "to_metadata_filter"):
            filter_options = filter.to_metadata_filter()
        cached = self._query_cache.get(
            CacheType.RETRIEVAL,
            query,
            filter_options,
            top_k=top_k,
            include_abolished=include_abolished,
        )
        if cached:
            results = cached.get("results")
            if results:
                logger.debug(
                    f"Cache HIT: Retrieved {len(results)} results for '{query[:30]}...'"
                )
                # Deserialize cached dicts back to SearchResult objects
                from ..domain.entities import Chunk, ChunkLevel, SearchResult

                deserialized = []
                for r in results:
                    if isinstance(r, dict):
                        # Reconstruct Chunk from cached dict
                        chunk = Chunk(
                            id=r.get("chunk_id", ""),
                            rule_code=r.get("rule_code", ""),
                            level=ChunkLevel.from_string(r.get("level", "text")),
                            title=r.get("title", ""),
                            text=r.get("text", ""),
                            embedding_text="",
                            full_text=r.get("text", ""),
                            parent_path=[],
                            token_count=0,
                            keywords=[],
                            is_searchable=True,
                            doc_type=r.get("doc_type", "regulation"),
                        )
                        # Reconstruct SearchResult
                        deserialized.append(
                            SearchResult(
                                chunk=chunk,
                                score=r.get("score", 0.0),
                                rank=r.get("rank", 0),
                            )
                        )
                    else:
                        # Already a SearchResult object
                        deserialized.append(r)
                return deserialized
        return None

    def _store_retrieval_cache(
        self,
        query: str,
        results: List["SearchResult"],
        filter: Optional["SearchFilter"] = None,
        top_k: int = 10,
        include_abolished: bool = False,
    ) -> None:
        """Store retrieval results in cache."""
        if not self._query_cache or not results:
            return
        filter_options = None
        if filter and hasattr(filter, "to_metadata_filter"):
            filter_options = filter.to_metadata_filter()
        serialized = [
            {
                "chunk_id": r.chunk.id,
                "score": r.score,
                "rank": r.rank,
                "rule_code": r.chunk.rule_code,
                "title": r.chunk.title,
                "text": r.chunk.text[:500],
                "level": r.chunk.level.value,
            }
            for r in results
        ]
        self._query_cache.set(
            CacheType.RETRIEVAL,
            query,
            {"results": serialized},
            filter_options,
            top_k=top_k,
            include_abolished=include_abolished,
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._query_cache:
            return self._query_cache.stats()
        return {"cache_enabled": False}

    def clear_cache(self) -> int:
        """Clear all cache entries."""
        if self._query_cache:
            return self._query_cache.clear_all()
        return 0

    def search_by_rule_code(
        self,
        rule_code: str,
        top_k: int = 50,
        include_abolished: bool = True,
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
        query = Query(text="규정", include_abolished=include_abolished)
        return self.store.search(query, filter, top_k)

    @staticmethod
    def _build_rule_code_filter(
        base_filter: Optional[SearchFilter],
        rule_code: str,
    ) -> SearchFilter:
        if base_filter is None:
            return SearchFilter(rule_codes=[rule_code])
        return SearchFilter(
            status=base_filter.status,
            levels=base_filter.levels,
            rule_codes=[rule_code],
            effective_date_from=base_filter.effective_date_from,
            effective_date_to=base_filter.effective_date_to,
        )

    def get_reranking_metrics(self) -> RerankingMetrics:
        """
        Get the current reranking metrics (Cycle 3).

        Returns:
            RerankingMetrics object with usage statistics and performance data.
        """
        return self._reranking_metrics

    def reset_reranking_metrics(self) -> None:
        """Reset reranking metrics to zero (Cycle 3)."""
        self._reranking_metrics = RerankingMetrics()
        logger.info("Reranking metrics reset")

    def print_reranking_metrics(self) -> None:
        """Print reranking metrics summary to log (Cycle 3)."""
        logger.info(self._reranking_metrics.get_summary())

    # --- Phase 3: Multilingual Answer Support ---

    @staticmethod
    def detect_language(query: str) -> str:
        """
        Detect the language of a query based on character composition.

        Args:
            query: The query text to analyze.

        Returns:
            "english" if query is primarily English, "korean" otherwise.
        """
        if not query:
            return "korean"

        # Count English alphabet characters (ASCII range)
        english_char_count = sum(1 for c in query if c.isalpha() and ord(c) < 128)
        total_alpha_count = sum(1 for c in query if c.isalpha())

        # If no alphabetic characters, default to Korean
        if total_alpha_count == 0:
            return "korean"

        # Calculate ratio of English characters
        english_ratio = english_char_count / total_alpha_count

        # If more than 50% English characters, classify as English query
        if english_ratio > 0.5:
            logger.debug(f"Detected English query (ratio={english_ratio:.2f})")
            return "english"

        return "korean"

    def ask_multilingual(
        self,
        question: str,
        filter: Optional[SearchFilter] = None,
        top_k: int = 5,
        include_abolished: bool = False,
        audience_override: Optional["Audience"] = None,
        history_text: Optional[str] = None,
        search_query: Optional[str] = None,
        language: Optional[str] = None,
        debug: bool = False,
    ) -> Answer:
        """
        Ask a question and get an LLM-generated answer in the detected language.

        Supports both Korean and English queries. Automatically detects query language
        and generates responses in the same language.

        Args:
            question: The user's question.
            filter: Optional metadata filters.
            top_k: Number of chunks to use as context.
            include_abolished: Whether to include abolished regulations.
            audience_override: Optional audience override for ranking penalties.
            history_text: Optional conversation context for the LLM.
            search_query: Optional override for retrieval query.
            language: Optional language override ("english" or "korean").
            debug: Whether to print debug info (prompt).

        Returns:
            Answer with generated text and sources.
        """
        if not self.llm:
            from ..exceptions import ConfigurationError

            raise ConfigurationError("LLM client not configured. Use search() instead.")

        # Auto-detect language if not specified
        detected_language = language or self.detect_language(question)

        # Generate answer in the detected language
        if detected_language == "english":
            logger.info("Generating English answer for query")
            return self._ask_english(
                question=question,
                filter=filter,
                top_k=top_k,
                include_abolished=include_abolished,
                audience_override=audience_override,
                history_text=history_text,
                search_query=search_query,
                debug=debug,
            )
        else:
            # Use standard Korean ask method
            return self.ask(
                question=question,
                filter=filter,
                top_k=top_k,
                include_abolished=include_abolished,
                audience_override=audience_override,
                history_text=history_text,
                search_query=search_query,
                debug=debug,
            )

    def _ask_english(
        self,
        question: str,
        filter: Optional[SearchFilter] = None,
        top_k: int = 5,
        include_abolished: bool = False,
        audience_override: Optional["Audience"] = None,
        history_text: Optional[str] = None,
        search_query: Optional[str] = None,
        debug: bool = False,
    ) -> Answer:
        """
        Generate an English answer for an English query.

        Uses a specialized English prompt that instructs the LLM to respond in English
        while still referencing Korean regulations.

        Args:
            question: The user's English question.
            filter: Optional metadata filters.
            top_k: Number of chunks to use as context.
            include_abolished: Whether to include abolished regulations.
            audience_override: Optional audience override.
            history_text: Optional conversation context.
            search_query: Optional override for retrieval query.
            debug: Whether to print debug info.

        Returns:
            Answer with English text and sources.
        """
        # Self-RAG: Check if retrieval is needed
        retrieval_query = search_query or question
        if self._enable_self_rag:
            self._ensure_self_rag()
            if self._self_rag_pipeline and not self._self_rag_pipeline.should_retrieve(
                question
            ):
                logger.debug("Self-RAG: Skipping retrieval for simple query")
                return Answer(
                    text="This question does not require regulation search. Please ask a specific regulation-related question.",
                    sources=[],
                    confidence=0.5,
                )

        # Get relevant chunks
        results = self.search(
            retrieval_query,
            filter=filter,
            top_k=top_k * 3,
            include_abolished=include_abolished,
            audience_override=audience_override,
        )

        if not results:
            return Answer(
                text="No relevant regulations found. Please try different search terms.",
                sources=[],
                confidence=0.0,
            )

        # Self-RAG: Apply relevance filtering
        results = self._apply_self_rag_relevance_filter(question, results)

        # Filter out low-signal headings
        filtered_results = self._select_answer_sources(results, top_k)
        if not filtered_results:
            filtered_results = self._normalize_search_results(results[:top_k])

        # TAG-001: Check confidence threshold before generating answer
        # If confidence is too low, return fallback message to prevent hallucination
        confidence = self._compute_confidence(filtered_results)
        if confidence < self._confidence_threshold:
            logger.warning(
                f"Low confidence ({confidence:.3f} < {self._confidence_threshold:.3f}), "
                f"returning fallback message for English query: {question[:50]}..."
            )
            return Answer(
                text=FALLBACK_MESSAGE_EN,
                sources=[],
                confidence=confidence,
            )

        # Build context from search results
        context = self._build_context(filtered_results)

        # Get English prompt
        english_prompt = self._get_english_prompt()

        # Build user message
        user_message = self._build_english_user_message(question, context, history_text)

        if debug:
            logger.debug("=" * 40 + " ENGLISH PROMPT " + "=" * 40)
            logger.debug(f"[System]\n{english_prompt}\n")
            logger.debug(f"[User]\n{user_message}")
            logger.debug("=" * 80)

        # Generate answer
        # TAG-005: For English queries, use international persona or detect from audience
        detected_audience = self._detect_audience(question, audience_override)
        # For English queries, prefer "international" persona unless audience explicitly suggests otherwise
        if audience_override is None:
            persona = "international"
        else:
            persona = self._get_persona_from_audience(detected_audience) or "international"

        answer_text = self._generate_with_fact_check(
            question=question,
            context=context,
            history_text=history_text,
            debug=debug,
            custom_prompt=english_prompt,
            custom_user_message=user_message,
            persona=persona,
        )

        # Self-RAG: Start async support verification
        if self._enable_self_rag and self._self_rag_pipeline:
            self._self_rag_pipeline.start_async_support_check(
                question, context, answer_text
            )

        # Apply hallucination filter (SPEC-RAG-Q-002)
        if self.hallucination_filter:
            context_texts = [
                r.chunk.text for r in filtered_results if hasattr(r, "chunk")
            ]
            filter_result = self.hallucination_filter.filter_response(
                response=answer_text, context=context_texts
            )
            answer_text = filter_result.sanitized_response
            if filter_result.issues:
                logger.info(
                    f"Hallucination filter applied: {len(filter_result.issues)} issues fixed"
                )

        # SPEC-RAG-QUALITY-004: Faithfulness check
        # Block answers with low faithfulness score (< 0.3)
        faithfulness_result: Optional[FaithfulnessResult] = None
        if self.hallucination_filter:
            context_texts = [
                r.chunk.text for r in filtered_results if hasattr(r, "chunk")
            ]
            faithfulness_result = self.hallucination_filter.calculate_faithfulness(
                response=answer_text, context=context_texts
            )
            logger.info(
                f"Faithfulness score: {faithfulness_result.score:.3f} "
                f"(block_threshold={FAITHFULNESS_BLOCK_THRESHOLD})"
            )

            if faithfulness_result.should_block:
                logger.warning(
                    f"Low faithfulness detected ({faithfulness_result.score:.3f} < "
                    f"{FAITHFULNESS_BLOCK_THRESHOLD}): {faithfulness_result.reason}"
                )
                # Generate safe response instead
                safe_response = self._generate_safe_response(
                    question=question,
                    sources=filtered_results,
                    faithfulness_score=faithfulness_result.score,
                )
                return Answer(
                    text=safe_response,
                    sources=filtered_results,
                    confidence=faithfulness_result.score,
                )

        # SPEC-RAG-Q-004: Verify citations against source chunks
        answer_text = self._verify_citations(answer_text, filtered_results)

        # Compute confidence
        confidence = self._compute_confidence(filtered_results)

        return Answer(
            text=answer_text,
            sources=filtered_results,
            confidence=confidence,
        )

    # Conversation Memory Methods

    def create_conversation_session(
        self, user_id: Optional[str] = None, expiry_hours: int = 24 * 7
    ) -> str:
        """
        Create a new conversation session for long-term memory.

        Args:
            user_id: Optional user identifier
            expiry_hours: Hours until memory expires (default: 7 days)

        Returns:
            Session ID for tracking conversation
        """
        if not self._enable_conversation_memory or self._memory_manager is None:
            logger.warning("Conversation memory is not enabled")
            return ""

        from .conversation_memory import MemoryExpiryPolicy

        expiry_policy = MemoryExpiryPolicy.HOURS_24
        if expiry_hours >= 24 * 30:
            expiry_policy = MemoryExpiryPolicy.DAYS_30
        elif expiry_hours >= 24 * 7:
            expiry_policy = MemoryExpiryPolicy.DAYS_7

        session_id = self._memory_manager.create_session(
            user_id=user_id, expiry_policy=expiry_policy
        )
        logger.info(f"Created conversation session {session_id} for user {user_id}")
        return session_id

    def add_conversation_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a message to conversation memory.

        Args:
            session_id: Conversation session ID
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional metadata

        Returns:
            True if message was added successfully
        """
        if not self._enable_conversation_memory or self._memory_manager is None:
            return False

        context = self._memory_manager.add_message(
            session_id=session_id, role=role, content=content, metadata=metadata
        )
        return context is not None

    def get_conversation_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation context for search enhancement.

        Args:
            session_id: Conversation session ID

        Returns:
            Dictionary with context information or None
        """
        if not self._enable_conversation_memory or self._memory_manager is None:
            return None

        return self._memory_manager.get_context_for_search(session_id)

    def expand_query_with_context(self, session_id: str, query: str) -> str:
        """
        Expand query using conversation context.

        This improves retrieval for context-dependent questions by:
        - Adding topic-related terms
        - Including key entities from conversation
        - Maintaining conversation continuity

        Args:
            session_id: Conversation session ID
            query: Original search query

        Returns:
            Context-expanded query
        """
        if not self._enable_conversation_memory or self._memory_manager is None:
            return query

        return self._memory_manager.expand_query(session_id, query)

    def cleanup_expired_conversations(self) -> int:
        """
        Remove expired conversation sessions from memory.

        Returns:
            Number of sessions cleaned up
        """
        if not self._enable_conversation_memory or self._memory_manager is None:
            return 0

        return self._memory_manager.cleanup_expired_sessions()

    @staticmethod
    def _get_english_prompt() -> str:
        """Get the system prompt for English Q&A."""
        return """You are a university regulation expert for Dong-A University.

Your task is to provide **detailed and helpful answers** in English to questions based on the provided Korean regulations.

## ⚠️ Strict Prohibition (Hallucination Prevention)
1. **NO phone number/contact creation**: Do NOT fabricate phone numbers like "02-XXXX-XXXX" or "02-1234-5678".
2. **NO other school examples**: Do NOT mention regulations or examples from Korea University, Seoul National University, etc.
3. **NO numerical fabrication**: Do NOT create percentages or deadlines like "40%", "30 days" that are not in the regulations.
4. **NO generic avoidance**: Do NOT say "it varies by university" or "generally..." to avoid answering.
5. **NO uncited information**: Never state facts without regulation citations. All information must be provided with proper citations.

## Basic Principles
- **Answer ONLY based on the provided regulation content.**
- Do NOT guess or mention general practices for content not in the regulations.
- Translate key Korean regulation terms accurately and provide context.
- If the regulation does not contain information to answer the question, state clearly that the regulation does not specify it.

## 📋 Required Response Format

All answers MUST follow this structure:

### 1. Core Answer
[Direct answer to the question - MUST include regulation citations]

### 2. Related Regulations
- **Regulation**: [Regulation Name]
- **Article**: [Article X, Section Y]
- **Content**: [Summary of relevant content]

### 3. Additional Notes
[Guidance for additional help if needed]

## ⚠️ Mandatory Citation Requirements

1. **ALL answers MUST include regulation name and article citations.**
2. **Citation Format**: "[Regulation Name] Article X" or "[Regulation Name] Article X, Section Y"
3. **Citation Location**: Place citations in parentheses immediately after the core content.
4. **Citation Examples**:
   - "Leave of absence must be applied for 1 month before semester starts (University Regulations Article 40, Section 1)."
   - "Tuition fees must be paid before each semester begins (Tuition Regulations Article 5)."
5. **No Uncited Answers**: Answers without regulation citations are considered incomplete.

## Important Notes
- The source text is in Korean, but you must respond in English.
- Preserve accuracy when translating regulation content.
- If uncertain about a translation, provide the original Korean term in parentheses.
"""

    @staticmethod
    def _build_english_user_message(
        question: str, context: str, history_text: Optional[str]
    ) -> str:
        """Build user message for English Q&A."""
        if history_text:
            return f"""Conversation History:
{history_text}

Current Question: {question}

Reference Regulations:
{context}

Based on the above regulations, please answer the question in English."""

        return f"""Question: {question}

Reference Regulations:
{context}

Based on the above regulations, please answer the question in English."""
