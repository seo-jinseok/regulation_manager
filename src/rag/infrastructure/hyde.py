"""
HyDE (Hypothetical Document Embeddings) for Regulation RAG System.

Cycle 8 Improvements:
- Optimized Korean prompt for better hypothetical document generation
- Enhanced validation with quality scoring
- Refined application conditions based on query characteristics
- Added performance metrics tracking

Priority 2 Performance Improvements:
- LRU cache eviction policy (REQ-PO-008)
- Cache file size limit (REQ-PO-010)
- Gzip compression for disk cache (REQ-PO-011)

Priority 4 Security Hardening:
- Input validation with Pydantic
- Malicious query pattern detection
- Query sanitization
"""

import gzip
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..domain.entities import SearchResult
    from ..domain.repositories import ILLMClient, IVectorStore
    from ..domain.value_objects import SearchFilter

logger = logging.getLogger(__name__)

# Cache configuration (REQ-PO-008, REQ-PO-010)
DEFAULT_MAX_CACHE_SIZE = 1000  # Maximum number of entries in memory
DEFAULT_MAX_CACHE_FILE_SIZE_MB = 10  # Maximum cache file size in MB


@dataclass
class HyDEResult:
    """Result of HyDE query expansion."""

    original_query: str
    hypothetical_doc: str
    from_cache: bool
    cache_key: Optional[str] = None
    quality_score: float = 0.0
    generation_time_ms: float = 0.0


@dataclass
class HyDEMetrics:
    """Performance metrics for HyDE generation (Cycle 8)."""

    total_generations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0  # LRU eviction count (REQ-PO-008)
    validation_failures: int = 0
    total_quality_score: float = 0.0
    total_generation_time_ms: float = 0.0

    def get_cache_hit_rate(self) -> float:
        if self.total_generations == 0:
            return 0.0
        return self.cache_hits / self.total_generations

    def get_average_quality(self) -> float:
        if self.total_generations == 0:
            return 0.0
        return self.total_quality_score / self.total_generations

    def get_average_generation_time_ms(self) -> float:
        if self.total_generations == 0:
            return 0.0
        return self.total_generation_time_ms / self.total_generations


class HyDEGenerator:
    """Generates hypothetical documents for improved retrieval."""

    SYSTEM_PROMPT = """당신은 대학 규정 전문가로, 사용자의 질문에 대해 가상의 대학 규정 조문을 작성합니다.

## 작성 원칙

### 1. 문체와 형식
- 실제 대학 규정처럼 공식적이고 형식적인 문체 사용
- "~할 수 있다", "~하여야 한다" 등의 규정 표현 사용

### 2. 내용 구성
- 질문의 핵심 의도를 파악하여 관련 규정 내용 작성
- 구체적인 용어와 키워드 포함 (예: 휴직, 휴학, 복학, 등록금, 장학금 등)
- 필요시 조건, 절차, 제한 사항 등을 언급

### 3. 길이와 구조
- 150-250자 내외로 작성
- 단락을 나누어 가독성 확보

### 4. 제한 사항
- 실제 규정명이나 조항 번호는 포함하지 않음

## 예시

질문: "학교 가기 싫어"
가상 답변: 학생의 휴학은 질병, 가사사정, 기타 부득이한 사유로 인하여 수학할 수 없는 경우에 한하여 허가할 수 있다. 휴학 기간은 1회에 1학기를 초과하지 못하며, 통산 3학기를 초과할 수 없다.

질문: "장학금 받고 싶어"
가상 답변: 성적 우수 장학금은 직전 학기 평점 3.5 이상인 학생 중에서 성적 순에 따라 지급한다. 보건 의료 장학금은 간호사 면허를 소지한 자로서 보건 의료 기관에서 근무하는 자에게 지급된다.

사용자의 질문에 대한 가상 대학 규정 조문을 작성하세요."""

    REGULATORY_TERMS = [
        "규정",
        "규칙",
        "조",
        "항",
        "호",
        "세칙",
        "지침",
        "시행세칙",
        "운영규정",
        "관리지침",
    ]

    VAGUE_INDICATORS = [
        "싶어",
        "하고 싶",
        "원해",
        "희망",
        "싫어",
        "안 하",
        "기피",
        "어떻게",
        "뭐야",
        "무엇",
        "가능",
        "수 있",
        "될까",
        "있어?",
        "해야",
        "할까",
        "해줘",
    ]

    EMOTIONAL_INDICATORS = [
        "힘들",
        "어렵",
        "고생",
        "스트레스",
        "피곤",
        "지치",
        "포기",
        "번아웃",
        "걱정",
        "불안",
        "후회",
    ]

    def __init__(
        self,
        llm_client: Optional["ILLMClient"] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
        max_cache_file_size_mb: int = DEFAULT_MAX_CACHE_FILE_SIZE_MB,
    ):
        """
        Initialize HyDE generator with enhanced cache management (REQ-PO-008, REQ-PO-010).

        Args:
            llm_client: LLM client for generating hypothetical documents.
            cache_dir: Directory for cache storage.
            enable_cache: Whether to enable caching.
            max_cache_size: Maximum number of cache entries (LRU eviction when exceeded).
            max_cache_file_size_mb: Maximum cache file size in MB (enforces limit).
        """
        self._llm_client = llm_client
        self._enable_cache = enable_cache
        self._max_cache_size = max_cache_size
        self._max_cache_file_size_bytes = max_cache_file_size_mb * 1024 * 1024

        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = (
                Path(__file__).parent.parent.parent.parent / "data" / "cache" / "hyde"
            )

        if self._enable_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            # Load cache with LRU tracking (REQ-PO-008)
            cache_data = self._load_cache()
            self._cache = OrderedDict(cache_data)
            self._enforce_cache_size_limit()
        else:
            self._cache = OrderedDict()

        self._metrics = HyDEMetrics()

    def _load_cache(self) -> Dict:
        """
        Load HyDE cache from disk with gzip decompression (REQ-PO-011).

        Returns:
            Dict containing cached data.
        """
        cache_file = self._cache_dir / "hyde_cache.json.gz"

        # Try compressed cache first (REQ-PO-011)
        if cache_file.exists():
            try:
                with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug(f"Loaded compressed HyDE cache ({len(data)} entries)")
                return data
            except Exception as e:
                logger.warning(f"Failed to load compressed cache: {e}")

        # Fallback to uncompressed cache
        cache_file_uncompressed = self._cache_dir / "hyde_cache.json"
        if cache_file_uncompressed.exists():
            try:
                with open(cache_file_uncompressed, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug(f"Loaded uncompressed HyDE cache ({len(data)} entries)")

                # Migrate to compressed format
                self._save_cache()
                return data
            except Exception as e:
                logger.warning(f"Failed to load HyDE cache: {e}")

        return {}

    def _save_cache(self) -> None:
        """
        Save HyDE cache to disk with gzip compression (REQ-PO-011).

        Enforces cache file size limit before saving (REQ-PO-010).
        """
        if not self._enable_cache:
            return

        # Enforce size limit before saving (REQ-PO-010)
        self._enforce_cache_size_limit()

        cache_file = self._cache_dir / "hyde_cache.json.gz"

        try:
            # Convert OrderedDict to dict for JSON serialization
            cache_data = dict(self._cache)

            # Save with gzip compression (REQ-PO-011)
            with gzip.open(cache_file, "wt", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)

            # Check file size and enforce limit
            file_size = cache_file.stat().st_size
            if file_size > self._max_cache_file_size_bytes:
                logger.warning(
                    f"Cache file size ({file_size / 1024 / 1024:.1f}MB) exceeds limit "
                    f"({self._max_cache_file_size_bytes / 1024 / 1024:.1f}MB), "
                    f"evicting oldest entries"
                )
                self._evict_entries_to_fit_size()
                self._save_cache()

            logger.debug(
                f"Saved HyDE cache ({len(cache_data)} entries, {file_size / 1024:.1f}KB)"
            )

        except Exception as e:
            logger.warning(f"Failed to save HyDE cache: {e}")

    def _enforce_cache_size_limit(self) -> None:
        """
        Enforce LRU cache size limit (REQ-PO-008).

        Evicts least recently used entries when cache exceeds max size.
        """
        while len(self._cache) > self._max_cache_size:
            # Pop oldest entry (LRU eviction - REQ-PO-008)
            oldest_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Evicted LRU cache entry: {oldest_key}")
            self._metrics.cache_evictions += 1

    def _evict_entries_to_fit_size(self) -> None:
        """
        Evict cache entries to fit within file size limit (REQ-PO-010).

        Removes oldest entries until estimated file size is within limit.
        """
        target_entries = max(
            1,
            int(self._max_cache_size * 0.8),  # Reduce to 80% of max
        )

        while len(self._cache) > target_entries:
            oldest_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Evicted cache entry for size limit: {oldest_key}")

    def _update_cache_lru(self, key: str) -> None:
        """
        Update cache entry access time for LRU tracking (REQ-PO-008).

        Args:
            key: Cache key to update.
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)

    def _add_to_cache_with_lru(self, key: str, value: dict) -> None:
        """
        Add entry to cache with LRU eviction (REQ-PO-008).

        Args:
            key: Cache key.
            value: Cache value.
        """
        # Enforce size limit before adding
        if len(self._cache) >= self._max_cache_size:
            # Evict oldest entry
            oldest_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Evicted LRU cache entry before add: {oldest_key}")

        self._cache[key] = value
        # Move to end (most recently used)
        self._cache.move_to_end(key)

    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode("utf-8")).hexdigest()[:16]

    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        self._llm_client = llm_client

    def get_metrics(self) -> HyDEMetrics:
        return self._metrics

    def reset_metrics(self) -> None:
        self._metrics = HyDEMetrics()
        logger.info("HyDE metrics reset")

    def _validate_hypothetical_doc(
        self, doc: str, original_query: str
    ) -> Tuple[bool, Optional[str], float]:
        quality_score = 0.0

        if not doc:
            return False, None, quality_score

        doc = doc.strip()

        if len(doc) < 30:
            return False, None, quality_score
        elif 50 <= len(doc) <= 400:
            quality_score += 0.3
        elif len(doc) > 400:
            quality_score += 0.1

        error_patterns = [
            "죄송합니다",
            "알 수 없습니다",
            "도움을 드릴 수 없습니다",
            "제공해 드릴 수 없",
            "확인할 수 없",
            "규정에 없",
        ]
        if any(pattern in doc for pattern in error_patterns):
            return False, None, quality_score

        regulatory_patterns = [
            r"할 수 있",
            r"하여야",
            r"한다",
            r"지급",
            r"신청",
            r"허가",
            r"승인",
        ]
        regulatory_count = sum(
            1 for pattern in regulatory_patterns if re.search(pattern, doc)
        )
        if regulatory_count >= 2:
            quality_score += 0.3

        education_keywords = [
            "학생",
            "교원",
            "수업",
            "학점",
            "등록",
            "졸업",
            "휴학",
            "복학",
            "휴직",
            "장학금",
            "등록금",
            "성적",
        ]
        keyword_count = sum(1 for kw in education_keywords if kw in doc)
        if keyword_count >= 2:
            quality_score += 0.2

        sentences = doc.split(". ")
        if 2 <= len(sentences) <= 5:
            quality_score += 0.2

        quality_score = min(quality_score, 1.0)

        if quality_score < 0.3:
            return False, None, quality_score

        return True, doc, quality_score

    def generate_hypothetical_doc(self, query: str) -> HyDEResult:
        start_time = time.time()

        # Security: Input validation (P4: Security Hardening)
        try:
            from .security import InputValidationError, sanitize_input

            # Sanitize input query
            sanitized_query = sanitize_input(query, max_length=500)
        except ImportError:
            # Security module not available, use basic validation
            sanitized_query = query.strip() if query else ""
        except InputValidationError as e:
            logger.warning(f"HyDE: Input validation failed: {e}")
            # Return original query as fallback (defense in depth)
            sanitized_query = query[:500] if query else ""

        if not sanitized_query:
            logger.warning("HyDE: Empty query received, skipping generation")
            return HyDEResult(
                original_query=query,
                hypothetical_doc="",
                from_cache=False,
            )

        query = query.strip()
        cache_key = self._get_cache_key(query)

        if cache_key in self._cache:
            self._metrics.total_generations += 1
            self._metrics.cache_hits += 1
            cached_doc = self._cache[cache_key]
            is_valid, validated_doc, quality_score = self._validate_hypothetical_doc(
                cached_doc, query
            )
            if is_valid:
                elapsed_ms = (time.time() - start_time) * 1000
                self._metrics.total_quality_score += quality_score
                self._metrics.total_generation_time_ms += elapsed_ms
                logger.debug(f"HyDE cache hit for query: {query[:30]}...")
                return HyDEResult(
                    original_query=query,
                    hypothetical_doc=validated_doc,
                    from_cache=True,
                    cache_key=cache_key,
                    quality_score=quality_score,
                    generation_time_ms=elapsed_ms,
                )

        self._metrics.total_generations += 1
        self._metrics.cache_misses += 1

        if not self._llm_client:
            logger.debug("No LLM client for HyDE, using original query")
            return HyDEResult(
                original_query=query,
                hypothetical_doc=query,
                from_cache=False,
            )

        try:
            hypothetical_doc = self._llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=f"질문: {query}",
                temperature=0.2,
            )

            is_valid, validated_doc, quality_score = self._validate_hypothetical_doc(
                hypothetical_doc, query
            )

            elapsed_ms = (time.time() - start_time) * 1000

            if not is_valid:
                self._metrics.validation_failures += 1
                logger.warning(
                    f"Invalid hypothetical doc for '{query}' (quality={quality_score:.2f})"
                )
                return HyDEResult(
                    original_query=query,
                    hypothetical_doc=query,
                    from_cache=False,
                    quality_score=quality_score,
                    generation_time_ms=elapsed_ms,
                )

            self._cache[cache_key] = validated_doc
            self._save_cache()

            self._metrics.total_quality_score += quality_score
            self._metrics.total_generation_time_ms += elapsed_ms

            logger.debug(
                f"Generated HyDE doc for: {query[:30]}... -> "
                f"{validated_doc[:50]}... (quality={quality_score:.2f})"
            )

            return HyDEResult(
                original_query=query,
                hypothetical_doc=validated_doc,
                from_cache=False,
                cache_key=cache_key,
                quality_score=quality_score,
                generation_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.validation_failures += 1
            logger.warning(f"HyDE generation failed: {e}")
            return HyDEResult(
                original_query=query,
                hypothetical_doc=query,
                from_cache=False,
                quality_score=0.0,
                generation_time_ms=elapsed_ms,
            )

    def should_use_hyde(
        self,
        query: str,
        complexity: str = "medium",
        query_length: int = 0,
        has_regulatory_terms: bool = False,
    ) -> bool:
        if not query:
            return False

        query = query.strip()

        if query_length == 0:
            query_length = len(query)

        # Check for emotional/vague indicators FIRST (Cycle 8: priority check)
        # Even short queries with emotional indicators should use HyDE
        if any(indicator in query for indicator in self.EMOTIONAL_INDICATORS):
            return True

        if any(indicator in query for indicator in self.VAGUE_INDICATORS):
            return True

        # Skip for very short queries (but not if matched above)
        if query_length < 5:
            return False

        if query_length > 100:
            return False

        if has_regulatory_terms or any(term in query for term in self.REGULATORY_TERMS):
            return False

        structural_patterns = [
            r"제\d+조",
            r"\d+-\d+-\d+",
            r"^\s*[\d\s]+$",
        ]
        if any(re.search(pattern, query) for pattern in structural_patterns):
            return False

        if complexity == "complex":
            return True

        if complexity == "medium" and query_length > 10:
            question_patterns = ["?", "까요", "인가", "방법", "어떻"]
            if any(pattern in query for pattern in question_patterns):
                return True

        return False


class HyDESearcher:
    """Search using HyDE-enhanced queries."""

    def __init__(
        self,
        hyde_generator: HyDEGenerator,
        store: "IVectorStore",
    ):
        self._hyde = hyde_generator
        self._store = store

    def search_with_hyde(
        self,
        query: str,
        filter: Optional["SearchFilter"] = None,
        top_k: int = 10,
    ) -> List["SearchResult"]:
        from ..domain.value_objects import Query

        hyde_result = self._hyde.generate_hypothetical_doc(query)

        if (
            not hyde_result.hypothetical_doc
            or hyde_result.hypothetical_doc == query
            or hyde_result.quality_score < 0.3
        ):
            logger.debug(
                f"HyDE quality too low ({hyde_result.quality_score:.2f}), using normal search"
            )
            normal_query = Query(text=query)
            return self._store.search(normal_query, filter, top_k)

        hyde_query = Query(text=hyde_result.hypothetical_doc)
        hyde_results = self._store.search(hyde_query, filter, top_k * 2)

        original_query = Query(text=query)
        original_results = self._store.search(original_query, filter, top_k * 2)

        return self._merge_results(
            hyde_results,
            original_results,
            top_k,
            hyde_quality=hyde_result.quality_score,
        )

    def _merge_results(
        self,
        hyde_results: List["SearchResult"],
        original_results: List["SearchResult"],
        top_k: int,
        hyde_quality: float,
    ) -> List["SearchResult"]:
        seen_ids = set()
        merged = []

        hyde_weight = 0.7 if hyde_quality > 0.6 else 0.5
        original_weight = 1.0 - hyde_weight

        for result in hyde_results:
            if result.chunk.id not in seen_ids:
                result.score *= hyde_weight
                seen_ids.add(result.chunk.id)
                merged.append(result)

        for result in original_results:
            if result.chunk.id not in seen_ids:
                result.score *= original_weight
                seen_ids.add(result.chunk.id)
                merged.append(result)

        merged.sort(key=lambda x: -x.score)
        return merged[:top_k]
