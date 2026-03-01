"""
Self-RAG Evaluator for Enhanced Retrieval-Augmented Generation.

Implements self-reflection mechanism where the LLM evaluates:
1. Whether retrieval is needed for a query
2. Whether retrieved documents are relevant
3. Whether the generated answer is supported by the context

9 RAG Architectures (#5): Self-RAG adds reflection capabilities to verify
retrieval necessity and answer grounding, improving response quality.

SPEC-RAG-QUALITY-011: Self-RAG Classification Fix
- Improved prompt with regulation domain context
- Keyword-based pre-filtering for fast classification
- Fallback mechanism with override capability
- Classification metrics tracking
"""

import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..config import get_config

if TYPE_CHECKING:
    from ..domain.entities import SearchResult
    from ..domain.repositories import ILLMClient


logger = logging.getLogger(__name__)


# SPEC-RAG-QUALITY-011 REQ-002: Regulation-related keywords for pre-filtering
# SPEC-RAG-003 Task 1.1: Added English keywords for bilingual support
REGULATION_KEYWORDS = [
    # Regulation types (Korean)
    "규정", "학칙", "지침", "요강", "준칙", "세칙", "규칙", "정관",
    # Structural references
    "제", "조", "항", "호", "장", "절",
    # Common topics (Korean)
    "등록", "휴학", "복학", "졸업", "장학금", "성적", "학점",
    "전공", "부전공", "복수전공", "학부", "학과", "대학원",
    "교수", "교원", "직원", "임용", "승진", "연구",
    "휴직", "복직", "재직", "정년",
    # Staff/Admin topics (Korean) - SPEC-RAG-QUALITY-014 EARS-U-001
    "급여", "보수", "수당", "서식", "양식",
    "복무", "근무", "출장", "휴가", "연수",
    "퇴직", "인사", "복리후생", "보험", "겸직",
    # Question words (Korean)
    "어떻게", "언제", "누가", "무엇", "어디서", "왜",
    # Action words (Korean)
    "신청", "제출", "변경", "취소", "이의", "합격",
    "자격", "요건", "기간", "절차", "방법", "기준",
    # English: Regulation types
    "regulation", "rule", "policy", "guideline", "procedure",
    # English: Common topics
    "tuition", "fee", "scholarship", "grant",
    "leave", "absence", "withdrawal", "return",
    "dormitory", "housing", "residence",
    "registration", "course", "credit", "grade",
    "graduation", "degree", "diploma", "thesis",
    "professor", "faculty", "instructor",
    "visa", "international",
    # English: Action/question words
    "apply", "submit", "requirement", "qualification", "deadline",
]


class SelfRAGEvaluator:
    """
    Self-RAG reflection mechanism.

    Uses LLM to evaluate retrieval necessity and result relevance,
    enabling more accurate and grounded responses.

    SPEC-RAG-QUALITY-011: Enhanced with keyword pre-filtering,
    fallback mechanism, and metrics tracking.
    """

    # SPEC-RAG-003 Task 1.2: Bilingual prompt for retrieval necessity evaluation
    RETRIEVAL_NEEDED_PROMPT = """You are a query classifier for a university regulation search system.
This system handles queries in BOTH Korean and English.

Query: {query}

Determine if this query is related to university regulations, rules, guidelines, or procedures.

**ALWAYS needs search (both Korean AND English):**
- Questions about regulations, rules, guidelines (규정, 학칙, 지침)
- Questions about procedures, methods, deadlines, qualifications (절차, 방법, 기간, 자격)
- Questions about tuition, scholarship, leave of absence, graduation
- Questions about dormitory, housing, registration, courses, grades
- Questions about professors, faculty, visa, international students
- Questions starting with "How", "When", "What", "Where", "Who"
- Questions starting with "어떻게", "언제", "누가", "무엇"
- Examples: "How do I apply for a leave of absence?", "What are the tuition fees?"
- Examples: "장학금 신청 방법", "졸업 요건이 뭔가요?"

**Does NOT need search (very rare):**
- Simple greetings ("Hello", "안녕하세요")
- Completely unrelated general knowledge questions

**IMPORTANT:** When in doubt, ALWAYS choose [RETRIEVE_YES].
In a regulation Q&A system, false positives (unnecessary search) are better than false negatives (missed search).
If the query mentions ANY university-related topic in ANY language → [RETRIEVE_YES]

Answer format: Output only [RETRIEVE_YES] or [RETRIEVE_NO]."""

    RELEVANCE_EVAL_PROMPT = """다음 문서가 질문에 답변하는 데 관련이 있는지 평가하세요.

질문: {query}

문서:
{context}

평가 기준:
- 문서가 질문의 주제와 직접적으로 관련이 있는가?
- 문서에 질문에 답하는 데 필요한 정보가 포함되어 있는가?

답변 형식: [RELEVANT] 또는 [IRRELEVANT] 중 하나만 출력하세요."""

    SUPPORT_EVAL_PROMPT = """다음 답변이 제공된 문서에 의해 충분히 뒷받침되는지 평가하세요.

질문: {query}

문서:
{context}

생성된 답변:
{answer}

평가 기준:
- 답변의 주요 주장이 문서에 근거하고 있는가?
- 문서에 없는 정보를 추측하거나 추가하지 않았는가?

답변 형식: [SUPPORTED], [PARTIALLY_SUPPORTED], 또는 [NOT_SUPPORTED] 중 하나만 출력하세요."""

    def __init__(self, llm_client: Optional["ILLMClient"] = None):
        """
        Initialize Self-RAG evaluator.

        Args:
            llm_client: LLM client for evaluation (can be set later)
        """
        self._llm_client = llm_client
        # SPEC-RAG-QUALITY-011 REQ-006: Metrics tracking
        self._metrics = {
            "retrieval_yes_count": 0,
            "retrieval_no_count": 0,
            "bypass_count": 0,
            "override_count": 0,
        }

    def _has_regulation_keywords(self, query: str) -> bool:
        """
        Check if query contains regulation-related keywords.

        SPEC-RAG-QUALITY-011 REQ-002: Keyword-based pre-filtering.
        Fast, deterministic classification for obvious cases.

        Args:
            query: User's question

        Returns:
            True if regulation keywords detected
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in REGULATION_KEYWORDS)

    # SPEC-RAG-003 Task 1.3: University topic words for override mechanism
    _UNIVERSITY_TOPIC_WORDS = {
        "tuition", "fee", "scholarship", "leave", "absence",
        "dormitory", "housing", "course", "registration",
        "grade", "graduation", "thesis", "professor", "visa",
        "international", "student", "university", "campus",
        "semester", "credit", "enroll", "withdraw", "degree",
        "academic", "faculty", "exam", "gpa",
    }

    def _has_university_topic_words(self, query: str) -> bool:
        """
        Check if query contains university-related topic words (English).

        SPEC-RAG-003 EARS-I-001: Override mechanism for English queries
        when LLM incorrectly returns RETRIEVE_NO.
        Requires at least 2 matching topic words for override.

        Args:
            query: User's question

        Returns:
            True if 2+ university topic words found
        """
        query_words = set(query.lower().split())
        matches = query_words & self._UNIVERSITY_TOPIC_WORDS
        return len(matches) >= 2

    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        """Set the LLM client for evaluation."""
        self._llm_client = llm_client

    def needs_retrieval(self, query: str) -> bool:
        """
        Evaluate if retrieval is needed for this query.

        SPEC-RAG-QUALITY-011 REQ-002, REQ-003:
        1. Keyword pre-filtering (bypass LLM if keywords found)
        2. LLM classification for non-obvious cases
        3. Override mechanism if LLM says NO but keywords exist

        For regulation Q&A systems, retrieval is almost always needed.
        Only skip retrieval for very simple greetings/chat.

        Args:
            query: User's question

        Returns:
            True if retrieval is recommended
        """
        # SPEC-RAG-QUALITY-011 REQ-002: Keyword pre-filtering
        if self._has_regulation_keywords(query):
            self._metrics["bypass_count"] += 1
            logger.debug(f"Query bypassed LLM due to keywords: {query[:50]}...")
            return True

        # SPEC-RAG-QUALITY-011 REQ-003: Fallback if no LLM
        if not self._llm_client:
            return True  # Default to retrieval if no LLM

        # Quick heuristic: very short queries without question words
        # are probably not factual questions
        if len(query) < 5 and not any(
            k in query for k in ["?", "뭐", "어떻", "왜", "언제"]
        ):
            return True  # Still default to retrieval to be safe

        prompt = self.RETRIEVAL_NEEDED_PROMPT.format(query=query)

        try:
            response = self._llm_client.generate(
                system_prompt="You are a retrieval necessity evaluator for a bilingual university regulation system.",
                user_message=prompt,
                temperature=0.0,
                max_tokens=256,
            )
            # Default to retrieval unless explicitly told not to
            if "[RETRIEVE_NO]" in response.upper():
                # SPEC-RAG-QUALITY-011 REQ-003: Override check (Korean keywords)
                if self._has_regulation_keywords(query):
                    self._metrics["override_count"] += 1
                    logger.info(
                        f"Override: LLM said NO but keywords found in query: {query[:50]}..."
                    )
                    return True
                # SPEC-RAG-003 Task 1.3: University topic override for English queries
                if self._has_university_topic_words(query):
                    self._metrics["override_count"] += 1
                    logger.info(
                        f"Override: LLM said NO but university topics found in query: {query[:50]}..."
                    )
                    return True
                self._metrics["retrieval_no_count"] += 1
                return False
            self._metrics["retrieval_yes_count"] += 1
            return True  # Default to retrieval
        except Exception:
            return True  # Default to retrieval on error

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get classification metrics.

        SPEC-RAG-QUALITY-011 REQ-006: Metrics tracking.

        Returns:
            Dictionary with classification counts and rates
        """
        total = (
            self._metrics["retrieval_yes_count"]
            + self._metrics["retrieval_no_count"]
            + self._metrics["bypass_count"]
        )

        metrics = self._metrics.copy()

        if total > 0:
            metrics["override_rate"] = self._metrics["override_count"] / total
            metrics["bypass_rate"] = self._metrics["bypass_count"] / total

        return metrics

    def evaluate_relevance(
        self,
        query: str,
        results: List["SearchResult"],
        max_context_chars: Optional[int] = None,
    ) -> tuple:
        """
        Evaluate relevance of search results.

        Args:
            query: User's question
            results: Search results to evaluate
            max_context_chars: Maximum context length to send to LLM (increased to 4000 for long procedure regulations)

        Returns:
            Tuple of (is_relevant: bool, relevant_results: List[SearchResult])
        """
        if max_context_chars is None:
            max_context_chars = get_config().max_context_chars
        if not self._llm_client or not results:
            return (bool(results), results)

        # Build context from top results
        context_parts = []
        current_len = 0

        for result in results[:5]:  # Evaluate top 5
            text = result.chunk.text[:500]
            context_parts.append(f"[{result.chunk.title}]\n{text}")
            current_len += len(text)

            if current_len >= max_context_chars:
                break

        context = "\n\n---\n\n".join(context_parts)
        prompt = self.RELEVANCE_EVAL_PROMPT.format(query=query, context=context)

        try:
            response = self._llm_client.generate(
                system_prompt="You are a document relevance evaluator.",
                user_message=prompt,
                temperature=0.0,
                max_tokens=256,
            )
            is_relevant = "[RELEVANT]" in response.upper()

            # If relevant, return all results; otherwise return empty
            return (is_relevant, results if is_relevant else [])
        except Exception:
            return (True, results)  # Default to relevant on error

    def evaluate_support(self, query: str, context: str, answer: str) -> str:
        """
        Evaluate if the answer is supported by the context.

        Args:
            query: User's question
            context: Retrieved context
            answer: Generated answer

        Returns:
            Support level: "SUPPORTED", "PARTIALLY_SUPPORTED", or "NOT_SUPPORTED"
        """
        if not self._llm_client:
            return "SUPPORTED"  # Default to supported if no LLM

        prompt = self.SUPPORT_EVAL_PROMPT.format(
            query=query, context=context[:2000], answer=answer[:1000]
        )

        try:
            response = self._llm_client.generate(
                system_prompt="You are an answer support evaluator.",
                user_message=prompt,
                temperature=0.0,
                max_tokens=256,
            )

            response_upper = response.upper()
            if "[SUPPORTED]" in response_upper:
                return "SUPPORTED"
            elif "[PARTIALLY_SUPPORTED]" in response_upper:
                return "PARTIALLY_SUPPORTED"
            else:
                return "NOT_SUPPORTED"
        except Exception:
            return "SUPPORTED"  # Default on error


class SelfRAGPipeline:
    """
    Complete Self-RAG pipeline integrating evaluation with search and generation.

    Features:
    - Retrieval necessity check: Skip retrieval for simple queries
    - Relevance filtering: Remove irrelevant results before answer generation
    - Support verification: Verify answer is grounded in retrieved context
    - Async support evaluation: Run support check in background for performance
    """

    def __init__(
        self,
        search_usecase=None,
        llm_client: Optional["ILLMClient"] = None,
        enable_retrieval_check: bool = True,
        enable_relevance_check: bool = True,
        enable_support_check: bool = True,  # Now enabled by default
        async_support_check: bool = True,  # Run support check asynchronously
    ):
        """
        Initialize Self-RAG pipeline.

        Args:
            search_usecase: SearchUseCase for retrieval
            llm_client: LLM client for evaluation and generation
            enable_retrieval_check: Whether to check if retrieval is needed
            enable_relevance_check: Whether to check result relevance
            enable_support_check: Whether to check answer support (now default True)
            async_support_check: Whether to run support check asynchronously
        """
        self._search_usecase = search_usecase
        self._llm_client = llm_client
        self._evaluator = SelfRAGEvaluator(llm_client)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_support_check: Optional[concurrent.futures.Future] = None

        self.enable_retrieval_check = enable_retrieval_check
        self.enable_relevance_check = enable_relevance_check
        self.enable_support_check = enable_support_check
        self.async_support_check = async_support_check

    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        """Set LLM client for both evaluation and generation."""
        self._llm_client = llm_client
        self._evaluator.set_llm_client(llm_client)

    def should_retrieve(self, query: str) -> bool:
        """Check if retrieval is needed for this query."""
        if not self.enable_retrieval_check:
            return True
        return self._evaluator.needs_retrieval(query)

    def filter_relevant_results(
        self, query: str, results: List["SearchResult"]
    ) -> List["SearchResult"]:
        """Filter results to only include relevant ones."""
        if not self.enable_relevance_check:
            return results
        _, relevant = self._evaluator.evaluate_relevance(query, results)
        return relevant

    def get_support_level(self, query: str, context: str, answer: str) -> str:
        """Get the support level of an answer."""
        if not self.enable_support_check:
            return "SUPPORTED"
        return self._evaluator.evaluate_support(query, context, answer)

    def start_async_support_check(
        self, query: str, context: str, answer: str
    ) -> Optional[concurrent.futures.Future]:
        """
        Start support check in background thread.

        Returns immediately with a Future that can be checked later.
        This allows the main response to be sent while support verification runs.

        Args:
            query: User's question
            context: Retrieved context
            answer: Generated answer

        Returns:
            Future object or None if async is disabled
        """
        if not self.enable_support_check:
            return None
        if not self.async_support_check:
            # Run synchronously
            return None

        try:
            future = self._executor.submit(
                self._evaluator.evaluate_support, query, context, answer
            )
            self._pending_support_check = future
            logger.debug("Started async support check")
            return future
        except Exception as e:
            logger.warning(f"Failed to start async support check: {e}")
            return None

    def get_async_support_result(self, timeout: float = 5.0) -> Optional[str]:
        """
        Get the result of async support check if available.

        Args:
            timeout: Maximum seconds to wait for result

        Returns:
            Support level or None if not available/timed out
        """
        if self._pending_support_check is None:
            return None

        try:
            result = self._pending_support_check.result(timeout=timeout)
            self._pending_support_check = None
            return result
        except concurrent.futures.TimeoutError:
            logger.debug("Async support check timed out")
            return None
        except Exception as e:
            logger.warning(f"Async support check failed: {e}")
            return None

    def evaluate_results_batch(
        self, query: str, results: List["SearchResult"]
    ) -> Tuple[bool, List["SearchResult"], float]:
        """
        Evaluate search results in batch for efficiency.

        Returns:
            Tuple of (is_relevant, filtered_results, confidence_score)
        """
        if not results:
            return (False, [], 0.0)

        # Defensive type checking: Convert dict to SearchResult if necessary
        processed_results = []
        for r in results:
            if isinstance(r, dict):
                # Convert dict representation to SearchResult
                # This handles cases where data was serialized/deserialized
                try:
                    from ..domain.entities import Chunk
                    from ..domain.entities import SearchResult as SR

                    chunk_data = r.get("chunk")
                    score = r.get("score", 0.0)
                    rank = r.get("rank", 0)

                    # Handle chunk as dict or Chunk object
                    if isinstance(chunk_data, dict):
                        chunk = Chunk(
                            id=chunk_data.get("id", ""),
                            rule_code=chunk_data.get("rule_code", ""),
                            level=chunk_data.get("level", "text"),
                            title=chunk_data.get("title", ""),
                            text=chunk_data.get("text", ""),
                            embedding_text=chunk_data.get("embedding_text", ""),
                            full_text=chunk_data.get("full_text", ""),
                            parent_path=chunk_data.get("parent_path", []),
                            token_count=chunk_data.get("token_count", 0),
                            keywords=chunk_data.get("keywords", []),
                            is_searchable=chunk_data.get("is_searchable", True),
                            doc_type=chunk_data.get("doc_type", "regulation"),
                            effective_date=chunk_data.get("effective_date"),
                            status=chunk_data.get("status", "active"),
                        )
                    else:
                        # chunk is already a Chunk object
                        chunk = chunk_data

                    processed_results.append(SR(chunk=chunk, score=score, rank=rank))
                except Exception as e:
                    logger.warning(f"Failed to convert dict to SearchResult: {e}")
                    continue
            else:
                # Already a SearchResult object
                processed_results.append(r)

        if not processed_results:
            return (False, [], 0.0)

        results = processed_results

        # Quick heuristic check first (no LLM call)
        top_score = results[0].score if results else 0.0

        # If top score is very high, skip LLM evaluation
        if top_score > 0.8:
            return (True, results, top_score)

        # Use LLM for borderline cases
        if self.enable_relevance_check:
            is_relevant, filtered = self._evaluator.evaluate_relevance(query, results)
            confidence = top_score if is_relevant else top_score * 0.5
            return (is_relevant, filtered, confidence)

        return (True, results, top_score)
