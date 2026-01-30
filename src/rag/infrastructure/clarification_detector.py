"""
Clarification Detector for RAG System.

Detects ambiguous or insufficient queries and generates clarification requests.
Reduces hallucinations by asking for more specific information when needed.
"""

import re
from dataclasses import dataclass
from typing import List

from ..domain.entities import SearchResult


@dataclass
class ClarificationRequest:
    """Clarification request result."""

    needs_clarification: bool
    reason: str
    clarification_questions: List[str]
    suggested_options: List[str]


@dataclass
class QueryAnalysis:
    """Query analysis result."""

    is_short: bool
    is_single_word: bool
    is_ambiguous: bool
    word_count: int
    char_count: int
    key_terms: List[str]


class ClarificationDetector:
    """
    Detects queries that need clarification and generates clarification requests.

    Features:
    - Detects single-word or short queries
    - Identifies ambiguous queries lacking context
    - Generates clarification questions based on retrieved context
    - Provides multiple choice options when applicable
    """

    # Korean character threshold for short queries
    SHORT_QUERY_CHAR_THRESHOLD = 10  # <= 10 Korean characters
    SHORT_QUERY_WORD_THRESHOLD = 2  # <= 2 words

    # Single-word pattern
    SINGLE_WORD_PATTERN = re.compile(r"^[\w\uAC00-\uD7AF]+$")

    # Ambiguous query patterns
    AMBIGUOUS_PATTERNS = [
        re.compile(r"^(어떻게|방법|절차|신청)$"),
        re.compile(r"^(자격|조건|기준)$"),
        re.compile(r"^(기간|언제|면제)$"),
        re.compile(r"^(서류|문서|증명)$"),
        re.compile(r"^(학점|성적|평점)$"),
        re.compile(r"^(장학|등록금|휴학)$"),
        re.compile(r"^(졸업|복학|재입학)$"),
        re.compile(r"^(규정|학칙|시행세칙)$"),
    ]

    def __init__(self):
        """Initialize ClarificationDetector."""
        pass

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query characteristics.

        Args:
            query: The user's query string

        Returns:
            QueryAnalysis with query characteristics
        """
        query = query.strip()
        words = query.split()

        # Count Korean characters (excluding spaces and punctuation)
        korean_chars = re.findall(r"[\uAC00-\uD7AF]", query)
        char_count = len(korean_chars)
        word_count = len(words)

        # Check if single word
        is_single_word = bool(self.SINGLE_WORD_PATTERN.match(query))

        # Check if short
        is_short = (
            char_count <= self.SHORT_QUERY_CHAR_THRESHOLD
            or word_count <= self.SHORT_QUERY_WORD_THRESHOLD
        )

        # Check if ambiguous
        is_ambiguous = bool(
            any(pattern.match(query) for pattern in self.AMBIGUOUS_PATTERNS)
        )

        # Extract key terms (remove common particles)
        key_terms = self._extract_key_terms(query)

        return QueryAnalysis(
            is_short=is_short,
            is_single_word=is_single_word,
            is_ambiguous=is_ambiguous,
            word_count=word_count,
            char_count=char_count,
            key_terms=key_terms,
        )

    def is_clarification_needed(self, query: str, results: List[SearchResult]) -> bool:
        """
        Determine if clarification is needed for the query.

        Args:
            query: The user's query string
            results: Search results from the query

        Returns:
            True if clarification is needed
        """
        analysis = self.analyze_query(query)

        # Immediate clarification needed for single-word or very short queries
        if analysis.is_single_word or (analysis.is_short and analysis.word_count == 1):
            return True

        # Clarification needed for ambiguous queries
        if analysis.is_ambiguous:
            return True

        # Clarification needed if no results found (query might be too general)
        if len(results) == 0:
            return True

        # Clarification needed if results are too diverse (ambiguous query)
        if len(results) > 5:
            # Check if results are from different regulations (high diversity)
            regulation_set = set()
            for r in results[:5]:
                if r.chunk.parent_path and len(r.chunk.parent_path) > 0:
                    regulation_set.add(r.chunk.parent_path[0])
            if len(regulation_set) > 2:
                return True

        return False

    def generate_clarification(
        self, query: str, results: List[SearchResult]
    ) -> ClarificationRequest:
        """
        Generate clarification request for the query.

        Args:
            query: The user's query string
            results: Search results from the query

        Returns:
            ClarificationRequest with clarification questions and options
        """
        analysis = self.analyze_query(query)

        # Determine reason for clarification
        reason = self._determine_clarification_reason(analysis, results)

        # Generate clarification questions
        questions = self._generate_clarification_questions(query, analysis, results)

        # Generate suggested options
        options = self._generate_suggested_options(query, analysis, results)

        return ClarificationRequest(
            needs_clarification=True,
            reason=reason,
            clarification_questions=questions,
            suggested_options=options,
        )

    def _extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms from query (remove common particles).

        Args:
            query: The query string

        Returns:
            List of key terms
        """
        # Remove common particles
        particles = [
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "의",
            "에",
            "에서",
            "로",
            "으로",
        ]
        terms = []

        for word in query.split():
            # Remove trailing particles
            cleaned = word
            for particle in particles:
                if cleaned.endswith(particle) and len(cleaned) > 1:
                    cleaned = cleaned[: -len(particle)]
            if cleaned:
                terms.append(cleaned)

        return terms

    def _determine_clarification_reason(
        self, analysis: QueryAnalysis, results: List[SearchResult]
    ) -> str:
        """
        Determine the reason for clarification.

        Args:
            analysis: Query analysis result
            results: Search results

        Returns:
            Reason string for clarification
        """
        if analysis.is_single_word:
            term = analysis.key_terms[0] if analysis.key_terms else "검색어"
            return f"'{term}' 검색은 여러 규정에서 관련 내용을 찾을 수 있어 구체적인 정보가 필요합니다."

        if analysis.is_short:
            return "검색어가 너무 짧아 정확한 답변을 드리기 어렵습니다."

        if analysis.is_ambiguous:
            return f"'{analysis.key_terms[0]}'에 대한 정보가 여러 규정에 있습니다."

        if len(results) == 0:
            return "검색 결과를 찾을 수 없습니다. 더 구체적인 검색어가 필요합니다."

        if len(results) > 5:
            return "여러 규정에서 관련 내용을 찾았습니다. 원하는 정보를 구체적으로 말씀해 주세요."

        return "정확한 답변을 위해 추가 정보가 필요합니다."

    def _generate_clarification_questions(
        self, query: str, analysis: QueryAnalysis, results: List[SearchResult]
    ) -> List[str]:
        """
        Generate clarification questions based on query and results.

        Args:
            query: The original query
            analysis: Query analysis result
            results: Search results

        Returns:
            List of clarification questions
        """
        questions = []

        if analysis.is_single_word or analysis.is_short:
            term = analysis.key_terms[0] if analysis.key_terms else query
            questions.append(f"'{term}'에 대해 어떤 것을 알고 싶으신가요?")
            questions.append(
                f"{term} 신청 방법, 자격 요건, 제출 서류 중 무엇이 궁금하신가요?"
            )

        elif analysis.is_ambiguous:
            term = analysis.key_terms[0] if analysis.key_terms else query
            if term in ["신청", "방법", "절차"]:
                questions.append("어떤 절차나 신청에 대해 알고 싶으신가요?")
                questions.append(
                    "특정 규정이나 항목을 지정해 주시면 정확한 답변이 가능합니다."
                )
            elif term in ["자격", "조건", "기준"]:
                questions.append("어떤 자격이나 요건을 확인하고 싶으신가요?")
                questions.append("특정 장학금, 휴학, 복학 등 구체적으로 말씀해 주세요.")
            elif term in ["기간", "언제", "면제"]:
                questions.append("어떤 기간이나 시기에 대해 알고 싶으신가요?")
                questions.append("신청 기간, 휴학 기간 등 구체적으로 말씀해 주세요.")

        elif len(results) > 5:
            # Generate questions based on diverse results
            regulations = set()
            for r in results[:5]:
                if r.chunk.parent_path and len(r.chunk.parent_path) > 0:
                    regulations.add(r.chunk.parent_path[0])

            if len(regulations) > 2:
                reg_list = list(regulations)[:3]
                questions.append(
                    f"어떤 규정에 대해 알고 싶으신가요? ({', '.join(reg_list)} 등)"
                )

        if not questions:
            questions.append(
                "더 구체적인 질문을 해 주시면 정확한 답변을 드릴 수 있습니다."
            )

        return questions

    def _generate_suggested_options(
        self, query: str, analysis: QueryAnalysis, results: List[SearchResult]
    ) -> List[str]:
        """
        Generate suggested options for the user.

        Args:
            query: The original query
            analysis: Query analysis result
            results: Search results

        Returns:
            List of suggested options
        """
        options = []

        if analysis.is_single_word or analysis.is_short:
            term = analysis.key_terms[0] if analysis.key_terms else query
            options.append(f"{term} 신청 방법")
            options.append(f"{term} 자격 요건")
            options.append(f"{term} 제출 서류")
            options.append(f"{term} 심사 기간")

        elif analysis.is_ambiguous:
            term = analysis.key_terms[0] if analysis.key_terms else query
            if term in ["신청", "방법", "절차"]:
                options.append("휴학 신청 방법")
                options.append("장학금 신청 방법")
                options.append("증명서 발급 방법")
            elif term in ["자격", "조건", "기준"]:
                options.append("장학금 수혜 자격")
                options.append("휴학 자격 요건")
                options.append("복학 자격 요건")
            elif term in ["기간", "언제"]:
                options.append("휴학 신청 기간")
                options.append("장학금 신청 기간")
                options.append("등록금 납부 기간")

        elif len(results) > 0:
            # Generate options based on search results
            titles = set()
            for r in results[:3]:
                if r.chunk.parent_path and len(r.chunk.parent_path) > 0:
                    regulation = r.chunk.parent_path[0]
                    titles.add(regulation)

            for title in list(titles)[:3]:
                options.append(f"{title} 관련 규정")

        if not options:
            options = ["구체적인 질문을 입력해 주세요"]

        return options
