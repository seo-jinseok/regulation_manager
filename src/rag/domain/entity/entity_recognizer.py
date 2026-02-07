"""
Enhanced Entity Recognizer for Regulation RAG System.

Implements SPEC-RAG-SEARCH-001 TAG-001: Enhanced Entity Recognition.

Recognizes 6 new entity types:
1. Regulation sections (조, 항, 호)
2. Procedures (신청, 절차, 방법, 발급, 제출)
3. Requirements (자격, 요건, 조건, 기준, 제한)
4. Benefits (혜택, 지급, 지원, 급여)
5. Deadlines (기한, 마감, 날짜, 기간)
6. Hypernyms (hierarchical expansion: 등록금→학사→행정)
"""

import logging
import re
from typing import Dict, List

from .entity_types import EntityMatch, EntityRecognitionResult, EntityType

logger = logging.getLogger(__name__)


class RegulationEntityRecognizer:
    """
    Enhanced entity recognizer for regulation queries.

    Part of SPEC-RAG-SEARCH-001 TAG-001: Enhanced Entity Recognition.

    This class provides structured entity recognition with:
    - 6 distinct entity types
    - Confidence scoring
    - Query expansion terms
    - Hypernym (hierarchical) expansion

    Example:
        recognizer = RegulationEntityRecognizer()
        result = recognizer.recognize("장학금 신청 방법")

        # Returns EntityRecognitionResult with:
        # - BENEFIT entity for "장학금"
        # - PROCEDURE entities for "신청", "방법"
        # - Expanded terms: [장학금, 장학, 성적기준, ...]
    """

    # ============ SECTION PATTERNS ============
    # Patterns for regulation sections (제N조, 제N항, 제N호)
    SECTION_PATTERNS = [
        re.compile(r"제(\d+)조"),  # 제15조
        re.compile(r"제(\d+)항"),  # 제2항
        re.compile(r"제(\d+)호"),  # 제1호
        re.compile(r"제\s*(\d+)\s*조"),  # With spaces: 제 15 조
        re.compile(r"제\s*(\d+)\s*조의(\d+)"),  # 제15조의2
    ]

    # ============ PROCEDURE KEYWORDS ============
    # Procedure-related keywords (REQ-ER-002)
    PROCEDURE_KEYWORDS = [
        "신청",
        "절차",
        "방법",
        "발급",
        "제출",
        "신고",
        "등록",
        "신청서",
        "서류",
        "구비서류",
        "처리",
        "심사",
        "승인",
        "승인",
        "결재",
        "접수",
    ]

    # Procedure expansion mappings
    PROCEDURE_EXPANSIONS: Dict[str, List[str]] = {
        "신청": ["신청", "신청서", "제출", "등록", "접수"],
        "절차": ["절차", "방법", "과정", "단계", "순서"],
        "방법": ["방법", "절차", "방식", "요령", "하법"],
        "발급": ["발급", "수령", "교부", "지급"],
        "제출": ["제출", "제서", "신고", "접수"],
        "서류": ["서류", "구비서류", "증명서", "확인서", "신청서"],
    }

    # ============ REQUIREMENT KEYWORDS ============
    # Requirement-related keywords (REQ-ER-003)
    REQUIREMENT_KEYWORDS = [
        "자격",
        "요건",
        "조건",
        "기준",
        "제한",
        "대상",
        "충족",
        "요구",
        "필수",
        "선택",
        "우선",
        "가능",
        "불가",
        "제외",
    ]

    # Requirement expansion mappings
    REQUIREMENT_EXPANSIONS: Dict[str, List[str]] = {
        "자격": ["자격", "자격요건", "대상", "충족요건", "선정기준"],
        "요건": ["요건", "자격", "조건", "기준", "요구사항"],
        "조건": ["조건", "요건", "자격", "기준", "제한"],
        "기준": ["기준", "요건", "조건", "자격", "선정기준"],
        "제한": ["제한", "조건", "요건", "불가", "제외"],
    }

    # ============ BENEFIT KEYWORDS ============
    # Benefit-related keywords (REQ-ER-004)
    BENEFIT_KEYWORDS = [
        "혜택",
        "지급",
        "지원",
        "급여",
        "보조",
        "장학",
        "장학금",
        "수당",
        "비용",
        "경비",
        "지원금",
        "보조금",
        "수혜",
    ]

    # Benefit expansion mappings
    BENEFIT_EXPANSIONS: Dict[str, List[str]] = {
        "혜택": ["혜택", "지급", "지원", "급여", "수혜"],
        "지급": ["지급", "발급", "지원", "수령", "지급액"],
        "지원": ["지원", "보조", "장학", "지원금", "혜택"],
        "장학금": ["장학금", "장학", "성적장학금", "재정지원", "등록금감면"],
        "급여": ["급여", "지급", "수당", "연급", "급부"],
    }

    # ============ DEADLINE KEYWORDS ============
    # Deadline-related keywords (REQ-ER-005)
    DEADLINE_KEYWORDS = [
        "기한",
        "마감",
        "날짜",
        "기간",
        "까지",
        "이내",
        "이전",
        "이후",
        "부터",
        "당일",
        "매월",
        "매년",
        "시기",
        "일시",
    ]

    # Deadline expansion mappings
    DEADLINE_EXPANSIONS: Dict[str, List[str]] = {
        "기한": ["기한", "마감", "신청기간", "접수마감", "종료일"],
        "마감": ["마감", "기한", "마감일", "종료", "신청마감"],
        "기간": ["기간", "기간내", "소요기간", "처리기간", "유효기간"],
        "날짜": ["날짜", "일자", "시기", "일시", "마감일"],
    }

    # ============ HYPERNYM MAPPINGS ============
    # Hierarchical term expansion (REQ-ER-006)
    HYPERNYM_MAPPINGS: Dict[str, List[str]] = {
        "등록금": ["등록금", "수업료", "학사", "행정"],
        "장학금": ["장학금", "장학", "재정", "지원", "급여"],
        "휴학": ["휴학", "학적", "행정", "학사"],
        "복학": ["복학", "학적", "행정", "학사"],
        "교수": ["교수", "교원", "교직원", "임용"],
        "조교": ["조교", "교육조교", "연구조교", "지원"],
        "연구년": ["연구년", "안식년", "교원", "휴직"],
        "안식년": ["안식년", "연구년", "교원", "휴직"],
    }

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize entity recognizer.

        Args:
            confidence_threshold: Minimum confidence for entity matches (default: 0.5)
        """
        self._confidence_threshold = confidence_threshold

    def recognize(self, query: str) -> EntityRecognitionResult:
        """
        Recognize entities in query.

        This is the main entry point for entity recognition.
        It scans the query for all 6 entity types and returns matches.

        Args:
            query: The search query text

        Returns:
            EntityRecognitionResult with all found entities and expanded terms

        Example:
            recognizer = RegulationEntityRecognizer()
            result = recognizer.recognize("장학금 신청 방법")

            # result.matches contains:
            # - EntityMatch for "장학금" (BENEFIT, confidence=0.95)
            # - EntityMatch for "신청" (PROCEDURE, confidence=0.90)
            # - EntityMatch for "방법" (PROCEDURE, confidence=0.85)

            # result.total_expanded_terms contains:
            # [장학금, 장학, 성적기준, 신청, 신청서, 제출, 방법, 절차, ...]
        """
        if not query:
            return EntityRecognitionResult.from_matches(query, [])

        matches: List[EntityMatch] = []

        # 1. SECTION recognition (highest priority, exact pattern match)
        matches.extend(self._recognize_sections(query))

        # 2. PROCEDURE recognition
        matches.extend(self._recognize_procedures(query))

        # 3. REQUIREMENT recognition
        matches.extend(self._recognize_requirements(query))

        # 4. BENEFIT recognition
        matches.extend(self._recognize_benefits(query))

        # 5. DEADLINE recognition
        matches.extend(self._recognize_deadlines(query))

        # 6. HYPERNYM recognition (must check after keywords for context)
        matches.extend(self._recognize_hypernyms(query))

        # Filter by confidence threshold
        filtered_matches = [
            m for m in matches if m.confidence >= self._confidence_threshold
        ]

        return EntityRecognitionResult.from_matches(query, filtered_matches)

    def _recognize_sections(self, query: str) -> List[EntityMatch]:
        """Recognize regulation section references (제N조, 제N항, 제N호)."""
        matches = []

        for pattern in self.SECTION_PATTERNS:
            for match in pattern.finditer(query):
                entity_match = EntityMatch(
                    entity_type=EntityType.SECTION,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,  # High confidence for exact pattern match
                    expanded_terms=[],  # Sections don't need expansion
                )
                matches.append(entity_match)

        return matches

    def _recognize_procedures(self, query: str) -> List[EntityMatch]:
        """Recognize procedure-related keywords (REQ-ER-002)."""
        matches = []

        for keyword in self.PROCEDURE_KEYWORDS:
            if keyword in query:
                start = query.index(keyword)
                expanded = self.PROCEDURE_EXPANSIONS.get(keyword, [])

                entity_match = EntityMatch(
                    entity_type=EntityType.PROCEDURE,
                    text=keyword,
                    start=start,
                    end=start + len(keyword),
                    confidence=0.85,  # High confidence for keyword match
                    expanded_terms=expanded,
                )
                matches.append(entity_match)

        return matches

    def _recognize_requirements(self, query: str) -> List[EntityMatch]:
        """Recognize requirement-related keywords (REQ-ER-003)."""
        matches = []

        for keyword in self.REQUIREMENT_KEYWORDS:
            if keyword in query:
                start = query.index(keyword)
                expanded = self.REQUIREMENT_EXPANSIONS.get(keyword, [])

                entity_match = EntityMatch(
                    entity_type=EntityType.REQUIREMENT,
                    text=keyword,
                    start=start,
                    end=start + len(keyword),
                    confidence=0.85,
                    expanded_terms=expanded,
                )
                matches.append(entity_match)

        return matches

    def _recognize_benefits(self, query: str) -> List[EntityMatch]:
        """Recognize benefit-related keywords (REQ-ER-004)."""
        matches = []

        for keyword in self.BENEFIT_KEYWORDS:
            if keyword in query:
                start = query.index(keyword)
                expanded = self.BENEFIT_EXPANSIONS.get(keyword, [])

                entity_match = EntityMatch(
                    entity_type=EntityType.BENEFIT,
                    text=keyword,
                    start=start,
                    end=start + len(keyword),
                    confidence=0.90,  # High confidence for benefit keywords
                    expanded_terms=expanded,
                )
                matches.append(entity_match)

        return matches

    def _recognize_deadlines(self, query: str) -> List[EntityMatch]:
        """Recognize deadline-related keywords (REQ-ER-005)."""
        matches = []

        for keyword in self.DEADLINE_KEYWORDS:
            if keyword in query:
                start = query.index(keyword)
                expanded = self.DEADLINE_EXPANSIONS.get(keyword, [])

                entity_match = EntityMatch(
                    entity_type=EntityType.DEADLINE,
                    text=keyword,
                    start=start,
                    end=start + len(keyword),
                    confidence=0.80,
                    expanded_terms=expanded,
                )
                matches.append(entity_match)

        return matches

    def _recognize_hypernyms(self, query: str) -> List[EntityMatch]:
        """
        Recognize hypernym expansion opportunities (REQ-ER-006).

        Hypernym expansion adds hierarchical terms:
        - 등록금 → [등록금, 학사, 행정]
        - 장학금 → [장학금, 재정, 지원]
        """
        matches = []

        for keyword, hypernyms in self.HYPERNYM_MAPPINGS.items():
            if keyword in query:
                start = query.index(keyword)

                entity_match = EntityMatch(
                    entity_type=EntityType.HYPERNYM,
                    text=keyword,
                    start=start,
                    end=start + len(keyword),
                    confidence=0.75,  # Medium confidence (context-dependent)
                    expanded_terms=hypernyms,
                )
                matches.append(entity_match)

        return matches

    def get_expanded_query(
        self, query: str, max_terms: int = 10, use_entities: bool = True
    ) -> str:
        """
        Get query expanded with entity terms.

        Convenience method that combines recognition and expansion.

        Args:
            query: Original query
            max_terms: Maximum number of expanded terms to include
            use_entities: Whether to use entity recognition (default: True)

        Returns:
            Expanded query string

        Example:
            recognizer = RegulationEntityRecognizer()
            expanded = recognizer.get_expanded_query("장학금 신청")
            # Returns: "장학금 신청 장학 성적기준 신청서 제출"
        """
        if not use_entities:
            return query

        result = self.recognize(query)
        return result.get_expanded_query(max_terms=max_terms)
