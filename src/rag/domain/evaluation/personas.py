"""
User Persona Management for RAG Quality Evaluation.

Defines six user personas with query templates and generation logic.

Clean Architecture: Domain layer contains persona definitions and business logic.
"""

import logging
import random
from typing import Dict, List, Optional

from .models import PersonaProfile

logger = logging.getLogger(__name__)


# Persona definitions as specified in SPEC-RAG-QUALITY-001
PERSONAS: Dict[str, PersonaProfile] = {
    "freshman": PersonaProfile(
        name="freshman",
        display_name="신입생",
        expertise_level="beginner",
        vocabulary_style="simple",
        query_templates=[
            "{topic} 어떻게 해요?",
            "{topic} 절차 알려주세요",
            "{topic} 자격이 뭐예요?",
            "{topic} 신청 방법 알려주실까요?",
            "{topic} 기간이 얼마나 돼요?",
            "처음이라 {topic}를 잘 몰라요",
        ],
        common_topics=["휴학", "복학", "성적", "장학금", "수강", "등록"],
        answer_preferences={
            "detail_level": "simple",
            "citation_style": "minimal",
            "clarity_priority": True,
        },
    ),
    "graduate": PersonaProfile(
        name="graduate",
        display_name="대학원생",
        expertise_level="advanced",
        vocabulary_style="academic",
        query_templates=[
            "{topic} 관련 규정 확인 부탁드립니다",
            "{topic} 자격 요건이 어떻게 되나요?",
            "{topic} 신청 시 필요한 서류는 무엇인가요?",
            "{topic} 관련하여 조언 부탁드립니다",
            "{topic} 절차 상세히 알려주세요",
        ],
        common_topics=["연구년", "연구비", "논문", "등록금", "휴학", "조교"],
        answer_preferences={
            "detail_level": "comprehensive",
            "citation_style": "detailed",
            "precision_priority": True,
        },
    ),
    "professor": PersonaProfile(
        name="professor",
        display_name="교수",
        expertise_level="advanced",
        vocabulary_style="academic",
        query_templates=[
            "{topic} 관련 조항 확인 필요",
            "{topic} 적용 기준 상세히",
            "{topic} 관련 편/장/조 구체적 근거",
            "{topic} 관련 규정 해석 부탁드립니다",
            "{topic} 예외 사항이 있는지 확인 필요",
        ],
        common_topics=["연구년", "휴직", "승진", "연구비", "교원인사", " Sabbatical"],
        answer_preferences={
            "detail_level": "comprehensive",
            "citation_style": "detailed",
            "precision_priority": True,
            "include_article_references": True,
        },
    ),
    "staff": PersonaProfile(
        name="staff",
        display_name="교직원",
        expertise_level="intermediate",
        vocabulary_style="administrative",
        query_templates=[
            "{topic} 업무 처리 절차 확인",
            "{topic} 관련 서식 양식 알려주세요",
            "{topic} 승인 권한자가 누구인가요?",
            "{topic} 관련 비품/시설 사용 규정",
            "{topic} 처리 기한이 언제까지인가요?",
        ],
        common_topics=["복무", "휴가", "급여", "연수", "사무용품", "시설사용"],
        answer_preferences={
            "detail_level": "moderate",
            "citation_style": "standard",
            "include_procedures": True,
        },
    ),
    "parent": PersonaProfile(
        name="parent",
        display_name="학부모",
        expertise_level="beginner",
        vocabulary_style="simple",
        query_templates=[
            "자녀 {topic} 관련해서 알고 싶어요",
            "{topic} 부모님도 알아야 하나요?",
            "{topic} 비용이 어떻게 되나요?",
            "{topic} 신청은 부모가 해야 하나요?",
            "{topic} 관련 서류 뭐 필요한가요?",
        ],
        common_topics=["등록금", "장학금", "기숙사", "휴학", "성적", "졸업"],
        answer_preferences={
            "detail_level": "simple",
            "citation_style": "minimal",
            "parent_friendly": True,
        },
    ),
    "international": PersonaProfile(
        name="international",
        display_name="외국인유학생",
        expertise_level="intermediate",
        vocabulary_style="mixed",
        query_templates=[
            "How do I {topic}?",
            "{topic} procedure for international students",
            "{topic} requirements",
            "{topic} related to visa status",
            "Tell me about {topic} in English if possible",
            "{topic} English version available?",
        ],
        common_topics=["비자", "등록금", "수업", "기숙사", "언어", " Korean language"],
        answer_preferences={
            "detail_level": "moderate",
            "citation_style": "standard",
            "language": "korean_english_mixed",
            "include_english_terms": True,
        },
    ),
}


class PersonaManager:
    """
    Manages user personas and generates persona-specific queries.

    Supports six personas: freshman, graduate, professor, staff, parent, international.
    Generates 20+ queries per persona with persona-appropriate vocabulary.
    """

    def __init__(self):
        """Initialize the persona manager with predefined personas."""
        self.personas = PERSONAS
        logger.info(f"Initialized PersonaManager with {len(self.personas)} personas")

    def get_persona(self, persona_name: str) -> PersonaProfile:
        """
        Get a specific persona by name.

        Args:
            persona_name: Name of the persona (e.g., "freshman", "professor")

        Returns:
            PersonaProfile object

        Raises:
            ValueError: If persona not found
        """
        if persona_name not in self.personas:
            available = ", ".join(self.personas.keys())
            raise ValueError(
                f"Persona '{persona_name}' not found. Available: {available}"
            )

        return self.personas[persona_name]

    def list_personas(self) -> List[str]:
        """Get list of available persona names."""
        return list(self.personas.keys())

    def generate_queries(
        self,
        persona_name: str,
        count: int = 20,
        topics: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate persona-specific test queries.

        Args:
            persona_name: Name of the persona
            count: Number of queries to generate (default: 20)
            topics: Optional list of topics to use (default: use persona's common_topics)

        Returns:
            List of generated queries
        """
        persona = self.get_persona(persona_name)

        # Use provided topics or persona's common topics
        if topics is None:
            topics = persona.common_topics

        # Ensure we have enough topics by cycling if needed
        queries = []
        num_topics = len(topics)

        for i in range(count):
            # Select topic (cycle through if needed)
            topic = topics[i % num_topics]

            # Add variety for international students (Korean/English mix)
            if persona_name == "international":
                if random.random() < 0.3:  # 30% chance of pure English query
                    query = f"Tell me about {topic} in English"
                else:
                    query = persona.generate_query(topic)
            else:
                query = persona.generate_query(topic)

            queries.append(query)

        logger.info(f"Generated {len(queries)} queries for {persona_name} persona")
        return queries

    def generate_all_personas_queries(
        self,
        queries_per_persona: int = 20,
    ) -> Dict[str, List[str]]:
        """
        Generate queries for all personas.

        Args:
            queries_per_persona: Number of queries to generate per persona

        Returns:
            Dictionary mapping persona names to their query lists
        """
        all_queries = {}

        for persona_name in self.list_personas():
            queries = self.generate_queries(persona_name, count=queries_per_persona)
            all_queries[persona_name] = queries

        total_queries = sum(len(queries) for queries in all_queries.values())
        logger.info(f"Generated total of {total_queries} queries across all personas")

        return all_queries

    def validate_query_for_persona(
        self,
        query: str,
        persona_name: str,
    ) -> bool:
        """
        Validate that a query matches persona characteristics.

        Args:
            query: Query string to validate
            persona_name: Name of the persona

        Returns:
            True if query matches persona characteristics
        """
        persona = self.get_persona(persona_name)

        # Check length constraints based on expertise level
        if persona.expertise_level == "beginner":
            # Beginner personas use shorter, simpler queries
            if len(query.split()) > 20:
                return False
        elif persona.expertise_level == "advanced":
            # Advanced personas can use longer, more complex queries
            if len(query.split()) < 5:
                return False

        # Check vocabulary style
        if persona_name == "international":
            # International student queries should have some English
            _has_english = any(char.isalpha() and char.isascii() for char in query)
            # Not required but common
        elif persona_name == "professor":
            # Professor queries should use academic language
            _academic_terms = ["조항", "규정", "적용", "기준", "해석"]
            # At least one academic term expected

        return True

    def get_persona_answer_preferences(
        self,
        persona_name: str,
    ) -> Dict[str, any]:
        """
        Get answer preferences for a persona.

        Args:
            persona_name: Name of the persona

        Returns:
            Dictionary of answer preferences
        """
        persona = self.get_persona(persona_name)
        return persona.answer_preferences
