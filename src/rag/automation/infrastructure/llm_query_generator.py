"""
Query Generator using LLM.

Infrastructure layer implementation that generates diverse queries
for RAG testing using Claude AI API.

Clean Architecture: Infrastructure implements domain interfaces and uses
external libraries (anthropic).
"""

import random
from typing import List

from src.rag.automation.domain.entities import (
    DifficultyLevel,
    Persona,
    PersonaType,
    QueryType,
    TestCase,
)
from src.rag.automation.domain.value_objects import IntentAnalysis


class QueryGenerator:
    """
    Generates test queries with 3-level intent analysis.

    Each query is tagged with persona, difficulty, query type, and
    includes intent analysis for RAG system validation.
    """

    # Sample query templates for each persona type
    _QUERY_TEMPLATES = {
        PersonaType.FRESHMAN: [
            ("휴학 신청은 어떻게 하나요?", QueryType.PROCEDURAL, DifficultyLevel.EASY),
            (
                "학생회비 내는 방법 알려주세요",
                QueryType.PROCEDURAL,
                DifficultyLevel.EASY,
            ),
            ("장학금 신청 자격이 뭐야?", QueryType.ELIGIBILITY, DifficultyLevel.MEDIUM),
            ("기숙사 입주 조건 알려줘", QueryType.ELIGIBILITY, DifficultyLevel.MEDIUM),
        ],
        PersonaType.JUNIOR: [
            (
                "졸업 요건이 어떻게 되나요?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.MEDIUM,
            ),
            (
                "복수 전공 신청하는 방법 알려주세요",
                QueryType.PROCEDURAL,
                DifficultyLevel.MEDIUM,
            ),
            (
                "교환 학생 갈 수 있는 조건이 뭐야?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
            (
                "대학원 진학하면 어떤 혜택 있어?",
                QueryType.COMPARISON,
                DifficultyLevel.MEDIUM,
            ),
        ],
        PersonaType.GRADUATE: [
            ("논문 심사 규정 알려주세요", QueryType.PROCEDURAL, DifficultyLevel.HARD),
            (
                "연구비 지원 신청하는 방법이 뭐야?",
                QueryType.PROCEDURAL,
                DifficultyLevel.HARD,
            ),
            (
                "박사 과정 수료 기준이 어떻게 돼?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
            (
                "조교 급여 지급 일정 알려줘",
                QueryType.FACT_CHECK,
                DifficultyLevel.MEDIUM,
            ),
        ],
        PersonaType.NEW_PROFESSOR: [
            (
                "교원 평가 기준이 어떻게 되나요?",
                QueryType.FACT_CHECK,
                DifficultyLevel.HARD,
            ),
            (
                "연구년 휴직 신청하는 방법 알려주세요",
                QueryType.PROCEDURAL,
                DifficultyLevel.HARD,
            ),
            (
                "지도 학생 수 제한이 있나요?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.MEDIUM,
            ),
            (
                "교권 보호 관련 규정 알려주세요",
                QueryType.FACT_CHECK,
                DifficultyLevel.HARD,
            ),
        ],
        PersonaType.PROFESSOR: [
            (
                "정년 보장 심사 기준이 뭐야?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
            (
                "학위 과정 수료 기준이 어떻게 돼?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
            (
                "조교 급여 지급 일정 알려줘",
                QueryType.FACT_CHECK,
                DifficultyLevel.MEDIUM,
            ),
        ],
        PersonaType.NEW_STAFF: [
            (
                "연차 사용하는 방법 알려주세요",
                QueryType.PROCEDURAL,
                DifficultyLevel.EASY,
            ),
            (
                "퇴직금 계산하는 방법이 뭐야?",
                QueryType.FACT_CHECK,
                DifficultyLevel.MEDIUM,
            ),
            (
                "야간 근무 수당 지급 기준 알려줘",
                QueryType.ELIGIBILITY,
                DifficultyLevel.MEDIUM,
            ),
            ("복지 혜택 종류 알려주세요", QueryType.FACT_CHECK, DifficultyLevel.EASY),
        ],
        PersonaType.STAFF_MANAGER: [
            (
                "부서 예산 집행 절차가 어떻게 되나요?",
                QueryType.PROCEDURAL,
                DifficultyLevel.HARD,
            ),
            ("인사 발령 기준 알려주세요", QueryType.FACT_CHECK, DifficultyLevel.HARD),
            ("검토 기한이 며칠이야?", QueryType.FACT_CHECK, DifficultyLevel.MEDIUM),
            (
                "재정 집행 승인 권한이 어디까지야?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
        ],
        PersonaType.PARENT: [
            (
                "자녀 휴학하면 등록금 환급받을 수 있나요?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.MEDIUM,
            ),
            (
                "성적 장학금 받는 방법 알려주세요",
                QueryType.PROCEDURAL,
                DifficultyLevel.MEDIUM,
            ),
            (
                "자녀가 졸업하려면 어떤 조건이 필요한가요?",
                QueryType.ELIGIBILITY,
                DifficultyLevel.HARD,
            ),
            (
                "학부모 상담 예약하는 방법이 뭐야?",
                QueryType.PROCEDURAL,
                DifficultyLevel.EASY,
            ),
        ],
        PersonaType.DISTRESSED_STUDENT: [
            (
                "학자금 대출 못 받았어요 어떡하죠?",
                QueryType.EMOTIONAL,
                DifficultyLevel.HARD,
            ),
            (
                "휴학하고 싶은데 마감 지났어요 도와주세요",
                QueryType.EMOTIONAL,
                DifficultyLevel.HARD,
            ),
            (
                "성적이 너무 안 좋아서 장학금 받을 수 있나요?",
                QueryType.COMPLEX,
                DifficultyLevel.HARD,
            ),
            ("간호학과 전공 바꾸고 싶어요", QueryType.COMPLEX, DifficultyLevel.MEDIUM),
        ],
        PersonaType.DISSATISFIED_MEMBER: [
            (
                "수강 신청 시스템 왜 이렇게 복잡해요?",
                QueryType.EMOTIONAL,
                DifficultyLevel.MEDIUM,
            ),
            (
                "교수님이 성적을 공정하게 매기지 않았어요",
                QueryType.EMOTIONAL,
                DifficultyLevel.HARD,
            ),
            (
                "학교가 학생들 혜택을 줄였잖아요 이거 게 아니죠",
                QueryType.EMOTIONAL,
                DifficultyLevel.HARD,
            ),
            (
                "규정이 자꾸 바뀌는 이유가 뭐야요?",
                QueryType.AMBIGUOUS,
                DifficultyLevel.MEDIUM,
            ),
        ],
    }

    @classmethod
    def generate_for_persona(
        cls, persona: Persona, count_per_difficulty: dict
    ) -> List[TestCase]:
        """
        Generate test queries for a specific persona.

        Args:
            persona: The Persona to generate queries for.
            count_per_difficulty: Dict with 'easy', 'medium', 'hard' counts.

        Returns:
            List of TestCase entities with queries and intent analysis.
        """
        templates = cls._QUERY_TEMPLATES.get(persona.persona_type, [])
        test_cases = []

        # Generate queries for each difficulty level
        for difficulty_str, count in count_per_difficulty.items():
            difficulty = DifficultyLevel(difficulty_str)
            difficulty_templates = [t for t in templates if t[2] == difficulty]

            if not difficulty_templates:
                # Fallback to any template if difficulty-specific ones not available
                difficulty_templates = templates

            for _ in range(count):
                if not difficulty_templates:
                    break

                # Select template randomly
                query, query_type, _ = random.choice(difficulty_templates)

                # Generate intent analysis
                intent = cls._generate_intent_analysis(query, persona)

                test_case = TestCase(
                    query=query,
                    persona_type=persona.persona_type,
                    difficulty=difficulty,
                    query_type=query_type,
                    intent_analysis=intent,
                )

                test_cases.append(test_case)

        return test_cases

    @classmethod
    def _generate_intent_analysis(cls, query: str, persona: Persona) -> IntentAnalysis:
        """
        Generate 3-level intent analysis for a query.

        Args:
            query: The user query.
            persona: The persona context.

        Returns:
            IntentAnalysis with surface, hidden, and behavioral intent.
        """
        # Simple rule-based intent generation
        # In production, this would use LLM API

        surface_intent = cls._extract_surface_intent(query)
        hidden_intent = cls._infer_hidden_intent(query, persona)
        behavioral_intent = cls._infer_behavioral_intent(query, persona)

        return IntentAnalysis(
            surface_intent=surface_intent,
            hidden_intent=hidden_intent,
            behavioral_intent=behavioral_intent,
        )

    @classmethod
    def _extract_surface_intent(cls, query: str) -> str:
        """Extract surface intent from query."""
        # Simple keyword matching
        if "신청" in query or "방법" in query:
            return "절차/신청 문의"
        elif "자격" in query or "조건" in query or "기준" in query:
            return "자격/요건 확인"
        elif "알려줘" in query or "어떻게" in query:
            return "정보 요청"
        elif "왜" in query or "이유" in query:
            return "원인/근거 문의"
        elif "안 돼요" in query or "없어요" in query or "어떡하죠" in query:
            return "불만/도움 요청"
        else:
            return "일반 문의"

    @classmethod
    def _infer_hidden_intent(cls, query: str, persona: Persona) -> str:
        """Infer hidden intent from query and persona context."""
        # Context-aware hidden intent inference
        if persona.persona_type == PersonaType.FRESHMAN:
            return f"{query} - 학교 시스템 적응 필요"
        elif persona.persona_type == PersonaType.JUNIOR:
            return f"{query} - 졸업/진로 준비"
        elif persona.persona_type == PersonaType.GRADUATE:
            return f"{query} - 연구/학위 진행"
        elif persona.persona_type in [PersonaType.NEW_PROFESSOR, PersonaType.PROFESSOR]:
            return f"{query} - 교수 업무 수행"
        elif persona.persona_type in [PersonaType.NEW_STAFF, PersonaType.STAFF_MANAGER]:
            return f"{query} - 직무 수행"
        elif persona.persona_type == PersonaType.PARENT:
            return f"{query} - 자녀 교육 지원"
        elif persona.persona_type == PersonaType.DISTRESSED_STUDENT:
            return f"{query} - 긴급 상황 해결 필요"
        elif persona.persona_type == PersonaType.DISSATISFIED_MEMBER:
            return f"{query} - 권리 주장 및 불만 해소"
        else:
            return f"{query} - 정보 필요"

    @classmethod
    def _infer_behavioral_intent(cls, query: str, persona: Persona) -> str:
        """Infer ultimate behavioral intent."""
        # Extract action the user wants to take
        if "신청" in query:
            return "신청서 제출"
        elif "방법" in query:
            return "절차 수행"
        elif "자격" in query or "조건" in query:
            return "자격 확인 후 행동"
        elif "알려줘" in query:
            return "정보 습득"
        elif "불만" in query or "별로" in query or "없어요" in query:
            return "문제 해결 또는 대안 요청"
        else:
            return "정보 확인"
