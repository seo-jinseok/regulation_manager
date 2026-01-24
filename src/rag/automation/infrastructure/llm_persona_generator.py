"""
Persona Generator using LLM.

Infrastructure layer implementation that generates diverse user personas
for RAG testing using Claude AI API.

Clean Architecture: Infrastructure implements domain interfaces and uses
external libraries (anthropic).
"""

from typing import List

from src.rag.automation.domain.entities import (
    Persona,
    PersonaType,
)
from src.rag.automation.domain.value_objects import DifficultyDistribution


class PersonaGenerator:
    """
    Generates 10 types of user personas for RAG testing.

    Each persona has specific characteristics, query styles, and context hints
    to simulate real user behavior.
    """

    # 10 persona definitions with their characteristics
    _PERSONA_DEFINITIONS = {
        PersonaType.FRESHMAN: Persona(
            persona_type=PersonaType.FRESHMAN,
            name="신입생",
            description="학교 시스템에 익숙하지 않음, 비공식적 표현",
            characteristics=[
                "학교 시스템에 익숙하지 않음",
                "공식 용어를 잘 모름",
                "단순하고 직접적인 질문",
            ],
            query_styles=[
                "간단한 질문",
                "구어체 표현",
                "모호한 용어 사용",
            ],
            context_hints=[
                "입학 초기",
                "campus life adaptation",
                "freshman orientation",
            ],
        ),
        PersonaType.JUNIOR: Persona(
            persona_type=PersonaType.JUNIOR,
            name="재학생 (3학년)",
            description="구체적 정보 필요, 졸업 준비",
            characteristics=[
                "구체적인 정보 필요",
                "졸업 요건 관심",
                "전공 심화 과정 관심",
            ],
            query_styles=[
                "구체적인 질문",
                "여러 조건 확인",
                "비교 질문",
            ],
            context_hints=[
                "졸업 준비",
                "major advancement",
                "career planning",
            ],
        ),
        PersonaType.GRADUATE: Persona(
            persona_type=PersonaType.GRADUATE,
            name="대학원생",
            description="연구/논문 중심, 전문적 질문",
            characteristics=[
                "연구/논문 중심",
                "전문적 용어 사용",
                "세부 규정 확인",
            ],
            query_styles=[
                "전문적인 질문",
                "연구 관련 질문",
                "학위 규정 문의",
            ],
            context_hints=[
                "연구 활동",
                "thesis preparation",
                "academic research",
            ],
        ),
        PersonaType.NEW_PROFESSOR: Persona(
            persona_type=PersonaType.NEW_PROFESSOR,
            name="신임 교수",
            description="제도 파악 필요, 공식적 표현",
            characteristics=[
                "제도 파악 필요",
                "공식적 표현",
                "권리와 의무 확인",
            ],
            query_styles=[
                "공식적인 질문",
                "조항 확인",
                "절차 문의",
            ],
            context_hints=[
                "신규 임용",
                "faculty orientation",
                "academic governance",
            ],
        ),
        PersonaType.PROFESSOR: Persona(
            persona_type=PersonaType.PROFESSOR,
            name="정교수",
            description="세부 규정 확인, 권리 주장",
            characteristics=[
                "세부 규정 확인",
                "권리 주장",
                "교수 업무 관련",
            ],
            query_styles=[
                "구체적 조항 확인",
                "권리 관련 질문",
                "업무 절차 문의",
            ],
            context_hints=[
                "교수 업무",
                "faculty rights",
                "academic administration",
            ],
        ),
        PersonaType.NEW_STAFF: Persona(
            persona_type=PersonaType.NEW_STAFF,
            name="신입 직원",
            description="복무규정 파악, 혜택 문의",
            characteristics=[
                "복무규정 파악",
                "혜택 문의",
                "근무 조건 확인",
            ],
            query_styles=[
                "복지 혜택 문의",
                "근무 시간 질문",
                "휴가 관련 질문",
            ],
            context_hints=[
                "신규 입사",
                "staff benefits",
                "employment conditions",
            ],
        ),
        PersonaType.STAFF_MANAGER: Persona(
            persona_type=PersonaType.STAFF_MANAGER,
            name="과장급 직원",
            description="부서 운영, 예산 관련",
            characteristics=[
                "부서 운영",
                "예산 관련",
                "인사 관리",
            ],
            query_styles=[
                "부서 운영 질문",
                "예산 집행 문의",
                "인사 관련 절차",
            ],
            context_hints=[
                "부서장 역할",
                "department management",
                "budget administration",
            ],
        ),
        PersonaType.PARENT: Persona(
            persona_type=PersonaType.PARENT,
            name="학부모",
            description="자녀 관련 정보, 외부 시선",
            characteristics=[
                "자녀 관련 정보",
                "외부 시선",
                "비용 관련",
            ],
            query_styles=[
                "자녀 관련 질문",
                "등록금 문의",
                "학사 경로 질문",
            ],
            context_hints=[
                "자녀 교육",
                "tuition payment",
                "academic progress",
            ],
        ),
        PersonaType.DISTRESSED_STUDENT: Persona(
            persona_type=PersonaType.DISTRESSED_STUDENT,
            name="어려운 상황의 학생",
            description="감정적, 급한 상황",
            characteristics=[
                "감정적 상태",
                "급한 상황",
                "복지/지원 필요",
            ],
            query_styles=[
                "감정적 표현",
                "긴급 질문",
                "도움 요청",
            ],
            context_hints=[
                "어려운 상황",
                "urgent situation",
                "seeking help",
            ],
        ),
        PersonaType.DISSATISFIED_MEMBER: Persona(
            persona_type=PersonaType.DISSATISFIED_MEMBER,
            name="불만있는 구성원",
            description="권리 주장, 신고 의향",
            characteristics=[
                "권리 주장",
                "불만 표출",
                "신고 의향",
            ],
            query_styles=[
                "항의성 질문",
                "권리 주장",
                "제보 의향",
            ],
            context_hints=[
                "불만 상황",
                "rights assertion",
                "filing complaint",
            ],
        ),
    }

    # Default difficulty distribution: 30% easy, 40% medium, 30% hard
    _DEFAULT_DIFFICULTY_DISTRIBUTION = DifficultyDistribution(
        easy_ratio=0.3,
        medium_ratio=0.4,
        hard_ratio=0.3,
    )

    @classmethod
    def get_all_personas(cls) -> List[Persona]:
        """
        Get all 10 defined personas.

        Returns:
            List of all Persona entities.
        """
        return list(cls._PERSONA_DEFINITIONS.values())

    @classmethod
    def get_persona(cls, persona_type: PersonaType) -> Persona:
        """
        Get a specific persona by type.

        Args:
            persona_type: The type of persona to retrieve.

        Returns:
            Persona entity for the specified type.

        Raises:
            KeyError: If persona_type is not found.
        """
        return cls._PERSONA_DEFINITIONS[persona_type]

    @classmethod
    def get_difficulty_distribution(cls) -> DifficultyDistribution:
        """
        Get the default difficulty distribution for test cases.

        Returns:
            DifficultyDistribution with 30% easy, 40% medium, 30% hard.
        """
        return cls._DEFAULT_DIFFICULTY_DISTRIBUTION

    @classmethod
    def calculate_test_case_counts(
        cls, total_tests: int, distribution: DifficultyDistribution
    ) -> dict:
        """
        Calculate number of test cases per difficulty level.

        Args:
            total_tests: Total number of test cases to generate.
            distribution: Difficulty distribution ratios.

        Returns:
            Dict with 'easy', 'medium', 'hard' counts.
        """
        easy_count = int(total_tests * distribution.easy_ratio)
        medium_count = int(total_tests * distribution.medium_ratio)
        hard_count = total_tests - easy_count - medium_count  # Remainder goes to hard

        return {
            "easy": easy_count,
            "medium": medium_count,
            "hard": hard_count,
        }
