"""
Generate Test Use Case.

Application layer orchestrator for generating automated test cases.
Coordinates persona generation and query generation to create test sessions.

Clean Architecture: Application layer orchestrates domain and infrastructure.
"""

from datetime import datetime
from typing import List, Optional

from src.rag.automation.domain.entities import EvaluationCase, TestSession
from src.rag.automation.domain.repository import SessionRepository
from src.rag.automation.domain.value_objects import DifficultyDistribution
from src.rag.automation.infrastructure.llm_persona_generator import PersonaGenerator
from src.rag.automation.infrastructure.llm_query_generator import QueryGenerator
from src.rag.domain.repositories import ILLMClient


class GenerateTestUseCase:
    """
    Use Case for generating automated RAG test sessions.

    Orchestrates the creation of test cases with diverse personas,
    queries, and difficulty levels.
    """

    def __init__(
        self,
        session_repository: SessionRepository,
        llm_client: Optional[ILLMClient] = None,
    ):
        """
        Initialize the use case.

        Args:
            session_repository: Repository for persisting test sessions.
            llm_client: Optional LLM client for diverse query generation.
                      If provided, enables LLM-based query generation.
        """
        self.session_repository = session_repository
        self.persona_generator = PersonaGenerator()
        # Initialize QueryGenerator with LLM client if provided
        self.query_generator = QueryGenerator(llm_client)

    def execute(
        self,
        session_id: str,
        tests_per_persona: int = 3,
        difficulty_distribution: Optional[DifficultyDistribution] = None,
        vary_queries: bool = False,  # Default to False for backward compatibility
        seed: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> TestSession:
        """
        Generate a complete test session.

        Args:
            session_id: Unique identifier for the test session.
            tests_per_persona: Number of test cases to generate per persona.
            difficulty_distribution: Custom difficulty distribution (optional).
            vary_queries: If True, use LLM for diverse queries; if False, use templates.
            seed: Random seed for reproducibility.
            metadata: Additional metadata for the session.

        Returns:
            TestSession entity with generated test cases.
        """
        # Use default distribution if not provided
        if difficulty_distribution is None:
            difficulty_distribution = (
                self.persona_generator.get_difficulty_distribution()
            )

        # Get all personas
        all_personas = self.persona_generator.get_all_personas()

        # Calculate test case counts per difficulty
        counts = self.persona_generator.calculate_test_case_counts(
            tests_per_persona, difficulty_distribution
        )

        # Generate test cases for all personas
        all_test_cases: List[EvaluationCase] = []

        for persona in all_personas:
            # Generate queries for this persona
            test_cases = self.query_generator.generate_for_persona(
                persona=persona,
                count_per_difficulty=counts,
                vary_queries=vary_queries,
                seed=seed,
            )

            all_test_cases.extend(test_cases)

        # Create test session
        session = TestSession(
            session_id=session_id,
            started_at=datetime.now(),
            total_test_cases=len(all_test_cases),
            test_cases=all_test_cases,
            metadata=metadata or {},
        )

        # Persist the session
        self.session_repository.save(session)

        return session

    def execute_for_persona(
        self,
        session_id: str,
        persona_type: str,  # PersonaType enum value
        tests_per_difficulty: dict,
        vary_queries: bool = False,
        seed: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> TestSession:
        """
        Generate test session for a specific persona only.

        Args:
            session_id: Unique identifier for the test session.
            persona_type: Type of persona to generate tests for.
            tests_per_difficulty: Dict with 'easy', 'medium', 'hard' counts.
            vary_queries: If True, use LLM for diverse queries; if False, use templates.
            seed: Random seed for reproducibility.
            metadata: Additional metadata for the session.

        Returns:
            TestSession entity with generated test cases.
        """
        # Get specific persona
        from src.rag.automation.domain.entities import PersonaType

        persona_enum = PersonaType(persona_type)
        persona = self.persona_generator.get_persona(persona_enum)

        # Generate queries for this persona
        test_cases = self.query_generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=tests_per_difficulty,
            vary_queries=vary_queries,
            seed=seed,
        )

        # Create test session
        session = TestSession(
            session_id=session_id,
            started_at=datetime.now(),
            total_test_cases=len(test_cases),
            test_cases=test_cases,
            metadata=metadata or {},
        )

        # Persist the session
        self.session_repository.save(session)

        return session

    def get_session(self, session_id: str) -> Optional[TestSession]:
        """
        Retrieve a previously generated test session.

        Args:
            session_id: Unique identifier for the test session.

        Returns:
            TestSession if found, None otherwise.
        """
        return self.session_repository.load(session_id)

    def list_all_sessions(self) -> List[TestSession]:
        """
        List all test sessions.

        Returns:
            List of all TestSession entities.
        """
        return self.session_repository.list_all()
