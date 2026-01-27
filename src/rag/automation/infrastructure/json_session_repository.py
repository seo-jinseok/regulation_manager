"""
JSON-based Session Repository Implementation.

Infrastructure layer implementation for persisting test sessions
to file system using JSON format.

Clean Architecture: Infrastructure implements domain interfaces.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.rag.automation.domain.entities import EvaluationCase, TestSession
from src.rag.automation.domain.repository import SessionRepository


class JSONSessionRepository(SessionRepository):
    """
    File-based repository for test sessions using JSON storage.

    Saves and loads test sessions from data/output/test_sessions/ directory.
    """

    def __init__(self, base_path: str = "data/output/test_sessions"):
        """
        Initialize the repository.

        Args:
            base_path: Base directory for storing test session files.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Get file path for a session."""
        return self.base_path / f"{session_id}.json"

    def _serialize_test_case(self, test_case: EvaluationCase) -> dict:
        """Serialize EvaluationCase to dict."""
        return {
            "query": test_case.query,
            "persona_type": test_case.persona_type.value,
            "difficulty": test_case.difficulty.value,
            "query_type": test_case.query_type.value,
            "intent_analysis": (
                {
                    "surface_intent": test_case.intent_analysis.surface_intent,
                    "hidden_intent": test_case.intent_analysis.hidden_intent,
                    "behavioral_intent": test_case.intent_analysis.behavioral_intent,
                }
                if test_case.intent_analysis
                else None
            ),
            "expected_topics": test_case.expected_topics,
            "expected_regulations": test_case.expected_regulations,
            "metadata": test_case.metadata,
        }

    def _deserialize_test_case(self, data: dict) -> EvaluationCase:
        """Deserialize dict to EvaluationCase."""
        from src.rag.automation.domain.entities import (
            DifficultyLevel,
            PersonaType,
            QueryType,
        )
        from src.rag.automation.domain.value_objects import IntentAnalysis

        intent = None
        if data.get("intent_analysis"):
            intent_data = data["intent_analysis"]
            intent = IntentAnalysis(
                surface_intent=intent_data["surface_intent"],
                hidden_intent=intent_data["hidden_intent"],
                behavioral_intent=intent_data["behavioral_intent"],
            )

        return EvaluationCase(
            query=data["query"],
            persona_type=PersonaType(data["persona_type"]),
            difficulty=DifficultyLevel(data["difficulty"]),
            query_type=QueryType(data["query_type"]),
            intent_analysis=intent,
            expected_topics=data.get("expected_topics", []),
            expected_regulations=data.get("expected_regulations", []),
            metadata=data.get("metadata", {}),
        )

    def save(self, session: TestSession) -> None:
        """
        Save a test session to JSON file.

        Args:
            session: The TestSession entity to save.
        """
        session_path = self._get_session_path(session.session_id)

        data = {
            "session_id": session.session_id,
            "started_at": session.started_at.isoformat(),
            "total_test_cases": session.total_test_cases,
            "test_cases": [self._serialize_test_case(tc) for tc in session.test_cases],
            "completed_at": session.completed_at.isoformat()
            if session.completed_at
            else None,
            "metadata": session.metadata,
        }

        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, session_id: str) -> Optional[TestSession]:
        """
        Load a test session from JSON file.

        Args:
            session_id: The unique session identifier.

        Returns:
            TestSession if found, None otherwise.
        """
        session_path = self._get_session_path(session_id)

        if not session_path.exists():
            return None

        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = [self._deserialize_test_case(tc) for tc in data["test_cases"]]

        return TestSession(
            session_id=data["session_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            total_test_cases=data["total_test_cases"],
            test_cases=test_cases,
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            metadata=data.get("metadata", {}),
        )

    def list_all(self) -> List[TestSession]:
        """
        List all test sessions.

        Returns:
            List of all TestSession entities.
        """
        sessions = []

        for session_file in self.base_path.glob("*.json"):
            session_id = session_file.stem
            session = self.load(session_id)
            if session:
                sessions.append(session)

        # Sort by started_at descending (newest first)
        sessions.sort(key=lambda s: s.started_at, reverse=True)

        return sessions

    def delete(self, session_id: str) -> bool:
        """
        Delete a test session.

        Args:
            session_id: The session to delete.

        Returns:
            True if deleted, False if not found.
        """
        session_path = self._get_session_path(session_id)

        if not session_path.exists():
            return False

        session_path.unlink()
        return True
