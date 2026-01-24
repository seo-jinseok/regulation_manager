"""
Repository Interfaces for RAG Testing Automation System.

These are abstract base classes defining the contracts that infrastructure
implementations must fulfill. The domain layer depends only on these
interfaces, not on concrete implementations.

Clean Architecture: Domain layer defines interfaces, infrastructure implements them.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .entities import TestSession


class SessionRepository(ABC):
    """
    Abstract interface for test session persistence.

    Implementations may use file system (JSON), database, etc.
    """

    @abstractmethod
    def save(self, session: TestSession) -> None:
        """
        Save a test session to persistent storage.

        Args:
            session: The TestSession entity to save.
        """
        pass

    @abstractmethod
    def load(self, session_id: str) -> Optional[TestSession]:
        """
        Load a test session from persistent storage.

        Args:
            session_id: The unique session identifier.

        Returns:
            TestSession if found, None otherwise.
        """
        pass

    @abstractmethod
    def list_all(self) -> List[TestSession]:
        """
        List all test sessions.

        Returns:
            List of all TestSession entities.
        """
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """
        Delete a test session.

        Args:
            session_id: The session to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass
