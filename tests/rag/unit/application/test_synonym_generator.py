"""
Unit tests for SynonymGeneratorService.

Tests synonym generation (with mocked LLM), CRUD operations,
and file I/O for synonyms.json.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.rag.application.synonym_generator_service import SynonymGeneratorService


class TestSynonymGeneratorService:
    """Tests for SynonymGeneratorService"""

    @pytest.fixture
    def temp_synonyms_file(self):
        """Create a temporary synonyms file for testing."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            initial_data = {
                "version": "1.0.0",
                "description": "Test synonyms",
                "last_updated": "2025-01-01",
                "terms": {
                    "휴학": ["휴학원", "휴학 신청", "학업 중단"],
                    "복학": ["복학원", "학업 복귀"],
                },
            }
            json.dump(initial_data, f, ensure_ascii=False)
            temp_path = Path(f.name)
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock = Mock()
        mock.generate.return_value = "입학정원, 학과정원, 모집인원, 정원수, 총정원"
        return mock

    @pytest.fixture
    def service(self, temp_synonyms_file):
        """Create a service with temp file."""
        return SynonymGeneratorService(synonyms_path=temp_synonyms_file)

    @pytest.fixture
    def service_with_llm(self, temp_synonyms_file, mock_llm_client):
        """Create a service with mock LLM."""
        return SynonymGeneratorService(
            llm_client=mock_llm_client,
            synonyms_path=temp_synonyms_file,
        )

    # ==================== Load/Save Tests ====================

    def test_load_synonyms_existing_file(self, service):
        """Test loading existing synonyms file."""
        data = service.load_synonyms()
        assert data["version"] == "1.0.0"
        assert "휴학" in data["terms"]
        assert "복학" in data["terms"]

    def test_load_synonyms_missing_file(self, tmp_path):
        """Test loading when file doesn't exist."""
        service = SynonymGeneratorService(
            synonyms_path=tmp_path / "nonexistent.json"
        )
        data = service.load_synonyms()
        assert "version" in data
        assert "terms" in data
        assert data["terms"] == {}

    def test_save_synonyms_updates_date(self, service, temp_synonyms_file):
        """Test that saving updates last_updated field."""
        data = service.load_synonyms()
        data["terms"]["테스트"] = ["test1", "test2"]
        service.save_synonyms(data)

        # Reload and check
        reloaded = service.load_synonyms()
        assert "테스트" in reloaded["terms"]
        # Date should be updated to today
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        assert reloaded["last_updated"] == today

    # ==================== Get/List Tests ====================

    def test_get_synonyms_existing_term(self, service):
        """Test getting synonyms for existing term."""
        synonyms = service.get_synonyms("휴학")
        assert "휴학원" in synonyms
        assert "휴학 신청" in synonyms
        assert len(synonyms) == 3

    def test_get_synonyms_nonexistent_term(self, service):
        """Test getting synonyms for non-existent term."""
        synonyms = service.get_synonyms("없는용어")
        assert synonyms == []

    def test_list_terms(self, service):
        """Test listing all terms."""
        terms = service.list_terms()
        assert "휴학" in terms
        assert "복학" in terms
        assert len(terms) == 2

    # ==================== Add Tests ====================

    def test_add_synonym_new_term(self, service):
        """Test adding synonym to new term."""
        result = service.add_synonym("정원", "입학정원")
        assert result is True

        synonyms = service.get_synonyms("정원")
        assert "입학정원" in synonyms

    def test_add_synonym_existing_term(self, service):
        """Test adding synonym to existing term."""
        result = service.add_synonym("휴학", "새로운 동의어")
        assert result is True

        synonyms = service.get_synonyms("휴학")
        assert "새로운 동의어" in synonyms
        assert len(synonyms) == 4  # 3 original + 1 new

    def test_add_synonym_duplicate(self, service):
        """Test adding duplicate synonym returns False."""
        result = service.add_synonym("휴학", "휴학원")  # Already exists
        assert result is False

    def test_add_synonyms_multiple(self, service):
        """Test adding multiple synonyms at once."""
        count = service.add_synonyms("정원", ["입학정원", "학과정원", "모집인원"])
        assert count == 3

        synonyms = service.get_synonyms("정원")
        assert len(synonyms) == 3

    def test_add_synonyms_with_duplicates(self, service):
        """Test adding synonyms with some duplicates."""
        service.add_synonym("정원", "입학정원")

        # Try adding with one duplicate
        count = service.add_synonyms("정원", ["입학정원", "학과정원", "모집인원"])
        assert count == 2  # Only 2 new ones

    # ==================== Remove Tests ====================

    def test_remove_synonym_success(self, service):
        """Test removing existing synonym."""
        result = service.remove_synonym("휴학", "휴학원")
        assert result is True

        synonyms = service.get_synonyms("휴학")
        assert "휴학원" not in synonyms
        assert len(synonyms) == 2

    def test_remove_synonym_not_found(self, service):
        """Test removing non-existent synonym."""
        result = service.remove_synonym("휴학", "없는동의어")
        assert result is False

    def test_remove_synonym_removes_empty_term(self, service):
        """Test that removing last synonym removes the term."""
        # Add a term with single synonym
        service.add_synonym("테스트", "유일한동의어")
        assert "테스트" in service.list_terms()

        # Remove the only synonym
        service.remove_synonym("테스트", "유일한동의어")
        assert "테스트" not in service.list_terms()

    def test_remove_term(self, service):
        """Test removing entire term."""
        result = service.remove_term("휴학")
        assert result is True
        assert "휴학" not in service.list_terms()

    def test_remove_term_not_found(self, service):
        """Test removing non-existent term."""
        result = service.remove_term("없는용어")
        assert result is False

    # ==================== Generate Tests ====================

    def test_generate_synonyms_returns_list(self, service_with_llm):
        """Test that generate_synonyms returns parsed list."""
        candidates = service_with_llm.generate_synonyms("정원")

        assert isinstance(candidates, list)
        assert len(candidates) == 5
        assert "입학정원" in candidates
        assert "학과정원" in candidates
        assert "모집인원" in candidates

    def test_generate_synonyms_filters_duplicates(
        self, service_with_llm, mock_llm_client
    ):
        """Test that generate_synonyms removes duplicates from response."""
        mock_llm_client.generate.return_value = "입학정원, 입학정원, 학과정원"
        candidates = service_with_llm.generate_synonyms("정원")

        # Should have only unique values
        assert candidates == ["입학정원", "학과정원"]

    def test_generate_synonyms_excludes_existing(self, service_with_llm):
        """Test that existing synonyms are excluded."""
        # First add some synonyms
        service_with_llm.add_synonym("정원", "입학정원")

        # Generate should exclude existing
        candidates = service_with_llm.generate_synonyms(
            "정원", exclude_existing=True
        )
        assert "입학정원" not in candidates

    def test_generate_synonyms_includes_existing_when_disabled(
        self, service_with_llm
    ):
        """Test that existing synonyms are included when exclude_existing=False."""
        service_with_llm.add_synonym("정원", "입학정원")

        candidates = service_with_llm.generate_synonyms(
            "정원", exclude_existing=False
        )
        assert "입학정원" in candidates

    def test_generate_synonyms_filters_term_itself(
        self, service_with_llm, mock_llm_client
    ):
        """Test that the term itself is filtered out from candidates."""
        mock_llm_client.generate.return_value = "정원, 입학정원, 학과정원"
        candidates = service_with_llm.generate_synonyms("정원")

        assert "정원" not in candidates

    def test_generate_synonyms_no_llm_raises_error(self, service):
        """Test that generate_synonyms raises error without LLM."""
        with pytest.raises(RuntimeError, match="LLM 클라이언트가 설정되지 않았습니다"):
            service.generate_synonyms("정원")

    def test_generate_synonyms_max_10(self, service_with_llm, mock_llm_client):
        """Test that generate_synonyms returns max 10 candidates."""
        # Return more than 10
        mock_llm_client.generate.return_value = ", ".join(
            [f"동의어{i}" for i in range(15)]
        )
        candidates = service_with_llm.generate_synonyms("테스트")

        assert len(candidates) <= 10
