"""
Test suite for Dictionary Automation (Phase 7).

Tests for:
- DictionaryManager LLM-based recommendations
- Conflict detection and resolution
- Batch update from failures
- ApplyImprovementUseCase automation features
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.rag.automation.application.apply_improvement_usecase import ApplyImprovementUseCase
from src.rag.infrastructure.dictionary_manager import (
    ConflictInfo,
    DictionaryManager,
    IntentEntry,
    RecommendationResult,
    SynonymEntry,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_intents_file():
    """Create a temporary intents.json file for testing."""
    data = {
        "version": "1.0.0",
        "description": "Test intents",
        "last_updated": "2026-01-26",
        "intents": [
            {
                "id": "test_intent_1",
                "label": "테스트 인텐트 1",
                "triggers": ["테스트 트리거"],
                "patterns": ["테스트.*패턴"],
                "keywords": ["테스트", "키워드"],
            }
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_synonyms_file():
    """Create a temporary synonyms.json file for testing."""
    data = {
        "version": "1.0.0",
        "description": "Test synonyms",
        "last_updated": "2026-01-26",
        "terms": {
            "테스트": ["시험", "검사"],
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = mock.MagicMock()
    
    # Mock response for intent recommendation
    client.generate.return_value = '''```json
{
  "intents": [
    {
      "id": "new_test_intent",
      "label": "새로운 테스트 인텐트",
      "triggers": ["새 테스트", "새로운 검사"],
      "patterns": ["새.*테스트"],
      "keywords": ["새로운", "테스트", "검사"],
      "audience": "all"
    }
  ],
  "synonyms": [
    {
      "term": "새로운",
      "synonyms": ["최신의", "갓만든"],
      "context": "regulation_query"
    }
  ]
}
```'''
    
    return client


# ============================================================================
# DictionaryManager Tests
# ============================================================================


class TestDictionaryManager:
    """Test suite for DictionaryManager."""

    def test_init_with_paths(self, temp_intents_file, temp_synonyms_file):
        """Test DictionaryManager initialization with custom paths."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        assert manager.intents_path == temp_intents_file
        assert manager.synonyms_path == temp_synonyms_file
        assert manager.get_all_intents() is not None
        assert manager.get_all_synonyms() is not None

    def test_get_all_intents(self, temp_intents_file, temp_synonyms_file):
        """Test getting all intents."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        intents = manager.get_all_intents()
        assert len(intents) == 1
        assert intents[0]["id"] == "test_intent_1"

    def test_get_all_synonyms(self, temp_intents_file, temp_synonyms_file):
        """Test getting all synonyms."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        synonyms = manager.get_all_synonyms()
        assert "테스트" in synonyms
        assert synonyms["테스트"] == ["시험", "검사"]

    def test_intent_id_exists(self, temp_intents_file, temp_synonyms_file):
        """Test checking if intent ID exists."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        assert manager.intent_id_exists("test_intent_1") is True
        assert manager.intent_id_exists("nonexistent") is False

    def test_synonym_term_exists(self, temp_intents_file, temp_synonyms_file):
        """Test checking if synonym term exists."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        assert manager.synonym_term_exists("테스트") is True
        assert manager.synonym_term_exists("없는용어") is False

    def test_find_similar_intents(self, temp_intents_file, temp_synonyms_file):
        """Test finding similar intents based on triggers/keywords."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        # Find intents with similar triggers
        similar = manager.find_similar_intents(
            triggers=["테스트 트리거"],
            keywords=["테스트"],
            threshold=0.3,
        )

        assert len(similar) > 0
        assert similar[0]["id"] == "test_intent_1"
        assert similar[0]["_similarity"] > 0

    def test_calculate_overlap(self, temp_intents_file, temp_synonyms_file):
        """Test overlap calculation between lists."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        # Test full overlap
        overlap1 = manager._calculate_overlap(["a", "b"], ["a", "b"])
        assert overlap1 == 1.0

        # Test partial overlap
        overlap2 = manager._calculate_overlap(["a", "b", "c"], ["b", "c", "d"])
        assert overlap2 == 2 / 3

        # Test no overlap
        overlap3 = manager._calculate_overlap(["a", "b"], ["c", "d"])
        assert overlap3 == 0.0

    def test_detect_conflicts_duplicate_id(self, temp_intents_file, temp_synonyms_file):
        """Test conflict detection for duplicate intent ID."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        intent = IntentEntry(
            id="test_intent_1",  # Duplicate ID
            label="중복 인텐트",
            triggers=["새 트리거"],
            keywords=["새", "키워드"],
        )

        conflicts = manager.detect_conflicts(intent, [])
        assert len(conflicts) > 0
        assert conflicts[0].conflict_type == "duplicate_id"
        assert conflicts[0].severity == "error"

    def test_detect_conflicts_duplicate_synonym(self, temp_intents_file, temp_synonyms_file):
        """Test conflict detection for duplicate synonym term."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        synonym = SynonymEntry(
            term="테스트",  # Duplicate term
            synonyms=["추가"],
        )

        conflicts = manager.detect_conflicts(
            IntentEntry("", "", []), [synonym]
        )
        assert len(conflicts) > 0
        assert conflicts[0].conflict_type == "duplicate_synonym_term"

    def test_recommend_from_failure_without_llm(
        self, temp_intents_file, temp_synonyms_file
    ):
        """Test recommendation generation without LLM (fallback)."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
            llm_client=None,
        )

        result = manager.recommend_from_failure(
            query="새로운 질문",
            root_cause="인텐트 부족",
            suggested_fix="인텐트 추가",
            use_llm=False,
        )

        assert result.llm_used is False
        assert len(result.recommended_intents) == 0
        assert len(result.recommended_synonyms) == 0

    def test_recommend_from_failure_with_llm(
        self, temp_intents_file, temp_synonyms_file, mock_llm_client
    ):
        """Test recommendation generation with LLM."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
            llm_client=mock_llm_client,
        )

        result = manager.recommend_from_failure(
            query="새로운 질문",
            root_cause="인텐트 부족",
            suggested_fix="인텐트 추가",
            use_llm=True,
        )

        assert result.llm_used is True
        assert len(result.recommended_intents) > 0
        assert result.recommended_intents[0].id == "new_test_intent"
        assert len(result.recommended_synonyms) > 0

    def test_add_intent_new(self, temp_intents_file, temp_synonyms_file):
        """Test adding a new intent."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        intent = IntentEntry(
            id="new_intent",
            label="새 인텐트",
            triggers=["새 트리거"],
            keywords=["새", "키워드"],
        )

        result = manager.add_intent(intent, merge=False)
        assert result is True

        # Verify it was added
        assert manager.intent_id_exists("new_intent") is True

    def test_add_intent_merge(self, temp_intents_file, temp_synonyms_file):
        """Test merging intent with existing entry."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        intent = IntentEntry(
            id="test_intent_1",  # Existing ID
            label="업데이트된 인텐트",
            triggers=["추가 트리거"],
            keywords=["추가"],
        )

        result = manager.add_intent(intent, merge=True)
        assert result is True

        # Verify merge happened
        intents = manager.get_all_intents()
        existing = next(i for i in intents if i["id"] == "test_intent_1")
        assert "추가 트리거" in existing["triggers"]

    def test_add_synonym_new(self, temp_intents_file, temp_synonyms_file):
        """Test adding a new synonym."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        synonym = SynonymEntry(
            term="새용어",
            synonyms=["새로운 말"],
        )

        result = manager.add_synonym(synonym, merge=False)
        assert result is True

        # Verify it was added
        assert manager.synonym_term_exists("새용어") is True

    def test_add_synonym_merge(self, temp_intents_file, temp_synonyms_file):
        """Test merging synonym with existing entry."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        synonym = SynonymEntry(
            term="테스트",  # Existing term
            synonyms=["점검"],  # Additional synonym
        )

        result = manager.add_synonym(synonym, merge=True)
        assert result is True

        # Verify merge happened
        synonyms = manager.get_all_synonyms()
        assert "점검" in synonyms["테스트"]

    def test_get_stats(self, temp_intents_file, temp_synonyms_file):
        """Test getting dictionary statistics."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        stats = manager.get_stats()
        assert "intents" in stats
        assert "synonyms" in stats
        assert stats["intents"]["count"] == 1
        assert stats["synonyms"]["count"] == 1


# ============================================================================
# ApplyImprovementUseCase Automation Tests
# ============================================================================


class TestApplyImprovementAutomation:
    """Test suite for ApplyImprovementUseCase automation features."""

    def test_init_with_llm(self, temp_intents_file, temp_synonyms_file, mock_llm_client):
        """Test ApplyImprovementUseCase initialization with LLM client."""
        use_case = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
            llm_client=mock_llm_client,
        )

        assert use_case._dict_manager is not None
        assert use_case._llm_client is not None

    def test_apply_improvements_with_llm(
        self, temp_intents_file, temp_synonyms_file, mock_llm_client
    ):
        """Test apply_improvements with LLM recommendations enabled."""
        from src.rag.automation.domain.entities import TestResult
        from src.rag.automation.domain.value_objects import FiveWhyAnalysis

        use_case = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
            llm_client=mock_llm_client,
        )

        test_result = TestResult(
            test_case_id="test_001",
            query="새로운 질문",
            persona_type="student",
            difficulty="easy",
            query_type="fact_check",
        )

        analysis = FiveWhyAnalysis(
            test_case_id="test_001",
            original_failure="검색 실패",
            why_chain=["왜?", "인텐트 없음", "인텐트 없음", "인텐트 없음", "인텐트 없음"],
            root_cause="인텐트 부족",
            suggested_fix="인텐트 추가",
            component_to_patch="intents.json",
            code_change_required=False,
        )

        results = use_case.apply_improvements(
            analyses=[analysis],
            test_results=[test_result],
            dry_run=True,
        )

        assert "llm_recommendations" in results
        assert results["statistics"]["llm_recommendations"] == 1

    def test_batch_update_from_failures(
        self, temp_intents_file, temp_synonyms_file, mock_llm_client
    ):
        """Test batch update from failure data."""
        use_case = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
            llm_client=mock_llm_client,
        )

        failure_data = [
            {
                "query": "첫번째 실패 쿼리",
                "root_cause":인텐트 부족",
                "suggested_fix": "인텐트 추가",
            },
            {
                "query": "두번째 실패 쿼리",
                "root_cause": "동의어 부족",
                "suggested_fix": "동의어 추가",
            },
        ]

        results = use_case.batch_update_from_failures(
            failure_data=failure_data,
            dry_run=True,
            auto_apply=False,
        )

        assert results["processed"] == 2
        assert "statistics" in results

    def test_get_dictionary_stats(
        self, temp_intents_file, temp_synonyms_file, mock_llm_client
    ):
        """Test getting dictionary statistics through use case."""
        use_case = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
            llm_client=mock_llm_client,
        )

        stats = use_case.get_dictionary_stats()
        assert "intents" in stats
        assert "synonyms" in stats


# ============================================================================
# Integration Tests
# ============================================================================


class TestDictionaryAutomationIntegration:
    """Integration tests for dictionary automation workflow."""

    def test_full_workflow_with_auto_apply(
        self, temp_intents_file, temp_synonyms_file, mock_llm_client
    ):
        """Test full workflow: recommend → detect conflicts → auto-apply."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
            llm_client=mock_llm_client,
        )

        # Get recommendations
        result = manager.recommend_from_failure(
            query="자동화 테스트",
            root_cause="새로운 질문",
            suggested_fix="인텐트와 동의어 추가",
            use_llm=True,
        )

        assert len(result.recommended_intents) > 0
        assert len(result.recommended_synonyms) > 0

        # Check for conflicts
        initial_conflicts = result.conflicts
        assert isinstance(initial_conflicts, list)

        # Add entries (should merge duplicates)
        for intent in result.recommended_intents:
            manager.add_intent(intent, merge=True)

        for synonym in result.recommended_synonyms:
            manager.add_synonym(synonym, merge=True)

        # Save
        intents_saved, synonyms_saved = manager.save(create_backup=True)
        assert intents_saved is True
        assert synonyms_saved is True

        # Verify saved data
        updated_intents = manager.get_all_intents()
        updated_synonyms = manager.get_all_synonyms()

        # Check that new entries were added
        intent_ids = [i["id"] for i in updated_intents]
        assert "new_test_intent" in intent_ids

        # Check that new synonyms were added
        assert "새로운" in updated_synonyms

    def test_conflict_resolution_strategy(
        self, temp_intents_file, temp_synonyms_file
    ):
        """Test different conflict resolution strategies."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
            llm_client=None,
        )

        # Test 1: Duplicate ID (should be error)
        intent1 = IntentEntry(
            id="test_intent_1",
            label="중복",
            triggers=["중복"],
            keywords=["중복"],
        )

        conflicts = manager.detect_conflicts(intent1, [])
        assert any(c.severity == "error" for c in conflicts)

        # Test 2: Similar triggers (should be warning)
        intent2 = IntentEntry(
            id="new_similar_intent",
            label="유사 인텐트",
            triggers=["테스트 트리거"],  # Similar to existing
            keywords=["유사"],
        )

        conflicts = manager.detect_conflicts(intent2, [])
        assert any(c.severity == "warning" for c in conflicts)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestDictionaryAutomationEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dictionary_files(self):
        """Test handling of empty dictionary files."""
        # Create empty temp files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1, \
                tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            json.dump({}, f1)
            json.dump({}, f2)
            
            manager = DictionaryManager(
                intents_path=Path(f1.name),
                synonyms_path=Path(f2.name),
            )

            # Should handle empty files gracefully
            assert len(manager.get_all_intents()) == 0
            assert len(manager.get_all_synonyms()) == 0

        # Cleanup
        Path(f1.name).unlink(missing_ok=True)
        Path(f2.name).unlink(missing_ok=True)

    def test_malformed_json_files(self):
        """Test handling of malformed JSON files."""
        # Create files with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1, \
                tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            f1.write("{ invalid json")
            f2.write("{ invalid json")

            manager = DictionaryManager(
                intents_path=Path(f1.name),
                synonyms_path=Path(f2.name),
            )

            # Should fallback to defaults
            assert manager.get_all_intents() == []
            assert manager.get_all_synonyms() == {}

        # Cleanup
        Path(f1.name).unlink(missing_ok=True)
        Path(f2.name).unlink(missing_ok=True)

    def test_llm_timeout_handling(self, temp_intents_file, temp_synonyms_file):
        """Test handling of LLM timeout or errors."""
        # Mock LLM client that raises exception
        failing_client = mock.MagicMock()
        failing_client.generate.side_effect = Exception("LLM timeout")

        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
            llm_client=failing_client,
        )

        result = manager.recommend_from_failure(
            query="테스트",
            root_cause="테스트",
            suggested_fix="테스트",
            use_llm=True,
        )

        # Should handle error gracefully
        assert result.llm_used is False
        assert len(result.recommended_intents) == 0

    def test_concurrent_updates(self, temp_intents_file, temp_synonyms_file):
        """Test handling of concurrent dictionary updates."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        # Simulate concurrent additions
        intent1 = IntentEntry(
            id="concurrent_1",
            label="동시 1",
            triggers=["동시1"],
            keywords=["동시"],
        )
        intent2 = IntentEntry(
            id="concurrent_2",
            label="동시 2",
            triggers=["동시2"],
            keywords=["동시"],
        )

        # Add both
        manager.add_intent(intent1, merge=False)
        manager.add_intent(intent2, merge=False)

        # Verify both were added
        assert manager.intent_id_exists("concurrent_1") is True
        assert manager.intent_id_exists("concurrent_2") is True

    def test_version_increment(self, temp_intents_file, temp_synonyms_file):
        """Test automatic version increment on save."""
        manager = DictionaryManager(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        # Check initial version
        initial_stats = manager.get_stats()
        initial_version = initial_stats["versions"]["intents"]

        # Add and save
        intent = IntentEntry(
            id="version_test",
            label="버전 테스트",
            triggers=["버전"],
            keywords=["버전"],
        )
        manager.add_intent(intent, merge=False)
        manager.save(create_backup=False)

        # Check version was incremented
        updated_stats = manager.get_stats()
        updated_version = updated_stats["versions"]["intents"]

        assert updated_version != initial_version


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
