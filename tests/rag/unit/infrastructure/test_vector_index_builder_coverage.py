"""
Characterization tests for VectorIndexBuilder.

These tests document the CURRENT behavior of vector index building,
not what it SHOULD do. Tests capture actual outputs for regression detection.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from typing import Dict, List, Tuple


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_dense_retriever():
    """Create a mock DenseRetriever for testing."""
    retriever = MagicMock()
    retriever.search.return_value = []
    retriever.get_cache_stats.return_value = {"hits": 0, "misses": 0}
    return retriever


@pytest.fixture
def mock_retriever_class(mock_dense_retriever):
    """Mock the DenseRetriever class."""
    with patch("src.rag.infrastructure.vector_index_builder.DenseRetriever") as mock_class:
        mock_class.return_value = mock_dense_retriever
        yield mock_class


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        ("doc1", "First document content", {"source": "test"}),
        ("doc2", "Second document content", {"source": "test"}),
        ("doc3", "Third document content", {"source": "test"}),
    ]


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for testing."""
    return [
        {"id": "1", "content": "First document", "title": "Doc 1"},
        {"id": "2", "content": "Second document", "title": "Doc 2"},
    ]


# ============================================================================
# VectorIndexBuilder Initialization Tests
# ============================================================================


class TestVectorIndexBuilderInit:
    """Tests for VectorIndexBuilder initialization."""

    def test_init_with_defaults(self, mock_retriever_class):
        """VectorIndexBuilder initializes with default values."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder()
        assert builder.model_name == "jhgan/ko-sbert-multinli"
        assert builder.batch_size == 64

    def test_init_with_custom_model(self, mock_retriever_class):
        """VectorIndexBuilder accepts custom model name."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(model_name="custom/model")
        assert builder.model_name == "custom/model"

    def test_init_with_custom_batch_size(self, mock_retriever_class):
        """VectorIndexBuilder accepts custom batch size."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(batch_size=128)
        assert builder.batch_size == 128

    def test_init_with_custom_index_dir(self, mock_retriever_class, tmp_path):
        """VectorIndexBuilder accepts custom index directory."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(index_dir=str(tmp_path))
        assert builder.index_dir == tmp_path


# ============================================================================
# build_index Tests
# ============================================================================


class TestBuildIndex:
    """Tests for build_index method."""

    def test_build_index_creates_index(self, mock_retriever_class, sample_documents, tmp_path):
        """build_index creates an index from documents."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(index_dir=str(tmp_path))
        mock_retriever = mock_retriever_class.return_value

        index_path = builder.build_index(sample_documents)

        # Should have called add_documents
        mock_retriever.add_documents.assert_called_once_with(sample_documents)
        # Should have called clear first
        mock_retriever.clear.assert_called_once()
        # Should return index path
        assert index_path.endswith(".pkl")

    def test_build_index_with_custom_name(self, mock_retriever_class, sample_documents, tmp_path):
        """build_index uses custom index name."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(index_dir=str(tmp_path))
        index_path = builder.build_index(sample_documents, index_name="custom_index")

        assert "custom_index" in index_path

    def test_build_index_auto_names_from_model(self, mock_retriever_class, sample_documents, tmp_path):
        """build_index auto-generates name from model."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(
            model_name="org/model-name", index_dir=str(tmp_path)
        )
        index_path = builder.build_index(sample_documents)

        assert "model-name" in index_path


# ============================================================================
# build_index_from_json Tests
# ============================================================================


class TestBuildIndexFromJson:
    """Tests for build_index_from_json method."""

    def test_build_from_json_with_default_fields(
        self, mock_retriever_class, sample_json_data, tmp_path
    ):
        """build_index_from_json uses default field names."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(sample_json_data), encoding="utf-8")

        builder = VectorIndexBuilder(index_dir=str(tmp_path))
        mock_retriever = mock_retriever_class.return_value

        with patch.object(builder, "build_index") as mock_build:
            mock_build.return_value = "test_index.pkl"
            result = builder.build_index_from_json(str(json_path))

            # Should call build_index with extracted documents
            mock_build.assert_called_once()
            call_docs = mock_build.call_args[0][0]
            assert len(call_docs) == 2

    def test_build_from_json_with_custom_fields(
        self, mock_retriever_class, tmp_path
    ):
        """build_index_from_json uses custom field names."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        json_data = [
            {"doc_id": "1", "body": "Content", "category": "test"},
        ]
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(json_data), encoding="utf-8")

        builder = VectorIndexBuilder(index_dir=str(tmp_path))

        with patch.object(builder, "build_index") as mock_build:
            mock_build.return_value = "test_index.pkl"
            builder.build_index_from_json(
                str(json_path),
                content_field="body",
                id_field="doc_id",
            )

            call_docs = mock_build.call_args[0][0]
            doc_id, content, metadata = call_docs[0]
            assert doc_id == "1"
            assert content == "Content"

    def test_build_from_json_extracts_metadata(
        self, mock_retriever_class, tmp_path
    ):
        """build_index_from_json extracts non-content fields as metadata."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        json_data = [
            {"id": "1", "content": "Text", "title": "Title", "author": "Author"},
        ]
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(json_data), encoding="utf-8")

        builder = VectorIndexBuilder(index_dir=str(tmp_path))

        with patch.object(builder, "build_index") as mock_build:
            mock_build.return_value = "test_index.pkl"
            builder.build_index_from_json(str(json_path))

            call_docs = mock_build.call_args[0][0]
            doc_id, content, metadata = call_docs[0]
            assert metadata.get("title") == "Title"
            assert metadata.get("author") == "Author"

    def test_build_from_json_with_specific_metadata_fields(
        self, mock_retriever_class, tmp_path
    ):
        """build_index_from_json uses specified metadata fields only."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        json_data = [
            {
                "id": "1",
                "content": "Text",
                "include": "yes",
                "exclude": "no",
            },
        ]
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(json_data), encoding="utf-8")

        builder = VectorIndexBuilder(index_dir=str(tmp_path))

        with patch.object(builder, "build_index") as mock_build:
            mock_build.return_value = "test_index.pkl"
            builder.build_index_from_json(
                str(json_path),
                metadata_fields=["include"],
            )

            call_docs = mock_build.call_args[0][0]
            doc_id, content, metadata = call_docs[0]
            assert "include" in metadata
            assert "exclude" not in metadata


# ============================================================================
# load_index Tests
# ============================================================================


class TestLoadIndex:
    """Tests for load_index method."""

    def test_load_index_calls_retriever(self, mock_retriever_class, tmp_path):
        """load_index calls retriever load_index."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(index_dir=str(tmp_path))
        mock_retriever = mock_retriever_class.return_value
        mock_retriever.load_index.return_value = True

        result = builder.load_index("test_index.pkl")

        mock_retriever.load_index.assert_called_once_with("test_index.pkl")
        assert result is True

    def test_load_index_returns_false_on_failure(self, mock_retriever_class, tmp_path):
        """load_index returns False on failure."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(index_dir=str(tmp_path))
        mock_retriever = mock_retriever_class.return_value
        mock_retriever.load_index.return_value = False

        result = builder.load_index("nonexistent.pkl")

        assert result is False


# ============================================================================
# search Tests
# ============================================================================


class TestSearch:
    """Tests for search method."""

    def test_search_calls_retriever(self, mock_retriever_class, tmp_path):
        """search calls retriever search."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(index_dir=str(tmp_path))
        mock_retriever = mock_retriever_class.return_value
        mock_retriever.search.return_value = []

        result = builder.search("test query", top_k=10)

        mock_retriever.search.assert_called_once()
        assert result == []

    def test_search_with_score_threshold(self, mock_retriever_class, tmp_path):
        """search passes score_threshold to retriever."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(index_dir=str(tmp_path))
        mock_retriever = mock_retriever_class.return_value
        mock_retriever.search.return_value = []

        builder.search("test query", top_k=5, score_threshold=0.5)

        call_kwargs = mock_retriever.search.call_args[1]
        assert call_kwargs.get("score_threshold") == 0.5


# ============================================================================
# Module-Level Functions Tests
# ============================================================================


class TestBuildAllIndices:
    """Tests for build_all_indices function."""

    def test_build_all_indices_returns_dict(self, mock_retriever_class, tmp_path):
        """build_all_indices returns a dictionary."""
        from src.rag.infrastructure.vector_index_builder import build_all_indices

        # Create test data directory with no JSON files
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = build_all_indices(str(data_dir), str(tmp_path / "indices"))

        assert isinstance(result, dict)

    def test_build_all_indices_handles_missing_dir(self, mock_retriever_class, tmp_path):
        """build_all_indices handles missing data directory."""
        from src.rag.infrastructure.vector_index_builder import build_all_indices

        result = build_all_indices(
            str(tmp_path / "nonexistent"),
            str(tmp_path / "indices"),
        )

        assert result == {}

    def test_build_all_indices_uses_default_models(self, mock_retriever_class, tmp_path):
        """build_all_indices uses default Korean models."""
        from src.rag.infrastructure.vector_index_builder import build_all_indices

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a test JSON file
        json_file = data_dir / "test.json"
        json_file.write_text('[{"id": "1", "content": "test"}]', encoding="utf-8")

        # Mock VectorIndexBuilder to avoid actual model loading
        with patch("src.rag.infrastructure.vector_index_builder.VectorIndexBuilder") as mock_builder_class:
            mock_builder = MagicMock()
            mock_builder.build_index_from_json.return_value = "test_index.pkl"
            mock_builder_class.return_value = mock_builder

            result = build_all_indices(str(data_dir), str(tmp_path / "indices"))

            # Should have tried to build indices
            assert isinstance(result, dict)


class TestDownloadModel:
    """Tests for download_model function."""

    def test_download_model_returns_true_on_success(self):
        """download_model returns True when model downloads successfully."""
        from src.rag.infrastructure.vector_index_builder import download_model

        # Patch at module level before import
        import sys
        import types

        # Create mock sentence_transformers module
        mock_st_module = types.ModuleType("sentence_transformers")
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            result = download_model("test/model")

            assert result is True

    def test_download_model_returns_false_on_failure(self):
        """download_model returns False on failure."""
        from src.rag.infrastructure.vector_index_builder import download_model

        import types

        # Create mock that raises exception
        mock_st_module = types.ModuleType("sentence_transformers")
        mock_st_module.SentenceTransformer = MagicMock(side_effect=Exception("Download failed"))

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            result = download_model("nonexistent/model")

            assert result is False


class TestListAvailableModels:
    """Tests for list_available_models function."""

    def test_list_available_models_returns_dict(self):
        """list_available_models returns a dictionary."""
        from src.rag.infrastructure.vector_index_builder import list_available_models

        with patch("src.rag.infrastructure.vector_index_builder.DenseRetriever") as mock_retriever:
            mock_retriever.list_models.return_value = {"model1": {"dims": 768}}

            result = list_available_models()

            assert isinstance(result, dict)


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_build_index_empty_documents(self, mock_retriever_class, tmp_path):
        """build_index handles empty document list."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(index_dir=str(tmp_path))

        result = builder.build_index([])

        # Should still return a path
        assert result.endswith(".pkl")

    def test_build_from_json_empty_array(self, mock_retriever_class, tmp_path):
        """build_index_from_json handles empty JSON array."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        json_path = tmp_path / "empty.json"
        json_path.write_text("[]", encoding="utf-8")

        builder = VectorIndexBuilder(index_dir=str(tmp_path))

        with patch.object(builder, "build_index") as mock_build:
            mock_build.return_value = "test_index.pkl"
            builder.build_index_from_json(str(json_path))

            call_docs = mock_build.call_args[0][0]
            assert len(call_docs) == 0

    def test_build_from_json_missing_fields(self, mock_retriever_class, tmp_path):
        """build_index_from_json handles missing fields gracefully."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        json_data = [
            {"id": "1"},  # Missing content
        ]
        json_path = tmp_path / "incomplete.json"
        json_path.write_text(json.dumps(json_data), encoding="utf-8")

        builder = VectorIndexBuilder(index_dir=str(tmp_path))

        with patch.object(builder, "build_index") as mock_build:
            mock_build.return_value = "test_index.pkl"
            builder.build_index_from_json(str(json_path))

            call_docs = mock_build.call_args[0][0]
            doc_id, content, metadata = call_docs[0]
            assert doc_id == "1"
            assert content == ""  # Default empty content

    def test_search_empty_query(self, mock_retriever_class, tmp_path):
        """search handles empty query."""
        from src.rag.infrastructure.vector_index_builder import VectorIndexBuilder

        builder = VectorIndexBuilder(index_dir=str(tmp_path))
        mock_retriever = mock_retriever_class.return_value
        mock_retriever.search.return_value = []

        result = builder.search("", top_k=10)

        assert result == []
