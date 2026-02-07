"""
Shared fixtures for characterization tests.

These fixtures provide common test data and mock objects
for characterization testing across multiple test modules.
"""

from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock_client = Mock()
    mock_client.generate = Mock(return_value="Test response")
    return mock_client


@pytest.fixture
def sample_chunks():
    """Sample regulation chunks for testing."""
    from src.rag.domain.entities import Chunk

    chunks = [
        Mock(spec=Chunk),
        Mock(spec=Chunk),
        Mock(spec=Chunk),
    ]

    chunks[
        0
    ].text = (
        "학생 휴학에 관한 규정입니다. 휴학은 학기 시작 30일 이전에 신청해야 합니다."
    )
    chunks[0].title = "휴학 규정"
    chunks[0].metadata = {"regulation": "학칙", "article": "제15조"}

    chunks[1].text = "장학금 지급 기준은 성적 평점이 3.0 이상이어야 합니다."
    chunks[1].title = "장학금 규정"
    chunks[1].metadata = {"regulation": "장학금지급규정", "article": "제8조"}

    chunks[2].text = "교원의 승진 및 인사에 관한 규정입니다."
    chunks[2].title = "교원인사규정"
    chunks[2].metadata = {"regulation": "교원인사규정", "article": "제5조"}

    return chunks


@pytest.fixture
def sample_search_results(sample_chunks):
    """Sample search results for testing."""
    from src.rag.domain.entities import SearchResult

    results = []
    for chunk in sample_chunks:
        result = Mock(spec=SearchResult)
        result.chunk = chunk
        result.score = 0.85
        result.citation = f"{chunk.metadata.get('regulation', '')} {chunk.metadata.get('article', '')}"
        results.append(result)

    return results


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return {
        "simple": "휴학",
        "article_reference": "학칙 제15조",
        "natural_question": "휴학 어떻게 하나요",
        "intent": "휴학하고 싶어",
        "complex": "장학금 받으면서 휴학 가능한가요",
    }


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory for testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_search_usecase():
    """Mock SearchUseCase for testing."""
    from src.rag.application.search_usecase import SearchUseCase

    mock_usecase = Mock(spec=SearchUseCase)
    mock_usecase.search = Mock(return_value=[])
    mock_usecase.search_with_answer = Mock(return_value=Mock(answer="Test answer"))
    return mock_usecase
