"""
Test scenarios for Multi-hop Question Handler.

This module contains test cases for validating the multi-hop question
answering functionality in the regulation RAG system.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.application.multi_hop_handler import (
    DependencyCycleDetector,
    MultiHopHandler,
    MultiHopQueryDecomposer,
    SubQuery,
)
from src.rag.domain.entities import Chunk, SearchResult


class TestSubQuery:
    """Test SubQuery dataclass."""

    def test_sub_query_creation(self):
        """Test creating a SubQuery instance."""
        sub_query = SubQuery(
            query_id="hop_1",
            query_text="졸업 요건 조회",
            hop_order=1,
            depends_on=[],
            reasoning="First step: identify graduation requirements",
        )
        assert sub_query.query_id == "hop_1"
        assert sub_query.query_text == "졸업 요건 조회"
        assert sub_query.hop_order == 1
        assert sub_query.depends_on == []
        assert sub_query.reasoning == "First step: identify graduation requirements"

    def test_sub_query_to_dict(self):
        """Test converting SubQuery to dictionary."""
        sub_query = SubQuery(
            query_id="hop_1",
            query_text="전공 필수 과목 조회",
            hop_order=1,
            depends_on=[],
            reasoning="Get required major courses",
        )
        data = sub_query.to_dict()
        assert data["query_id"] == "hop_1"
        assert data["query_text"] == "전공 필수 과목 조회"
        assert data["hop_order"] == 1
        assert data["depends_on"] == []
        assert data["reasoning"] == "Get required major courses"

    def test_sub_query_from_dict(self):
        """Test creating SubQuery from dictionary."""
        data = {
            "query_id": "hop_2",
            "query_text": "선이수 과목 조회",
            "hop_order": 2,
            "depends_on": ["hop_1"],
            "context_from": "hop_1",
            "reasoning": "Get prerequisites for required courses",
        }
        sub_query = SubQuery.from_dict(data)
        assert sub_query.query_id == "hop_2"
        assert sub_query.query_text == "선이수 과목 조회"
        assert sub_query.hop_order == 2
        assert sub_query.depends_on == ["hop_1"]
        assert sub_query.context_from == "hop_1"


class TestDependencyCycleDetector:
    """Test dependency cycle detection."""

    def test_no_cycle_simple_chain(self):
        """Test simple linear chain has no cycle."""
        detector = DependencyCycleDetector(max_hops=5)
        dependencies = {
            "hop_1": [],
            "hop_2": ["hop_1"],
            "hop_3": ["hop_2"],
        }
        assert detector.detect_cycle(dependencies) is None

    def test_detect_direct_cycle(self):
        """Test detection of direct cycle (A -> B -> A)."""
        detector = DependencyCycleDetector(max_hops=5)
        dependencies = {
            "hop_1": ["hop_2"],
            "hop_2": ["hop_1"],
        }
        cycle = detector.detect_cycle(dependencies)
        assert cycle is not None
        assert "hop_1" in cycle

    def test_detect_complex_cycle(self):
        """Test detection of complex cycle (A -> B -> C -> A)."""
        detector = DependencyCycleDetector(max_hops=5)
        dependencies = {
            "hop_1": ["hop_2"],
            "hop_2": ["hop_3"],
            "hop_3": ["hop_1"],
        }
        cycle = detector.detect_cycle(dependencies)
        assert cycle is not None
        assert len(cycle) >= 3

    def test_max_hops_validation(self):
        """Test max hops validation."""
        detector = DependencyCycleDetector(max_hops=3)

        # Valid: 3 hops
        sub_queries = [
            SubQuery(query_id="hop_1", query_text="Q1", hop_order=1, depends_on=[]),
            SubQuery(
                query_id="hop_2", query_text="Q2", hop_order=2, depends_on=["hop_1"]
            ),
            SubQuery(
                query_id="hop_3", query_text="Q3", hop_order=3, depends_on=["hop_2"]
            ),
        ]
        assert detector.validate_max_hops(sub_queries) is True

        # Invalid: 5 hops (exceeds max_hops=3)
        sub_queries = [
            SubQuery(
                query_id=f"hop_{i}", query_text=f"Q{i}", hop_order=i, depends_on=[]
            )
            for i in range(1, 6)
        ]
        assert detector.validate_max_hops(sub_queries) is False


class TestMultiHopQueryDecomposer:
    """Test multi-hop query decomposition."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = MagicMock()
        return mock_client

    @pytest.fixture
    def decomposer(self, mock_llm_client):
        """Create a MultiHopQueryDecomposer instance."""
        return MultiHopQueryDecomposer(mock_llm_client)

    @pytest.mark.asyncio
    async def test_decompose_simple_query(self, decomposer, mock_llm_client):
        """Test decomposition of a simple single-hop query."""
        mock_llm_client.generate.return_value = """```json
{
  "sub_queries": [
    {
      "query_id": "hop_1",
      "query_text": "졸업 학점",
      "hop_order": 1,
      "depends_on": [],
      "reasoning": "Simple query about graduation credits"
    }
  ]
}
```"""

        sub_queries = await decomposer.decompose("졸업하려면 몇 학점이 필요한가요?")
        assert len(sub_queries) == 1
        assert sub_queries[0].hop_order == 1
        assert sub_queries[0].query_text == "졸업 학점"

    @pytest.mark.asyncio
    async def test_decompose_multi_hop_query(self, decomposer, mock_llm_client):
        """Test decomposition of a multi-hop query."""
        mock_llm_client.generate.return_value = """```json
{
  "sub_queries": [
    {
      "query_id": "hop_1",
      "query_text": "졸업 요건에 따른 전공 필수 과목 목록",
      "hop_order": 1,
      "depends_on": [],
      "reasoning": "First identify required courses for graduation"
    },
    {
      "query_id": "hop_2",
      "query_text": "전공 필수 과목들의 선이수 과목 조회",
      "hop_order": 2,
      "depends_on": ["hop_1"],
      "context_from": "hop_1",
      "reasoning": "Then find prerequisites for each required course"
    }
  ]
}
```"""

        sub_queries = await decomposer.decompose(
            "졸업 요건을 충족하려면 어떤 전공 필수 과목을 이수해야 하고, "
            "그 과목들의 선이수 과목은 무엇인가요?"
        )
        assert len(sub_queries) == 2
        assert sub_queries[0].hop_order == 1
        assert sub_queries[1].hop_order == 2
        assert sub_queries[1].depends_on == ["hop_1"]
        assert sub_queries[1].context_from == "hop_1"

    @pytest.mark.asyncio
    async def test_decompose_fallback_on_error(self, decomposer, mock_llm_client):
        """Test fallback to single-hop when decomposition fails."""
        mock_llm_client.generate.side_effect = Exception("LLM error")

        sub_queries = await decomposer.decompose("test query")
        assert len(sub_queries) == 1
        assert sub_queries[0].hop_order == 1
        assert sub_queries[0].query_text == "test query"
        assert "fallback" in sub_queries[0].reasoning.lower()

    @pytest.mark.asyncio
    async def test_decompose_fallback_on_invalid_json(
        self, decomposer, mock_llm_client
    ):
        """Test fallback when LLM returns invalid JSON."""
        mock_llm_client.generate.return_value = "This is not valid JSON"

        sub_queries = await decomposer.decompose("test query")
        assert len(sub_queries) == 1
        assert sub_queries[0].hop_order == 1
        assert "parsing error" in sub_queries[0].reasoning.lower()


class TestMultiHopHandler:
    """Test multi-hop query handler."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = MagicMock()
        return mock_store

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = MagicMock()
        return mock_client

    @pytest.fixture
    def handler(self, mock_vector_store, mock_llm_client):
        """Create a MultiHopHandler instance."""
        return MultiHopHandler(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            max_hops=5,
            hop_timeout_seconds=30,
            enable_self_rag=False,
        )

    @pytest.mark.asyncio
    async def test_execute_single_hop(
        self, handler, mock_vector_store, mock_llm_client
    ):
        """Test execution of a single-hop query."""
        # Mock decomposer to return single sub-query
        handler.decomposer.decompose = AsyncMock(
            return_value=[
                SubQuery(
                    query_id="hop_1", query_text="test", hop_order=1, depends_on=[]
                )
            ]
        )

        # Mock vector store search
        mock_chunk = MagicMock(spec=Chunk)
        mock_chunk.id = "chunk_1"
        mock_chunk.text = "Test content"
        mock_chunk.title = "Test Regulation"

        mock_vector_store.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        # Mock LLM generation
        mock_llm_client.generate.return_value = "Test answer"

        result = await handler.execute_multi_hop("test query")

        assert result.success is True
        assert result.hop_count == 1
        assert result.final_answer == "Test answer"

    @pytest.mark.asyncio
    async def test_cycle_detection(self, handler, mock_vector_store, mock_llm_client):
        """Test that cyclic dependencies are detected and rejected."""
        # Create sub-queries with a cycle
        sub_queries = [
            SubQuery(
                query_id="hop_1", query_text="Q1", hop_order=1, depends_on=["hop_2"]
            ),
            SubQuery(
                query_id="hop_2", query_text="Q2", hop_order=2, depends_on=["hop_1"]
            ),
        ]

        handler.decomposer.decompose = AsyncMock(return_value=sub_queries)

        result = await handler.execute_multi_hop("cyclic query")

        assert result.success is False
        assert "cyclic" in result.final_answer.lower()

    @pytest.mark.asyncio
    async def test_max_hops_exceeded(self, handler, mock_vector_store, mock_llm_client):
        """Test that exceeding max hops is rejected."""
        # Create sub-queries exceeding max_hops (5)
        sub_queries = [
            SubQuery(
                query_id=f"hop_{i}", query_text=f"Q{i}", hop_order=i, depends_on=[]
            )
            for i in range(1, 7)  # 6 hops
        ]

        handler.decomposer.decompose = AsyncMock(return_value=sub_queries)

        result = await handler.execute_multi_hop("complex query")

        assert result.success is False
        assert (
            "exceeds maximum" in result.final_answer.lower()
            or "hop count" in result.final_answer.lower()
        )


class TestMultiHopIntegration:
    """Integration tests for multi-hop functionality."""

    @pytest.mark.integration
    def test_multi_hop_scenario_graduation_prerequisites(self):
        """
        Test realistic multi-hop scenario: graduation requirements -> prerequisites.

        Query: "졸업 요건을 충족하려면 어떤 전공 필수 과목을 이수해야 하고,
                 그 과목들의 선이수 과목은 무엇인가요?"

        Expected:
        - Hop 1: Query graduation requirements -> Get required major courses
        - Hop 2: Query each required course -> Get prerequisites
        - Synthesize: Combine all information into comprehensive answer
        """
        # This test requires actual vector store and LLM
        # Mark as integration test and skip in unit test runs
        pytest.skip("Integration test - requires actual vector store and LLM")

    @pytest.mark.integration
    def test_multi_hop_scenario_scholarship_eligibility(self):
        """
        Test multi-hop scenario for scholarship eligibility.

        Query: "장학금을 받으려면 성적 기준이 얼마이고, "
                 "해당 성적을 유지하기 위해 어떤 과목을 우선 들어야 하나요?"

        Expected:
        - Hop 1: Query scholarship requirements -> Get GPA threshold
        - Hop 2: Query course difficulty/ratings -> Identify easier courses
        - Synthesize: Recommend course sequence for maintaining scholarship eligibility
        """
        pytest.skip("Integration test - requires actual vector store and LLM")


# Test scenarios documentation
TEST_SCENARIOS = """
## Multi-hop Test Scenarios

### Scenario 1: Graduation Requirements (2-hop)
**Query:** "졸업 요건을 충족하려면 어떤 전공 필수 과목을 이수해야 하고, 그 과목들의 선이수 과목은 무엇인가요?"

**Hop 1:** "졸업 요건에 따른 전공 필수 과목 목록"
- Expected: Retrieve regulation specifying required major courses for graduation
- Output: List of required courses (e.g., "전공필수 12학점 이수")

**Hop 2:** "전공 필수 과목들의 선이수 과목 조회"
- Context from Hop 1: List of required courses
- Expected: For each required course, find its prerequisites
- Output: Prerequisite mapping for each course

**Synthesis:** Combine all information into comprehensive course plan

---

### Scenario 2: Scholarship Eligibility (2-hop)
**Query:** "장학금을 받으려면 성적 기준이 얼마이고, 해당 성적을 유지하기 위해 어떤 과목을 우선 들어야 하나요?"

**Hop 1:** "장학금 성적 기준"
- Expected: GPA threshold for scholarship eligibility
- Output: "3.0/4.5 이상"

**Hop 2:** "성적 유지를 위한 수강 추천 과목"
- Context from Hop 1: Need to maintain 3.0 GPA
- Expected: Find courses with higher success rates or easier grading
- Output: Recommended courses for maintaining GPA

**Synthesis:** Actionable scholarship maintenance plan

---

### Scenario 3: Faculty Promotion (3-hop)
**Query:** "교수가 정교수로 승진하려면 어떤 연구 업적이 필요하고, 해당 업적을 충족하기 위한 연구비 지원 제도는 무엇인가요?"

**Hop 1:** "정교수 승진 요건"
- Expected: Research publication requirements for promotion
- Output: "SCI 논문 3편 이상"

**Hop 2:** "연구비 지원 제도 검색"
- Context from Hop 1: Need to publish SCI papers
- Expected: Find research grant programs
- Output: List of available grants

**Hop 3:** "연구비 신청 자격 및 절차"
- Context from Hop 2: Specific grant programs
- Expected: Application requirements for each grant
- Output: Application guide

**Synthesis:** Comprehensive promotion pathway with funding strategy

---

### Scenario 4: Course Registration Conflict (2-hop)
**Query:** "교양 과목과 전공 과목이 시간표가 겹칠 때 수강 변경 절차는 무엇인가요?"

**Hop 1:** "수강 변경 (수강 철회, 변경) 규정"
- Expected: Course change/drop regulations
- Output: Procedures and deadlines

**Hop 2:** "시간표 겹침 시 예외 규정"
- Context from Hop 1: Standard change procedures
- Expected: Special provisions for conflicts
- Output: Exception handling process

**Synthesis:** Step-by-step guide for resolving schedule conflicts

---

### Scenario 5: Student Leave of Absence (2-hop)
**Query:** "휴학 후 복학할 때 학점 인정 기준과 복학 신청 절차는 어떻게 되나요?"

**Hop 1:** "복학 시 학점 인정 기준"
- Expected: Credit recognition rules after leave
- Output: Valid credits and expiration rules

**Hop 2:** "복학 신청 절차 및 서류"
- Context from Hop 1: Need to apply for readmission
- Expected: Application process and required documents
- Output: Step-by-step application guide

**Synthesis:** Complete readmission guide with credit planning
"""
