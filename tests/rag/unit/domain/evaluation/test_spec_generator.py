"""
Unit tests for SPECGenerator.

Tests for SPEC-RAG-EVAL-001 Milestone 4: Report Enhancement.
"""

import tempfile
from pathlib import Path
from typing import List

import pytest

from src.rag.domain.evaluation.failure_classifier import FailureSummary, FailureType
from src.rag.domain.evaluation.recommendation_engine import Priority, Recommendation
from src.rag.domain.evaluation.spec_generator import SPECDocument, SPECGenerator


@pytest.fixture
def generator():
    """Create a SPECGenerator instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SPECGenerator(template_dir=tmpdir)


@pytest.fixture
def sample_failures():
    """Create sample failure summaries."""
    return [
        FailureSummary(
            failure_type=FailureType.HALLUCINATION,
            count=10,
            examples=["query1", "query2"],
            affected_personas=["freshman", "professor"],
            avg_score=0.45,
        ),
        FailureSummary(
            failure_type=FailureType.MISSING_INFO,
            count=5,
            examples=["query3"],
            affected_personas=["staff"],
            avg_score=0.55,
        ),
    ]


@pytest.fixture
def sample_recommendations():
    """Create sample recommendations."""
    return [
        Recommendation(
            id="REC-001",
            title="Reduce Hallucinations",
            description="Implement stricter context grounding.",
            priority=Priority.CRITICAL,
            failure_types=[FailureType.HALLUCINATION],
            actions=[
                "Add citation requirements",
                "Implement fact checking",
                "Add hallucination detection",
            ],
            spec_mapping="SPEC-RAG-Q-001",
        ),
        Recommendation(
            id="REC-002",
            title="Improve Completeness",
            description="Enhance prompts for complete information.",
            priority=Priority.HIGH,
            failure_types=[FailureType.MISSING_INFO],
            actions=[
                "Add completeness checklist",
                "Implement multi-hop retrieval",
            ],
            spec_mapping="SPEC-RAG-Q-002",
        ),
    ]


class TestSPECDocument:
    """Tests for SPECDocument."""

    def test_creation(self):
        """Test creating a SPECDocument."""
        spec = SPECDocument(
            spec_id="SPEC-TEST-001",
            title="Test SPEC",
            status="Draft",
            created_at="2024-01-01T00:00:00",
            priority="HIGH",
            description="Test description",
            requirements=[],
            acceptance_criteria=["Criteria 1", "Criteria 2"],
            technical_approach=["Approach 1"],
        )

        assert spec.spec_id == "SPEC-TEST-001"
        assert spec.title == "Test SPEC"
        assert spec.status == "Draft"
        assert len(spec.acceptance_criteria) == 2

    def test_to_markdown(self):
        """Test markdown generation."""
        spec = SPECDocument(
            spec_id="SPEC-TEST-001",
            title="Test SPEC",
            status="Draft",
            created_at="2024-01-01T00:00:00",
            priority="HIGH",
            description="This is a test SPEC document.",
            requirements=[
                {
                    "title": "REQ-1",
                    "type": "Functional",
                    "format": "Ubiquitous",
                    "description": "The system shall do X.",
                }
            ],
            acceptance_criteria=["All tests pass", "Coverage > 85%"],
            technical_approach=["Implement feature", "Add tests"],
            related_specs=["SPEC-OTHER-001"],
        )

        markdown = spec.to_markdown()

        assert "SPEC-TEST-001" in markdown
        assert "Test SPEC" in markdown
        assert "Draft" in markdown
        assert "HIGH" in markdown
        assert "Description" in markdown
        assert "Requirements" in markdown
        assert "REQ-1" in markdown
        assert "Acceptance Criteria" in markdown
        assert "All tests pass" in markdown
        assert "Technical Approach" in markdown
        assert "Related SPECs" in markdown

    def test_to_dict(self):
        """Test serialization to dictionary."""
        spec = SPECDocument(
            spec_id="SPEC-001",
            title="Test",
            status="Draft",
            created_at="2024-01-01",
            priority="MEDIUM",
            description="Description",
            requirements=[],
            acceptance_criteria=[],
            technical_approach=[],
            metadata={"key": "value"},
        )

        data = spec.to_dict()

        assert data["spec_id"] == "SPEC-001"
        assert data["title"] == "Test"
        assert data["metadata"]["key"] == "value"


class TestSPECGenerator:
    """Tests for SPECGenerator."""

    def test_init(self, generator):
        """Test initialization."""
        assert generator.template_dir is not None
        assert generator._spec_counter == 0

    def test_generate_spec(self, generator, sample_failures, sample_recommendations):
        """Test generating a SPEC document."""
        spec = generator.generate_spec(
            failures=sample_failures,
            recommendations=sample_recommendations,
        )

        assert spec is not None
        assert spec.spec_id.startswith("SPEC-RAG-Q-")
        assert spec.title is not None
        assert spec.status == "Draft"
        assert len(spec.requirements) > 0
        assert len(spec.acceptance_criteria) > 0
        assert "failure_types" in spec.metadata

    def test_generate_spec_empty_inputs(self, generator):
        """Test generating SPEC with empty inputs."""
        spec = generator.generate_spec(
            failures=[],
            recommendations=[],
        )

        assert spec is not None
        assert spec.title == "Improve RAG System Quality"  # Default template

    def test_generate_spec_custom_prefix(self, generator, sample_failures, sample_recommendations):
        """Test generating SPEC with custom prefix."""
        spec = generator.generate_spec(
            failures=sample_failures,
            recommendations=sample_recommendations,
            spec_prefix="SPEC-CUSTOM",
        )

        assert spec.spec_id.startswith("SPEC-CUSTOM-")

    def test_generate_spec_increments_counter(self, generator, sample_failures, sample_recommendations):
        """Test that counter increments with each SPEC."""
        spec1 = generator.generate_spec(sample_failures, sample_recommendations)
        spec2 = generator.generate_spec(sample_failures, sample_recommendations)

        # Extract numbers from spec_ids
        num1 = int(spec1.spec_id.split("-")[-1])
        num2 = int(spec2.spec_id.split("-")[-1])

        assert num2 > num1

    def test_generate_spec_determines_priority(self, generator, sample_failures):
        """Test that SPEC priority is determined from recommendations."""
        critical_rec = Recommendation(
            id="REC-CRIT",
            title="Critical",
            description="Critical issue",
            priority=Priority.CRITICAL,
            failure_types=[FailureType.HALLUCINATION],
            actions=["Fix"],
        )

        spec = generator.generate_spec(
            failures=sample_failures,
            recommendations=[critical_rec],
        )

        assert spec.priority == "CRITICAL"

    def test_generate_spec_extracts_technical_approach(self, generator, sample_failures, sample_recommendations):
        """Test that technical approach is extracted from recommendations."""
        spec = generator.generate_spec(
            failures=sample_failures,
            recommendations=sample_recommendations,
        )

        assert len(spec.technical_approach) > 0
        # Should include actions from recommendations
        assert any("citation" in action.lower() for action in spec.technical_approach)

    def test_generate_spec_includes_related_specs(self, generator, sample_failures, sample_recommendations):
        """Test that related specs are included."""
        spec = generator.generate_spec(
            failures=sample_failures,
            recommendations=sample_recommendations,
        )

        assert len(spec.related_specs) > 0
        assert "SPEC-RAG-Q-001" in spec.related_specs


class TestSPECGeneratorTemplates:
    """Tests for SPEC templates by failure type."""

    def test_hallucination_template(self, generator):
        """Test hallucination template is used correctly."""
        failures = [
            FailureSummary(
                failure_type=FailureType.HALLUCINATION,
                count=10,
                examples=[],
                affected_personas=[],
                avg_score=0.4,
            )
        ]

        spec = generator.generate_spec(failures=failures, recommendations=[])

        assert "hallucination" in spec.title.lower()
        assert len(spec.acceptance_criteria) >= 2

    def test_missing_info_template(self, generator):
        """Test missing info template is used correctly."""
        failures = [
            FailureSummary(
                failure_type=FailureType.MISSING_INFO,
                count=5,
                examples=[],
                affected_personas=[],
                avg_score=0.5,
            )
        ]

        spec = generator.generate_spec(failures=failures, recommendations=[])

        assert "complete" in spec.title.lower() or "information" in spec.title.lower()

    def test_citation_error_template(self, generator):
        """Test citation error template is used correctly."""
        failures = [
            FailureSummary(
                failure_type=FailureType.CITATION_ERROR,
                count=5,
                examples=[],
                affected_personas=[],
                avg_score=0.5,
            )
        ]

        spec = generator.generate_spec(failures=failures, recommendations=[])

        assert "citation" in spec.title.lower()

    def test_retrieval_failure_template(self, generator):
        """Test retrieval failure template is used correctly."""
        failures = [
            FailureSummary(
                failure_type=FailureType.RETRIEVAL_FAILURE,
                count=5,
                examples=[],
                affected_personas=[],
                avg_score=0.5,
            )
        ]

        spec = generator.generate_spec(failures=failures, recommendations=[])

        assert "retrieval" in spec.title.lower()

    def test_unknown_failure_type_uses_default(self, generator):
        """Test unknown failure type uses default template."""
        failures = [
            FailureSummary(
                failure_type=FailureType.UNKNOWN,
                count=5,
                examples=[],
                affected_personas=[],
                avg_score=0.5,
            )
        ]

        spec = generator.generate_spec(failures=failures, recommendations=[])

        assert spec.title == "Improve RAG System Quality"


class TestSPECGeneratorSave:
    """Tests for SPEC save functionality."""

    def test_save_spec_default_path(self, generator, sample_failures, sample_recommendations):
        """Test saving SPEC to default path."""
        spec = generator.generate_spec(sample_failures, sample_recommendations)

        path = generator.save_spec(spec)

        assert path is not None
        assert Path(path).exists()
        assert path.endswith("spec.md")

        # Read and verify content
        content = Path(path).read_text()
        assert spec.spec_id in content

    def test_save_spec_custom_path(self, generator, sample_failures, sample_recommendations):
        """Test saving SPEC to custom path."""
        spec = generator.generate_spec(sample_failures, sample_recommendations)

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = str(Path(tmpdir) / "custom_spec.md")
            path = generator.save_spec(spec, path=custom_path)

            assert path == custom_path
            assert Path(path).exists()

    def test_save_spec_creates_directories(self, generator, sample_failures, sample_recommendations):
        """Test that save creates parent directories."""
        spec = generator.generate_spec(sample_failures, sample_recommendations)

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = str(Path(tmpdir) / "nested" / "dir" / "spec.md")
            path = generator.save_spec(spec, path=custom_path)

            assert Path(path).exists()


class TestSPECGeneratorBatch:
    """Tests for batch SPEC generation."""

    def test_generate_batch_specs(self, generator):
        """Test generating multiple SPECs for different failure types."""
        failures = [
            FailureSummary(
                failure_type=FailureType.HALLUCINATION,
                count=10,
                examples=[],
                affected_personas=[],
                avg_score=0.4,
            ),
            FailureSummary(
                failure_type=FailureType.MISSING_INFO,
                count=5,
                examples=[],
                affected_personas=[],
                avg_score=0.5,
            ),
            FailureSummary(
                failure_type=FailureType.CITATION_ERROR,
                count=3,
                examples=[],
                affected_personas=[],
                avg_score=0.5,
            ),
        ]

        recommendations = [
            Recommendation(
                id="REC-001",
                title="Fix Hallucination",
                description="",
                priority=Priority.CRITICAL,
                failure_types=[FailureType.HALLUCINATION],
                actions=["Action 1"],
            ),
            Recommendation(
                id="REC-002",
                title="Fix Missing Info",
                description="",
                priority=Priority.HIGH,
                failure_types=[FailureType.MISSING_INFO],
                actions=["Action 2"],
            ),
        ]

        specs = generator.generate_batch_specs(
            failures=failures,
            recommendations=recommendations,
            max_specs=3,
        )

        assert len(specs) <= 3
        assert all(isinstance(s, SPECDocument) for s in specs)

    def test_generate_batch_specs_limits_count(self, generator):
        """Test that batch generation respects max_specs limit."""
        failures = [
            FailureSummary(
                failure_type=FailureType.HALLUCINATION,
                count=10,
                examples=[],
                affected_personas=[],
                avg_score=0.4,
            ),
            FailureSummary(
                failure_type=FailureType.MISSING_INFO,
                count=5,
                examples=[],
                affected_personas=[],
                avg_score=0.5,
            ),
            FailureSummary(
                failure_type=FailureType.CITATION_ERROR,
                count=3,
                examples=[],
                affected_personas=[],
                avg_score=0.5,
            ),
            FailureSummary(
                failure_type=FailureType.RETRIEVAL_FAILURE,
                count=2,
                examples=[],
                affected_personas=[],
                avg_score=0.5,
            ),
        ]

        specs = generator.generate_batch_specs(
            failures=failures,
            recommendations=[],
            max_specs=2,
        )

        assert len(specs) == 2

    def test_generate_batch_specs_empty(self, generator):
        """Test batch generation with empty inputs."""
        specs = generator.generate_batch_specs(
            failures=[],
            recommendations=[],
        )

        assert specs == []


class TestSPECGeneratorEdgeCases:
    """Edge case tests for SPECGenerator."""

    def test_generate_spec_with_many_recommendations(self, generator, sample_failures):
        """Test with many recommendations (should limit to 5)."""
        recommendations = [
            Recommendation(
                id=f"REC-{i}",
                title=f"Recommendation {i}",
                description="",
                priority=Priority.MEDIUM,
                failure_types=[FailureType.HALLUCINATION],
                actions=[f"Action {i}"],
            )
            for i in range(10)
        ]

        spec = generator.generate_spec(
            failures=sample_failures,
            recommendations=recommendations,
        )

        # Should limit requirements to 5
        assert len(spec.requirements) <= 5

    def test_generate_spec_with_no_spec_mapping(self, generator, sample_failures):
        """Test with recommendations that have no SPEC mapping."""
        recommendations = [
            Recommendation(
                id="REC-NONE",
                title="No Mapping",
                description="",
                priority=Priority.LOW,
                failure_types=[FailureType.LOW_QUALITY],
                actions=["Action"],
                spec_mapping=None,
            )
        ]

        spec = generator.generate_spec(
            failures=sample_failures,
            recommendations=recommendations,
        )

        assert spec is not None
        assert spec.related_specs == []

    def test_ears_format_determination(self, generator):
        """Test EARS format determination from failure types."""
        # Event-driven type
        rec_event = Recommendation(
            id="REC-1",
            title="Test",
            description="",
            priority=Priority.HIGH,
            failure_types=[FailureType.MISSING_INFO],
            actions=[],
        )

        format_type = generator._determine_ears_format(rec_event)
        assert format_type == "Event-driven"

        # Unwanted type
        rec_unwanted = Recommendation(
            id="REC-2",
            title="Test",
            description="",
            priority=Priority.HIGH,
            failure_types=[FailureType.HALLUCINATION],
            actions=[],
        )

        format_type = generator._determine_ears_format(rec_unwanted)
        assert format_type == "Unwanted"
