"""
Unit tests for Flip-the-RAG synthetic data generation.

Tests the enhanced synthetic data generation system including:
- Section classification
- Question generation patterns
- Ground truth extraction
- Semantic validation
- Quality filtering
- Dataset generation and persistence
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.domain.evaluation.models import TestCase
from src.rag.domain.evaluation.synthetic_data import (
    GroundTruthExtractor,
    QuestionGenerator,
    SectionClassifier,
    SectionType,
    SemanticValidator,
    SyntheticDataGenerator,
    pd_timestamp,
)


class TestQuestionGenerator:
    """Test QuestionGenerator class."""

    def test_generate_procedural_questions(self):
        """Test procedural question generation."""
        title = "휴학 신청"
        content = "1. 서류 제출\n2. 학과장 승인\n3. 교무처 등록"

        questions = QuestionGenerator.generate_procedural(title, content, count=3)

        assert len(questions) > 0
        assert any("절차" in q for q in questions)
        assert any("방법" in q or "신청" in q for q in questions)
        assert any("단계" in q for q in questions)

    def test_generate_conditional_questions(self):
        """Test conditional question generation."""
        title = "장학금 신청"
        content = "자격: 학점 3.0 이상, 성실한 학생"

        questions = QuestionGenerator.generate_conditional(title, content, count=3)

        assert len(questions) > 0
        assert any("자격" in q or "요건" in q for q in questions)
        assert any("제한" in q or "조건" in q for q in questions)

    def test_generate_factual_questions(self):
        """Test factual question generation."""
        title = "학칙"
        content = "본교의 교육 목표와 학사 운영에 관한 규정"

        questions = QuestionGenerator.generate_factual(title, content, count=3)

        assert len(questions) > 0
        assert any("뭐예요" in q or "알려주세요" in q for q in questions)
        assert any("설명" in q or "정의" in q for q in questions)

    def test_extract_numbered_steps(self):
        """Test extraction of numbered steps from content."""
        content = "1. 신청서 작성\n2. 서류 제출\n3. 심사"

        steps = QuestionGenerator._extract_numbered_steps(content)

        assert len(steps) == 3
        assert "신청서 작성" in steps[0]
        assert "서류 제출" in steps[1]
        assert "심사" in steps[2]

    def test_extract_criteria(self):
        """Test extraction of eligibility criteria."""
        content = (
            "자격 요건은 학점 3.0 이상이어야 한다. "
            "제한 사항으로는 중복 지원이 불가능하다. "
            "대상은 2학년 이상 학생이다."
        )

        criteria = QuestionGenerator._extract_criteria(content)

        assert len(criteria) >= 2
        assert any("자격" in c or "요건" in c for c in criteria)

    def test_extract_key_terms(self):
        """Test extraction of key terms."""
        content = '"학점"과 "평점"은 서로 다른 개념이다'

        terms = QuestionGenerator._extract_key_terms(content)

        assert len(terms) > 0


class TestSectionClassifier:
    """Test SectionClassifier class."""

    def test_classify_procedural_section(self):
        """Test classification of procedural sections."""
        section = {
            "title": "휴학 절차",
            "text": "1. 신청서 제출\n2. 승인\n3. 등록",
            "type": "article",
        }

        section_type = SectionClassifier.classify(section)

        assert section_type == SectionType.PROCEDURAL

    def test_classify_conditional_section(self):
        """Test classification of conditional sections."""
        section = {
            "title": "장학금 자격",
            "text": "자격 요건: 학점 3.0 이상인 학생",
            "type": "article",
        }

        section_type = SectionClassifier.classify(section)

        assert section_type == SectionType.CONDITIONAL

    def test_classify_factual_section(self):
        """Test classification of factual sections."""
        section = {
            "title": "학칙의 목적",
            "text": "본 규정은 학사 운영에 관한 사항을 정함을 목적으로 한다",
            "type": "article",
        }

        section_type = SectionClassifier.classify(section)

        assert section_type == SectionType.FACTUAL

    def test_calculate_score(self):
        """Test score calculation for classification."""
        content = "신청 절차는 다음과 같다. 1. 서류 작성"

        score = SectionClassifier._calculate_score(
            content, ["절차", "신청"], [r"\d+[\.)]"]
        )

        assert score > 0


class TestGroundTruthExtractor:
    """Test GroundTruthExtractor class."""

    def test_extract_procedural_ground_truth(self):
        """Test ground truth extraction for procedural questions."""
        section = {
            "title": "휴학 신청",
            "text": "1. 신청서 작성\n2. 학과장 승인\n3. 교무처 제출",
            "display_no": "제5조",
        }

        ground_truth = GroundTruthExtractor.extract(
            section, "휴학 신청 절차가 어떻게 되나요?", SectionType.PROCEDURAL
        )

        assert len(ground_truth) > 0
        assert "휴학 신청" in ground_truth

    def test_extract_conditional_ground_truth(self):
        """Test ground truth extraction for conditional questions."""
        section = {
            "title": "장학금 자격",
            "text": "자격 요건은 학점 3.0 이상이어야 한다",
            "display_no": "제10조",
        }

        ground_truth = GroundTruthExtractor.extract(
            section, "장학금 자격 자격 요건이 뭐예요?", SectionType.CONDITIONAL
        )

        assert len(ground_truth) > 0
        assert "장학금" in ground_truth

    def test_extract_factual_ground_truth(self):
        """Test ground truth extraction for factual questions."""
        section = {
            "title": "학칙 목적",
            "text": "본 규정은 학사 운영에 관한 사항을 정함",
            "display_no": "제1조",
        }

        ground_truth = GroundTruthExtractor.extract(
            section, "학칙 목적가(이) 뭐예요?", SectionType.FACTUAL
        )

        assert len(ground_truth) > 0


class TestSemanticValidator:
    """Test SemanticValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a mock semantic validator."""
        return SemanticValidator()

    def test_init_with_valid_model(self):
        """Test initialization with a valid model name."""
        with patch("src.rag.domain.evaluation.synthetic_data.SentenceTransformer"):
            validator = SemanticValidator(model_name="test-model")
            assert validator is not None

    def test_init_with_invalid_model(self):
        """Test initialization with invalid model (should gracefully degrade)."""
        with patch(
            "src.rag.domain.evaluation.synthetic_data.SentenceTransformer",
            side_effect=Exception("Model load failed"),
        ):
            validator = SemanticValidator(model_name="invalid-model")
            assert validator.model is None

    def test_validate_with_model(self):
        """Test validation with model loaded."""
        validator = SemanticValidator()
        validator.model = MagicMock()

        # Mock embeddings
        validator.model.encode = MagicMock(return_value=np.array([1.0, 0.5, 0.3]))

        result = validator.validate(
            "휴학 절차가 어떻게 되나요?",
            "휴학 절차는 신청서 작성, 학과장 승인, 교무처 제출 순으로 진행됩니다",
            threshold=0.5,
        )

        assert result is True

    def test_validate_without_model(self):
        """Test validation when model is not loaded (should pass)."""
        validator = SemanticValidator()
        validator.model = None

        result = validator.validate(
            "휴학 절차가 어떻게 되나요?",
            "휴학 절차는 신청서 작성, 학과장 승인, 교무처 제출 순으로 진행됩니다",
            threshold=0.5,
        )

        assert result is True  # Should pass when model is not loaded

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 0.5, 0.3])
        vec2 = np.array([0.8, 0.6, 0.2])

        similarity = SemanticValidator._cosine_similarity(vec1, vec2)

        assert 0 <= similarity <= 1
        assert similarity > 0.9  # Should be high for similar vectors


class TestSyntheticDataGenerator:
    """Test SyntheticDataGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        return SyntheticDataGenerator(
            min_question_length=10,
            max_question_length=200,
            min_answer_length=50,
            semantic_threshold=0.5,
        )

    @pytest.fixture
    def sample_regulation(self):
        """Create a sample regulation document."""
        return {
            "file_name": "test_regulation.json",
            "docs": [
                {
                    "title": "휴학 규정",
                    "content": [
                        {
                            "id": "1",
                            "type": "article",
                            "display_no": "제5조",
                            "title": "휴학 절차",
                            "text": "휴학 절차는 다음과 같습니다.\n1. 휴학 신청서 작성\n2. 학과장 승인\n3. 교무처 제출\n4. 총장 승인",
                        },
                        {
                            "id": "2",
                            "type": "article",
                            "display_no": "제6조",
                            "title": "휴학 자격",
                            "text": "휴학 자격 요건은 다음과 같습니다. 성적이 우수해야 하며",
                        },
                    ],
                }
            ],
        }

    def test_init(self, generator):
        """Test generator initialization."""
        assert generator.min_question_length == 10
        assert generator.max_question_length == 200
        assert generator.semantic_threshold == 0.5
        assert generator.semantic_validator is not None
        assert generator.question_generator is not None
        assert generator.section_classifier is not None
        assert generator.ground_truth_extractor is not None

    def test_load_regulation_document(self, generator, tmp_path):
        """Test loading regulation document from file."""
        # Create a test file
        test_file = tmp_path / "test_regulation.json"
        test_data = {"file_name": "test.json", "docs": []}
        test_file.write_text(
            json.dumps(test_data, ensure_ascii=False), encoding="utf-8"
        )

        regulation = generator.load_regulation_document(str(test_file))

        assert regulation["file_name"] == "test.json"

    def test_extract_sections(self, generator, sample_regulation):
        """Test section extraction from regulation."""
        sections = generator.extract_sections(sample_regulation)

        assert len(sections) == 2
        assert sections[0]["title"] == "휴학 절차"
        assert sections[1]["title"] == "휴학 자격"

    def test_extract_sections_skips_index(self, generator):
        """Test that index documents are skipped."""
        regulation = {
            "docs": [
                {
                    "title": "차례",
                    "content": [],
                },
                {
                    "title": "찾아보기",
                    "content": [],
                },
            ],
        }

        sections = generator.extract_sections(regulation)

        assert len(sections) == 0

    def test_validate_test_case_valid(self, generator):
        """Test validation of a valid test case."""
        # Mock semantic validator to always pass for this test
        generator.semantic_validator.validate = MagicMock(return_value=True)

        question = "휴학 절차가 어떻게 되나요?"
        ground_truth = (
            "휴학 절차는 다음과 같습니다. "
            "1. 휴학 신청서를 작성하여 제출합니다. "
            "2. 학과장의 승인을 받습니다. "
            "3. 교무처에 최종 제출합니다."
        )

        result = generator._validate_test_case(question, ground_truth)

        assert result is True

    def test_validate_test_case_question_too_short(self, generator):
        """Test validation fails for too short question."""
        question = "휴학?"
        ground_truth = "휴학 절차는 다음과 같습니다." * 10

        result = generator._validate_test_case(question, ground_truth)

        assert result is False

    def test_validate_test_case_question_too_long(self, generator):
        """Test validation fails for too long question."""
        question = "휴학" * 100  # Very long
        ground_truth = "휴학 절차는 다음과 같습니다." * 10

        result = generator._validate_test_case(question, ground_truth)

        assert result is False

    def test_validate_test_case_ground_truth_too_short(self, generator):
        """Test validation fails for too short ground truth."""
        question = "휴학 절차가 어떻게 되나요?"
        ground_truth = "짧음"

        result = generator._validate_test_case(question, ground_truth)

        assert result is False

    def test_validate_test_case_empty_ground_truth(self, generator):
        """Test validation fails for empty ground truth."""
        question = "휴학 절차가 어떻게 되나요?"
        ground_truth = "   "

        result = generator._validate_test_case(question, ground_truth)

        assert result is False

    @pytest.mark.asyncio
    async def test_generate_from_regulation(self, generator, sample_regulation):
        """Test generating test cases from regulation."""
        # Mock semantic validation to always pass
        generator.semantic_validator.validate = MagicMock(return_value=True)

        test_cases = await generator.generate_from_regulation(sample_regulation)

        assert len(test_cases) > 0
        assert all(isinstance(tc, TestCase) for tc in test_cases)
        assert all(tc.valid for tc in test_cases)

    @pytest.mark.asyncio
    async def test_generate_from_regulation_skips_short_sections(self, generator):
        """Test that sections with short text are skipped."""
        regulation = {
            "file_name": "test.json",
            "docs": [
                {
                    "title": "짧은 규정",
                    "content": [
                        {
                            "id": "1",
                            "type": "article",
                            "title": "짧은 조항",
                            "text": "짧음",  # Less than 50 chars
                        }
                    ],
                }
            ],
        }

        # Mock semantic validation
        generator.semantic_validator.validate = MagicMock(return_value=True)

        test_cases = await generator.generate_from_regulation(regulation)

        assert len(test_cases) == 0

    @pytest.mark.asyncio
    async def test_generate_dataset(self, generator, sample_regulation, tmp_path):
        """Test generating a complete dataset."""
        # Create temporary regulation file
        reg_file = tmp_path / "regulation.json"
        reg_file.write_text(
            json.dumps(sample_regulation, ensure_ascii=False), encoding="utf-8"
        )

        output_file = tmp_path / "output.json"

        # Mock semantic validation
        generator.semantic_validator.validate = MagicMock(return_value=True)

        test_cases, stats = await generator.generate_dataset(
            regulation_paths=[str(reg_file)],
            target_size=10,
            output_path=str(output_file),
        )

        assert len(test_cases) >= 0
        assert "total_sections" in stats
        assert "valid_test_cases" in stats
        assert "section_type_distribution" in stats

        # Check output file was created
        assert output_file.exists()

        # Check output file contents
        with open(output_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        assert "metadata" in output_data
        assert "test_cases" in output_data

    def test_save_dataset(self, generator, tmp_path):
        """Test saving dataset to file."""
        test_cases = [
            TestCase(
                question="휴학 절차가 어떻게 되나요?",
                ground_truth="휴학 절차는 신청서 작성 후 승인을 받아야 합니다",
                question_type="procedural",
                valid=True,
            )
        ]

        stats = {
            "total_sections": 10,
            "valid_test_cases": 1,
            "section_type_distribution": {"procedural": 1},
        }

        output_file = tmp_path / "dataset.json"

        generator.save_dataset(test_cases, str(output_file), stats)

        assert output_file.exists()

        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["total_test_cases"] == 1
        assert len(data["test_cases"]) == 1
        assert data["test_cases"][0]["question"] == "휴학 절차가 어떻게 되나요?"

    @pytest.mark.asyncio
    async def test_generate_dataset_stops_at_target_size(
        self, generator, sample_regulation, tmp_path
    ):
        """Test that dataset generation stops at target size."""
        # Create a regulation that would generate many test cases
        reg_file = tmp_path / "regulation.json"
        reg_file.write_text(
            json.dumps(sample_regulation, ensure_ascii=False), encoding="utf-8"
        )

        # Mock to generate many test cases
        async def mock_generate(reg):
            # Return many test cases
            return [
                TestCase(
                    question=f"질문 {i}",
                    ground_truth=f"답변 {i}" * 20,
                    question_type="procedural",
                    valid=True,
                )
                for i in range(100)
            ]

        generator.generate_from_regulation = mock_generate

        output_file = tmp_path / "output.json"

        test_cases, stats = await generator.generate_dataset(
            regulation_paths=[str(reg_file)],
            target_size=10,
            output_path=str(output_file),
        )

        assert len(test_cases) == 10  # Should stop at target size


class TestHelperFunctions:
    """Test helper functions."""

    def test_pd_timestamp(self):
        """Test timestamp generation."""
        timestamp = pd_timestamp()

        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format contains T


@pytest.mark.integration
class TestIntegration:
    """Integration tests for synthetic data generation."""

    @pytest.mark.asyncio
    async def test_end_to_end_generation(self, tmp_path):
        """Test complete end-to-end generation workflow."""
        # Create sample regulation file
        regulation = {
            "file_name": "integration_test.json",
            "docs": [
                {
                    "title": "테스트 규정",
                    "content": [
                        {
                            "id": "1",
                            "type": "article",
                            "display_no": "제1조",
                            "title": "휴학 절차",
                            "text": (
                                "휴학 절차는 다음과 같습니다.\n"
                                "1. 휴학 신청서를 작성합니다.\n"
                                "2. 학과장의 승인을 받습니다.\n"
                                "3. 교무처에 제출합니다.\n"
                                "4. 총장의 승인을 받습니다."
                            ),
                        },
                        {
                            "id": "2",
                            "type": "article",
                            "display_no": "제2조",
                            "title": "휴학 자격",
                            "text": (
                                "휴학 자격 요건은 다음과 같습니다. "
                                "학점이 3.0 이상이어야 합니다. "
                                "등록금을 완납해야 합니다."
                            ),
                        },
                    ],
                }
            ],
        }

        reg_file = tmp_path / "regulation.json"
        reg_file.write_text(
            json.dumps(regulation, ensure_ascii=False), encoding="utf-8"
        )

        output_file = tmp_path / "dataset.json"

        # Create generator
        generator = SyntheticDataGenerator(
            min_question_length=10,
            max_question_length=200,
            min_answer_length=50,
            semantic_threshold=0.5,
        )

        # Mock semantic validation
        generator.semantic_validator.validate = MagicMock(return_value=True)

        # Generate dataset
        test_cases, stats = await generator.generate_dataset(
            regulation_paths=[str(reg_file)],
            target_size=20,
            output_path=str(output_file),
        )

        # Verify results
        assert len(test_cases) > 0
        assert stats["valid_test_cases"] > 0
        assert "procedural" in stats["section_type_distribution"]
        assert "conditional" in stats["section_type_distribution"]

        # Verify output file
        assert output_file.exists()
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["total_test_cases"] == len(test_cases)
        assert len(data["test_cases"]) == len(test_cases)

        # Verify test case structure
        for tc_data in data["test_cases"]:
            assert "question" in tc_data
            assert "ground_truth" in tc_data
            assert "question_type" in tc_data
            assert "regulation_id" in tc_data
            assert tc_data["valid"] is True
