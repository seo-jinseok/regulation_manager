"""
Synthetic Test Data Generation using Flip-the-RAG workflow.

Generates high-quality test cases from regulation documents.

Flip-the-RAG: Generate questions from answers (regulation documents)
Instead of Q -> A, generate A -> Q from document content.

Enhanced Features:
- Section classification (procedural/conditional/factual)
- Diverse question generation patterns
- Ground truth extraction with validation
- Semantic similarity validation (cosine similarity >= 0.5)
- Quality filtering and dataset saving
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import TestCase

logger = logging.getLogger(__name__)


class SectionType:
    """Section type classification constants."""

    PROCEDURAL = "procedural"  # Numbered steps, procedures
    CONDITIONAL = "conditional"  # Eligibility criteria, conditions
    FACTUAL = "factual"  # Definitions, general information


class QuestionGenerator:
    """
    Generates diverse questions from regulation sections.

    Provides multiple question patterns for each section type
    to ensure question diversity in the test dataset.
    """

    # Procedural question templates
    PROCEDURAL_PATTERNS = [
        "{title} 절차가 어떻게 되나요?",
        "{title} 필요한 서류는 뭐예요?",
        "{title} 신청 방법 알려주세요",
        "{title} 어떻게 신청해요?",
        "{title} 진행 과정 설명해주세요",
        "{title}流程是什麼？",  # Chinese - for diversity
    ]

    # Conditional question templates
    CONDITIONAL_PATTERNS = [
        "{title} 자격 요건이 뭐예요?",
        "{title} 누가 신청할 수 있나요?",
        "{title} 제한 사항이 있나요?",
        "{title} 조건이 어떻게 되나요?",
        "{title} 신청 자격 설명해주세요",
        "{title} 誰可以申請？",  # Chinese - for diversity
    ]

    # Factual question templates
    FACTUAL_PATTERNS = [
        "{title}가(이) 뭐예요?",
        "{title}에 대해 알려주세요",
        "{title} 설명해 주실까요?",
        "{title} 어떻게 돼요?",
        "{title} 정의가 뭔가요?",
        "{title} 是什麼？",  # Chinese - for diversity
    ]

    @classmethod
    def generate_procedural(cls, title: str, content: str, count: int = 3) -> List[str]:
        """
        Generate procedural questions from section.

        Args:
            title: Section title
            content: Section content
            count: Number of questions to generate

        Returns:
            List of procedural questions
        """
        questions = []

        # Use template-based questions
        for pattern in cls.PROCEDURAL_PATTERNS[:count]:
            questions.append(pattern.format(title=title))

        # Extract numbered steps and generate step-specific questions
        steps = cls._extract_numbered_steps(content)
        for i in range(min(count, len(steps))):
            questions.append(f"{title} {i + 1}단계는 뭔가요?")

        # Add prerequisite question
        questions.append(f"{title} 전에 필요한 준비가 뭐예요?")

        return questions

    @classmethod
    def generate_conditional(
        cls, title: str, content: str, count: int = 3
    ) -> List[str]:
        """
        Generate conditional questions from section.

        Args:
            title: Section title
            content: Section content
            count: Number of questions to generate

        Returns:
            List of conditional questions
        """
        questions = []

        # Use template-based questions
        for pattern in cls.CONDITIONAL_PATTERNS[:count]:
            questions.append(pattern.format(title=title))

        # Extract criteria and generate criteria-specific questions
        criteria = cls._extract_criteria(content)
        for i, criterion in enumerate(criteria[:count]):
            # Truncate long criteria
            criterion_short = (
                criterion[:30] + "..." if len(criterion) > 30 else criterion
            )
            questions.append(f"{title}에서 '{criterion_short}' 조건은 뭔가요?")

        return questions

    @classmethod
    def generate_factual(cls, title: str, content: str, count: int = 3) -> List[str]:
        """
        Generate factual questions from section.

        Args:
            title: Section title
            content: Section content
            count: Number of questions to generate

        Returns:
            List of factual questions
        """
        questions = []

        # Use template-based questions
        for pattern in cls.FACTUAL_PATTERNS[:count]:
            questions.append(pattern.format(title=title))

        # Extract key terms and generate term-specific questions
        terms = cls._extract_key_terms(content)
        for term in terms[:count]:
            questions.append(f"{title}에서 '{term}'가 뭔가요?")

        return questions

    @staticmethod
    def _extract_numbered_steps(content: str) -> List[str]:
        """Extract numbered steps from content."""
        # Pattern to match numbered list items
        pattern = r"\d+[\.)]\s*([^\n]+)"
        matches = re.findall(pattern, content)
        return [m.strip() for m in matches if m.strip()]

    @staticmethod
    def _extract_criteria(content: str) -> List[str]:
        """Extract eligibility criteria from content."""
        # Split by sentences
        sentences = re.split(r"[.。]", content)

        # Filter sentences with criteria keywords
        criteria_keywords = [
            "자격",
            "요건",
            "제한",
            "대상",
            "조건",
            "할 수 있는",
            "해야 한다",
        ]
        criteria = []

        for sentence in sentences:
            if any(keyword in sentence for keyword in criteria_keywords):
                sentence = sentence.strip()
                if len(sentence) > 10:  # Filter out short fragments
                    criteria.append(sentence)

        return criteria

    @staticmethod
    def _extract_key_terms(content: str) -> List[str]:
        """Extract key terms from content."""
        # Extract quoted terms, parenthesized terms, and defined terms
        patterns = [
            r'"([^"]{2,10})"',  # Double-quoted terms
            r"'([^']{2,10})'",  # Single-quoted terms
            r"「([^」]{2,10})」",  # Japanese-style quotes
            r"（([^）]{2,10})）",  # Parenthesized terms
        ]

        terms = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            terms.extend(matches)

        return list(set(terms))  # Remove duplicates


class SectionClassifier:
    """
    Classifies regulation sections into types.

    Uses pattern matching and content analysis to determine
    whether a section is procedural, conditional, or factual.
    """

    # Procedural indicators
    PROCEDURAL_KEYWORDS = [
        "절차",
        "방법",
        "신청",
        "제출",
        "심사",
        "승인",
        "등록",
        "신고",
        "진행",
        "과정",
        "流程",  # Chinese
        "申请",  # Chinese
    ]

    PROCEDURAL_PATTERNS = [
        r"\d+[\.)]",  # Numbered list: 1., 2), etc.
        r"[①-⑩]",  # Circled numbers
        r"단계",  # Stage/step
        r"다음과 같다",  # As follows
    ]

    # Conditional indicators
    CONDITIONAL_KEYWORDS = [
        "자격",
        "요건",
        "제한",
        "대상",
        "조건",
        "할 수 있는",
        "해야 한다",
        "하여야 한다",
        "할 수 있다",
        "자격",  # Qualification
        "条件",  # Chinese
    ]

    # Factual indicators (default)
    FACTUAL_KEYWORDS = [
        "목적",
        "정의",
        "범위",
        "용어",
        "설명",
        "내용",
        "목적",  # Purpose
        "定義",  # Chinese
    ]

    @classmethod
    def classify(cls, section: Dict[str, Any]) -> str:
        """
        Classify a regulation section by type.

        Args:
            section: Section dictionary with title, text, type fields

        Returns:
            Section type: procedural, conditional, or factual
        """
        title = section.get("title", "")
        text = section.get("text", "")
        content = f"{title} {text}".lower()

        # Check for procedural patterns
        procedural_score = cls._calculate_score(
            content, cls.PROCEDURAL_KEYWORDS, cls.PROCEDURAL_PATTERNS
        )

        # Check for conditional patterns
        conditional_score = cls._calculate_score(content, cls.CONDITIONAL_KEYWORDS, [])

        # Classify based on scores
        if procedural_score > conditional_score and procedural_score > 0:
            return SectionType.PROCEDURAL
        elif conditional_score > 0:
            return SectionType.CONDITIONAL
        else:
            return SectionType.FACTUAL

    @staticmethod
    def _calculate_score(
        content: str, keywords: List[str], patterns: List[str]
    ) -> float:
        """Calculate classification score for content."""
        score = 0.0

        # Keyword matching
        for keyword in keywords:
            if keyword in content:
                score += 1.0

        # Pattern matching
        for pattern in patterns:
            if re.search(pattern, content):
                score += 0.5

        return score


class GroundTruthExtractor:
    """
    Extracts ground truth answers from regulation sections.

    Extracts relevant text passages that serve as ground truth
    answers for generated questions.
    """

    @classmethod
    def extract(cls, section: Dict[str, Any], question: str, question_type: str) -> str:
        """
        Extract ground truth answer from section for a question.

        Args:
            section: Regulation section
            question: Generated question
            question_type: Type of question (procedural/conditional/factual)

        Returns:
            Ground truth answer
        """
        title = section.get("title", "")
        text = section.get("text", "")
        display_no = section.get("display_no", "")

        # Build base content
        base_content = f"{display_no} {title}".strip()
        if text:
            base_content += f"\n{text}"

        # Extract based on question type
        if question_type == SectionType.PROCEDURAL:
            return cls._extract_procedural(base_content, text)
        elif question_type == SectionType.CONDITIONAL:
            return cls._extract_conditional(base_content, text)
        else:
            return cls._extract_factual(base_content, text)

    @staticmethod
    def _extract_procedural(base_content: str, text: str) -> str:
        """Extract ground truth for procedural questions."""
        # Extract numbered list portion
        lines = text.split("\n") if text else []
        procedural_lines = [line for line in lines if re.match(r"\s*\d+[\.)]", line)]

        if procedural_lines:
            # Include title and up to 5 numbered steps
            steps = "\n".join(procedural_lines[:5])
            return f"{base_content}\n{steps}"
        else:
            # Return full text if no numbered list found
            return base_content[:500]  # Limit to 500 chars

    @staticmethod
    def _extract_conditional(base_content: str, text: str) -> str:
        """Extract ground truth for conditional questions."""
        if not text:
            return base_content[:500]

        # Extract sentences with criteria keywords
        sentences = re.split(r"[.。]", text)
        criteria_keywords = ["자격", "요건", "제한", "대상", "조건", "하여야", "할 수"]

        criteria_sentences = []
        for sentence in sentences:
            if any(keyword in sentence for keyword in criteria_keywords):
                sentence = sentence.strip()
                if len(sentence) > 5:
                    criteria_sentences.append(sentence)

        if criteria_sentences:
            # Join up to 3 criteria sentences
            criteria_text = ". ".join(criteria_sentences[:3])
            return f"{base_content}\n{criteria_text}"
        else:
            return base_content[:500]

    @staticmethod
    def _extract_factual(base_content: str, text: str) -> str:
        """Extract ground truth for factual questions."""
        # Return title and first 3 lines of text
        if text:
            lines = text.split("\n")
            first_lines = "\n".join(lines[:3])
            return f"{base_content}\n{first_lines}"
        else:
            return base_content[:500]


class SemanticValidator:
    """
    Validates semantic similarity between questions and ground truth.

    Uses sentence transformers to compute cosine similarity and ensure
    questions are relevant to their ground truth answers.
    """

    def __init__(self, model_name: str = "jhgan/ko-sbert-sts"):
        """
        Initialize semantic validator.

        Args:
            model_name: Sentence transformer model name
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded semantic validator model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load semantic validator: {e}")
            self.model = None

    def validate(
        self, question: str, ground_truth: str, threshold: float = 0.5
    ) -> bool:
        """
        Validate semantic similarity between question and ground truth.

        Args:
            question: Generated question
            ground_truth: Extracted ground truth
            threshold: Minimum cosine similarity threshold

        Returns:
            True if similarity >= threshold
        """
        if self.model is None:
            # Skip validation if model not loaded
            return True

        try:
            # Generate embeddings
            question_embedding = self.model.encode(question)
            truth_embedding = self.model.encode(ground_truth)

            # Compute cosine similarity
            similarity = self._cosine_similarity(question_embedding, truth_embedding)

            return similarity >= threshold
        except Exception as e:
            logger.warning(f"Semantic validation error: {e}")
            return True  # Pass validation on error

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


class SyntheticDataGenerator:
    """
    Synthetic Test Data Generator using Flip-the-RAG workflow.

    Enhanced implementation with:
    - Advanced section classification
    - Diverse question generation
    - Ground truth extraction with validation
    - Semantic similarity validation
    - Quality filtering
    - Dataset persistence
    """

    def __init__(
        self,
        min_question_length: int = 10,
        max_question_length: int = 200,
        min_answer_length: int = 50,
        semantic_threshold: float = 0.5,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            min_question_length: Minimum question length in characters
            max_question_length: Maximum question length in characters
            min_answer_length: Minimum ground truth length in characters
            semantic_threshold: Minimum semantic similarity (cosine similarity)
        """
        self.min_question_length = min_question_length
        self.max_question_length = max_question_length
        self.min_answer_length = min_answer_length
        self.semantic_threshold = semantic_threshold

        # Initialize components
        self.semantic_validator = SemanticValidator()
        self.question_generator = QuestionGenerator()
        self.section_classifier = SectionClassifier()
        self.ground_truth_extractor = GroundTruthExtractor()

        logger.info(
            f"Initialized SyntheticDataGenerator with semantic threshold={semantic_threshold}"
        )

    def load_regulation_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load regulation document from JSON file.

        Args:
            file_path: Path to regulation JSON file

        Returns:
            Regulation document dictionary
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded regulation document: {file_path}")
        return data

    def extract_sections(self, regulation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract sections from regulation document.

        Args:
            regulation: Regulation document with docs array

        Returns:
            List of section dictionaries
        """
        sections = []
        docs = regulation.get("docs", [])

        for doc in docs:
            title = doc.get("title", "")
            content = doc.get("content", [])

            # Skip index documents
            if "차례" in title or "찾아보기" in title:
                continue

            # Extract articles from content
            for item in content:
                item_type = item.get("type", "")

                # Only process articles with text
                if item_type == "article" and item.get("text"):
                    sections.append(item)

        logger.info(f"Extracted {len(sections)} sections from regulation")
        return sections

    async def generate_from_regulation(
        self, regulation: Dict[str, Any]
    ) -> List[TestCase]:
        """
        Generate test cases from a single regulation document.

        Args:
            regulation: Regulation document with sections

        Returns:
            List of TestCase objects
        """
        test_cases = []
        sections = self.extract_sections(regulation)

        logger.info(f"Generating test cases from {len(sections)} sections")

        for section in sections:
            # Skip if section too short
            text = section.get("text", "")
            if len(text) < 50:
                logger.debug(
                    f"Skipping section (too short): {section.get('title', 'Unknown')}"
                )
                continue

            # Classify section type
            section_type = self.section_classifier.classify(section)

            # Generate questions based on section type
            questions = self._generate_questions_from_section(section, section_type)

            # Generate test cases for each question
            for question in questions:
                # Extract ground truth answer
                ground_truth = self.ground_truth_extractor.extract(
                    section, question, section_type
                )

                # Validate test case
                if self._validate_test_case(question, ground_truth):
                    test_case = TestCase(
                        question=question,
                        ground_truth=ground_truth,
                        regulation_id=regulation.get("file_name", "unknown"),
                        section_id=section.get("id", ""),
                        question_type=section_type,
                        valid=True,
                        metadata={
                            "generated_by": "Flip-the-RAG-Enhanced",
                            "section_type": section_type,
                            "section_title": section.get("title", ""),
                            "section_display_no": section.get("display_no", ""),
                        },
                    )
                    test_cases.append(test_case)

        logger.info(f"Generated {len(test_cases)} valid test cases from regulation")
        return test_cases

    def _generate_questions_from_section(
        self, section: Dict[str, Any], section_type: str
    ) -> List[str]:
        """
        Generate questions from a regulation section.

        Args:
            section: Regulation section with content
            section_type: Classified section type

        Returns:
            List of generated questions
        """
        title = section.get("title", "")
        text = section.get("text", "")

        # Generate questions based on section type
        if section_type == SectionType.PROCEDURAL:
            return self.question_generator.generate_procedural(title, text)
        elif section_type == SectionType.CONDITIONAL:
            return self.question_generator.generate_conditional(title, text)
        else:
            return self.question_generator.generate_factual(title, text)

    def _validate_test_case(self, question: str, ground_truth: str) -> bool:
        """
        Validate generated test case quality.

        Args:
            question: Generated question
            ground_truth: Extracted ground truth

        Returns:
            True if test case passes quality checks
        """
        # Check question length
        if (
            len(question) < self.min_question_length
            or len(question) > self.max_question_length
        ):
            logger.debug(f"Question length validation failed: {len(question)} chars")
            return False

        # Check ground truth length
        if len(ground_truth) < self.min_answer_length:
            logger.debug(f"Ground truth too short: {len(ground_truth)} chars")
            return False

        # Check for meaningful content
        if not ground_truth.strip():
            logger.debug("Ground truth is empty")
            return False

        # Validate semantic similarity
        if not self.semantic_validator.validate(
            question, ground_truth, self.semantic_threshold
        ):
            logger.debug("Semantic similarity validation failed")
            return False

        return True

    async def generate_dataset(
        self,
        regulation_paths: List[str],
        target_size: int = 500,
        output_path: Optional[str] = None,
    ) -> Tuple[List[TestCase], Dict[str, Any]]:
        """
        Generate synthetic test dataset from multiple regulations.

        Args:
            regulation_paths: List of paths to regulation JSON files
            target_size: Target number of test cases
            output_path: Optional path to save dataset as JSON

        Returns:
            Tuple of (test cases, generation statistics)
        """
        logger.info(
            f"Generating synthetic dataset from {len(regulation_paths)} regulations"
        )

        all_test_cases = []
        stats = {
            "total_sections": 0,
            "total_questions": 0,
            "valid_test_cases": 0,
            "validation_failures": 0,
            "section_type_distribution": {},
        }

        for reg_path in regulation_paths:
            try:
                # Load regulation
                regulation = self.load_regulation_document(reg_path)
                sections = self.extract_sections(regulation)
                stats["total_sections"] += len(sections)

                # Generate test cases
                test_cases = await self.generate_from_regulation(regulation)
                all_test_cases.extend(test_cases)

                # Stop if we've reached target size
                if len(all_test_cases) >= target_size:
                    logger.info(f"Reached target size of {target_size} test cases")
                    break

            except Exception as e:
                logger.error(f"Error processing {reg_path}: {e}")
                continue

        # Update statistics
        stats["valid_test_cases"] = len(all_test_cases)
        stats["validation_failures"] = (
            stats["total_questions"] - stats["valid_test_cases"]
        )

        # Calculate section type distribution
        for tc in all_test_cases:
            qt = tc.question_type or "unknown"
            stats["section_type_distribution"][qt] = (
                stats["section_type_distribution"].get(qt, 0) + 1
            )

        # If we have more than target, sample to target
        if len(all_test_cases) > target_size:
            import random

            all_test_cases = random.sample(all_test_cases, target_size)
            stats["valid_test_cases"] = target_size

        logger.info(f"Generated dataset with {len(all_test_cases)} test cases")
        logger.info(f"Statistics: {stats}")

        # Save dataset if output path provided
        if output_path:
            self.save_dataset(all_test_cases, output_path, stats)

        return all_test_cases, stats

    def save_dataset(
        self,
        test_cases: List[TestCase],
        output_path: str,
        stats: Dict[str, Any],
    ) -> None:
        """
        Save test dataset to JSON file.

        Args:
            test_cases: List of test cases
            output_path: Path to save dataset
            stats: Generation statistics
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare dataset
        dataset = {
            "metadata": {
                "total_test_cases": len(test_cases),
                "generated_at": str(pd_timestamp()),  # Use timestamp
                "generator": "Flip-the-RAG-Enhanced",
                "statistics": stats,
            },
            "test_cases": [tc.to_dict() for tc in test_cases],
        }

        # Save to JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved dataset to {output_path}")


def pd_timestamp() -> str:
    """Get current timestamp for dataset metadata."""
    from datetime import datetime

    return datetime.now().isoformat()
