"""
Synthetic Test Data Generation using Flip-the-RAG workflow.

Generates high-quality test cases from regulation documents.

Clean Architecture: Domain layer contains generation logic and validation rules.
"""

import logging
import re
from typing import Dict, List

from .models import TestCase

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Synthetic Test Data Generator using Flip-the-RAG workflow.

    Flip-the-RAG: Generate questions from answers (regulation documents)
    Instead of Q -> A, generate A -> Q from document content.
    """

    def __init__(self, min_question_length: int = 10, max_question_length: int = 200):
        """
        Initialize the synthetic data generator.

        Args:
            min_question_length: Minimum question length in characters
            max_question_length: Maximum question length in characters
        """
        self.min_question_length = min_question_length
        self.max_question_length = max_question_length
        logger.info("Initialized SyntheticDataGenerator with Flip-the-RAG workflow")

    async def generate_from_regulation(
        self,
        regulation: Dict[str, any],
    ) -> List[TestCase]:
        """
        Generate test cases from a single regulation document.

        Args:
            regulation: Regulation document with sections

        Returns:
            List of TestCase objects
        """
        test_cases = []
        sections = regulation.get("sections", [])

        logger.info(
            f"Generating test cases from regulation with {len(sections)} sections"
        )

        for section in sections:
            # Skip if section too short
            content = section.get("content", "")
            if len(content) < 100:
                logger.debug(
                    f"Skipping section (too short): {section.get('title', 'Unknown')}"
                )
                continue

            # Generate questions based on section type
            questions = self._generate_questions_from_section(section)

            # Create test cases for each question
            for question in questions:
                # Extract ground truth answer
                ground_truth = self._extract_answer(section, question)

                # Validate test case
                if self._validate_test_case(question, ground_truth):
                    test_case = TestCase(
                        question=question,
                        ground_truth=ground_truth,
                        regulation_id=regulation.get("id"),
                        section_id=section.get("id"),
                        question_type=self._classify_question_type(question),
                        valid=True,
                        metadata={
                            "generated_by": "Flip-the-RAG",
                            "section_type": section.get("type", "unknown"),
                            "section_title": section.get("title", ""),
                        },
                    )
                    test_cases.append(test_case)

        logger.info(f"Generated {len(test_cases)} valid test cases from regulation")
        return test_cases

    def _generate_questions_from_section(self, section: Dict[str, any]) -> List[str]:
        """
        Generate questions from a regulation section.

        Args:
            section: Regulation section with content

        Returns:
            List of generated questions
        """
        content = section.get("content", "")
        title = section.get("title", "")

        questions = []

        # Detect section type and generate appropriate questions
        if self._has_numbered_list(content):
            questions.extend(self._generate_procedural_questions(title, content))
        elif self._has_eligibility_criteria(content):
            questions.extend(self._generate_conditional_questions(title, content))
        else:
            questions.extend(self._generate_factual_questions(title, content))

        return questions

    def _has_numbered_list(self, content: str) -> bool:
        """Check if content contains numbered list (procedural steps)."""
        # Look for patterns like "1.", "2)", "①"
        patterns = [r"\d+\.", r"\d+\)", r"[①-⑭]"]
        return any(re.search(pattern, content) for pattern in patterns)

    def _has_eligibility_criteria(self, content: str) -> bool:
        """Check if content contains eligibility criteria."""
        keywords = ["자격", "요건", "제한", "대상", "조건", "할 수 있는", "해야 한다"]
        return any(keyword in content for keyword in keywords)

    def _generate_procedural_questions(self, title: str, content: str) -> List[str]:
        """
        Generate questions for procedural content (numbered lists).

        Args:
            title: Section title
            content: Section content

        Returns:
            List of procedural questions
        """
        questions = [
            f"{title} 절차가 어떻게 되나요?",
            f"{title} 필요한 서류는 뭐예요?",
            f"{title} 신청 방법 알려주세요",
            f"{title} 어떻게 신청해요?",
        ]

        # Extract numbered steps and generate step-specific questions
        steps = self._extract_numbered_steps(content)
        for i in range(min(3, len(steps))):  # Max 3 step-specific questions
            questions.append(f"{title} {i + 1}단계는 뭔가요?")

        # Add prerequisite question
        questions.append(f"{title} 전에 필요한 준비가 뭐예요?")

        return questions

    def _generate_conditional_questions(self, title: str, content: str) -> List[str]:
        """
        Generate questions for eligibility criteria content.

        Args:
            title: Section title
            content: Section content

        Returns:
            List of conditional questions
        """
        return [
            f"{title} 자격 요건이 뭐예요?",
            f"{title} 누가 신청할 수 있나요?",
            f"{title} 제한 사항이 있나요?",
            f"{title} 조건이 어떻게 되나요?",
        ]

    def _generate_factual_questions(self, title: str, content: str) -> List[str]:
        """
        Generate questions for factual content.

        Args:
            title: Section title
            content: Section content

        Returns:
            List of factual questions
        """
        return [
            f"{title}가(이) 뭐예요?",
            f"{title}에 대해 알려주세요",
            f"{title} 설명해 주실까요?",
            f"{title} 어떻게 돼요?",
        ]

    def _extract_numbered_steps(self, content: str) -> List[str]:
        """
        Extract numbered steps from content.

        Args:
            content: Section content

        Returns:
            List of step descriptions
        """
        # Pattern to match numbered list items
        pattern = r"\d+[\.)]\s*([^\n]+)"
        matches = re.findall(pattern, content)
        return matches

    def _extract_answer(self, section: Dict[str, any], question: str) -> str:
        """
        Extract ground truth answer from section for a question.

        Args:
            section: Regulation section
            question: Generated question

        Returns:
            Ground truth answer
        """
        content = section.get("content", "")
        title = section.get("title", "")

        # For procedural questions, include the numbered list
        if "절차" in question or "방법" in question:
            # Extract numbered list portion
            lines = content.split("\n")
            procedural_lines = [line for line in lines if re.match(r"\d+[\.)]", line)]
            if procedural_lines:
                return f"{title}\n" + "\n".join(procedural_lines[:5])  # Max 5 steps

        # For conditional questions, extract criteria
        if "자격" in question or "조건" in question:
            # Extract sentences with criteria keywords
            sentences = content.split(". ")
            criteria_sentences = [
                s
                for s in sentences
                if any(kw in s for kw in ["자격", "요건", "제한", "대상", "조건"])
            ]
            if criteria_sentences:
                return ". ".join(criteria_sentences[:3])  # Max 3 sentences

        # Default: Return first portion of content
        lines = content.split("\n")
        return f"{title}\n" + "\n".join(lines[:3])  # Title + first 3 lines

    def _classify_question_type(self, question: str) -> str:
        """
        Classify question type.

        Args:
            question: Question text

        Returns:
            Question type: "procedural", "conditional", or "factual"
        """
        if any(keyword in question for keyword in ["절차", "방법", "어떻게", "신청"]):
            return "procedural"
        elif any(keyword in question for keyword in ["자격", "조건", "제한", "대상"]):
            return "conditional"
        else:
            return "factual"

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
        if len(ground_truth) < 50:
            logger.debug(f"Ground truth too short: {len(ground_truth)} chars")
            return False

        # Check for meaningful content
        if not ground_truth.strip():
            logger.debug("Ground truth is empty")
            return False

        return True

    async def generate_dataset(
        self,
        regulations: List[Dict[str, any]],
        target_size: int = 500,
    ) -> List[TestCase]:
        """
        Generate synthetic test dataset from multiple regulations.

        Args:
            regulations: List of regulation documents
            target_size: Target number of test cases

        Returns:
            List of TestCase objects
        """
        logger.info(f"Generating synthetic dataset from {len(regulations)} regulations")

        all_test_cases = []

        for regulation in regulations:
            test_cases = await self.generate_from_regulation(regulation)
            all_test_cases.extend(test_cases)

            # Stop if we've reached target size
            if len(all_test_cases) >= target_size:
                logger.info(f"Reached target size of {target_size} test cases")
                break

        # If we have more than target, sample to target
        if len(all_test_cases) > target_size:
            import random

            all_test_cases = random.sample(all_test_cases, target_size)

        logger.info(f"Generated dataset with {len(all_test_cases)} test cases")

        # Log statistics
        question_types = {}
        for tc in all_test_cases:
            qt = tc.question_type or "unknown"
            question_types[qt] = question_types.get(qt, 0) + 1

        logger.info(f"Question type distribution: {question_types}")

        return all_test_cases
