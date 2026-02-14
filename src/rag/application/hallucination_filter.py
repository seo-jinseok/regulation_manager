"""
HallucinationFilter service for validating LLM responses against context.

This module implements SPEC-RAG-Q-002: Hallucination Prevention.
It validates that LLM responses are grounded in the retrieved context.

Key features:
- Contact information validation (phone, email) - REQ-001
- Department name validation - REQ-002
- Citation grounding validation - REQ-003
- Multiple filter modes (warn, sanitize, block, passthrough)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class FilterMode(Enum):
    """Filter operation mode."""

    WARN = "warn"  # Log warnings only
    SANITIZE = "sanitize"  # Auto-fix issues
    BLOCK = "block"  # Block response with issues
    PASSTHROUGH = "passthrough"  # No filtering


@dataclass
class FilterResult:
    """Result of hallucination filtering."""

    original_response: str
    sanitized_response: str
    is_modified: bool
    blocked: bool
    block_reason: Optional[str]
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class HallucinationFilter:
    """
    Service for validating LLM responses against context.

    This filter ensures that responses only contain information
    that can be verified against the retrieved context documents.

    Usage:
        filter_service = HallucinationFilter(mode=FilterMode.SANITIZE)
        result = filter_service.filter_response(response, context)
        if not result.blocked:
            return result.sanitized_response
    """

    # Phone number patterns (Korean format)
    PHONE_PATTERNS = [
        re.compile(r"\d{2,3}-\d{3,4}-\d{4}"),  # 02-1234-5678, 010-123-4567
        re.compile(r"(?<!\d)\d{10,11}(?!\d)"),  # 0212345678, 01012345678 (standalone)
        re.compile(r"\(\d{2,3}\)\s*\d{3,4}-\d{4}"),  # (02) 1234-5678
    ]

    # Email patterns
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    # Common department/team names pattern (Korean)
    # Matches: 학적팀, 장학팀, 교무처, 학술연구지원팀, etc.
    DEPARTMENT_PATTERN = re.compile(
        r"([가-힣]{1,20}(?:팀|부|처|실|원|센터|국|소))"
    )

    # Pattern for departments ending with "과" (e.g., 학생과)
    DEPARTMENT_WITH_GWA_PATTERN = re.compile(
        r"([가-힣]{1,20}과)"
    )

    # Citation patterns - extract just the article number part
    CITATION_PATTERNS = [
        re.compile(r"(학칙|규정|지침|령)\s*제\d+조"),
        re.compile(r"제\d+조"),
    ]

    # Article number extraction pattern for normalization
    ARTICLE_NUMBER_PATTERN = re.compile(r"제\d+조")

    # Fallback messages
    CONTACT_FALLBACK = "자세한 연락처는 해당 부서에 직접 문의해 주시기 바랍니다"
    DEPARTMENT_FALLBACK = "담당 부서"
    CITATION_FALLBACK = "관련 규정"

    def __init__(self, mode: FilterMode = FilterMode.SANITIZE):
        """
        Initialize the filter with specified mode.

        Args:
            mode: Filter operation mode (default: SANITIZE)
        """
        self.mode = mode

    def filter_response(self, response: str, context: List[str]) -> FilterResult:
        """
        Filter response by validating against context.

        Args:
            response: The LLM response to validate
            context: List of context documents from retrieval

        Returns:
            FilterResult with validation results and sanitized response
        """
        if self.mode == FilterMode.PASSTHROUGH:
            return FilterResult(
                original_response=response,
                sanitized_response=response,
                is_modified=False,
                blocked=False,
                block_reason=None,
                issues=[],
                warnings=[],
            )

        issues: List[str] = []
        warnings: List[str] = []
        sanitized = response

        # Validate contacts (phone and email)
        sanitized, contact_issues = self.validate_contacts(sanitized, context)
        issues.extend(contact_issues)
        if contact_issues:
            warnings.append(f"Contact validation issues: {len(contact_issues)}")

        # Validate department names
        sanitized, dept_issues = self.validate_departments(sanitized, context)
        issues.extend(dept_issues)
        if dept_issues:
            warnings.append(f"Department validation issues: {len(dept_issues)}")

        # Validate citations
        sanitized, citation_issues = self.validate_citations(sanitized, context)
        issues.extend(citation_issues)
        if citation_issues:
            warnings.append(f"Citation validation issues: {len(citation_issues)}")

        is_modified = sanitized != response

        # Handle different modes
        blocked = False
        block_reason = None

        if self.mode == FilterMode.WARN:
            # Just log warnings, don't modify
            for warning in warnings:
                logger.warning(f"Hallucination detected: {warning}")
            return FilterResult(
                original_response=response,
                sanitized_response=response,  # Unchanged in WARN mode
                is_modified=False,
                blocked=False,
                block_reason=None,
                issues=issues,
                warnings=warnings,
            )

        if self.mode == FilterMode.BLOCK and issues:
            blocked = True
            block_reason = f"Response contains unverified information: {'; '.join(issues[:3])}"

        return FilterResult(
            original_response=response,
            sanitized_response=sanitized,
            is_modified=is_modified,
            blocked=blocked,
            block_reason=block_reason,
            issues=issues,
            warnings=warnings,
        )

    def validate_contacts(
        self, response: str, context: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Validate phone numbers and emails against context.

        REQ-001: Contact Information Validation

        Args:
            response: The response to validate
            context: List of context documents

        Returns:
            Tuple of (sanitized_response, list_of_issues)
        """
        issues: List[str] = []
        sanitized = response
        context_text = " ".join(context)

        # Normalize context phone numbers for comparison
        context_phones = self._normalize_phones(context_text)
        context_emails = set(self.EMAIL_PATTERN.findall(context_text))

        # Check phone numbers
        for pattern in self.PHONE_PATTERNS:
            matches = list(pattern.finditer(sanitized))
            for match in reversed(matches):
                phone = match.group(0)
                normalized = self._normalize_phone(phone)

                if normalized not in context_phones:
                    issues.append(f"Unverified phone number: {phone}")
                    sanitized = (
                        sanitized[: match.start()]
                        + self.CONTACT_FALLBACK
                        + sanitized[match.end() :]
                    )

        # Check emails
        email_matches = list(self.EMAIL_PATTERN.finditer(sanitized))
        for match in reversed(email_matches):
            email = match.group(0)
            if email not in context_emails:
                issues.append(f"Unverified email: {email}")
                sanitized = (
                    sanitized[: match.start()]
                    + self.CONTACT_FALLBACK
                    + sanitized[match.end() :]
                )

        return sanitized, issues

    def validate_departments(
        self, response: str, context: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Validate department names against context.

        REQ-002: Department Name Validation

        Args:
            response: The response to validate
            context: List of context documents

        Returns:
            Tuple of (sanitized_response, list_of_issues)
        """
        issues: List[str] = []
        sanitized = response
        context_text = " ".join(context)

        # Extract department names from context
        context_depts = set()
        for match in self.DEPARTMENT_PATTERN.finditer(context_text):
            full_dept = match.group(1)
            context_depts.add(full_dept)
            context_depts.add(self._normalize_department(full_dept))

        # Also check for departments ending with "과"
        for match in self.DEPARTMENT_WITH_GWA_PATTERN.finditer(context_text):
            full_dept = match.group(1)
            context_depts.add(full_dept)
            context_depts.add(self._normalize_department(full_dept))

        # Collect all department matches with their positions
        # Use a dict to track which positions have been matched
        all_matches = {}  # (start, end) -> dept_name

        # Process DEPARTMENT_PATTERN first (higher priority)
        for match in self.DEPARTMENT_PATTERN.finditer(sanitized):
            start, end = match.start(), match.end()
            dept = match.group(1)
            all_matches[(start, end)] = dept

        # Process DEPARTMENT_WITH_GWA_PATTERN only if not overlapping
        for match in self.DEPARTMENT_WITH_GWA_PATTERN.finditer(sanitized):
            start, end = match.start(), match.end()
            dept = match.group(1)

            # Check if this overlaps with an existing match
            overlaps = False
            for (s, e) in all_matches:
                if not (end <= s or start >= e):  # Overlaps
                    overlaps = True
                    break

            if not overlaps:
                all_matches[(start, end)] = dept

        # Sort by start position descending
        sorted_matches = sorted(all_matches.items(), key=lambda x: x[0][0], reverse=True)

        for (start, end), dept in sorted_matches:
            normalized = self._normalize_department(dept)

            # Check both normalized and raw form
            if normalized not in context_depts and dept not in context_depts:
                issues.append(f"Unverified department: {dept}")
                sanitized = (
                    sanitized[:start]
                    + self.DEPARTMENT_FALLBACK
                    + sanitized[end:]
                )

        return sanitized, issues

    def validate_citations(
        self, response: str, context: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Validate article citations against context.

        REQ-003: Citation Grounding

        Args:
            response: The response to validate
            context: List of context documents

        Returns:
            Tuple of (sanitized_response, list_of_issues)
        """
        issues: List[str] = []
        sanitized = response
        context_text = " ".join(context)

        # Extract article numbers from context (e.g., "제10조" from "학칙 제10조: ...")
        context_article_numbers = set(self.ARTICLE_NUMBER_PATTERN.findall(context_text))

        # Check citations in response
        for pattern in self.CITATION_PATTERNS:
            matches = list(pattern.finditer(sanitized))
            for match in reversed(matches):
                citation = match.group(0)

                # Extract article number from the citation
                article_nums = self.ARTICLE_NUMBER_PATTERN.findall(citation)

                # Check if at least one article number exists in context
                is_verified = any(num in context_article_numbers for num in article_nums)

                if not is_verified:
                    issues.append(f"Unverified citation: {citation}")
                    sanitized = (
                        sanitized[: match.start()]
                        + self.CITATION_FALLBACK
                        + sanitized[match.end() :]
                    )

        return sanitized, issues

    def _normalize_phone(self, phone: str) -> str:
        """
        Normalize phone number for comparison.

        Removes all non-digit characters.
        """
        return re.sub(r"\D", "", phone)

    def _normalize_phones(self, text: str) -> set:
        """
        Extract and normalize all phone numbers from text.

        Returns:
            Set of normalized phone numbers
        """
        phones = set()
        for pattern in self.PHONE_PATTERNS:
            for match in pattern.finditer(text):
                phones.add(self._normalize_phone(match.group(0)))
        return phones

    def _normalize_department(self, dept: str) -> str:
        """
        Normalize department name for comparison.

        Removes spaces and converts to lowercase.
        """
        return dept.replace(" ", "").lower()
