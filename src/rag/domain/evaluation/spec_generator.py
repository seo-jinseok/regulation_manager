"""
SPEC Generator for RAG Evaluation Improvements.

Auto-generates SPEC documents from failure patterns and recommendations.
Part of SPEC-RAG-EVAL-001 Milestone 4: Report Enhancement.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .failure_classifier import FailureSummary, FailureType
from .recommendation_engine import Recommendation, Priority

logger = logging.getLogger(__name__)


@dataclass
class SPECDocument:
    """A generated SPEC document."""

    spec_id: str
    title: str
    status: str
    created_at: str
    priority: str
    description: str
    requirements: List[Dict[str, Any]]
    acceptance_criteria: List[str]
    technical_approach: List[str]
    related_specs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"# {self.spec_id}: {self.title}",
            "",
            f"**Status**: {self.status}",
            f"**Priority**: {self.priority}",
            f"**Created**: {self.created_at}",
            "",
            "## Description",
            "",
            self.description,
            "",
            "## Requirements",
            "",
        ]

        for i, req in enumerate(self.requirements, 1):
            lines.append(f"### REQ-{i}: {req.get('title', 'Untitled')}")
            lines.append("")
            lines.append(f"**Type**: {req.get('type', 'Functional')}")
            lines.append(f"**Format**: {req.get('format', 'Ubiquitous')}")
            lines.append("")
            lines.append(req.get('description', ''))
            lines.append("")

        lines.extend([
            "## Acceptance Criteria",
            "",
        ])

        for criteria in self.acceptance_criteria:
            lines.append(f"- [ ] {criteria}")

        lines.extend([
            "",
            "## Technical Approach",
            "",
        ])

        for approach in self.technical_approach:
            lines.append(f"- {approach}")

        if self.related_specs:
            lines.extend([
                "",
                "## Related SPECs",
                "",
            ])
            for spec in self.related_specs:
                lines.append(f"- {spec}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spec_id": self.spec_id,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "priority": self.priority,
            "description": self.description,
            "requirements": self.requirements,
            "acceptance_criteria": self.acceptance_criteria,
            "technical_approach": self.technical_approach,
            "related_specs": self.related_specs,
            "metadata": self.metadata,
        }


class SPECGenerator:
    """
    Generates SPEC documents from evaluation failures.

    Provides:
    - Automatic SPEC document creation from failure patterns
    - EARS format requirement generation
    - Template-based document structure
    - Markdown output for MoAI integration
    """

    # Templates for different failure types
    FAILURE_TEMPLATES = {
        FailureType.HALLUCINATION: {
            "title": "Prevent Hallucination in RAG Responses",
            "description": (
                "The RAG system shall generate responses that are strictly grounded "
                "in retrieved context documents, avoiding fabrication of phone numbers, "
                "email addresses, URLs, or other specific details not present in source material."
            ),
            "ears_type": "Unwanted",
            "acceptance_criteria": [
                "No fabricated phone numbers in responses",
                "No fabricated email addresses in responses",
                "All specific claims traceable to source documents",
                "Hallucination detection score > 0.95",
            ],
        },
        FailureType.MISSING_INFO: {
            "title": "Ensure Complete Information in Responses",
            "description": (
                "When the user asks about procedures, deadlines, or requirements, "
                "the system shall provide complete information including all relevant "
                "steps, timeframes, and necessary documentation."
            ),
            "ears_type": "Event-driven",
            "acceptance_criteria": [
                "Deadline queries return specific date/time information",
                "Procedure queries include all required steps",
                "Requirement queries list all necessary qualifications",
                "Completeness score > 0.85",
            ],
        },
        FailureType.CITATION_ERROR: {
            "title": "Enforce Proper Citation Format",
            "description": (
                "The system shall cite all regulation references using the standard "
                "format: regulation name in corner brackets followed by article number."
            ),
            "ears_type": "Ubiquitous",
            "acceptance_criteria": [
                "All regulation references use proper format",
                "Citations match actual document content",
                "No orphan citations without source document",
                "Citation format compliance > 0.95",
            ],
        },
        FailureType.RETRIEVAL_FAILURE: {
            "title": "Improve Document Retrieval Accuracy",
            "description": (
                "The retrieval system shall return documents that are highly relevant "
                "to the user query, with proper ranking and minimal irrelevant content."
            ),
            "ears_type": "Ubiquitous",
            "acceptance_criteria": [
                "Top-3 retrieved documents are relevant to query",
                "Contextual precision score > 0.80",
                "Contextual recall score > 0.80",
                "Irrelevant document rate < 20%",
            ],
        },
        FailureType.AMBIGUITY: {
            "title": "Reduce Ambiguous Responses",
            "description": (
                "The system shall provide specific, actionable responses rather than "
                "generic statements or deflections to contact departments."
            ),
            "ears_type": "Ubiquitous",
            "acceptance_criteria": [
                "No generic deflection phrases",
                "Specific information provided when available",
                "Clear indication when information is uncertain",
                "Ambiguity rate < 10%",
            ],
        },
        FailureType.IRRELEVANCE: {
            "title": "Improve Query-Response Alignment",
            "description": (
                "The system shall directly address the user's question without "
                "tangential information or off-topic content."
            ),
            "ears_type": "Event-driven",
            "acceptance_criteria": [
                "Answer directly addresses the question asked",
                "Answer relevancy score > 0.85",
                "No off-topic content in responses",
                "Query-response alignment verified",
            ],
        },
    }

    def __init__(self, template_dir: str = ".moai/templates/spec"):
        """
        Initialize the SPEC generator.

        Args:
            template_dir: Directory containing SPEC templates
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self._spec_counter = 0

    def generate_spec(
        self,
        failures: List[FailureSummary],
        recommendations: List[Recommendation],
        spec_prefix: str = "SPEC-RAG-Q",
    ) -> SPECDocument:
        """
        Generate a SPEC document from failure patterns.

        Args:
            failures: List of failure summaries
            recommendations: List of recommendations
            spec_prefix: Prefix for SPEC ID

        Returns:
            Generated SPECDocument
        """
        self._spec_counter += 1
        spec_id = f"{spec_prefix}-{self._spec_counter:03d}"

        # Determine primary failure type (most frequent)
        if failures:
            primary_failure = failures[0].failure_type
        else:
            primary_failure = FailureType.UNKNOWN

        # Get template for primary failure
        template = self.FAILURE_TEMPLATES.get(
            primary_failure,
            {
                "title": "Improve RAG System Quality",
                "description": "Address quality issues identified through evaluation.",
                "ears_type": "Ubiquitous",
                "acceptance_criteria": [
                    "Overall quality score > 0.85",
                    "Pass rate > 80%",
                ],
            },
        )

        # Build requirements from recommendations
        requirements = self._build_requirements(recommendations)

        # Build technical approach from recommendations' actions
        technical_approach = []
        for rec in recommendations[:3]:  # Top 3 recommendations
            technical_approach.extend(rec.actions[:2])  # Top 2 actions each

        # Determine priority
        if recommendations:
            highest_priority = min(
                recommendations,
                key=lambda r: {
                    Priority.CRITICAL: 0,
                    Priority.HIGH: 1,
                    Priority.MEDIUM: 2,
                    Priority.LOW: 3,
                }.get(r.priority, 99),
            )
            priority = highest_priority.priority.value.upper()
        else:
            priority = "MEDIUM"

        # Get related specs
        related_specs = list(set(
            rec.spec_mapping
            for rec in recommendations
            if rec.spec_mapping
        ))

        spec = SPECDocument(
            spec_id=spec_id,
            title=template["title"],
            status="Draft",
            created_at=datetime.now().isoformat(),
            priority=priority,
            description=template["description"],
            requirements=requirements,
            acceptance_criteria=template["acceptance_criteria"],
            technical_approach=technical_approach,
            related_specs=related_specs,
            metadata={
                "failure_types": [f.failure_type.value for f in failures],
                "total_failures": sum(f.count for f in failures),
                "recommendations_count": len(recommendations),
            },
        )

        logger.info(f"Generated SPEC: {spec_id}")
        return spec

    def _build_requirements(
        self,
        recommendations: List[Recommendation],
    ) -> List[Dict[str, Any]]:
        """Build EARS format requirements from recommendations."""
        requirements = []

        for i, rec in enumerate(recommendations[:5], 1):  # Max 5 requirements
            req = {
                "id": f"REQ-{i}",
                "title": rec.title,
                "type": "Functional",
                "format": self._determine_ears_format(rec),
                "description": rec.description,
                "actions": rec.actions,
            }
            requirements.append(req)

        return requirements

    def _determine_ears_format(self, recommendation: Recommendation) -> str:
        """Determine EARS format based on recommendation type."""
        for failure_type in recommendation.failure_types:
            if failure_type in self.FAILURE_TEMPLATES:
                return self.FAILURE_TEMPLATES[failure_type]["ears_type"]
        return "Ubiquitous"

    def save_spec(
        self,
        spec: SPECDocument,
        path: Optional[str] = None,
    ) -> str:
        """
        Save SPEC document to file.

        Args:
            spec: SPECDocument to save
            path: Optional file path (uses default if not provided)

        Returns:
            Path where SPEC was saved
        """
        if path is None:
            # Default path in .moai/specs directory
            spec_dir = Path(".moai/specs") / spec.spec_id
            spec_dir.mkdir(parents=True, exist_ok=True)
            path = str(spec_dir / "spec.md")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        content = spec.to_markdown()

        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved SPEC to: {path}")
        return str(path_obj)

    def generate_batch_specs(
        self,
        failures: List[FailureSummary],
        recommendations: List[Recommendation],
        max_specs: int = 3,
    ) -> List[SPECDocument]:
        """
        Generate multiple SPEC documents for different failure categories.

        Args:
            failures: List of failure summaries
            recommendations: List of recommendations
            max_specs: Maximum number of SPECs to generate

        Returns:
            List of generated SPECDocuments
        """
        specs = []

        # Group failures by type
        failures_by_type: Dict[FailureType, List[FailureSummary]] = {}
        for failure in failures:
            ft = failure.failure_type
            if ft not in failures_by_type:
                failures_by_type[ft] = []
            failures_by_type[ft].append(failure)

        # Filter recommendations by failure type
        for failure_type, type_failures in list(failures_by_type.items())[:max_specs]:
            type_recommendations = [
                r for r in recommendations
                if failure_type in r.failure_types
            ]

            spec = self.generate_spec(
                failures=type_failures,
                recommendations=type_recommendations,
            )
            specs.append(spec)

        logger.info(f"Generated {len(specs)} SPEC documents")
        return specs

    def create_spec_from_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
    ) -> SPECDocument:
        """
        Create a SPEC from a named template.

        Args:
            template_name: Name of the template to use
            variables: Variables to substitute in template

        Returns:
            Generated SPECDocument
        """
        # This would load from template files in a real implementation
        # For now, use the default generation
        return self.generate_spec(
            failures=[],
            recommendations=[],
        )
