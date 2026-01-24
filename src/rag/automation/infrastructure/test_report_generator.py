"""
Test Report Generator.

Infrastructure service for generating comprehensive test reports
in markdown format.

Clean Architecture: Infrastructure layer handles output formatting.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..domain.entities import MultiTurnScenario, TestResult

logger = logging.getLogger(__name__)


class TestReportGenerator:
    """
    Infrastructure service for generating test reports.

    Creates comprehensive markdown reports covering all aspects of
    automated RAG testing.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory for output reports (default: ./test_reports).
        """
        self.output_dir = output_dir or Path("./test_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_report(
        self,
        session_id: str,
        test_results: List["TestResult"],
        multi_turn_scenarios: Optional[List["MultiTurnScenario"]] = None,
        component_analyses: Optional[Dict[str, any]] = None,
        failure_analyses: Optional[Dict[str, any]] = None,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Generate a comprehensive test report.

        Args:
            session_id: Test session identifier.
            test_results: List of test results.
            multi_turn_scenarios: Optional multi-turn scenarios.
            component_analyses: Optional component analysis results.
            failure_analyses: Optional failure analysis results.
            metadata: Optional session metadata.

        Returns:
            Path to generated report file.
        """
        self.logger.info(f"Generating report for session: {session_id}")

        # Build report sections
        sections = []

        # 1. Summary
        sections.append(self._generate_summary(session_id, test_results, metadata))

        # 2. Test Results Overview
        sections.append(self._generate_test_overview(test_results))

        # 3. Detailed Test Results
        sections.append(self._generate_detailed_results(test_results))

        # 4. Multi-Turn Analysis
        if multi_turn_scenarios:
            sections.append(self._generate_multi_turn_analysis(multi_turn_scenarios))

        # 5. Component Analysis
        if component_analyses:
            sections.append(self._generate_component_analysis(component_analyses))

        # 6. Failure Analysis
        if failure_analyses:
            sections.append(self._generate_failure_analysis(failure_analyses))

        # 7. Quality Metrics
        sections.append(self._generate_quality_metrics(test_results))

        # 8. Recommendations
        sections.append(self._generate_recommendations(test_results, component_analyses, failure_analyses))

        # 9. Appendix
        sections.append(self._generate_appendix(metadata))

        # Combine sections
        report_content = "\n\n".join(sections)

        # Write to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"test_report_{session_id}_{timestamp}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        self.logger.info(f"Report generated: {report_path}")

        return report_path

    def _generate_summary(
        self,
        session_id: str,
        test_results: List["TestResult"],
        metadata: Optional[Dict],
    ) -> str:
        """Generate summary section."""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.passed)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        avg_confidence = sum(r.confidence for r in test_results) / total_tests if total_tests > 0 else 0
        avg_time = sum(r.execution_time_ms for r in test_results) / total_tests if total_tests > 0 else 0

        lines = [
            "# Test Summary",
            "",
            f"**Session ID:** {session_id}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overall Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tests | {total_tests} |",
            f"| Passed | {passed_tests} |",
            f"| Failed | {total_tests - passed_tests} |",
            f"| Pass Rate | {pass_rate:.1f}% |",
            f"| Avg Confidence | {avg_confidence:.2f} |",
            f"| Avg Execution Time | {avg_time:.0f}ms |",
            "",
        ]

        if metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        return "\n".join(lines)

    def _generate_test_overview(self, test_results: List["TestResult"]) -> str:
        """Generate test results overview section."""
        lines = [
            "# Test Results Overview",
            "",
            "## Results by Status",
            "",
        ]

        # Count by status
        passed = [r for r in test_results if r.passed]
        failed = [r for r in test_results if not r.passed]

        lines.extend([
            f"✅ **Passed:** {len(passed)}",
            f"❌ **Failed:** {len(failed)}",
            "",
        ])

        # Performance metrics
        times = [r.execution_time_ms for r in test_results]
        if times:
            lines.extend([
                "## Performance Metrics",
                "",
                f"- **Min Time:** {min(times)}ms",
                f"- **Max Time:** {max(times)}ms",
                f"- **Avg Time:** {sum(times) / len(times):.0f}ms",
                "",
            ])

        return "\n".join(lines)

    def _generate_detailed_results(self, test_results: List["TestResult"]) -> str:
        """Generate detailed test results section."""
        lines = [
            "# Detailed Test Results",
            "",
        ]

        for idx, result in enumerate(test_results, 1):
            status_icon = "✅" if result.passed else "❌"
            lines.extend([
                f"## {idx}. {result.test_case_id} {status_icon}",
                "",
                f"**Query:** {result.query}",
                "",
                f"**Answer:** {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}",
                "",
                f"**Confidence:** {result.confidence:.2f}",
                f"**Execution Time:** {result.execution_time_ms}ms",
                "",
            ])

            if result.sources:
                lines.extend([
                    "**Sources:**",
                    "",
                ])
                for source in result.sources[:5]:
                    lines.append(f"- {source}")
                lines.append("")

            if result.quality_score:
                qs = result.quality_score
                lines.extend([
                    "**Quality Score:**",
                    "",
                    f"- Total: {qs.total_score:.1f}/5.0",
                    f"- Accuracy: {qs.dimensions.accuracy:.2f}",
                    f"- Completeness: {qs.dimensions.completeness:.2f}",
                    f"- Relevance: {qs.dimensions.relevance:.2f}",
                    f"- Source Citation: {qs.dimensions.source_citation:.2f}",
                    "",
                ])

            if result.error_message:
                lines.extend([
                    "**Error:**",
                    "",
                    "```",
                    result.error_message,
                    "```",
                    "",
                ])

        return "\n".join(lines)

    def _generate_multi_turn_analysis(self, scenarios: List["MultiTurnScenario"]) -> str:
        """Generate multi-turn analysis section."""
        lines = [
            "# Multi-Turn Conversation Analysis",
            "",
        ]

        for scenario in scenarios:
            lines.extend([
                f"## Scenario: {scenario.scenario_id}",
                "",
                f"**Persona:** {scenario.persona_type.value}",
                f"**Total Turns:** {scenario.total_turns}",
                f"**Difficulty:** {scenario.difficulty.value}",
                f"**Context Preservation Rate:** {scenario.context_preservation_rate:.1%}",
                "",
            ])

            # Follow-up distribution
            if scenario.follow_up_distribution:
                lines.extend([
                    "**Follow-Up Question Distribution:**",
                    "",
                ])
                for follow_type, count in scenario.follow_up_distribution.items():
                    lines.append(f"- {follow_type.value}: {count}")
                lines.append("")

            # Turn details
            lines.extend([
                "**Turn Details:**",
                "",
            ])
            for turn in scenario.turns:
                context_icon = "✅" if turn.context_preserved else "❌"
                lines.extend([
                    f"### Turn {turn.turn_number} {context_icon}",
                    "",
                    f"**Query:** {turn.query}",
                    f"**Answer:** {turn.answer[:100]}...",
                    "",
                    f"- **Follow-up Type:** {turn.follow_up_type.value if turn.follow_up_type else 'N/A'}",
                    f"- **Confidence:** {turn.confidence:.2f}",
                    f"- **Context Preserved:** {turn.context_preserved}",
                    "",
                ])

        return "\n".join(lines)

    def _generate_component_analysis(self, analyses: Dict[str, any]) -> str:
        """Generate component analysis section."""
        lines = [
            "# RAG Component Analysis",
            "",
        ]

        for test_id, analysis in analyses.items():
            lines.extend([
                f"## Test: {test_id}",
                "",
                f"**Overall Impact:** {analysis.get('overall_impact', 'N/A')}",
                f"**Net Impact Score:** {analysis.get('net_impact_score', 0)}",
                "",
                "**Component Contributions:**",
                "",
            ])

            contributions = analysis.get("contributions", [])
            for contrib in contributions:
                component = contrib.get("component", "unknown")
                score = contrib.get("score", 0)
                reason = contrib.get("reason", "")
                executed = "✅" if contrib.get("was_executed", False) else "❌"

                score_icon = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"

                lines.append(f"- {score_icon} **{component}** ({executed}): {score} - {reason}")

            lines.append("")

        return "\n".join(lines)

    def _generate_failure_analysis(self, analyses: Dict[str, any]) -> str:
        """Generate failure analysis section."""
        lines = [
            "# Failure Analysis (5-Why)",
            "",
        ]

        for test_id, analysis in analyses.items():
            lines.extend([
                f"## Test: {test_id}",
                "",
                f"**Original Failure:** {analysis.get('original_failure', 'N/A')}",
                f"**Root Cause:** {analysis.get('root_cause', 'N/A')}",
                "",
                "**5-Why Chain:**",
                "",
            ])

            why_chain = analysis.get("why_chain", [])
            for idx, why in enumerate(why_chain, 1):
                lines.append(f"{idx}. Why? → {why}")

            lines.extend([
                "",
                "**Suggested Fix:**",
                "",
                analysis.get("suggested_fix", "No suggestion"),
                "",
                f"**Target:** {analysis.get('component_to_patch', 'N/A')}",
                f"**Code Change Required:** {analysis.get('code_change_required', False)}",
                "",
            ])

        return "\n".join(lines)

    def _generate_quality_metrics(self, test_results: List["TestResult"]) -> str:
        """Generate quality metrics section."""
        lines = [
            "# Quality Metrics",
            "",
            "## Dimension Breakdown",
            "",
        ]

        # Collect quality dimensions
        dimensions_list = []
        for result in test_results:
            if result.quality_score:
                dimensions_list.append(result.quality_score.dimensions)

        if dimensions_list:
            # Calculate averages
            avg_accuracy = sum(d.accuracy for d in dimensions_list) / len(dimensions_list)
            avg_completeness = sum(d.completeness for d in dimensions_list) / len(dimensions_list)
            avg_relevance = sum(d.relevance for d in dimensions_list) / len(dimensions_list)
            avg_citation = sum(d.source_citation for d in dimensions_list) / len(dimensions_list)
            avg_practicality = sum(d.practicality for d in dimensions_list) / len(dimensions_list)
            avg_actionability = sum(d.actionability for d in dimensions_list) / len(dimensions_list)

            lines.extend([
                "| Dimension | Average |",
                "|-----------|---------|",
                f"| Accuracy | {avg_accuracy:.2f} |",
                f"| Completeness | {avg_completeness:.2f} |",
                f"| Relevance | {avg_relevance:.2f} |",
                f"| Source Citation | {avg_citation:.2f} |",
                f"| Practicality | {avg_practicality:.2f} |",
                f"| Actionability | {avg_actionability:.2f} |",
                "",
            ])

        # Score distribution
        score_ranges = {
            "Excellent (4.5-5.0)": 0,
            "Good (4.0-4.4)": 0,
            "Partial (3.0-3.9)": 0,
            "Poor (<3.0)": 0,
        }

        for result in test_results:
            if result.quality_score:
                score = result.quality_score.total_score
                if score >= 4.5:
                    score_ranges["Excellent (4.5-5.0)"] += 1
                elif score >= 4.0:
                    score_ranges["Good (4.0-4.4)"] += 1
                elif score >= 3.0:
                    score_ranges["Partial (3.0-3.9)"] += 1
                else:
                    score_ranges["Poor (<3.0)"] += 1

        lines.extend([
            "**Score Distribution:**",
            "",
        ])
        for range_name, count in score_ranges.items():
            if count > 0:
                lines.append(f"- {range_name}: {count}")
        lines.append("")

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        test_results: List["TestResult"],
        component_analyses: Optional[Dict],
        failure_analyses: Optional[Dict],
    ) -> str:
        """Generate recommendations section."""
        lines = [
            "# Recommendations",
            "",
            "## Priority Actions",
            "",
        ]

        # High priority recommendations from failure analyses
        if failure_analyses:
            for test_id, analysis in failure_analyses.items():
                if analysis.get("code_change_required"):
                    lines.extend([
                        f"### 🚨 {test_id}",
                        "",
                        f"**Issue:** {analysis.get('root_cause', 'Unknown')}",
                        f"**Fix:** {analysis.get('suggested_fix', 'No suggestion')}",
                        "",
                    ])

        # Component improvement suggestions
        if component_analyses:
            lines.extend([
                "## Component Improvements",
                "",
            ])

            for test_id, analysis in component_analyses.items():
                suggestions = analysis.get("suggestions", [])
                if suggestions:
                    lines.extend([
                        f"### {test_id}",
                        "",
                    ])
                    for suggestion in suggestions:
                        lines.append(f"- {suggestion}")
                    lines.append("")

        return "\n".join(lines)

    def _generate_appendix(self, metadata: Optional[Dict]) -> str:
        """Generate appendix section."""
        lines = [
            "# Appendix",
            "",
            "## Report Information",
            "",
            f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "- **Format:** Markdown",
            "",
        ]

        if metadata:
            lines.extend([
                "## Session Metadata",
                "",
            ])
            for key, value in metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        return "\n".join(lines)
