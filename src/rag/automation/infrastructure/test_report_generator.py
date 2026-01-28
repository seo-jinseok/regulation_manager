"""
Test Report Generator.

Infrastructure service for generating comprehensive test reports
in markdown and HTML formats with interactive visualizations.

Clean Architecture: Infrastructure layer handles output formatting.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..domain.entities import MultiTurnScenario, QualityTestResult

logger = logging.getLogger(__name__)


class ReportGenerator:
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
        test_results: List["QualityTestResult"],
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
        sections.append(
            self._generate_recommendations(
                test_results, component_analyses, failure_analyses
            )
        )

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
        test_results: List["QualityTestResult"],
        metadata: Optional[Dict],
    ) -> str:
        """Generate summary section."""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.passed)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        avg_confidence = (
            sum(r.confidence for r in test_results) / total_tests
            if total_tests > 0
            else 0
        )
        avg_time = (
            sum(r.execution_time_ms for r in test_results) / total_tests
            if total_tests > 0
            else 0
        )

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

    def _generate_test_overview(self, test_results: List["QualityTestResult"]) -> str:
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

        lines.extend(
            [
                f"✅ **Passed:** {len(passed)}",
                f"❌ **Failed:** {len(failed)}",
                "",
            ]
        )

        # Performance metrics
        times = [r.execution_time_ms for r in test_results]
        if times:
            lines.extend(
                [
                    "## Performance Metrics",
                    "",
                    f"- **Min Time:** {min(times)}ms",
                    f"- **Max Time:** {max(times)}ms",
                    f"- **Avg Time:** {sum(times) / len(times):.0f}ms",
                    "",
                ]
            )

        return "\n".join(lines)

    def _generate_detailed_results(
        self, test_results: List["QualityTestResult"]
    ) -> str:
        """Generate detailed test results section."""
        lines = [
            "# Detailed Test Results",
            "",
        ]

        for idx, result in enumerate(test_results, 1):
            status_icon = "✅" if result.passed else "❌"
            lines.extend(
                [
                    f"## {idx}. {result.test_case_id} {status_icon}",
                    "",
                    f"**Query:** {result.query}",
                    "",
                    f"**Answer:** {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}",
                    "",
                    f"**Confidence:** {result.confidence:.2f}",
                    f"**Execution Time:** {result.execution_time_ms}ms",
                    "",
                ]
            )

            if result.sources:
                lines.extend(
                    [
                        "**Sources:**",
                        "",
                    ]
                )
                for source in result.sources[:5]:
                    lines.append(f"- {source}")
                lines.append("")

            if result.quality_score:
                qs = result.quality_score
                lines.extend(
                    [
                        "**Quality Score:**",
                        "",
                        f"- Total: {qs.total_score:.1f}/5.0",
                        f"- Accuracy: {qs.dimensions.accuracy:.2f}",
                        f"- Completeness: {qs.dimensions.completeness:.2f}",
                        f"- Relevance: {qs.dimensions.relevance:.2f}",
                        f"- Source Citation: {qs.dimensions.source_citation:.2f}",
                        "",
                    ]
                )

            if result.error_message:
                lines.extend(
                    [
                        "**Error:**",
                        "",
                        "```",
                        result.error_message,
                        "```",
                        "",
                    ]
                )

        return "\n".join(lines)

    def _generate_multi_turn_analysis(
        self, scenarios: List["MultiTurnScenario"]
    ) -> str:
        """Generate multi-turn analysis section."""
        lines = [
            "# Multi-Turn Conversation Analysis",
            "",
        ]

        for scenario in scenarios:
            lines.extend(
                [
                    f"## Scenario: {scenario.scenario_id}",
                    "",
                    f"**Persona:** {scenario.persona_type.value}",
                    f"**Total Turns:** {scenario.total_turns}",
                    f"**Difficulty:** {scenario.difficulty.value}",
                    f"**Context Preservation Rate:** {scenario.context_preservation_rate:.1%}",
                    "",
                ]
            )

            # Follow-up distribution
            if scenario.follow_up_distribution:
                lines.extend(
                    [
                        "**Follow-Up Question Distribution:**",
                        "",
                    ]
                )
                for follow_type, count in scenario.follow_up_distribution.items():
                    lines.append(f"- {follow_type.value}: {count}")
                lines.append("")

            # Turn details
            lines.extend(
                [
                    "**Turn Details:**",
                    "",
                ]
            )
            for turn in scenario.turns:
                context_icon = "✅" if turn.context_preserved else "❌"
                lines.extend(
                    [
                        f"### Turn {turn.turn_number} {context_icon}",
                        "",
                        f"**Query:** {turn.query}",
                        f"**Answer:** {turn.answer[:100]}...",
                        "",
                        f"- **Follow-up Type:** {turn.follow_up_type.value if turn.follow_up_type else 'N/A'}",
                        f"- **Confidence:** {turn.confidence:.2f}",
                        f"- **Context Preserved:** {turn.context_preserved}",
                        "",
                    ]
                )

        return "\n".join(lines)

    def _generate_component_analysis(self, analyses: Dict[str, any]) -> str:
        """Generate component analysis section."""
        lines = [
            "# RAG Component Analysis",
            "",
        ]

        for test_id, analysis in analyses.items():
            lines.extend(
                [
                    f"## Test: {test_id}",
                    "",
                    f"**Overall Impact:** {analysis.get('overall_impact', 'N/A')}",
                    f"**Net Impact Score:** {analysis.get('net_impact_score', 0)}",
                    "",
                    "**Component Contributions:**",
                    "",
                ]
            )

            contributions = analysis.get("contributions", [])
            for contrib in contributions:
                component = contrib.get("component", "unknown")
                score = contrib.get("score", 0)
                reason = contrib.get("reason", "")
                executed = "✅" if contrib.get("was_executed", False) else "❌"

                score_icon = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"

                lines.append(
                    f"- {score_icon} **{component}** ({executed}): {score} - {reason}"
                )

            lines.append("")

        return "\n".join(lines)

    def _generate_failure_analysis(self, analyses: Dict[str, any]) -> str:
        """Generate failure analysis section."""
        lines = [
            "# Failure Analysis (5-Why)",
            "",
        ]

        for test_id, analysis in analyses.items():
            lines.extend(
                [
                    f"## Test: {test_id}",
                    "",
                    f"**Original Failure:** {analysis.get('original_failure', 'N/A')}",
                    f"**Root Cause:** {analysis.get('root_cause', 'N/A')}",
                    "",
                    "**5-Why Chain:**",
                    "",
                ]
            )

            why_chain = analysis.get("why_chain", [])
            for idx, why in enumerate(why_chain, 1):
                lines.append(f"{idx}. Why? → {why}")

            lines.extend(
                [
                    "",
                    "**Suggested Fix:**",
                    "",
                    analysis.get("suggested_fix", "No suggestion"),
                    "",
                    f"**Target:** {analysis.get('component_to_patch', 'N/A')}",
                    f"**Code Change Required:** {analysis.get('code_change_required', False)}",
                    "",
                ]
            )

        return "\n".join(lines)

    def _generate_quality_metrics(self, test_results: List["QualityTestResult"]) -> str:
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
            avg_accuracy = sum(d.accuracy for d in dimensions_list) / len(
                dimensions_list
            )
            avg_completeness = sum(d.completeness for d in dimensions_list) / len(
                dimensions_list
            )
            avg_relevance = sum(d.relevance for d in dimensions_list) / len(
                dimensions_list
            )
            avg_citation = sum(d.source_citation for d in dimensions_list) / len(
                dimensions_list
            )
            avg_practicality = sum(d.practicality for d in dimensions_list) / len(
                dimensions_list
            )
            avg_actionability = sum(d.actionability for d in dimensions_list) / len(
                dimensions_list
            )

            lines.extend(
                [
                    "| Dimension | Average |",
                    "|-----------|---------|",
                    f"| Accuracy | {avg_accuracy:.2f} |",
                    f"| Completeness | {avg_completeness:.2f} |",
                    f"| Relevance | {avg_relevance:.2f} |",
                    f"| Source Citation | {avg_citation:.2f} |",
                    f"| Practicality | {avg_practicality:.2f} |",
                    f"| Actionability | {avg_actionability:.2f} |",
                    "",
                ]
            )

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

        lines.extend(
            [
                "**Score Distribution:**",
                "",
            ]
        )
        for range_name, count in score_ranges.items():
            if count > 0:
                lines.append(f"- {range_name}: {count}")
        lines.append("")

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        test_results: List["QualityTestResult"],
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
                    lines.extend(
                        [
                            f"### 🚨 {test_id}",
                            "",
                            f"**Issue:** {analysis.get('root_cause', 'Unknown')}",
                            f"**Fix:** {analysis.get('suggested_fix', 'No suggestion')}",
                            "",
                        ]
                    )

        # Component improvement suggestions
        if component_analyses:
            lines.extend(
                [
                    "## Component Improvements",
                    "",
                ]
            )

            for test_id, analysis in component_analyses.items():
                suggestions = analysis.get("suggestions", [])
                if suggestions:
                    lines.extend(
                        [
                            f"### {test_id}",
                            "",
                        ]
                    )
                    for suggestion in suggestions:
                        lines.append(f"- {suggestion}")
                    lines.append("")

        return "\n".join(lines)

    def generate_html_report(
        self,
        session_id: str,
        test_results: List["QualityTestResult"],
        multi_turn_scenarios: Optional[List["MultiTurnScenario"]] = None,
        component_analyses: Optional[Dict[str, any]] = None,
        failure_analyses: Optional[Dict[str, any]] = None,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Generate an interactive HTML report with Chart.js visualizations.

        Args:
            session_id: Test session identifier.
            test_results: List of test results.
            multi_turn_scenarios: Optional multi-turn scenarios.
            component_analyses: Optional component analysis results.
            failure_analyses: Optional failure analysis results.
            metadata: Optional session metadata.

        Returns:
            Path to generated HTML report file.
        """
        self.logger.info(f"Generating HTML report for session: {session_id}")

        # Calculate statistics
        stats = self._calculate_statistics(test_results)

        # Generate chart data
        chart_data = self._generate_chart_data(test_results, multi_turn_scenarios)

        # Build HTML content
        html_content = self._build_html_template(
            session_id=session_id,
            stats=stats,
            chart_data=chart_data,
            test_results=test_results,
            multi_turn_scenarios=multi_turn_scenarios,
            component_analyses=component_analyses,
            failure_analyses=failure_analyses,
            metadata=metadata,
        )

        # Write to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"test_report_{session_id}_{timestamp}.html"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(f"HTML report generated: {report_path}")

        return report_path

    def _calculate_statistics(self, test_results: List["QualityTestResult"]) -> Dict:
        """Calculate comprehensive statistics from test results."""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.passed)
        failed_tests = total_tests - passed_tests

        # Quality score statistics
        quality_scores = [
            r.quality_score.total_score for r in test_results if r.quality_score
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        # Confidence statistics
        avg_confidence = (
            sum(r.confidence for r in test_results) / total_tests
            if total_tests > 0
            else 0
        )

        # Execution time statistics
        times = [r.execution_time_ms for r in test_results]
        avg_time = sum(times) / len(times) if times else 0

        # Dimension averages
        dimensions_list = [
            r.quality_score.dimensions for r in test_results if r.quality_score
        ]
        dimension_avgs = {}
        if dimensions_list:
            dimension_avgs = {
                "accuracy": sum(d.accuracy for d in dimensions_list)
                / len(dimensions_list),
                "completeness": sum(d.completeness for d in dimensions_list)
                / len(dimensions_list),
                "relevance": sum(d.relevance for d in dimensions_list)
                / len(dimensions_list),
                "source_citation": sum(d.source_citation for d in dimensions_list)
                / len(dimensions_list),
                "practicality": sum(d.practicality for d in dimensions_list)
                / len(dimensions_list),
                "actionability": sum(d.actionability for d in dimensions_list)
                / len(dimensions_list),
            }

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

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "avg_quality": avg_quality,
            "avg_confidence": avg_confidence,
            "avg_time": avg_time,
            "min_time": min(times) if times else 0,
            "max_time": max(times) if times else 0,
            "dimension_avgs": dimension_avgs,
            "score_ranges": score_ranges,
        }

    def _generate_chart_data(
        self,
        test_results: List["QualityTestResult"],
        multi_turn_scenarios: Optional[List["MultiTurnScenario"]],
    ) -> Dict:
        """Generate data structures for Chart.js visualizations."""
        chart_data = {
            "dimensions": {},
            "personas": {},
            "query_types": {},
            "timeline": {},
        }

        # Dimension radar chart data
        dimension_avgs = {}
        dimensions_list = [
            r.quality_score.dimensions for r in test_results if r.quality_score
        ]
        if dimensions_list:
            dimension_avgs = {
                "accuracy": sum(d.accuracy for d in dimensions_list)
                / len(dimensions_list),
                "completeness": sum(d.completeness for d in dimensions_list)
                / len(dimensions_list),
                "relevance": sum(d.relevance for d in dimensions_list)
                / len(dimensions_list),
                "source_citation": sum(d.source_citation for d in dimensions_list)
                / len(dimensions_list),
                "practicality": sum(d.practicality for d in dimensions_list)
                / len(dimensions_list),
                "actionability": sum(d.actionability for d in dimensions_list)
                / len(dimensions_list),
            }
        chart_data["dimensions"] = dimension_avgs

        # Persona performance bar chart
        persona_stats = {}
        for result in test_results:
            # Extract persona from test_case_id or metadata
            persona = (
                result.test_case_id.split("_")[0]
                if "_" in result.test_case_id
                else "Unknown"
            )
            if persona not in persona_stats:
                persona_stats[persona] = {"passed": 0, "failed": 0, "scores": []}
            if result.passed:
                persona_stats[persona]["passed"] += 1
            else:
                persona_stats[persona]["failed"] += 1
            if result.quality_score:
                persona_stats[persona]["scores"].append(
                    result.quality_score.total_score
                )

        chart_data["personas"] = {
            persona: {
                "pass_rate": (
                    stats["passed"] / (stats["passed"] + stats["failed"]) * 100
                )
                if (stats["passed"] + stats["failed"]) > 0
                else 0,
                "avg_score": sum(stats["scores"]) / len(stats["scores"])
                if stats["scores"]
                else 0,
            }
            for persona, stats in persona_stats.items()
        }

        # Query type heatmap data
        query_type_stats = {}
        for result in test_results:
            # Determine query type from metadata or patterns
            query_type = result.rag_pipeline_log.get("query_type", "general")
            if query_type not in query_type_stats:
                query_type_stats[query_type] = {"passed": 0, "failed": 0}
            if result.passed:
                query_type_stats[query_type]["passed"] += 1
            else:
                query_type_stats[query_type]["failed"] += 1

        chart_data["query_types"] = query_type_stats

        # Timeline chart data
        timeline_data = []
        for idx, result in enumerate(test_results):
            timeline_data.append(
                {
                    "index": idx + 1,
                    "score": result.quality_score.total_score
                    if result.quality_score
                    else 0,
                    "confidence": result.confidence,
                    "time_ms": result.execution_time_ms,
                    "passed": result.passed,
                }
            )

        chart_data["timeline"] = timeline_data

        return chart_data

    def _build_html_template(
        self,
        session_id: str,
        stats: Dict,
        chart_data: Dict,
        test_results: List["QualityTestResult"],
        multi_turn_scenarios: Optional[List["MultiTurnScenario"]],
        component_analyses: Optional[Dict],
        failure_analyses: Optional[Dict],
        metadata: Optional[Dict],
    ) -> str:
        """Build the complete HTML template with embedded CSS, JS, and Chart.js."""

        # Escape JSON data for JavaScript
        stats_json = json.dumps(stats, ensure_ascii=False)
        chart_data_json = json.dumps(chart_data, ensure_ascii=False)
        test_results_json = json.dumps(
            [
                {
                    "id": r.test_case_id,
                    "query": r.query,
                    "answer": r.answer[:200] + "..."
                    if len(r.answer) > 200
                    else r.answer,
                    "confidence": r.confidence,
                    "time_ms": r.execution_time_ms,
                    "passed": r.passed,
                    "quality_score": r.quality_score.total_score
                    if r.quality_score
                    else 0,
                    "dimensions": (
                        {
                            "accuracy": r.quality_score.dimensions.accuracy,
                            "completeness": r.quality_score.dimensions.completeness,
                            "relevance": r.quality_score.dimensions.relevance,
                            "source_citation": r.quality_score.dimensions.source_citation,
                        }
                        if r.quality_score
                        else None
                    ),
                    "sources": r.sources[:3],
                }
                for r in test_results
            ],
            ensure_ascii=False,
        )

        html = (
            f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Quality Test Report - {session_id}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}

        .header p {{
            opacity: 0.9;
            font-size: 14px;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}

        .card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        }}

        .card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}

        .card .value.success {{ color: #10b981; }}
        .card .value.warning {{ color: #f59e0b; }}
        .card .value.danger {{ color: #ef4444; }}
        .card .value.info {{ color: #3b82f6; }}

        .card .label {{
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .filters {{
            padding: 20px 30px;
            background: white;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .filter-group label {{
            font-size: 14px;
            color: #4b5563;
        }}

        .filter-group select {{
            padding: 8px 12px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 14px;
            background: white;
            cursor: pointer;
        }}

        .charts-section {{
            padding: 30px;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }}

        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        .chart-container h3 {{
            font-size: 16px;
            color: #1f2937;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
        }}

        .chart-wrapper {{
            position: relative;
            height: 300px;
        }}

        .results-table {{
            padding: 0 30px 30px;
        }}

        .results-table h2 {{
            font-size: 20px;
            color: #1f2937;
            margin-bottom: 20px;
        }}

        .table-container {{
            overflow-x: auto;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}

        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        td {{
            padding: 15px;
            border-bottom: 1px solid #e5e7eb;
            font-size: 14px;
        }}

        tbody tr:hover {{
            background: #f3f4f6;
        }}

        tbody tr.hidden {{
            display: none;
        }}

        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}

        .status-badge.passed {{
            background: #d1fae5;
            color: #065f46;
        }}

        .status-badge.failed {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .score-bar {{
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            width: 100px;
        }}

        .score-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}

        .score-bar-fill.excellent {{ background: #10b981; }}
        .score-bar-fill.good {{ background: #3b82f6; }}
        .score-bar-fill.partial {{ background: #f59e0b; }}
        .score-bar-fill.poor {{ background: #ef4444; }}

        .quality-indicator {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}

        .dot.excellent {{ background: #10b981; }}
        .dot.good {{ background: #3b82f6; }}
        .dot.partial {{ background: #f59e0b; }}
        .dot.poor {{ background: #ef4444; }}

        @media (max-width: 768px) {{
            .summary-cards {{
                grid-template-columns: 1fr;
            }}
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .filters {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG Quality Test Report</h1>
            <p>Session: {session_id} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="summary-cards">
            <div class="card">
                <div class="label">Total Tests</div>
                <div class="value info" id="total-tests">-</div>
            </div>
            <div class="card">
                <div class="label">Pass Rate</div>
                <div class="value success" id="pass-rate">-</div>
            </div>
            <div class="card">
                <div class="label">Avg Quality Score</div>
                <div class="value" id="avg-quality">-</div>
            </div>
            <div class="card">
                <div class="label">Avg Confidence</div>
                <div class="value info" id="avg-confidence">-</div>
            </div>
            <div class="card">
                <div class="label">Avg Execution Time</div>
                <div class="value" id="avg-time">-</div>
            </div>
        </div>

        <div class="filters">
            <div class="filter-group">
                <label for="status-filter">Status:</label>
                <select id="status-filter">
                    <option value="all">All</option>
                    <option value="passed">Passed</option>
                    <option value="failed">Failed</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="sort-by">Sort by:</label>
                <select id="sort-by">
                    <option value="index">Default</option>
                    <option value="score-desc">Score (High to Low)</option>
                    <option value="score-asc">Score (Low to High)</option>
                    <option value="time-desc">Time (Slow to Fast)</option>
                    <option value="time-asc">Time (Fast to Slow)</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="search-input">Search:</label>
                <input type="text" id="search-input" placeholder="Search queries..."
                       style="padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px;">
            </div>
        </div>

        <div class="charts-section">
            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Overall Quality Gauge</h3>
                    <div class="chart-wrapper">
                        <canvas id="quality-gauge"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h3>Quality Dimensions (Radar)</h3>
                    <div class="chart-wrapper">
                        <canvas id="dimensions-radar"></canvas>
                    </div>
                </div>
            </div>
            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Persona Performance</h3>
                    <div class="chart-wrapper">
                        <canvas id="persona-bar"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h3>Test Timeline</h3>
                    <div class="chart-wrapper">
                        <canvas id="timeline-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div class="results-table">
            <h2>Detailed Test Results</h2>
            <div class="table-container">
                <table id="results-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Status</th>
                            <th>Query</th>
                            <th>Quality Score</th>
                            <th>Confidence</th>
                            <th>Time (ms)</th>
                        </tr>
                    </thead>
                    <tbody id="results-tbody">
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Data from Python
        const stats = """
            + stats_json
            + """;
        const chartData = """
            + chart_data_json
            + """;
        const testResults = """
            + test_results_json
            + """;

        // Update summary cards
        document.getElementById('total-tests').textContent = stats.total_tests;
        document.getElementById('pass-rate').textContent = stats.pass_rate.toFixed(1) + '%';
        document.getElementById('avg-quality').textContent = stats.avg_quality.toFixed(2) + '/5.0';
        document.getElementById('avg-confidence').textContent = (stats.avg_confidence * 100).toFixed(1) + '%';
        document.getElementById('avg-time').textContent = stats.avg_time.toFixed(0) + 'ms';

        // Set color based on quality score
        const qualityEl = document.getElementById('avg-quality');
        if (stats.avg_quality >= 4.5) qualityEl.classList.add('success');
        else if (stats.avg_quality >= 4.0) qualityEl.classList.add('info');
        else if (stats.avg_quality >= 3.0) qualityEl.classList.add('warning');
        else qualityEl.classList.add('danger');

        // Chart.js defaults
        Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        Chart.defaults.color = '#6b7280';

        // 1. Quality Gauge Chart (Doughnut)
        const qualityCtx = document.getElementById('quality-gauge').getContext('2d');
        const qualityScore = stats.avg_quality;
        const qualityColor = qualityScore >= 4.5 ? '#10b981' : qualityScore >= 4.0 ? '#3b82f6' : qualityScore >= 3.0 ? '#f59e0b' : '#ef4444';

        new Chart(qualityCtx, {{
            type: 'doughnut',
            data: {{
                datasets: [{{
                    data: [qualityScore, 5 - qualityScore],
                    backgroundColor: [qualityColor, '#e5e7eb'],
                    borderWidth: 0,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{ enabled: false }},
                    title: {{
                        display: true,
                        text: qualityScore.toFixed(2) + '/5.0',
                        position: 'bottom',
                        font: {{ size: 24, weight: 'bold' }},
                        color: qualityColor,
                        padding: 20
                    }}
                }}
            }}
        }});

        // 2. Dimensions Radar Chart
        const radarCtx = document.getElementById('dimensions-radar').getContext('2d');
        const dimensions = chartData.dimensions || {{}};
        const dimensionLabels = Object.keys(dimensions).map(d => d.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase()));
        const dimensionValues = Object.values(dimensions).map(v => (v * 100).toFixed(1));

        new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: dimensionLabels,
                datasets: [{{
                    label: 'Score %',
                    data: dimensionValues,
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(118, 75, 162, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(118, 75, 162, 1)',
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            stepSize: 20,
                            backdropColor: 'transparent'
                        }},
                        grid: {{ color: '#e5e7eb' }},
                        angleLines: {{ color: '#e5e7eb' }}
                    }}
                }},
                plugins: {{
                    legend: {{ display: false }}
                }}
            }}
        }});

        // 3. Persona Performance Bar Chart
        const personaCtx = document.getElementById('persona-bar').getContext('2d');
        const personas = chartData.personas || {{}};
        const personaLabels = Object.keys(personas);
        const personaPassRates = Object.values(personas).map(p => p.pass_rate);
        const personaColors = personaPassRates.map(r => r >= 80 ? '#10b981' : r >= 60 ? '#f59e0b' : '#ef4444');

        new Chart(personaCtx, {{
            type: 'bar',
            data: {{
                labels: personaLabels,
                datasets: [{{
                    label: 'Pass Rate %',
                    data: personaPassRates,
                    backgroundColor: personaColors,
                    borderRadius: 6,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {{
                    x: {{
                        beginAtZero: true,
                        max: 100,
                        grid: {{ color: '#e5e7eb' }}
                    }},
                    y: {{
                        grid: {{ display: false }}
                    }}
                }},
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (context) => context.parsed.x.toFixed(1) + '%'
                        }}
                    }}
                }}
            }}
        }});

        // 4. Timeline Chart
        const timelineCtx = document.getElementById('timeline-chart').getContext('2d');
        const timelineData = chartData.timeline || [];
        const timelineLabels = timelineData.map(d => d.index);
        const timelineScores = timelineData.map(d => d.score);

        new Chart(timelineCtx, {{
            type: 'line',
            data: {{
                labels: timelineLabels,
                datasets: [{{
                    label: 'Quality Score',
                    data: timelineScores,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 5,
                        grid: {{ color: '#e5e7eb' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'Test Number'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{ display: false }}
                }}
            }}
        }});

        // Populate results table
        function renderResults(results) {{
            const tbody = document.getElementById('results-tbody');
            tbody.innerHTML = '';

            results.forEach((result, index) => {{
                const scoreClass = result.quality_score >= 4.5 ? 'excellent' : result.quality_score >= 4.0 ? 'good' : result.quality_score >= 3.0 ? 'partial' : 'poor';
                const scorePercent = (result.quality_score / 5) * 100;

                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${{index + 1}}</td>
                    <td>
                        <span class="status-badge ${{result.passed ? 'passed' : 'failed'}}">
                            ${{result.passed ? 'PASS' : 'FAIL'}}
                        </span>
                    </td>
                    <td>
                        <div style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${{result.query}}">
                            ${{result.query}}
                        </div>
                    </td>
                    <td>
                        <div class="quality-indicator">
                            <div class="dot ${{scoreClass}}"></div>
                            <span>${{result.quality_score.toFixed(1)}}</span>
                            <div class="score-bar">
                                <div class="score-bar-fill ${{scoreClass}}" style="width: ${{scorePercent}}%"></div>
                            </div>
                        </div>
                    </td>
                    <td>${{(result.confidence * 100).toFixed(1)}}%</td>
                    <td>${{result.time_ms}}</td>
                `;
                tbody.appendChild(row);
            }});
        }}

        // Initial render
        renderResults(testResults);

        // Filter functionality
        const statusFilter = document.getElementById('status-filter');
        const sortBy = document.getElementById('sort-by');
        const searchInput = document.getElementById('search-input');

        function applyFilters() {{
            let filtered = [...testResults];

            // Status filter
            if (statusFilter.value !== 'all') {{
                filtered = filtered.filter(r => r.passed === (statusFilter.value === 'passed'));
            }}

            // Search filter
            if (searchInput.value) {{
                const search = searchInput.value.toLowerCase();
                filtered = filtered.filter(r => r.query.toLowerCase().includes(search));
            }}

            // Sort
            switch (sortBy.value) {{
                case 'score-desc':
                    filtered.sort((a, b) => b.quality_score - a.quality_score);
                    break;
                case 'score-asc':
                    filtered.sort((a, b) => a.quality_score - b.quality_score);
                    break;
                case 'time-desc':
                    filtered.sort((a, b) => b.time_ms - a.time_ms);
                    break;
                case 'time-asc':
                    filtered.sort((a, b) => a.time_ms - b.time_ms);
                    break;
            }}

            renderResults(filtered);
        }}

        statusFilter.addEventListener('change', applyFilters);
        sortBy.addEventListener('change', applyFilters);
        searchInput.addEventListener('input', applyFilters);
    </script>
</body>
</html>"""
        )
        return html

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
            lines.extend(
                [
                    "## Session Metadata",
                    "",
                ]
            )
            for key, value in metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        return "\n".join(lines)


# Type alias for backward compatibility with tests
TestReportGenerator = ReportGenerator
