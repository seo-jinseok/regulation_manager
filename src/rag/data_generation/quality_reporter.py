"""
Quality Reporter

Ground Truth 데이터셋 품질 보고서 생성기
HTML 형식의 상세 보고서를 생성합니다.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class QualityReporter:
    """
    품질 보고서 생성기

    검증 결과를 기반으로 HTML 형식의 상세 보고서를 생성합니다.
    """

    def __init__(self):
        """QualityReporter 초기화"""
        self.report_data: dict[str, Any] = {}

    def generate_html_report(
        self,
        validation_results: dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        HTML 형식 품질 보고서 생성

        Args:
            validation_results: 검증 결과
            output_path: 출력 파일 경로
        """
        html_content = self._build_html_report(validation_results)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML 보고서 생성 완료: {output_path}")

    def _build_html_report(self, results: dict[str, Any]) -> str:
        """HTML 보고서 빌드"""
        assessment = results.get("overall_assessment", {})

        status_colors = {
            "EXCELLENT": "#10b981",
            "GOOD": "#3b82f6",
            "FAIR": "#f59e0b",
            "POOR": "#ef4444",
        }

        status_color = status_colors.get(assessment.get("status", "UNKNOWN"), "#6b7280")

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ground Truth Dataset Quality Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background: #f9fafb;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #111827; margin-bottom: 10px; }}
        h2 {{ color: #374151; margin-top: 30px; margin-bottom: 15px; border-bottom: 2px solid #e5e7eb; padding-bottom: 5px; }}
        h3 {{ color: #4b5563; margin-top: 20px; margin-bottom: 10px; }}
        .summary-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f3f4f6;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3b82f6;
        }}
        .metric-label {{ font-size: 12px; color: #6b7280; margin-bottom: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #111827; }}
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .progress-fill {{
            height: 100%;
            background: #3b82f6;
            transition: width 0.3s ease;
        }}
        .issue-list {{ list-style: none; margin: 10px 0; }}
        .issue-item {{
            padding: 8px 12px;
            background: #fee2e2;
            border-left: 4px solid #ef4444;
            margin-bottom: 5px;
            border-radius: 4px;
        }}
        .recommendation-item {{
            padding: 8px 12px;
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            margin-bottom: 5px;
            border-radius: 4px;
        }}
        .distribution-chart {{
            margin: 15px 0;
        }}
        .bar-container {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        .bar-label {{ width: 120px; font-size: 14px; }}
        .bar-area {{ flex: 1; height: 30px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }}
        .bar-fill {{ height: 100%; background: #3b82f6; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; color: white; font-size: 12px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f9fafb; font-weight: 600; color: #374151; }}
        tr:hover {{ background: #f9fafb; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Ground Truth Dataset Quality Report</h1>
        <p style="color: #6b7280;">RAG 시스템 평가용 데이터셋 품질 분석 보고서</p>

        <div class="summary-card">
            <h2>Overall Assessment</h2>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div>
                    <span class="status-badge" style="background: {status_color};">
                        {assessment.get("status", "UNKNOWN")}
                    </span>
                </div>
                <div>
                    <div class="metric-label">Quality Score</div>
                    <div class="metric-value">{assessment.get("quality_score", 0):.2f}/1.00</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {assessment.get("quality_score", 0) * 100}%"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Pairs</div>
                <div class="metric-value">{results.get("total_pairs", 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Categories</div>
                <div class="metric-value">{results.get("category_balance", {}).get("total_categories", 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Answer Quality</div>
                <div class="metric-value">{results.get("answer_quality", {}).get("avg_score", 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Completeness</div>
                <div class="metric-value">{results.get("answer_completeness", {}).get("completeness_ratio", 0):.2%}</div>
            </div>
        </div>

        <div class="summary-card">
            <h2>Question Diversity</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Questions</td>
                    <td>{results.get("question_diversity", {}).get("total_questions", 0)}</td>
                </tr>
                <tr>
                    <td>Unique Questions</td>
                    <td>{results.get("question_diversity", {}).get("unique_questions", 0)}</td>
                </tr>
                <tr>
                    <td>Duplicate Ratio</td>
                    <td>{results.get("question_diversity", {}).get("duplicate_ratio", 0):.2%}</td>
                </tr>
                <tr>
                    <td>Average Length</td>
                    <td>{results.get("question_diversity", {}).get("avg_question_length", 0):.1f} characters</td>
                </tr>
            </table>
        </div>

        <div class="summary-card">
            <h2>Category Distribution</h2>
            {self._build_distribution_html(results.get("category_balance", {}).get("category_distribution", {}))}
        </div>

        <div class="summary-card">
            <h2>Difficulty Distribution</h2>
            {self._build_distribution_html(results.get("difficulty_distribution", {}).get("difficulty_distribution", {}))}
        </div>

        <div class="summary-card">
            <h2>Query Type Distribution</h2>
            {self._build_distribution_html(results.get("query_types", {}).get("query_type_distribution", {}))}
        </div>

"""

        # Issues & Recommendations
        if assessment.get("issues"):
            html += """
        <div class="summary-card">
            <h2>Issues Found</h2>
            <ul class="issue-list">
"""
            for issue in assessment["issues"]:
                html += f'                <li class="issue-item">{issue}</li>\n'

            html += """            </ul>
        </div>
"""

        if assessment.get("recommendations"):
            html += """
        <div class="summary-card">
            <h2>Recommendations</h2>
            <ul class="issue-list">
"""
            for rec in assessment["recommendations"]:
                html += f'                <li class="recommendation-item">{rec}</li>\n'

            html += """            </ul>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""

        return html

    def _build_distribution_html(self, distribution: dict[str, int]) -> str:
        """분포 차트 HTML 빌드"""
        if not distribution:
            return "<p>No data available</p>"

        total = sum(distribution.values())
        max_count = max(distribution.values())

        html = '<div class="distribution-chart">\n'

        for label, count in sorted(
            distribution.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / total) * 100
            bar_width = (count / max_count) * 100

            html += f"""
            <div class="bar-container">
                <div class="bar-label">{label}</div>
                <div class="bar-area">
                    <div class="bar-fill" style="width: {bar_width}%">{count}</div>
                </div>
            </div>
"""

        html += "</div>\n"

        return html

    def generate_markdown_report(
        self,
        validation_results: dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Markdown 형식 품질 보고서 생성

        Args:
            validation_results: 검증 결과
            output_path: 출력 파일 경로
        """
        md_content = self._build_markdown_report(validation_results)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Markdown 보고서 생성 완료: {output_path}")

    def _build_markdown_report(self, results: dict[str, Any]) -> str:
        """Markdown 보고서 빌드"""
        assessment = results.get("overall_assessment", {})

        md = f"""# Ground Truth Dataset Quality Report

RAG 시스템 평가용 데이터셋 품질 분석 보고서

## Overall Assessment

- **Status**: {assessment.get("status", "UNKNOWN")}
- **Quality Score**: {assessment.get("quality_score", 0):.2f}/1.00

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Pairs | {results.get("total_pairs", 0)} |
| Categories | {results.get("category_balance", {}).get("total_categories", 0)} |
| Avg Answer Quality | {results.get("answer_quality", {}).get("avg_score", 0):.2f} |
| Completeness | {results.get("answer_completeness", {}).get("completeness_ratio", 0):.2%} |

## Question Diversity

| Metric | Value |
|--------|-------|
| Total Questions | {results.get("question_diversity", {}).get("total_questions", 0)} |
| Unique Questions | {results.get("question_diversity", {}).get("unique_questions", 0)} |
| Duplicate Ratio | {results.get("question_diversity", {}).get("duplicate_ratio", 0):.2%} |
| Avg Length | {results.get("question_diversity", {}).get("avg_question_length", 0):.1f} chars |

## Category Distribution

"""

        # Category distribution
        cat_dist = results.get("category_balance", {}).get("category_distribution", {})
        for cat, count in sorted(cat_dist.items(), key=lambda x: x[1], reverse=True):
            md += f"- {cat}: {count}\n"

        md += "\n## Difficulty Distribution\n\n"

        # Difficulty distribution
        diff_dist = results.get("difficulty_distribution", {}).get(
            "difficulty_distribution", {}
        )
        for diff, count in diff_dist.items():
            md += f"- {diff}: {count}\n"

        md += "\n## Query Type Distribution\n\n"

        # Query type distribution
        type_dist = results.get("query_types", {}).get("query_type_distribution", {})
        for qtype, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
            md += f"- {qtype}: {count}\n"

        # Issues & Recommendations
        if assessment.get("issues"):
            md += "\n## Issues Found\n\n"
            for issue in assessment["issues"]:
                md += f"- ❌ {issue}\n"

        if assessment.get("recommendations"):
            md += "\n## Recommendations\n\n"
            for rec in assessment["recommendations"]:
                md += f"- 💡 {rec}\n"

        return md


def main():
    """테스트 메인 함수"""
    reporter = QualityReporter()

    # 테스트용 검증 결과
    test_results = {
        "total_pairs": 500,
        "overall_assessment": {
            "status": "GOOD",
            "quality_score": 0.75,
            "issues": ["일부 답변의 길이가 짧음"],
            "recommendations": ["답변 길이를 100자 이상으로 권장"],
        },
        "question_diversity": {
            "total_questions": 500,
            "unique_questions": 480,
            "duplicate_ratio": 0.04,
            "avg_question_length": 35.5,
        },
        "category_balance": {
            "total_categories": 13,
            "category_distribution": {
                "졸업": 80,
                "성적": 70,
                "장학금": 65,
                "휴학": 60,
                "등록": 55,
            },
        },
        "difficulty_distribution": {
            "difficulty_distribution": {
                "초급": 200,
                "중급": 200,
                "고급": 100,
            },
        },
        "query_types": {
            "query_type_distribution": {
                "정확한 쿼리": 250,
                "구어체 쿼리": 100,
                "복합 질문": 80,
                "모호한 쿼리": 70,
            },
        },
        "answer_quality": {
            "avg_score": 0.75,
        },
        "answer_completeness": {
            "completeness_ratio": 0.95,
        },
    }

    output_dir = Path(
        "/Users/truestone/Dropbox/repo/University/regulation_manager/data/ground_truth"
    )

    reporter.generate_html_report(test_results, output_dir / "quality_report.html")
    reporter.generate_markdown_report(test_results, output_dir / "QUALITY_REPORT.md")


if __name__ == "__main__":
    main()
