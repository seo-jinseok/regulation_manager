#!/usr/bin/env python3
"""
HTML Report Generator for RAGAS Evaluation Results.

Generates interactive HTML reports with:
- Score distribution histograms
- Category-wise comparison
- Failed samples analysis
- Metric correlations

Usage:
    python scripts/generate_ragas_report.py --input test_reports/ragas_evaluation_20250129_120000.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import click

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGAS Evaluation Report - Regulation Manager</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}

        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            text-align: center;
            transition: transform 0.2s;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        }}

        .metric-card.passed {{
            border-left: 5px solid #10b981;
        }}

        .metric-card.failed {{
            border-left: 5px solid #ef4444;
        }}

        .metric-card h3 {{
            font-size: 0.9em;
            color: #6b7280;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .metric-card .score {{
            font-size: 2.5em;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 5px;
        }}

        .metric-card .score.passed {{
            color: #10b981;
        }}

        .metric-card .score.failed {{
            color: #ef4444;
        }}

        .metric-card .threshold {{
            font-size: 0.85em;
            color: #9ca3af;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            font-size: 1.8em;
            color: #1f2937;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
        }}

        .chart {{
            margin: 20px 0;
        }}

        .category-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }}

        .category-table th {{
            background: #f3f4f6;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: #374151;
            border-bottom: 2px solid #e5e7eb;
        }}

        .category-table td {{
            padding: 15px;
            border-bottom: 1px solid #e5e7eb;
        }}

        .category-table tr:hover {{
            background: #f9fafb;
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}

        .progress-fill.passed {{
            background: linear-gradient(90deg, #10b981, #34d399);
        }}

        .progress-fill.failed {{
            background: linear-gradient(90deg, #ef4444, #f87171);
        }}

        .failed-sample {{
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
        }}

        .failed-sample h4 {{
            color: #991b1b;
            margin-bottom: 10px;
        }}

        .failed-sample .query {{
            font-weight: 500;
            color: #1f2937;
            margin-bottom: 10px;
        }}

        .failed-sample .failure-reasons {{
            color: #dc2626;
            font-size: 0.95em;
            margin-bottom: 10px;
        }}

        .failed-sample .scores {{
            display: flex;
            gap: 15px;
            font-size: 0.9em;
            color: #6b7280;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }}

        .badge.pass {{
            background: #d1fae5;
            color: #065f46;
        }}

        .badge.fail {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #6b7280;
            font-size: 0.9em;
            border-top: 1px solid #e5e7eb;
        }}

        .threshold-info {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .threshold-info h4 {{
            color: #1e40af;
            margin-bottom: 10px;
        }}

        .threshold-info ul {{
            list-style: none;
            padding-left: 0;
        }}

        .threshold-info li {{
            padding: 5px 0;
            color: #1e3a8a;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 RAGAS Evaluation Report</h1>
            <p>Regulation Manager RAG System Quality Assessment</p>
            <p style="margin-top: 10px; font-size: 0.95em;">
                📅 {timestamp} | 📊 {total_samples} Samples Evaluated
            </p>
        </div>

        <div class="summary">
            <div class="metric-card {faithfulness_class}">
                <h3>Faithfulness (환각 감지)</h3>
                <div class="score {faithfulness_class}">{faithfulness_score:.3f}</div>
                <div class="threshold">Threshold: 0.90</div>
            </div>

            <div class="metric-card {relevancy_class}">
                <h3>Answer Relevancy (답변 관련성)</h3>
                <div class="score {relevancy_class}">{relevancy_score:.3f}</div>
                <div class="threshold">Threshold: 0.85</div>
            </div>

            <div class="metric-card {precision_class}">
                <h3>Contextual Precision (검색 정밀도)</h3>
                <div class="score {precision_class}">{precision_score:.3f}</div>
                <div class="threshold">Threshold: 0.80</div>
            </div>

            <div class="metric-card {recall_class}">
                <h3>Contextual Recall (검색 재현율)</h3>
                <div class="score {recall_class}">{recall_score:.3f}</div>
                <div class="threshold">Threshold: 0.80</div>
            </div>

            <div class="metric-card">
                <h3>Pass Rate</h3>
                <div class="score {pass_rate_class}">{pass_rate:.1%}</div>
                <div class="threshold">{passed_samples}/{total_samples} Passed</div>
            </div>
        </div>

        <div class="content">
            <div class="section">
                <h2>📊 Score Distribution</h2>
                <div id="histogram" class="chart"></div>
            </div>

            <div class="section">
                <h2>📈 Metric Correlations</h2>
                <div id="correlation" class="chart"></div>
            </div>

            <div class="section">
                <h2>🏷️ Results by Category</h2>
                {category_table}
            </div>

            {failed_samples_section}

            <div class="section">
                <h2>🎯 Evaluation Thresholds</h2>
                <div class="threshold-info">
                    <h4>Quality Standards</h4>
                    <ul>
                        <li><strong>Faithfulness ≥ 0.90:</strong> Minimal hallucination in generated answers</li>
                        <li><strong>Answer Relevancy ≥ 0.85:</strong> Answers directly address user queries</li>
                        <li><strong>Contextual Precision ≥ 0.80:</strong> Retrieved contexts are relevant and ranked well</li>
                        <li><strong>Contextual Recall ≥ 0.80:</strong> Retrieved contexts contain all necessary information</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Generated by Regulation Manager RAG Quality Evaluation System</p>
            <p>Powered by RAGAS LLM-as-Judge Framework</p>
        </div>
    </div>

    <script>
        // Score distributions
        const scores = {scores_json};

        // Histogram
        const trace1 = {{
            x: scores.faithfulness,
            name: 'Faithfulness',
            type: 'histogram',
            marker: {{ color: 'rgba(102, 126, 234, 0.7)' }},
            xbins: {{ size: 0.05 }}
        }};

        const trace2 = {{
            x: scores.answer_relevancy,
            name: 'Answer Relevancy',
            type: 'histogram',
            marker: {{ color: 'rgba(118, 75, 162, 0.7)' }},
            xbins: {{ size: 0.05 }}
        }};

        const trace3 = {{
            x: scores.contextual_precision,
            name: 'Contextual Precision',
            type: 'histogram',
            marker: {{ color: 'rgba(16, 185, 129, 0.7)' }},
            xbins: {{ size: 0.05 }}
        }};

        const trace4 = {{
            x: scores.contextual_recall,
            name: 'Contextual Recall',
            type: 'histogram',
            marker: {{ color: 'rgba(239, 68, 68, 0.7)' }},
            xbins: {{ size: 0.05 }}
        }};

        const layout1 = {{
            title: 'Score Distribution',
            barmode: 'overlay',
            xaxis: {{ title: 'Score', range: [0, 1] }},
            yaxis: {{ title: 'Count' }},
            hovermode: 'closest',
            margin: {{ t: 50, b: 50, l: 50, r: 50 }}
        }};

        Plotly.newPlot('histogram', [trace1, trace2, trace3, trace4], layout1);

        // Correlation scatter plot
        const correlationTrace = {{
            x: scores.faithfulness,
            y: scores.answer_relevancy,
            mode: 'markers',
            type: 'scatter',
            name: 'Faithfulness vs Relevancy',
            text: scores.queries,
            marker: {{
                size: 8,
                color: scores.overall,
                colorscale: 'RdYlGn',
                showscale: true,
                colorbar: {{
                    title: 'Overall Score',
                    x: 1.15
                }}
            }}
        }};

        const layout2 = {{
            title: 'Faithfulness vs Answer Relevancy',
            xaxis: {{ title: 'Faithfulness', range: [0, 1] }},
            yaxis: {{ title: 'Answer Relevancy', range: [0, 1] }},
            hovermode: 'closest',
            margin: {{ t: 50, b: 50, l: 50, r: 50 }}
        }};

        Plotly.newPlot('correlation', [correlationTrace], layout2);
    </script>
</body>
</html>
"""


def load_results(input_file: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores(results: List[Dict[str, Any]]) -> Dict[str, List]:
    """Extract scores from results for visualization."""
    scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "contextual_precision": [],
        "contextual_recall": [],
        "overall": [],
        "queries": [],
    }

    for result in results:
        scores["faithfulness"].append(result["faithfulness"])
        scores["answer_relevancy"].append(result["answer_relevancy"])
        scores["contextual_precision"].append(result["contextual_precision"])
        scores["contextual_recall"].append(result["contextual_recall"])
        scores["overall"].append(result["overall_score"])
        scores["queries"].append(result["query"][:50])

    return scores


def generate_category_table(summary: Dict[str, Any]) -> str:
    """Generate HTML table for category-wise results."""
    if "by_category" not in summary:
        return "<p>No category data available.</p>"

    rows = []
    for category, stats in sorted(
        summary["by_category"].items(),
        key=lambda x: x[1]["pass_rate"],
        reverse=True,
    ):
        pass_class = "passed" if stats["pass_rate"] >= 0.8 else "failed"
        badge_class = "pass" if stats["pass_rate"] >= 0.8 else "fail"

        rows.append(f"""
            <tr>
                <td><strong>{category}</strong></td>
                <td>{stats["count"]}</td>
                <td>{stats["avg_overall"]:.3f}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill {pass_class}" style="width: {stats["pass_rate"] * 100:.1f}%"></div>
                    </div>
                </td>
                <td>
                    <span class="badge {badge_class}">{stats["pass_rate"]:.1%}</span>
                </td>
            </tr>
        """)

    return f"""
        <table class="category-table">
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Samples</th>
                    <th>Avg Overall</th>
                    <th>Pass Rate</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    """


def generate_failed_samples_section(
    results: List[Dict[str, Any]], max_samples: int = 10
) -> str:
    """Generate HTML section for failed samples."""
    failed_samples = [r for r in results if not r["passed"]]

    if not failed_samples:
        return """
            <div class="section">
                <h2>✅ Failed Samples Analysis</h2>
                <p style="color: #10b981; font-size: 1.1em;">
                    🎉 All samples passed evaluation! No failures to analyze.
                </p>
            </div>
        """

    samples_html = []
    for i, sample in enumerate(failed_samples[:max_samples], 1):
        failure_reasons = "<br>".join(
            f"• {r}" for r in sample.get("failure_reasons", [])
        )

        samples_html.append(f"""
            <div class="failed-sample">
                <h4>Sample {i}</h4>
                <div class="query">📝 {sample["query"]}</div>
                <div class="failure-reasons">
                    <strong>Failure Reasons:</strong><br>
                    {failure_reasons}
                </div>
                <div class="scores">
                    <span>Faithfulness: <strong>{sample["faithfulness"]:.3f}</strong></span>
                    <span>Relevancy: <strong>{sample["answer_relevancy"]:.3f}</strong></span>
                    <span>Precision: <strong>{sample["contextual_precision"]:.3f}</strong></span>
                    <span>Recall: <strong>{sample["contextual_recall"]:.3f}</strong></span>
                </div>
            </div>
        """)

    more_text = ""
    if len(failed_samples) > max_samples:
        more_text = f"<p><em>... and {len(failed_samples) - max_samples} more failed samples</em></p>"

    return f"""
        <div class="section">
            <h2>❌ Failed Samples Analysis ({len(failed_samples)} total)</h2>
            {"".join(samples_html)}
            {more_text}
        </div>
    """


def generate_html_report(
    results_data: Dict[str, Any],
    output_file: Path,
) -> None:
    """Generate HTML report from evaluation results."""
    summary = results_data["summary"]
    results = results_data["results"]
    timestamp = results_data["timestamp"]

    # Extract average scores
    avg_scores = summary["average_scores"]

    # Determine pass/fail classes
    faithfulness_class = "passed" if avg_scores["faithfulness"] >= 0.90 else "failed"
    relevancy_class = "passed" if avg_scores["answer_relevancy"] >= 0.85 else "failed"
    precision_class = (
        "passed" if avg_scores["contextual_precision"] >= 0.80 else "failed"
    )
    recall_class = "passed" if avg_scores["contextual_recall"] >= 0.80 else "failed"

    pass_rate = summary["pass_rate"]
    pass_rate_class = "passed" if pass_rate >= 0.8 else "failed"

    # Extract scores for visualization
    scores = extract_scores(results)
    scores_json = json.dumps(scores, ensure_ascii=False)

    # Generate category table
    category_table = generate_category_table(summary)

    # Generate failed samples section
    failed_samples_section = generate_failed_samples_section(results)

    # Format timestamp
    formatted_timestamp = datetime.fromisoformat(timestamp).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Render HTML
    html_content = HTML_TEMPLATE.format(
        timestamp=formatted_timestamp,
        total_samples=summary["total_samples"],
        passed_samples=summary["passed_samples"],
        faithfulness_score=avg_scores["faithfulness"],
        faithfulness_class=faithfulness_class,
        relevancy_score=avg_scores["answer_relevancy"],
        relevancy_class=relevancy_class,
        precision_score=avg_scores["contextual_precision"],
        precision_class=precision_class,
        recall_score=avg_scores["contextual_recall"],
        recall_class=recall_class,
        pass_rate=pass_rate,
        pass_rate_class=pass_rate_class,
        scores_json=scores_json,
        category_table=category_table,
        failed_samples_section=failed_samples_section,
    )

    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML report generated: {output_file}")


@click.command()
@click.option(
    "--input",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSON file from RAGAS evaluation",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output HTML file path (default: test_reports/ragas_report_<timestamp>.html)",
)
def main(input: Path, output: Optional[Path]) -> None:
    """
    Generate HTML report from RAGAS evaluation results.

    This script creates an interactive HTML report with:
    - Score distribution histograms
    - Category-wise comparison tables
    - Failed samples analysis
    - Metric correlation visualizations

    Example:
        python scripts/generate_ragas_report.py --input test_reports/ragas_evaluation_20250129.json
    """
    print("📊 Generating HTML Report from RAGAS Evaluation Results")
    print("=" * 80)
    print()

    # Load results
    print(f"📂 Loading results from: {input}")
    results_data = load_results(input)

    # Determine output path
    if output is None:
        output_dir = input.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = output_dir / f"ragas_report_{timestamp}.html"

    # Generate report
    generate_html_report(results_data, output)

    print()
    print("=" * 80)
    print("✅ Report generation complete!")
    print(f"📄 Open in browser: file://{output.absolute()}")


if __name__ == "__main__":
    main()
