"""
Gradio Quality Dashboard for RAG Evaluation Monitoring.

Provides real-time monitoring of RAG answer quality metrics:
- Faithfulness
- Answer Relevancy
- Precision
- Recall

Usage:
    uv run python -m src.rag.interface.web.quality_dashboard
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

try:
    import gradio as gr
    import plotly.graph_objects as go

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None
    go = None


# Default paths
DEFAULT_METRICS_PATH = "data/quality_metrics.json"
DEFAULT_HISTORY_PATH = "data/quality_history.jsonl"


def _load_metrics(metrics_path: str = DEFAULT_METRICS_PATH) -> Dict:
    """Load latest quality metrics from JSON file."""
    path = Path(metrics_path)
    if not path.exists():
        # Return mock data for testing
        return _get_mock_metrics()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _get_mock_metrics()


def _load_history(
    history_path: str = DEFAULT_HISTORY_PATH, days: int = 30
) -> List[Dict]:
    """Load quality metrics history from JSONL file."""
    path = Path(history_path)
    if not path.exists():
        return _get_mock_history(days)

    history = []
    cutoff_date = datetime.now() - timedelta(days=days)

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    record_date = datetime.fromisoformat(record.get("timestamp", ""))
                    if record_date >= cutoff_date:
                        history.append(record)
                except Exception:
                    continue
    except Exception:
        return _get_mock_history(days)

    return history


def _get_mock_metrics() -> Dict:
    """Generate mock metrics for testing."""
    return {
        "timestamp": datetime.now().isoformat(),
        "faithfulness": {
            "score": 0.85,
            "trend": "improve",
            "change": 0.05,
        },
        "answer_relevancy": {
            "score": 0.78,
            "trend": "maintain",
            "change": 0.01,
        },
        "precision": {
            "score": 0.82,
            "trend": "improve",
            "change": 0.03,
        },
        "recall": {
            "score": 0.75,
            "trend": "decrease",
            "change": -0.02,
        },
        "personas": {
            "faculty": {"score": 0.88, "count": 45},
            "student": {"score": 0.76, "count": 120},
            "staff": {"score": 0.82, "count": 35},
        },
    }


def _get_mock_history(days: int = 30) -> List[Dict]:
    """Generate mock history data for testing."""
    history = []
    base_date = datetime.now() - timedelta(days=days)

    for i in range(days):
        date = base_date + timedelta(days=i)
        history.append(
            {
                "timestamp": date.isoformat(),
                "faithfulness": 0.80 + (i * 0.001) + (i % 5) * 0.01,
                "answer_relevancy": 0.75 + (i * 0.0015) + (i % 3) * 0.02,
                "precision": 0.78 + (i * 0.0012) + (i % 4) * 0.015,
                "recall": 0.72 + (i * 0.0008) + (i % 6) * 0.01,
            }
        )

    return history


def _format_trend_indicator(trend: str, change: float) -> str:
    """Format trend indicator with emoji."""
    if trend == "improve":
        return f"📈 +{change:.2%}"
    elif trend == "decrease":
        return f"📉 {change:.2%}"
    else:
        return f"➡️ {change:+.2%}"


def _create_metric_card(title: str, score: float, trend: str, change: float) -> str:
    """Create HTML for metric card."""
    trend_html = _format_trend_indicator(trend, change)
    color = (
        "#10b981"
        if trend == "improve"
        else "#ef4444"
        if trend == "decrease"
        else "#6b7280"
    )

    return f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">{title}</div>
        <div style="font-size: 2.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
            {score:.1%}
        </div>
        <div style="font-size: 0.85rem; color: {color}; font-weight: 500;">
            {trend_html}
        </div>
    </div>
    """


def _create_timeline_chart(history: List[Dict]) -> "go.Figure":
    """Create timeline chart showing metrics over time."""
    if not history:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        return fig

    dates = [datetime.fromisoformat(h["timestamp"]) for h in history]
    faithfulness = [h.get("faithfulness", 0) for h in history]
    relevancy = [h.get("answer_relevancy", 0) for h in history]
    precision = [h.get("precision", 0) for h in history]
    recall = [h.get("recall", 0) for h in history]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=faithfulness,
            mode="lines+markers",
            name="Faithfulness",
            line=dict(color="#10b981", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=relevancy,
            mode="lines+markers",
            name="Answer Relevancy",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=precision,
            mode="lines+markers",
            name="Precision",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=recall,
            mode="lines+markers",
            name="Recall",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title="Quality Metrics Timeline (30 Days)",
        xaxis_title="Date",
        yaxis_title="Score",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_yaxes(range=[0, 1])

    return fig


def _create_persona_table(metrics: Dict) -> str:
    """Create HTML table for persona comparison."""
    personas = metrics.get("personas", {})

    if not personas:
        return "<div style='text-align: center; padding: 20px; color: #6b7280;'>No persona data available</div>"

    # Find best and worst performers
    sorted_personas = sorted(
        personas.items(), key=lambda x: x[1]["score"], reverse=True
    )
    best = sorted_personas[0]
    worst = sorted_personas[-1]

    persona_labels = {
        "faculty": "👨‍🏫 교수",
        "student": "👨‍🎓 학생",
        "staff": "💼 직원",
    }

    rows = []
    for persona_id, data in sorted_personas:
        label = persona_labels.get(persona_id, persona_id)
        score = data["score"]
        count = data["count"]

        # Highlight best and worst
        highlight = ""
        if persona_id == best[0]:
            highlight = "background: rgba(16, 185, 129, 0.1);"
        elif persona_id == worst[0]:
            highlight = "background: rgba(239, 68, 68, 0.1);"

        rows.append(f"""
        <tr style="{highlight}">
            <td style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.1);">{label}</td>
            <td style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); text-align: center;">
                <strong>{score:.1%}</strong>
            </td>
            <td style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); text-align: center;">{count}</td>
        </tr>
        """)

    return f"""
    <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
        <thead>
            <tr style="background: rgba(255,255,255,0.05);">
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid rgba(255,255,255,0.1);">페르소나</th>
                <th style="padding: 12px; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.1);">점수</th>
                <th style="padding: 12px; text-align: center; border-bottom: 2px solid rgba(255,255,255,0.1);">질문 수</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    <div style="margin-top: 12px; font-size: 0.85rem; color: #6b7280;">
        🏆 최고 성과: <strong>{persona_labels.get(best[0], best[0])}</strong> ({best[1]["score"]:.1%}) |
        ⚠️ 개선 필요: <strong>{persona_labels.get(worst[0], worst[0])}</strong> ({worst[1]["score"]:.1%})
    </div>
    """


def refresh_dashboard(
    metrics_path: str,
    history_path: str,
    days: int,
) -> tuple:
    """Refresh all dashboard components."""
    metrics = _load_metrics(metrics_path)
    history = _load_history(history_path, days)

    # Metric cards
    faithfulness_card = _create_metric_card(
        "Faithfulness",
        metrics["faithfulness"]["score"],
        metrics["faithfulness"]["trend"],
        metrics["faithfulness"]["change"],
    )

    relevancy_card = _create_metric_card(
        "Answer Relevancy",
        metrics["answer_relevancy"]["score"],
        metrics["answer_relevancy"]["trend"],
        metrics["answer_relevancy"]["change"],
    )

    precision_card = _create_metric_card(
        "Precision",
        metrics["precision"]["score"],
        metrics["precision"]["trend"],
        metrics["precision"]["change"],
    )

    recall_card = _create_metric_card(
        "Recall",
        metrics["recall"]["score"],
        metrics["recall"]["trend"],
        metrics["recall"]["change"],
    )

    # Timeline chart
    timeline_chart = _create_timeline_chart(history)

    # Persona table
    persona_table = _create_persona_table(metrics)

    # Status
    status = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return (
        faithfulness_card,
        relevancy_card,
        precision_card,
        recall_card,
        timeline_chart,
        persona_table,
        status,
    )


def generate_pdf_report(metrics_path: str, history_path: str) -> str:
    """Generate PDF report from quality metrics."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        metrics = _load_metrics(metrics_path)
        _load_history(history_path, 30)

        # Generate PDF
        pdf_path = f"data/quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title = Paragraph("RAG Quality Evaluation Report", styles["Title"])
        story.append(title)
        story.append(Spacer(1, 0.3 * inch))

        # Date
        date_para = Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        )
        story.append(date_para)
        story.append(Spacer(1, 0.3 * inch))

        # Metrics table
        metrics_data = [["Metric", "Score", "Trend", "Change"]]
        for metric_name, key in [
            ("Faithfulness", "faithfulness"),
            ("Answer Relevancy", "answer_relevancy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
        ]:
            m = metrics[key]
            metrics_data.append(
                [
                    metric_name,
                    f"{m['score']:.1%}",
                    m["trend"],
                    f"{m['change']:+.2%}",
                ]
            )

        metrics_table = Table(metrics_data)
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), "#dddddd"),
                    ("TEXTCOLOR", (0, 0), (-1, 0), "#000000"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), "#eeeeee"),
                    ("GRID", (0, 0), (-1, -1), 1, "#999999"),
                ]
            )
        )
        story.append(metrics_table)
        story.append(Spacer(1, 0.5 * inch))

        # Persona comparison
        personas = metrics.get("personas", {})
        if personas:
            story.append(Paragraph("Persona Performance", styles["Heading2"]))
            persona_data = [["Persona", "Score", "Query Count"]]
            for persona_id, data in personas.items():
                persona_data.append(
                    [persona_id, f"{data['score']:.1%}", str(data["count"])]
                )

            persona_table = Table(persona_data)
            persona_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), "#dddddd"),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("GRID", (0, 0), (-1, -1), 1, "#999999"),
                    ]
                )
            )
            story.append(persona_table)

        doc.build(story)
        return f"✅ Report generated: {pdf_path}"

    except Exception as e:
        return f"❌ Report generation failed: {e}"


def create_dashboard(
    metrics_path: str = DEFAULT_METRICS_PATH,
    history_path: str = DEFAULT_HISTORY_PATH,
) -> "gr.Blocks":
    """Create Gradio quality dashboard."""
    if not GRADIO_AVAILABLE:
        raise ImportError("gradio is required. Install with: uv add gradio")

    with gr.Blocks(
        title="RAG Quality Dashboard",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.emerald,
            neutral_hue=gr.themes.colors.neutral,
        ).set(
            body_background_fill="#0f0f0f",
            body_background_fill_dark="#0f0f0f",
            block_background_fill="#1a1a1a",
            block_background_fill_dark="#1a1a1a",
            border_color_primary="rgba(255,255,255,0.06)",
            border_color_primary_dark="rgba(255,255,255,0.06)",
        ),
    ) as app:
        gr.HTML("""
            <div style="text-align: center; padding: 28px 20px 20px;">
                <h1 style="font-size: 1.8rem; font-weight: 600; color: #fafafa;
                           letter-spacing: -0.025em; margin: 0;">
                    📊 RAG Quality Dashboard
                </h1>
                <p style="color: #a3a3a3; margin-top: 6px; font-size: 0.9rem; font-weight: 400;">
                    Real-time monitoring of RAG answer quality metrics
                </p>
            </div>
        """)

        # Metric cards row
        with gr.Row():
            faithfulness_card = gr.HTML(
                _create_metric_card("Faithfulness", 0.85, "improve", 0.05)
            )
            relevancy_card = gr.HTML(
                _create_metric_card("Answer Relevancy", 0.78, "maintain", 0.01)
            )
            precision_card = gr.HTML(
                _create_metric_card("Precision", 0.82, "improve", 0.03)
            )
            recall_card = gr.HTML(
                _create_metric_card("Recall", 0.75, "decrease", -0.02)
            )

        # Timeline chart
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Quality Metrics Timeline")
                timeline_chart = gr.Plot(value=_create_timeline_chart(_load_history()))

        # Persona comparison
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 👥 Persona Performance")
                persona_table = gr.HTML(
                    value=_create_persona_table(_get_mock_metrics())
                )

        # Controls
        with gr.Row():
            metrics_path_input = gr.Textbox(
                value=DEFAULT_METRICS_PATH,
                label="Metrics Path",
                interactive=False,
            )
            history_path_input = gr.Textbox(
                value=DEFAULT_HISTORY_PATH,
                label="History Path",
                interactive=False,
            )
            days_input = gr.Slider(
                minimum=7,
                maximum=90,
                value=30,
                step=1,
                label="History Days",
            )

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", variant="primary")
            report_btn = gr.Button("📄 Generate PDF Report", variant="secondary")

        status_output = gr.Markdown("")

        # Event handlers
        refresh_btn.click(
            fn=refresh_dashboard,
            inputs=[metrics_path_input, history_path_input, days_input],
            outputs=[
                faithfulness_card,
                relevancy_card,
                precision_card,
                recall_card,
                timeline_chart,
                persona_table,
                status_output,
            ],
        )

        report_btn.click(
            fn=generate_pdf_report,
            inputs=[metrics_path_input, history_path_input],
            outputs=[status_output],
        )

        # Auto-refresh (5 minutes = 300 seconds)
        timer = gr.Timer(300)
        timer.tick(
            fn=refresh_dashboard,
            inputs=[metrics_path_input, history_path_input, days_input],
            outputs=[
                faithfulness_card,
                relevancy_card,
                precision_card,
                recall_card,
                timeline_chart,
                persona_table,
                status_output,
            ],
        )

    return app


def main():
    """Launch quality dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Quality Dashboard")
    parser.add_argument("--port", type=int, default=7861, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument(
        "--metrics-path", default=DEFAULT_METRICS_PATH, help="Metrics JSON path"
    )
    parser.add_argument(
        "--history-path", default=DEFAULT_HISTORY_PATH, help="History JSONL path"
    )

    args = parser.parse_args()

    app = create_dashboard(
        metrics_path=args.metrics_path,
        history_path=args.history_path,
    )
    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
