"""
FastAPI Routes for RAG Quality Metrics.

Provides REST API endpoints for:
- Latest quality metrics
- Historical metrics data
- Persona-based metrics comparison
- PDF report generation

Usage:
    These routes are integrated into the main FastAPI application.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

# Default paths
DEFAULT_METRICS_PATH = "data/quality_metrics.json"
DEFAULT_HISTORY_PATH = "data/quality_history.jsonl"

router = APIRouter(prefix="/quality", tags=["quality"])


def _load_metrics(metrics_path: str = DEFAULT_METRICS_PATH) -> Dict:
    """Load latest quality metrics from JSON file."""
    path = Path(metrics_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Failed to load metrics: {err}"
        ) from err


def _load_history(
    history_path: str = DEFAULT_HISTORY_PATH, days: int = 30
) -> List[Dict]:
    """Load quality metrics history from JSONL file."""
    path = Path(history_path)
    if not path.exists():
        return []

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
        return []

    return history


def _save_metrics(metrics: Dict, metrics_path: str = DEFAULT_METRICS_PATH) -> bool:
    """Save metrics to JSON file."""
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _append_history(record: Dict, history_path: str = DEFAULT_HISTORY_PATH) -> bool:
    """Append record to history JSONL file."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


@router.get("/metrics/latest")
async def get_latest_metrics(
    metrics_path: str = Query(
        DEFAULT_METRICS_PATH, description="Path to metrics JSON file"
    ),
) -> Dict:
    """
    Get the latest quality metrics.

    Returns:
        JSON with latest metrics including:
        - faithfulness: Score, trend, and change
        - answer_relevancy: Score, trend, and change
        - precision: Score, trend, and change
        - recall: Score, trend, and change
        - personas: Performance by persona type
        - timestamp: When metrics were recorded
    """
    return _load_metrics(metrics_path)


@router.get("/metrics/history")
async def get_metrics_history(
    days: int = Query(30, ge=1, le=90, description="Number of days to retrieve"),
    history_path: str = Query(
        DEFAULT_HISTORY_PATH, description="Path to history JSONL file"
    ),
) -> List[Dict]:
    """
    Get historical quality metrics.

    Args:
        days: Number of days of history to retrieve (1-90)

    Returns:
        List of historical metric records, each containing:
        - timestamp: When the record was created
        - faithfulness: Faithfulness score
        - answer_relevancy: Answer relevancy score
        - precision: Precision score
        - recall: Recall score
    """
    return _load_history(history_path, days)


@router.get("/metrics/personas")
async def get_persona_metrics(
    metrics_path: str = Query(
        DEFAULT_METRICS_PATH, description="Path to metrics JSON file"
    ),
) -> Dict:
    """
    Get quality metrics broken down by persona.

    Returns:
        JSON with persona-specific metrics:
        - personas: Dictionary with persona IDs as keys
        - Each persona contains:
            - score: Average quality score
            - count: Number of queries evaluated
        - best: Best performing persona
        - worst: Worst performing persona
    """
    metrics = _load_metrics(metrics_path)
    personas = metrics.get("personas", {})

    if not personas:
        return {"personas": {}, "best": None, "worst": None}

    # Sort by score
    sorted_personas = sorted(
        personas.items(), key=lambda x: x[1]["score"], reverse=True
    )
    best = {"id": sorted_personas[0][0], **sorted_personas[0][1]}
    worst = {"id": sorted_personas[-1][0], **sorted_personas[-1][1]}

    return {
        "personas": personas,
        "best": best,
        "worst": worst,
    }


@router.post("/metrics/record")
async def record_metrics(
    faithfulness: float = Query(..., ge=0, le=1, description="Faithfulness score"),
    answer_relevancy: float = Query(
        ..., ge=0, le=1, description="Answer relevancy score"
    ),
    precision: float = Query(..., ge=0, le=1, description="Precision score"),
    recall: float = Query(..., ge=0, le=1, description="Recall score"),
    persona: Optional[str] = Query(
        None, description="Persona type (faculty, student, staff)"
    ),
    metrics_path: str = Query(
        DEFAULT_METRICS_PATH, description="Path to metrics JSON file"
    ),
    history_path: str = Query(
        DEFAULT_HISTORY_PATH, description="Path to history JSONL file"
    ),
) -> Dict:
    """
    Record new quality metrics.

    This endpoint updates the latest metrics and appends to history.
    It also updates persona-specific statistics.

    Args:
        faithfulness: Faithfulness score (0-1)
        answer_relevancy: Answer relevancy score (0-1)
        precision: Precision score (0-1)
        recall: Recall score (0-1)
        persona: Optional persona type for categorization

    Returns:
        Success status and updated metrics
    """
    # Load existing metrics
    try:
        existing_metrics = _load_metrics(metrics_path)
    except HTTPException:
        existing_metrics = {
            "faithfulness": {"score": 0, "trend": "maintain", "change": 0},
            "answer_relevancy": {"score": 0, "trend": "maintain", "change": 0},
            "precision": {"score": 0, "trend": "maintain", "change": 0},
            "recall": {"score": 0, "trend": "maintain", "change": 0},
            "personas": {},
        }

    # Calculate trends
    def calculate_trend(old_score: float, new_score: float) -> tuple:
        change = new_score - old_score
        if change > 0.01:
            trend = "improve"
        elif change < -0.01:
            trend = "decrease"
        else:
            trend = "maintain"
        return trend, change

    # Update metrics with trends
    new_metrics = {
        "timestamp": datetime.now().isoformat(),
        "faithfulness": {
            "score": faithfulness,
            "trend": calculate_trend(
                existing_metrics["faithfulness"]["score"], faithfulness
            )[0],
            "change": calculate_trend(
                existing_metrics["faithfulness"]["score"], faithfulness
            )[1],
        },
        "answer_relevancy": {
            "score": answer_relevancy,
            "trend": calculate_trend(
                existing_metrics["answer_relevancy"]["score"], answer_relevancy
            )[0],
            "change": calculate_trend(
                existing_metrics["answer_relevancy"]["score"], answer_relevancy
            )[1],
        },
        "precision": {
            "score": precision,
            "trend": calculate_trend(existing_metrics["precision"]["score"], precision)[
                0
            ],
            "change": calculate_trend(
                existing_metrics["precision"]["score"], precision
            )[1],
        },
        "recall": {
            "score": recall,
            "trend": calculate_trend(existing_metrics["recall"]["score"], recall)[0],
            "change": calculate_trend(existing_metrics["recall"]["score"], recall)[1],
        },
        "personas": existing_metrics.get("personas", {}),
    }

    # Update persona stats if provided
    if persona:
        personas = new_metrics["personas"]
        if persona not in personas:
            personas[persona] = {"score": 0, "count": 0}

        # Update persona average score
        old_count = personas[persona]["count"]
        old_avg = personas[persona]["score"]
        new_avg = (old_avg * old_count + faithfulness) / (old_count + 1)
        personas[persona] = {"score": new_avg, "count": old_count + 1}

    # Save metrics
    if not _save_metrics(new_metrics, metrics_path):
        raise HTTPException(status_code=500, detail="Failed to save metrics")

    # Append to history
    history_record = {
        "timestamp": datetime.now().isoformat(),
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "precision": precision,
        "recall": recall,
    }

    if not _append_history(history_record, history_path):
        raise HTTPException(status_code=500, detail="Failed to append to history")

    return {"status": "success", "metrics": new_metrics}


@router.post("/report/generate")
async def generate_report(
    metrics_path: str = Query(
        DEFAULT_METRICS_PATH, description="Path to metrics JSON file"
    ),
    history_path: str = Query(
        DEFAULT_HISTORY_PATH, description="Path to history JSONL file"
    ),
) -> Dict:
    """
    Generate a PDF quality report.

    Returns:
        JSON with:
        - status: Success or error
        - path: Path to generated PDF file
        - url: Download URL for the report
    """
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
        pdf_filename = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = Path("data") / pdf_filename
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
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

        return {
            "status": "success",
            "path": str(pdf_path),
            "url": f"/api/quality/report/download/{pdf_filename}",
        }

    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Report generation failed: {err}"
        ) from err


@router.get("/report/download/{filename}")
async def download_report(filename: str) -> Dict:
    """
    Get download information for a report.

    Args:
        filename: Name of the report file

    Returns:
        JSON with download URL and file information
    """
    pdf_path = Path("data") / filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "status": "success",
        "filename": filename,
        "path": str(pdf_path),
        "size": pdf_path.stat().st_size,
        "created": datetime.fromtimestamp(pdf_path.stat().st_ctime).isoformat(),
    }


@router.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint for the quality metrics service.

    Returns:
        JSON with service status and configuration
    """
    metrics_file_exists = Path(DEFAULT_METRICS_PATH).exists()
    history_file_exists = Path(DEFAULT_HISTORY_PATH).exists()

    return {
        "status": "healthy",
        "service": "quality-metrics",
        "metrics_available": metrics_file_exists,
        "history_available": history_file_exists,
        "timestamp": datetime.now().isoformat(),
    }
