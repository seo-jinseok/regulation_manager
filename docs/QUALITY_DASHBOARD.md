# RAG Quality Dashboard

Real-time monitoring dashboard for RAG answer quality metrics.

## Features

### Dashboard UI (Gradio)

- **4 Metric Cards**: Faithfulness, Answer Relevancy, Precision, Recall
  - Real-time score display
  - Trend indicators (📈 improve, 📉 decrease, ➡️ maintain)
  - Percentage change visualization

- **Timeline Chart**: 30-day historical view
  - Interactive Plotly chart
  - All 4 metrics on single graph
  - Dark theme optimized
  - Hover tooltips for detailed values

- **Persona Comparison Table**:
  - Faculty (👨‍🏫 교수)
  - Student (👨‍🎓 학생)
  - Staff (💼 직원)
  - Best/worst performer highlights

- **Auto-Refresh**: Every 5 minutes
- **PDF Export**: Generate downloadable reports
- **Fast Load**: Under 2 seconds

### FastAPI Endpoints

#### `GET /quality/metrics/latest`
Get the latest quality metrics.

**Response:**
```json
{
  "timestamp": "2025-01-29T12:00:00",
  "faithfulness": {"score": 0.85, "trend": "improve", "change": 0.05},
  "answer_relevancy": {"score": 0.78, "trend": "maintain", "change": 0.01},
  "precision": {"score": 0.82, "trend": "improve", "change": 0.03},
  "recall": {"score": 0.75, "trend": "decrease", "change": -0.02},
  "personas": {
    "faculty": {"score": 0.88, "count": 45},
    "student": {"score": 0.76, "count": 120},
    "staff": {"score": 0.82, "count": 35}
  }
}
```

#### `GET /quality/metrics/history?days=30`
Get historical quality metrics.

**Query Parameters:**
- `days`: Number of days (1-90, default: 30)

**Response:**
```json
[
  {"timestamp": "2025-01-01T00:00:00", "faithfulness": 0.80, ...},
  {"timestamp": "2025-01-02T00:00:00", "faithfulness": 0.81, ...},
  ...
]
```

#### `GET /quality/metrics/personas`
Get persona-specific metrics.

**Response:**
```json
{
  "personas": {
    "faculty": {"score": 0.88, "count": 45},
    "student": {"score": 0.76, "count": 120},
    "staff": {"score": 0.82, "count": 35}
  },
  "best": {"id": "faculty", "score": 0.88, "count": 45},
  "worst": {"id": "student", "score": 0.76, "count": 120}
}
```

#### `POST /quality/metrics/record`
Record new quality metrics.

**Query Parameters:**
- `faithfulness`: Score (0-1)
- `answer_relevancy`: Score (0-1)
- `precision`: Score (0-1)
- `recall`: Score (0-1)
- `persona`: Optional persona type (faculty, student, staff)

#### `POST /quality/report/generate`
Generate PDF quality report.

**Response:**
```json
{
  "status": "success",
  "path": "data/quality_report_20250129_120000.pdf",
  "url": "/api/quality/report/download/quality_report_20250129_120000.pdf"
}
```

#### `GET /quality/health`
Health check endpoint.

## Usage

### Running the Dashboard

```bash
# Generate test data (optional)
uv run python scripts/generate_quality_test_data.py

# Start the dashboard
uv run python -m src.rag.interface.web.quality_dashboard

# With custom options
uv run python -m src.rag.interface.web.quality_dashboard --port 7861 --share
```

### Running Tests

```bash
# Run all dashboard tests
uv run pytest tests/test_quality_dashboard.py

# Run specific test class
uv run pytest tests/test_quality_dashboard.py::TestQualityDashboard

# Run integration tests
uv run pytest tests/test_quality_dashboard.py -m integration
```

## Data Files

### `data/quality_metrics.json`
Latest quality metrics snapshot.

```json
{
  "timestamp": "2025-01-29T12:00:00",
  "faithfulness": {"score": 0.85, "trend": "improve", "change": 0.05},
  ...
}
```

### `data/quality_history.jsonl`
Historical metrics (one JSON object per line).

```jsonl
{"timestamp": "2025-01-01T00:00:00", "faithfulness": 0.80, ...}
{"timestamp": "2025-01-02T00:00:00", "faithfulness": 0.81, ...}
```

## Integration with RAG System

### Recording Metrics

```python
from src.rag.interface.web.routes.quality import _save_metrics, _append_history

# After RAG query evaluation
metrics = {
    "timestamp": datetime.now().isoformat(),
    "faithfulness": {"score": 0.85, "trend": "improve", "change": 0.05},
    # ... other metrics
}

_save_metrics(metrics, "data/quality_metrics.json")

# Append to history
history_record = {
    "timestamp": datetime.now().isoformat(),
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "precision": 0.82,
    "recall": 0.75,
}
_append_history(history_record, "data/quality_history.jsonl")
```

### Using with RAGAS/DeepEval

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Evaluate RAG outputs
result = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy]
)

# Extract scores and record
faithfulness_score = result['faithfulness']
answer_relevancy_score = result['answer_relevancy']

# Record to dashboard
_save_metrics({...})
```

## Dependencies

- `gradio>=6.2.0` - Dashboard UI
- `plotly>=5.18.0` - Interactive charts
- `reportlab>=4.0.0` - PDF report generation
- `fastapi` - API routes
- `ragas>=0.4.3` - RAG evaluation
- `deepeval>=3.8.1` - Alternative evaluation

## Architecture

```
src/rag/interface/web/
├── __init__.py
├── quality_dashboard.py    # Gradio UI
└── routes/
    ├── __init__.py
    └── quality.py          # FastAPI routes
```

## Success Criteria

- ✅ Dashboard loads in < 2 seconds
- ✅ All 4 metric cards display correctly
- ✅ Timeline chart shows 30-day history
- ✅ Persona comparison table highlights best/worst
- ✅ Auto-refresh every 5 minutes
- ✅ PDF report generation works
- ✅ FastAPI endpoints return correct data
- ✅ Tests cover core functionality
