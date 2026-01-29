"""
Web Interface for RAG Quality Dashboard.

This package contains the Gradio dashboard and FastAPI routes
for real-time quality monitoring.
"""

from .quality_dashboard import create_dashboard, main
from .routes import router

__all__ = ["create_dashboard", "main", "router"]
