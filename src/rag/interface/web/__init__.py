"""
Web Interface for RAG Quality Dashboard and Live Monitoring.

This package contains:
- LiveMonitor: Real-time RAG pipeline event monitoring (SPEC-RAG-MONITOR-001 Phase 4)
- Quality Dashboard: RAG quality evaluation dashboard
"""

from .live_monitor import LiveMonitor, create_monitor_tab

# Optional imports (may not be available)
try:
    from .quality_dashboard import create_dashboard, main

    QUALITY_DASHBOARD_AVAILABLE = True
except ImportError:
    QUALITY_DASHBOARD_AVAILABLE = False
    create_dashboard = None
    main = None

__all__ = [
    # Live Monitor (Phase 4)
    "LiveMonitor",
    "create_monitor_tab",
    # Quality Dashboard (optional)
    "create_dashboard",
    "main",
]
