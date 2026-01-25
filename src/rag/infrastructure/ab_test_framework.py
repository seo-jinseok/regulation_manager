"""
A/B Testing Framework for Reranker Models (Cycle 5).

Provides framework for comparing multiple reranker models:
- Korean-specific models (Dongjin-kr/kr-reranker, NLPai/ko-reranker)
- Multilingual models (BAAI/bge-reranker-v2-m3)
- Performance metrics tracking
- Statistical analysis
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class RerankerModelType(Enum):
    """Types of reranker models."""
    MULTILINGUAL = "multilingual"
    KOREAN = "korean"


@dataclass
class ABTestMetrics:
    """Metrics for a single reranker model in A/B testing."""
    
    model_name: str
    model_type: RerankerModelType
    
    # Query metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Performance metrics
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Quality metrics (to be filled by external evaluation)
    avg_relevance_score: float = 0.0
    ndcg_score: float = 0.0  # Normalized Discounted Cumulative Gain
    
    # Timestamps
    first_query_time: Optional[datetime] = None
    last_query_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_relevance_score": self.avg_relevance_score,
            "ndcg_score": self.ndcg_score,
            "first_query_time": self.first_query_time.isoformat() if self.first_query_time else None,
            "last_query_time": self.last_query_time.isoformat() if self.last_query_time else None,
        }


@dataclass
class ABTestSession:
    """A/B testing session tracking multiple models."""
    
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Model metrics
    model_metrics: Dict[str, ABTestMetrics] = field(default_factory=dict)
    
    # Configuration
    test_ratio: float = 0.5  # Ratio of traffic to test model
    
    def get_metrics(self, model_name: str) -> ABTestMetrics:
        """Get or create metrics for a model."""
        if model_name not in self.model_metrics:
            # Detect Korean models by checking for korean, kr, or ko prefixes
            model_lower = model_name.lower()
            is_korean = (
                "korean" in model_lower or 
                "kr-" in model_lower or 
                model_lower.startswith("ko-") or
                "/kr/" in model_lower
            )
            
            self.model_metrics[model_name] = ABTestMetrics(
                model_name=model_name,
                model_type=RerankerModelType.KOREAN if is_korean else RerankerModelType.MULTILINGUAL,
                first_query_time=datetime.now(),
            )
        return self.model_metrics[model_name]
    
    def record_query(
        self,
        model_name: str,
        latency_ms: float,
        success: bool = True,
        relevance_score: float = 0.0,
    ) -> None:
        """Record a query result for a model."""
        metrics = self.get_metrics(model_name)
        metrics.total_queries += 1
        if success:
            metrics.successful_queries += 1
        else:
            metrics.failed_queries += 1
        
        metrics.total_latency_ms += latency_ms
        metrics.avg_latency_ms = metrics.total_latency_ms / metrics.total_queries
        metrics.last_query_time = datetime.now()
        
        if relevance_score > 0:
            # Update running average for relevance score
            n = metrics.successful_queries
            metrics.avg_relevance_score = (
                (metrics.avg_relevance_score * (n - 1) + relevance_score) / n
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "test_ratio": self.test_ratio,
            "model_metrics": {
                name: metrics.to_dict()
                for name, metrics in self.model_metrics.items()
            },
        }


class ABTestRepository:
    """Repository for storing A/B test results."""
    
    def __init__(self, storage_dir: str = ".metrics/reranker_ab"):
        """
        Initialize repository.
        
        Args:
            storage_dir: Directory to store test results.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_session(self, session: ABTestSession) -> str:
        """
        Save A/B test session to file.
        
        Args:
            session: ABTestSession to save.
            
        Returns:
            Path to saved file.
        """
        filename = f"ab_test_{session.session_id}.json"
        filepath = self.storage_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved A/B test session to {filepath}")
        return str(filepath)
    
    def load_session(self, session_id: str) -> Optional[ABTestSession]:
        """
        Load A/B test session from file.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            ABTestSession if found, None otherwise.
        """
        filepath = self.storage_dir / f"ab_test_{session_id}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError, KeyError):
            return None
        
        session = ABTestSession(
            session_id=data["session_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            test_ratio=data.get("test_ratio", 0.5),
        )
        
        for model_name, metrics_data in data.get("model_metrics", {}).items():
            metrics = ABTestMetrics(
                model_name=metrics_data["model_name"],
                model_type=RerankerModelType(metrics_data["model_type"]),
                total_queries=metrics_data.get("total_queries", 0),
                successful_queries=metrics_data.get("successful_queries", 0),
                failed_queries=metrics_data.get("failed_queries", 0),
                total_latency_ms=metrics_data.get("total_latency_ms", 0.0),
                avg_latency_ms=metrics_data.get("avg_latency_ms", 0.0),
                avg_relevance_score=metrics_data.get("avg_relevance_score", 0.0),
                ndcg_score=metrics_data.get("ndcg_score", 0.0),
                first_query_time=datetime.fromisoformat(metrics_data["first_query_time"]) if metrics_data.get("first_query_time") else None,
                last_query_time=datetime.fromisoformat(metrics_data["last_query_time"]) if metrics_data.get("last_query_time") else None,
            )
            session.model_metrics[model_name] = metrics
        
        return session
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        sessions = []
        for filepath in self.storage_dir.glob("ab_test_*.json"):
            session_id = filepath.stem.replace("ab_test_", "")
            sessions.append(session_id)
        return sorted(sessions)


class ABTestManager:
    """Manager for A/B testing reranker models."""
    
    def __init__(
        self,
        control_model: str,
        test_models: List[str],
        test_ratio: float = 0.5,
        repository: Optional[ABTestRepository] = None,
    ):
        """
        Initialize A/B test manager.
        
        Args:
            control_model: Primary model name (multilingual).
            test_models: List of Korean model names to test.
            test_ratio: Ratio of traffic to test models (0.0-1.0).
            repository: Optional repository for persistence.
        """
        self.control_model = control_model
        self.test_models = test_models
        self.test_ratio = max(0.0, min(1.0, test_ratio))
        self.repository = repository or ABTestRepository()
        
        # Current session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session = ABTestSession(
            session_id=self.session_id,
            test_ratio=self.test_ratio,
        )
        
        # Initialize metrics for all models
        self.session.get_metrics(control_model)
        for model in test_models:
            self.session.get_metrics(model)
    
    def select_model(self) -> str:
        """
        Select a model for the current query using A/B testing.
        
        Returns:
            Selected model name.
        """
        if random.random() < self.test_ratio and self.test_models:
            # Select a random test model
            return random.choice(self.test_models)
        return self.control_model
    
    def record_result(
        self,
        model_name: str,
        latency_ms: float,
        success: bool = True,
        relevance_score: float = 0.0,
    ) -> None:
        """
        Record query result for the selected model.
        
        Args:
            model_name: Model that was used.
            latency_ms: Query latency in milliseconds.
            success: Whether the query succeeded.
            relevance_score: Optional relevance score (0-1).
        """
        self.session.record_query(model_name, latency_ms, success, relevance_score)
        
        # Auto-save every 10 queries
        if self.session.model_metrics[model_name].total_queries % 10 == 0:
            self.repository.save_session(self.session)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of A/B test results.
        
        Returns:
            Summary dictionary with comparison metrics.
        """
        summary = {
            "session_id": self.session_id,
            "test_ratio": self.test_ratio,
            "models": {},
        }
        
        for model_name, metrics in self.session.model_metrics.items():
            summary["models"][model_name] = metrics.to_dict()
        
        # Calculate comparison
        if len(self.session.model_metrics) >= 2:
            control = self.session.get_metrics(self.control_model)
            
            for test_model in self.test_models:
                if test_model in self.session.model_metrics:
                    test = self.session.get_metrics(test_model)
                    
                    # Calculate relative performance
                    latency_improvement = (
                        (control.avg_latency_ms - test.avg_latency_ms) / control.avg_latency_ms * 100
                        if control.avg_latency_ms > 0
                        else 0
                    )
                    
                    relevance_improvement = (
                        (test.avg_relevance_score - control.avg_relevance_score) / control.avg_relevance_score * 100
                        if control.avg_relevance_score > 0
                        else 0
                    )
                    
                    summary[f"{test_model}_vs_{self.control_model}"] = {
                        "latency_improvement_percent": latency_improvement,
                        "relevance_improvement_percent": relevance_improvement,
                        "recommendation": self._get_recommendation(latency_improvement, relevance_improvement),
                    }
        
        return summary
    
    def _get_recommendation(self, latency_improvement: float, relevance_improvement: float) -> str:
        """Generate recommendation based on metrics."""
        if relevance_improvement > 10 and latency_improvement > -20:
            return "ADOPT: Test model shows significant improvement"
        elif relevance_improvement > 5:
            return "CONSIDER: Test model shows moderate improvement"
        elif latency_improvement > 20 and relevance_improvement > -5:
            return "CONSIDER: Test model is much faster"
        elif relevance_improvement < -10:
            return "REJECT: Test model performs worse"
        elif relevance_improvement < 0 and latency_improvement < 0:
            return "REJECT: Test model performs worse on both metrics"
        else:
            return "NEUTRAL: No significant difference"
    
    def save_session(self) -> str:
        """Save current session to repository."""
        self.session.end_time = datetime.now()
        return self.repository.save_session(self.session)
    
    def load_session(self, session_id: str) -> Optional[ABTestSession]:
        """Load a previous session."""
        return self.repository.load_session(session_id)


def create_ab_manager(
    control_model: str = "BAAI/bge-reranker-v2-m3",
    test_models: Optional[List[str]] = None,
    test_ratio: float = 0.5,
) -> ABTestManager:
    """
    Create an A/B test manager with default Korean models.
    
    Args:
        control_model: Primary multilingual model.
        test_models: List of Korean models to test.
        test_ratio: Ratio of traffic to test models.
        
    Returns:
        Configured ABTestManager.
    """
    if test_models is None:
        test_models = ["Dongjin-kr/kr-reranker", "NLPai/ko-reranker"]
    
    return ABTestManager(
        control_model=control_model,
        test_models=test_models,
        test_ratio=test_ratio,
    )
