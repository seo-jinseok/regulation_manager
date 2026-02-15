"""
Unit tests for evaluation CLI commands (P2).

Tests for SPEC-RAG-EVAL-001 P2: Interface Integration.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.rag.interface.unified_cli import (
    _cmd_quality_generate_spec,
    _cmd_quality_resume,
    _cmd_quality_run,
    _cmd_quality_status,
)


@pytest.fixture
def mock_args():
    """Create mock args object."""
    class MockArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return MockArgs


@pytest.fixture
def mock_console():
    """Create mock console."""
    return MagicMock()


@pytest.fixture
def mock_store():
    """Create mock evaluation store."""
    store = MagicMock()
    store.get_statistics.return_value = MagicMock(
        total_evaluations=100,
        avg_overall_score=0.75,
        pass_rate=0.7,
        min_score=0.3,
        max_score=0.95,
        std_deviation=0.15,
        trend="improving",
        avg_faithfulness=0.80,
        avg_answer_relevancy=0.78,
        avg_contextual_precision=0.72,
        avg_contextual_recall=0.70,
    )
    return store


@pytest.fixture
def mock_evaluator():
    """Create mock evaluator."""
    evaluator = MagicMock()
    result = MagicMock()
    result.overall_score = 0.85
    result.faithfulness = 0.90
    result.answer_relevancy = 0.88
    result.contextual_precision = 0.82
    result.contextual_recall = 0.80
    result.persona = "test"
    evaluator.evaluate_single_turn.return_value = result
    return evaluator


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestCmdQualityRun:
    """Tests for _cmd_quality_run command."""

    def test_run_basic_flow(self, mock_args, mock_console, mock_evaluator, mock_store):
        """Test basic evaluation run flow."""
        args = mock_args(
            session_id=None,
            personas=[],
            queries_per_persona=2,
            batch_size=5,
            no_checkpoint=True,
            output=None,
        )

        persona_mgr = MagicMock()
        persona_mgr.list_personas.return_value = ["freshman"]
        persona_mgr.generate_queries.return_value = ["query1", "query2"]

        search_usecase = MagicMock()
        search_usecase.search.return_value = [
            MagicMock(chunk=MagicMock(text="context1"))
        ]

        llm_client = MagicMock()

        with patch('src.rag.application.evaluation.CheckpointManager') as mock_checkpoint:
            mock_checkpoint.return_value = MagicMock()

            with patch('src.rag.application.evaluation.ProgressReporter') as mock_reporter:
                mock_reporter_instance = MagicMock()
                mock_reporter_instance.get_progress.return_value = MagicMock(
                    completed=2, total=2
                )
                mock_reporter_instance.get_eta.return_value = 0
                mock_reporter.return_value = mock_reporter_instance

                with patch('src.rag.infrastructure.tool_executor.ToolExecutor') as mock_tool:
                    mock_tool_instance = MagicMock()
                    mock_tool_instance._handle_generate_answer.return_value = "answer"
                    mock_tool.return_value = mock_tool_instance

                    result = _cmd_quality_run(
                        args, mock_console, mock_evaluator, mock_store,
                        persona_mgr, search_usecase, llm_client
                    )

                    assert result == 0

    def test_run_with_session_id(self, mock_args, mock_console, mock_evaluator, mock_store):
        """Test run with existing session ID."""
        args = mock_args(
            session_id="test-session",
            personas=[],
            queries_per_persona=2,
            batch_size=5,
            no_checkpoint=False,
            output=None,
        )

        persona_mgr = MagicMock()
        persona_mgr.list_personas.return_value = []
        persona_mgr.generate_queries.return_value = []

        search_usecase = MagicMock()
        llm_client = MagicMock()

        with patch('src.rag.application.evaluation.CheckpointManager') as mock_checkpoint:
            checkpoint_mgr = MagicMock()
            progress = MagicMock()
            progress.session_id = "test-session"
            checkpoint_mgr.load_checkpoint.return_value = progress
            mock_checkpoint.return_value = checkpoint_mgr

            result = _cmd_quality_run(
                args, mock_console, mock_evaluator, mock_store,
                persona_mgr, search_usecase, llm_client
            )

            # Should exit early since no personas
            assert result == 0

    def test_run_saves_report(self, mock_args, mock_console, mock_evaluator, mock_store):
        """Test that run saves report to output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.json")

            args = mock_args(
                session_id=None,
                personas=[],
                queries_per_persona=1,
                batch_size=5,
                no_checkpoint=True,
                output=output_path,
            )

            persona_mgr = MagicMock()
            persona_mgr.list_personas.return_value = ["freshman"]
            persona_mgr.generate_queries.return_value = ["query1"]

            search_usecase = MagicMock()
            search_usecase.search.return_value = [
                MagicMock(chunk=MagicMock(text="context1"))
            ]

            llm_client = MagicMock()

            with patch('src.rag.application.evaluation.CheckpointManager'):
                with patch('src.rag.application.evaluation.ProgressReporter') as mock_reporter:
                    mock_reporter_instance = MagicMock()
                    mock_reporter_instance.get_progress.return_value = MagicMock(
                        completed=1, total=1
                    )
                    mock_reporter_instance.get_eta.return_value = 0
                    mock_reporter.return_value = mock_reporter_instance

                    with patch('src.rag.infrastructure.tool_executor.ToolExecutor') as mock_tool:
                        mock_tool_instance = MagicMock()
                        mock_tool_instance._handle_generate_answer.return_value = "answer"
                        mock_tool.return_value = mock_tool_instance

                        result = _cmd_quality_run(
                            args, mock_console, mock_evaluator, mock_store,
                            persona_mgr, search_usecase, llm_client
                        )

                        # Report file should exist
                        assert Path(output_path).exists()


class TestCmdQualityResume:
    """Tests for _cmd_quality_resume command."""

    def test_resume_list_sessions(self, mock_args, mock_console, mock_store):
        """Test listing resumable sessions."""
        args = mock_args(
            list=True,
            session_id=None,
        )

        with patch('src.rag.application.evaluation.CheckpointManager'):
            with patch('src.rag.application.evaluation.ResumeController') as mock_resume:
                resume_ctrl = MagicMock()
                resume_ctrl.find_interrupted_sessions.return_value = [
                    {"session_id": "session-1", "completion_rate": 50, "updated_at": "2024-01-01"},
                ]
                mock_resume.return_value = resume_ctrl

                result = _cmd_quality_resume(args, mock_console, mock_store)

                assert result == 0

    def test_resume_no_sessions_available(self, mock_args, mock_console, mock_store):
        """Test resume when no sessions available."""
        args = mock_args(
            list=False,
            session_id=None,
        )

        with patch('src.rag.application.evaluation.CheckpointManager'):
            with patch('src.rag.application.evaluation.ResumeController') as mock_resume:
                resume_ctrl = MagicMock()
                resume_ctrl.get_resume_recommendation.return_value = None
                mock_resume.return_value = resume_ctrl

                result = _cmd_quality_resume(args, mock_console, mock_store)

                assert result == 1

    def test_resume_specific_session(self, mock_args, mock_console, mock_store):
        """Test resuming specific session."""
        args = mock_args(
            list=False,
            session_id="session-123",
        )

        with patch('src.rag.application.evaluation.CheckpointManager'):
            with patch('src.rag.application.evaluation.ResumeController') as mock_resume:
                resume_ctrl = MagicMock()
                resume_ctrl.can_resume.return_value = (True, "Can resume")
                resume_ctrl.get_resume_context.return_value = MagicMock(
                    completion_rate=50.0,
                    completed_count=50,
                    total_count=100,
                    failed_count=5,
                    remaining_personas=["professor"],
                )
                mock_resume.return_value = resume_ctrl

                result = _cmd_quality_resume(args, mock_console, mock_store)

                assert result == 0


class TestCmdQualityGenerateSpec:
    """Tests for _cmd_quality_generate_spec command."""

    def test_generate_spec_no_failures(self, mock_args, mock_console, mock_store):
        """Test SPEC generation when no failures exist."""
        args = mock_args(
            session_id=None,
            output=None,
            threshold=0.6,
        )

        with patch('src.rag.infrastructure.storage.evaluation_store.EvaluationStore') as mock_eval_store:
            mock_eval_store.return_value.get_evaluations.return_value = []

            result = _cmd_quality_generate_spec(args, mock_console, mock_store)

            assert result == 0

    def test_generate_spec_with_failures(self, mock_args, mock_console, mock_store):
        """Test SPEC generation with failures."""
        from src.rag.domain.evaluation import FailureType

        args = mock_args(
            session_id=None,
            output=None,
            threshold=0.6,
        )

        # Mock failed evaluations
        failed_eval = MagicMock()
        failed_eval.overall_score = 0.5
        failed_eval.query = "test query"
        failed_eval.answer = "test answer"
        failed_eval.contexts = ["context1"]

        with patch('src.rag.infrastructure.storage.evaluation_store.EvaluationStore') as mock_eval_store:
            mock_eval_store.return_value.get_evaluations.return_value = [failed_eval]

            with patch('src.rag.domain.evaluation.FailureClassifier') as mock_classifier:
                mock_classifier_instance = MagicMock()
                # Use actual FailureType enum
                mock_classifier_instance.classify_batch.return_value = [
                    MagicMock(
                        failure_type=FailureType.HALLUCINATION,
                        count=5,
                        avg_score=0.5,
                    )
                ]
                mock_classifier.return_value = mock_classifier_instance

                with patch('src.rag.domain.evaluation.RecommendationEngine') as mock_engine:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.generate_recommendations.return_value = []
                    mock_engine_instance.get_action_plan.return_value = {
                        "immediate_actions": [],
                        "short_term_actions": [],
                        "long_term_actions": [],
                    }
                    mock_engine.return_value = mock_engine_instance

                    with patch('src.rag.domain.evaluation.SPECGenerator') as mock_gen:
                        mock_spec = MagicMock()
                        mock_spec.to_markdown.return_value = "# Test SPEC"
                        mock_gen_instance = MagicMock()
                        mock_gen_instance.generate_spec.return_value = mock_spec
                        mock_gen.return_value = mock_gen_instance

                        result = _cmd_quality_generate_spec(args, mock_console, mock_store)

                        assert result == 0

    def test_generate_spec_with_output(self, mock_args, mock_console, mock_store):
        """Test SPEC generation with output file."""
        from src.rag.domain.evaluation import FailureType

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "spec.md")

            args = mock_args(
                session_id=None,
                output=output_path,
                threshold=0.6,
            )

            failed_eval = MagicMock()
            failed_eval.overall_score = 0.5
            failed_eval.query = "test query"
            failed_eval.answer = "test answer"
            failed_eval.contexts = ["context1"]

            with patch('src.rag.infrastructure.storage.evaluation_store.EvaluationStore') as mock_eval_store:
                mock_eval_store.return_value.get_evaluations.return_value = [failed_eval]

                with patch('src.rag.domain.evaluation.FailureClassifier') as mock_classifier_class:
                    # Create a proper mock summary with explicit attribute assignment
                    mock_summary = MagicMock()
                    mock_summary.failure_type = FailureType.HALLUCINATION
                    mock_summary.count = 5
                    mock_summary.avg_score = 0.5

                    # Mock the classifier instance
                    mock_classifier_instance = MagicMock()
                    mock_classifier_instance.classify_batch.return_value = [mock_summary]

                    # When FailureClassifier() is called, return our mock instance
                    mock_classifier_class.return_value = mock_classifier_instance

                    with patch('src.rag.domain.evaluation.RecommendationEngine') as mock_engine_class:
                        mock_engine_instance = MagicMock()
                        mock_engine_instance.generate_recommendations.return_value = []
                        mock_engine_instance.get_action_plan.return_value = {
                            "immediate_actions": [],
                            "short_term_actions": [],
                            "long_term_actions": [],
                        }
                        mock_engine_class.return_value = mock_engine_instance

                        with patch('src.rag.domain.evaluation.SPECGenerator') as mock_gen_class:
                            mock_spec = MagicMock()
                            mock_spec.to_markdown.return_value = "# Test SPEC"
                            mock_gen_instance = MagicMock()
                            mock_gen_instance.generate_spec.return_value = mock_spec
                            mock_gen_instance.save_spec.return_value = output_path
                            mock_gen_class.return_value = mock_gen_instance

                            result = _cmd_quality_generate_spec(args, mock_console, mock_store)

                            assert result == 0


class TestCmdQualityStatus:
    """Tests for _cmd_quality_status command."""

    def test_status_cleanup(self, mock_args, mock_console, mock_store):
        """Test cleanup option."""
        args = mock_args(
            cleanup=True,
            session_id=None,
            all=False,
        )

        with patch('src.rag.application.evaluation.CheckpointManager') as mock_checkpoint:
            checkpoint_mgr = MagicMock()
            checkpoint_mgr.cleanup_completed_sessions.return_value = 3
            mock_checkpoint.return_value = checkpoint_mgr

            result = _cmd_quality_status(args, mock_console, mock_store)

            assert result == 0

    def test_status_specific_session(self, mock_args, mock_console, mock_store):
        """Test status for specific session."""
        args = mock_args(
            cleanup=False,
            session_id="session-123",
            all=False,
        )

        with patch('src.rag.application.evaluation.CheckpointManager') as mock_checkpoint:
            checkpoint_mgr = MagicMock()
            progress = MagicMock()
            progress.session_id = "session-123"
            progress.status = "running"
            progress.started_at = "2024-01-01"
            progress.updated_at = "2024-01-02"
            progress.completed_queries = 50
            progress.total_queries = 100
            progress.completion_rate = 50.0
            progress.personas = {"freshman": MagicMock(completed_queries=50, total_queries=50, failed_queries=0)}
            checkpoint_mgr.load_checkpoint.return_value = progress
            mock_checkpoint.return_value = checkpoint_mgr

            result = _cmd_quality_status(args, mock_console, mock_store)

            assert result == 0

    def test_status_session_not_found(self, mock_args, mock_console, mock_store):
        """Test status for non-existent session."""
        args = mock_args(
            cleanup=False,
            session_id="nonexistent",
            all=False,
        )

        with patch('src.rag.application.evaluation.CheckpointManager') as mock_checkpoint:
            checkpoint_mgr = MagicMock()
            checkpoint_mgr.load_checkpoint.return_value = None
            mock_checkpoint.return_value = checkpoint_mgr

            result = _cmd_quality_status(args, mock_console, mock_store)

            assert result == 1

    def test_status_all_sessions(self, mock_args, mock_console, mock_store):
        """Test status for all sessions."""
        args = mock_args(
            cleanup=False,
            session_id=None,
            all=True,
        )

        with patch('src.rag.application.evaluation.CheckpointManager') as mock_checkpoint:
            checkpoint_mgr = MagicMock()
            checkpoint_mgr.list_sessions.return_value = [
                {"session_id": "s1", "status": "running", "completion_rate": 50, "updated_at": "2024-01-01"},
                {"session_id": "s2", "status": "completed", "completion_rate": 100, "updated_at": "2024-01-02"},
            ]
            mock_checkpoint.return_value = checkpoint_mgr

            result = _cmd_quality_status(args, mock_console, mock_store)

            assert result == 0

    def test_status_no_sessions(self, mock_args, mock_console, mock_store):
        """Test status when no sessions exist."""
        args = mock_args(
            cleanup=False,
            session_id=None,
            all=False,
        )

        with patch('src.rag.application.evaluation.CheckpointManager') as mock_checkpoint:
            checkpoint_mgr = MagicMock()
            checkpoint_mgr.list_sessions.return_value = []
            mock_checkpoint.return_value = checkpoint_mgr

            result = _cmd_quality_status(args, mock_console, mock_store)

            assert result == 0
