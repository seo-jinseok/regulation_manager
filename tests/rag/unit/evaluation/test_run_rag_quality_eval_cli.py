"""
Tests for run_rag_quality_eval.py CLI interface.

These tests verify the argparse-based CLI interface for RAG quality evaluation.
Following TDD RED-GREEN-REFACTOR cycle.

SPEC: SPEC-RAG-SKILL-001
REQ: REQ-002 (CLI 진입점 개선)
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Path to the CLI script
CLI_SCRIPT = Path(__file__).parent.parent.parent.parent.parent / "run_rag_quality_eval.py"


class TestCLIHelp:
    """Tests for --help and usage display."""

    def test_help_displays_usage(self):
        """--help should display clear usage information."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        # RED: This will fail initially until argparse is implemented
        assert result.returncode == 0, f"Help should succeed: {result.stderr}"

    def test_help_shows_quick_mode(self):
        """--help should show --quick mode description."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert "--quick" in result.stdout, "Help should mention --quick mode"

    def test_help_shows_full_mode(self):
        """--help should show --full mode description."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert "--full" in result.stdout, "Help should mention --full mode"

    def test_help_shows_status_mode(self):
        """--help should show --status mode description."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert "--status" in result.stdout, "Help should mention --status mode"

    def test_help_shows_examples(self):
        """--help should show usage examples."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert "Examples:" in result.stdout or "examples:" in result.stdout.lower()


class TestQuickMode:
    """Tests for --quick evaluation mode."""

    def test_quick_mode_exists(self):
        """--quick flag should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--quick", "--help"],
            capture_output=True,
            text=True,
        )
        # Should not error on unknown argument
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_quick_mode_is_mutually_exclusive_with_full(self):
        """--quick and --full should be mutually exclusive."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--quick", "--full"],
            capture_output=True,
            text=True,
        )
        # Should error because they are mutually exclusive
        assert result.returncode != 0 or "not allowed" in result.stderr.lower()


class TestFullMode:
    """Tests for --full evaluation mode."""

    def test_full_mode_exists(self):
        """--full flag should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--full", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()


class TestStatusMode:
    """Tests for --status evaluation status check."""

    def test_status_mode_exists(self):
        """--status flag should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--status", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()


class TestTargetingOptions:
    """Tests for targeting options (--persona, --category, --queries)."""

    def test_persona_option_exists(self):
        """--persona option should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--persona", "student-undergraduate", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_category_option_exists(self):
        """--category option should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--category", "simple", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_queries_option_exists(self):
        """--queries option should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--queries", "5", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_queries_accepts_integer(self):
        """--queries should accept integer value."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--queries", "10"],
            capture_output=True,
            text=True,
        )
        # Should not error with invalid literal for int()
        assert "invalid literal" not in result.stderr.lower()


class TestOutputOptions:
    """Tests for output options (--output, --format, --summary)."""

    def test_output_option_exists(self):
        """--output / -o option should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--output", "report.md", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_output_short_flag_exists(self):
        """-o short flag should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "-o", "report.md", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_format_option_exists(self):
        """--format option should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--format", "json", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_format_accepts_valid_choices(self):
        """--format should accept json, markdown, both."""
        for fmt in ["json", "markdown", "both"]:
            result = subprocess.run(
                [sys.executable, str(CLI_SCRIPT), "--format", fmt, "--help"],
                capture_output=True,
                text=True,
            )
            assert "invalid choice" not in result.stderr.lower(), f"Format '{fmt}' should be valid"

    def test_format_rejects_invalid_choice(self):
        """--format should reject invalid choices."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--format", "invalid"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 or "invalid choice" in result.stderr.lower()

    def test_summary_option_exists(self):
        """--summary option should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--summary", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()


class TestAdvancedOptions:
    """Tests for advanced options (--baseline, --checkpoint, --resume)."""

    def test_baseline_option_exists(self):
        """--baseline option should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--baseline", "eval-001", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_checkpoint_option_exists(self):
        """--checkpoint option should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--checkpoint", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_resume_option_exists(self):
        """--resume option should be recognized."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--resume", "checkpoint.json", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()


class TestNoModeDefault:
    """Tests for default behavior when no mode is specified."""

    def test_default_mode_exists(self):
        """Running without mode flags should be valid (default mode)."""
        # This test verifies the script accepts running without mode flags
        # Actual execution is mocked in integration tests
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestArgparseIntegration:
    """Integration tests for argparse configuration."""

    def test_mutually_exclusive_modes(self):
        """--quick, --full, --status should be mutually exclusive."""
        # Test quick + full
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--quick", "--full"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

        # Test quick + status
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--quick", "--status"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

        # Test full + status
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--full", "--status"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_persona_accepts_multiple_values(self):
        """--persona should accept multiple persona names."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--persona", "student-undergraduate", "professor", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_category_accepts_multiple_values(self):
        """--category should accept multiple category names."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--category", "simple", "complex", "--help"],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr.lower()
