"""Unit tests for System Health module."""

import ast
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.rag.domain.evaluation.system_health import (
    BareExceptVisitor,
    CodeFinding,
    CodeQualityReport,
    CodeQualityScanner,
    ConfigDriftDetector,
    ConfigDriftResult,
    CoverageDelta,
    CoverageDeltaChecker,
    FunctionLengthVisitor,
    SystemHealthReport,
    run_health_scan,
)


class TestBareExceptVisitor:
    """Test AST visitor for bare except detection."""

    def test_detect_bare_except(self):
        code = """
try:
    x = 1
except:
    pass
"""
        tree = ast.parse(code)
        visitor = BareExceptVisitor()
        visitor.visit(tree)
        assert len(visitor.findings) == 1
        assert visitor.findings[0][1] == "bare except"

    def test_detect_except_exception_pass(self):
        code = """
try:
    x = 1
except Exception:
    pass
"""
        tree = ast.parse(code)
        visitor = BareExceptVisitor()
        visitor.visit(tree)
        assert len(visitor.findings) == 1
        assert "Exception: pass" in visitor.findings[0][1]

    def test_no_detection_for_specific_except(self):
        code = """
try:
    x = 1
except ValueError as e:
    logger.error(e)
"""
        tree = ast.parse(code)
        visitor = BareExceptVisitor()
        visitor.visit(tree)
        assert len(visitor.findings) == 0

    def test_no_detection_for_except_with_body(self):
        code = """
try:
    x = 1
except Exception as e:
    logger.error(e)
    raise
"""
        tree = ast.parse(code)
        visitor = BareExceptVisitor()
        visitor.visit(tree)
        assert len(visitor.findings) == 0


class TestFunctionLengthVisitor:
    """Test function length detection."""

    def test_detect_function_length(self):
        code = """
def short_fn():
    return 1

def long_fn():
    x = 1
    y = 2
    z = 3
    return x + y + z
"""
        tree = ast.parse(code)
        visitor = FunctionLengthVisitor()
        visitor.visit(tree)
        assert len(visitor.functions) == 2
        names = [f[0] for f in visitor.functions]
        assert "short_fn" in names
        assert "long_fn" in names


class TestCodeQualityScanner:
    """Test CodeQualityScanner."""

    @pytest.fixture
    def src_dir(self, tmp_path):
        """Create a test source directory with sample files."""
        src = tmp_path / "src"
        src.mkdir()
        module = src / "sample.py"
        module.write_text(
            """
import os

# TODO: fix this
def short_fn():
    return 1

def handle_error():
    try:
        x = 1
    except:
        pass

# FIXME: also fix this
if x == 42:
    print("magic number")
""",
            encoding="utf-8",
        )
        return tmp_path

    def test_scan_finds_bare_except(self, src_dir):
        scanner = CodeQualityScanner(src_dir=str(src_dir / "src"))
        report = scanner.scan()
        assert report.bare_except_count >= 1

    def test_scan_finds_todos(self, src_dir):
        scanner = CodeQualityScanner(src_dir=str(src_dir / "src"))
        report = scanner.scan()
        assert report.todo_count >= 1
        assert report.fixme_count >= 1

    def test_scan_counts_files(self, src_dir):
        scanner = CodeQualityScanner(src_dir=str(src_dir / "src"))
        report = scanner.scan()
        assert report.files_scanned >= 1

    def test_scan_nonexistent_dir(self):
        scanner = CodeQualityScanner(src_dir="/nonexistent")
        report = scanner.scan()
        assert report.files_scanned == 0

    def test_report_to_dict(self):
        report = CodeQualityReport(bare_except_count=3, todo_count=5)
        d = report.to_dict()
        assert d["bare_except_count"] == 3
        assert d["todo_count"] == 5
        assert "findings_count" in d


class TestCoverageDeltaChecker:
    """Test CoverageDeltaChecker."""

    @pytest.fixture
    def coverage_file(self, tmp_path):
        data = {
            "totals": {"percent_covered": 87.5},
            "files": {
                "src/module_a.py": {"summary": {"percent_covered": 95.0}},
                "src/module_b.py": {"summary": {"percent_covered": 60.0}},
            },
        }
        path = tmp_path / "coverage.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    @pytest.fixture
    def history_file(self, tmp_path):
        data = {"coverage": {"current_coverage": 85.0}}
        path = tmp_path / "health_scan.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_check_coverage(self, coverage_file, history_file):
        checker = CoverageDeltaChecker(
            coverage_path=str(coverage_file),
            history_path=str(history_file),
        )
        result = checker.check()
        assert result.current_coverage == 87.5
        assert result.previous_coverage == 85.0
        assert result.delta == pytest.approx(2.5)
        assert result.passed is True

    def test_low_coverage_files(self, coverage_file, history_file):
        checker = CoverageDeltaChecker(
            coverage_path=str(coverage_file),
            history_path=str(history_file),
        )
        result = checker.check()
        assert len(result.low_coverage_files) == 1
        assert result.low_coverage_files[0]["file"] == "src/module_b.py"

    def test_missing_coverage_file(self, tmp_path):
        checker = CoverageDeltaChecker(
            coverage_path=str(tmp_path / "missing.json"),
            history_path=str(tmp_path / "missing_history.json"),
        )
        result = checker.check()
        assert result.passed is False

    def test_coverage_result_to_dict(self):
        result = CoverageDelta(current_coverage=90.0, delta=2.0)
        d = result.to_dict()
        assert d["current_coverage"] == 90.0


class TestConfigDriftDetector:
    """Test ConfigDriftDetector."""

    def test_detect_missing_env_var(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "empty.py").write_text("pass", encoding="utf-8")

        with patch.dict(os.environ, {}, clear=True):
            detector = ConfigDriftDetector(src_dir=str(src))
            result = detector.check()
            assert "OPENROUTER_API_KEY" in result.missing_env_vars
            assert result.status == "DRIFT_DETECTED"

    def test_no_drift_with_env_set(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "empty.py").write_text("pass", encoding="utf-8")

        env = {"OPENROUTER_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            detector = ConfigDriftDetector(src_dir=str(src))
            result = detector.check()
            assert result.missing_env_vars == []
            assert result.status == "OK"

    def test_detect_hardcoded_secret(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "bad_code.py").write_text(
            'api_key = "sk-1234567890abcdef"',
            encoding="utf-8",
        )

        env = {"OPENROUTER_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            detector = ConfigDriftDetector(src_dir=str(src))
            result = detector.check()
            assert len(result.hardcoded_secrets) >= 1
            assert result.status == "DRIFT_DETECTED"

    def test_result_to_dict(self):
        result = ConfigDriftResult(status="OK")
        d = result.to_dict()
        assert d["status"] == "OK"


class TestSystemHealthReport:
    """Test SystemHealthReport."""

    def test_to_dict(self):
        report = SystemHealthReport(overall_status="HEALTHY")
        d = report.to_dict()
        assert d["overall_status"] == "HEALTHY"

    def test_save(self, tmp_path):
        report = SystemHealthReport(overall_status="WARNING")
        path = tmp_path / "health.json"
        report.save(str(path))
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["overall_status"] == "WARNING"


class TestRunHealthScan:
    """Test run_health_scan integration."""

    def test_run_scan(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "module.py").write_text("pass", encoding="utf-8")

        coverage = tmp_path / "coverage.json"
        coverage.write_text(
            json.dumps({"totals": {"percent_covered": 90.0}, "files": {}}),
            encoding="utf-8",
        )

        env = {"OPENROUTER_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            report = run_health_scan(
                src_dir=str(src),
                coverage_path=str(coverage),
                history_path=str(tmp_path / "history.json"),
            )
            assert report.overall_status in ("HEALTHY", "WARNING", "CRITICAL")
            assert report.code_quality is not None
            assert report.coverage is not None
            assert report.config_drift is not None
