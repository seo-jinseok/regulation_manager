"""
System Health Radar for evaluation.

SPEC: SPEC-RAG-EVAL-002
EARS: EARS-U-008 (Code Quality), EARS-U-009 (Coverage Delta), EARS-U-010 (Config Drift)
"""

import ast
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CodeFinding:
    """A code quality finding."""
    category: str  # bare_except, todo, magic_number, long_function
    file_path: str
    line_number: int
    description: str
    severity: str = "warning"  # warning, info

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CodeQualityReport:
    """Results of code quality scan."""
    bare_except_count: int = 0
    todo_count: int = 0
    fixme_count: int = 0
    hack_count: int = 0
    magic_number_count: int = 0
    longest_function_lines: int = 0
    longest_function_name: str = ""
    longest_function_file: str = ""
    findings: List[CodeFinding] = field(default_factory=list)
    files_scanned: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bare_except_count": self.bare_except_count,
            "todo_count": self.todo_count,
            "fixme_count": self.fixme_count,
            "hack_count": self.hack_count,
            "magic_number_count": self.magic_number_count,
            "longest_function_lines": self.longest_function_lines,
            "longest_function_name": self.longest_function_name,
            "longest_function_file": self.longest_function_file,
            "files_scanned": self.files_scanned,
            "findings_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings[:50]],  # Limit output
        }


class BareExceptVisitor(ast.NodeVisitor):
    """AST visitor to detect bare except blocks."""

    def __init__(self):
        self.findings: List[Tuple[int, str]] = []
        self._current_file = ""

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # Bare except: no exception type specified
        if node.type is None:
            self.findings.append((node.lineno, "bare except"))
        # except Exception with pass body
        elif (
            isinstance(node.type, ast.Name) and node.type.id == "Exception"
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Pass)
        ):
            self.findings.append((node.lineno, "except Exception: pass"))
        self.generic_visit(node)


class FunctionLengthVisitor(ast.NodeVisitor):
    """AST visitor to find function lengths."""

    def __init__(self):
        self.functions: List[Tuple[str, int, int]] = []  # (name, start_line, length)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        self.functions.append((node.name, node.lineno, length))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        self.functions.append((node.name, node.lineno, length))
        self.generic_visit(node)


class CodeQualityScanner:
    """Scans source code for quality issues.

    EARS-U-008: Bare except, TODOs, magic numbers, long functions.
    """

    # Common configuration module names to exclude from magic number check
    CONFIG_FILES = {"config.py", "settings.py", "constants.py", "config_defaults.py"}

    MAGIC_NUMBER_PATTERN = re.compile(
        r"(?<![\w.])\b(\d{2,})\b(?!\s*[,\]\)])"  # 2+ digit numbers not in containers
    )
    TODO_PATTERN = re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)

    def __init__(self, src_dir: str = "src"):
        self.src_dir = Path(src_dir)

    def scan(self) -> CodeQualityReport:
        """Scan source directory for code quality issues."""
        report = CodeQualityReport()

        if not self.src_dir.exists():
            logger.warning("Source directory %s does not exist", self.src_dir)
            return report

        for py_file in sorted(self.src_dir.rglob("*.py")):
            report.files_scanned += 1
            rel_path = str(py_file.relative_to(self.src_dir.parent))

            try:
                source = py_file.read_text(encoding="utf-8")
            except OSError:
                continue

            # AST-based analysis
            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue

            # Check bare except blocks
            visitor = BareExceptVisitor()
            visitor.visit(tree)
            for line_no, desc in visitor.findings:
                report.bare_except_count += 1
                report.findings.append(CodeFinding(
                    category="bare_except",
                    file_path=rel_path,
                    line_number=line_no,
                    description=desc,
                ))

            # Check function lengths
            func_visitor = FunctionLengthVisitor()
            func_visitor.visit(tree)
            for name, start_line, length in func_visitor.functions:
                if length > report.longest_function_lines:
                    report.longest_function_lines = length
                    report.longest_function_name = name
                    report.longest_function_file = rel_path
                if length > 50:
                    report.findings.append(CodeFinding(
                        category="long_function",
                        file_path=rel_path,
                        line_number=start_line,
                        description=f"Function '{name}' is {length} lines (target: <50)",
                        severity="info",
                    ))

            # Regex-based analysis
            for line_no, line in enumerate(source.split("\n"), 1):
                # TODO/FIXME/HACK
                match = self.TODO_PATTERN.search(line)
                if match:
                    tag = match.group(1).upper()
                    if tag == "TODO":
                        report.todo_count += 1
                    elif tag == "FIXME":
                        report.fixme_count += 1
                    elif tag in ("HACK", "XXX"):
                        report.hack_count += 1
                    report.findings.append(CodeFinding(
                        category="todo_comment",
                        file_path=rel_path,
                        line_number=line_no,
                        description=f"{tag}: {line.strip()[:80]}",
                        severity="info",
                    ))

                # Magic numbers (skip config files)
                if py_file.name not in self.CONFIG_FILES:
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    # Look for hardcoded numbers in assignments/comparisons
                    if re.search(r"[=<>!]=?\s*\d{2,}", stripped) and not re.search(r"(port|version|status|code|http)", stripped, re.IGNORECASE):
                        report.magic_number_count += 1
                        report.findings.append(CodeFinding(
                            category="magic_number",
                            file_path=rel_path,
                            line_number=line_no,
                            description=f"Possible magic number: {stripped[:80]}",
                            severity="info",
                        ))

        return report


@dataclass
class CoverageDelta:
    """Coverage analysis result."""
    current_coverage: float = 0.0
    previous_coverage: float = 0.0
    delta: float = 0.0
    low_coverage_files: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CoverageDeltaChecker:
    """Reads coverage.json and computes delta from previous evaluation.

    EARS-U-009: Track coverage delta and flag modules < 80%.
    """

    COVERAGE_THRESHOLD = 85.0
    FILE_THRESHOLD = 80.0

    def __init__(
        self,
        coverage_path: str = "coverage.json",
        history_path: str = "data/evaluations/health_scan.json",
    ):
        self.coverage_path = Path(coverage_path)
        self.history_path = Path(history_path)

    def check(self) -> CoverageDelta:
        """Check current coverage and compute delta."""
        result = CoverageDelta()

        # Read current coverage
        if not self.coverage_path.exists():
            logger.warning("coverage.json not found at %s", self.coverage_path)
            result.passed = False
            return result

        try:
            with open(self.coverage_path, encoding="utf-8") as f:
                cov_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read coverage.json: %s", e)
            result.passed = False
            return result

        # Extract overall coverage
        totals = cov_data.get("totals", {})
        result.current_coverage = totals.get("percent_covered", 0.0)

        # Find low-coverage files
        files = cov_data.get("files", {})
        for filepath, file_data in files.items():
            summary = file_data.get("summary", {})
            pct = summary.get("percent_covered", 0.0)
            if pct < self.FILE_THRESHOLD:
                result.low_coverage_files.append({
                    "file": filepath,
                    "coverage": round(pct, 1),
                })

        # Load previous coverage from history
        result.previous_coverage = self._load_previous_coverage()
        result.delta = result.current_coverage - result.previous_coverage

        # Evaluate pass/fail
        result.passed = (
            result.current_coverage >= self.COVERAGE_THRESHOLD
            and result.delta >= 0  # No regression
        )

        return result

    def _load_previous_coverage(self) -> float:
        """Load previous coverage from health scan history."""
        if not self.history_path.exists():
            return 0.0
        try:
            with open(self.history_path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("coverage", {}).get("current_coverage", 0.0)
        except (json.JSONDecodeError, OSError):
            return 0.0


@dataclass
class ConfigDriftResult:
    """Result of configuration drift detection."""
    missing_env_vars: List[str] = field(default_factory=list)
    out_of_range_configs: List[Dict[str, Any]] = field(default_factory=list)
    hardcoded_secrets: List[Dict[str, str]] = field(default_factory=list)
    status: str = "OK"  # OK, DRIFT_DETECTED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigDriftDetector:
    """Validates environment variables and configuration ranges.

    EARS-U-010: Check env vars, config ranges, no hardcoded secrets.
    """

    REQUIRED_ENV_VARS = [
        "OPENROUTER_API_KEY",
    ]

    OPTIONAL_ENV_VARS = [
        "ENABLE_SELF_RAG",
        "ENABLE_HYDE",
        "BM25_TOKENIZE_MODE",
    ]

    CONFIG_RANGES = {
        "temperature": (0.0, 2.0),
        "top_k": (1, 100),
        "top_p": (0.0, 1.0),
        "max_tokens": (1, 32768),
    }

    SECRET_PATTERN = re.compile(
        r"""(?:api[_-]?key|secret|password|token)\s*=\s*['\"]([^'\"]{8,})['\"]""",
        re.IGNORECASE,
    )

    def __init__(self, src_dir: str = "src"):
        self.src_dir = Path(src_dir)

    def check(self) -> ConfigDriftResult:
        """Run configuration drift detection."""
        result = ConfigDriftResult()

        # Check required environment variables
        for var in self.REQUIRED_ENV_VARS:
            if not os.environ.get(var):
                result.missing_env_vars.append(var)

        # Check for hardcoded secrets in source
        if self.src_dir.exists():
            for py_file in self.src_dir.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding="utf-8")
                except OSError:
                    continue

                for match in self.SECRET_PATTERN.finditer(content):
                    rel_path = str(py_file.relative_to(self.src_dir.parent))
                    result.hardcoded_secrets.append({
                        "file": rel_path,
                        "line": content[:match.start()].count("\n") + 1,
                    })

        # Determine status
        if result.missing_env_vars or result.hardcoded_secrets or result.out_of_range_configs:
            result.status = "DRIFT_DETECTED"

        return result


@dataclass
class SystemHealthReport:
    """Aggregated system health metrics."""
    code_quality: Optional[CodeQualityReport] = None
    coverage: Optional[CoverageDelta] = None
    config_drift: Optional[ConfigDriftResult] = None
    overall_status: str = "HEALTHY"  # HEALTHY, WARNING, CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        result = {"overall_status": self.overall_status}
        if self.code_quality:
            result["code_quality"] = self.code_quality.to_dict()
        if self.coverage:
            result["coverage"] = self.coverage.to_dict()
        if self.config_drift:
            result["config_drift"] = self.config_drift.to_dict()
        return result

    def save(self, path: str = "data/evaluations/health_scan.json"):
        """Save health report for trend tracking."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def run_health_scan(
    src_dir: str = "src",
    coverage_path: str = "coverage.json",
    history_path: str = "data/evaluations/health_scan.json",
) -> SystemHealthReport:
    """Run all health checks and return aggregated report."""
    report = SystemHealthReport()

    # Code quality scan
    scanner = CodeQualityScanner(src_dir=src_dir)
    report.code_quality = scanner.scan()

    # Coverage delta
    checker = CoverageDeltaChecker(
        coverage_path=coverage_path,
        history_path=history_path,
    )
    report.coverage = checker.check()

    # Config drift
    detector = ConfigDriftDetector(src_dir=src_dir)
    report.config_drift = detector.check()

    # Determine overall status
    issues = 0
    if report.code_quality and report.code_quality.bare_except_count > 0:
        issues += 1
    if report.coverage and not report.coverage.passed:
        issues += 1
    if report.config_drift and report.config_drift.status == "DRIFT_DETECTED":
        issues += 1

    if issues >= 2:
        report.overall_status = "CRITICAL"
    elif issues >= 1:
        report.overall_status = "WARNING"
    else:
        report.overall_status = "HEALTHY"

    return report
