"""Tests for reparse-hwpx CLI command.

This module tests the CLI command for re-parsing HWPX files with:
- File discovery and sorting
- File validation (size, readability, ZIP structure)
- Backup creation with timestamps
- Dry-run and verbose modes
"""
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.commands.reparse_hwpx import (
    create_backup,
    discover_hwpx_files,
    main,
    validate_hwpx_file,
)


# ============================================================================
# File Discovery Tests (REQ-002)
# ============================================================================


def test_discover_hwpx_files_finds_all(tmp_path: Path) -> None:
    """REQ-002: Should discover all hwpx files in input directory."""
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "file1.hwpx").touch()
    (input_dir / "file2.hwpx").touch()
    (input_dir / "file3.hwpx").touch()

    # Act
    files = discover_hwpx_files(input_dir)

    # Assert
    assert len(files) == 3


def test_discover_hwpx_files_sorted_alphabetically(tmp_path: Path) -> None:
    """REQ-002: Should return files sorted alphabetically."""
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "z_file.hwpx").touch()
    (input_dir / "a_file.hwpx").touch()
    (input_dir / "m_file.hwpx").touch()

    # Act
    files = discover_hwpx_files(input_dir)

    # Assert
    assert files[0].name == "a_file.hwpx"
    assert files[1].name == "m_file.hwpx"
    assert files[2].name == "z_file.hwpx"


def test_discover_hwpx_files_empty_directory(tmp_path: Path) -> None:
    """REQ-002: Should return empty list for empty directory."""
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Act
    files = discover_hwpx_files(input_dir)

    # Assert
    assert files == []


def test_discover_hwpx_files_ignores_non_hwpx(tmp_path: Path) -> None:
    """REQ-002: Should ignore non-hwpx files."""
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "file1.hwpx").touch()
    (input_dir / "file2.txt").touch()
    (input_dir / "file3.json").touch()

    # Act
    files = discover_hwpx_files(input_dir)

    # Assert
    assert len(files) == 1
    assert files[0].name == "file1.hwpx"


# ============================================================================
# File Validation Tests (REQ-001)
# ============================================================================


def test_validate_hwpx_file_valid(tmp_path: Path) -> None:
    """REQ-001: Should pass validation for valid hwpx file."""
    # Arrange
    hwpx_file = tmp_path / "valid.hwpx"
    with zipfile.ZipFile(hwpx_file, "w") as zf:
        zf.writestr("Content/content.xml", "<xml>test</xml>")

    # Act
    is_valid, error = validate_hwpx_file(hwpx_file)

    # Assert
    assert is_valid is True
    assert error == ""


def test_validate_hwpx_file_zero_size(tmp_path: Path) -> None:
    """REQ-001: Should fail for zero-size file."""
    # Arrange
    hwpx_file = tmp_path / "empty.hwpx"
    hwpx_file.touch()

    # Act
    is_valid, error = validate_hwpx_file(hwpx_file)

    # Assert
    assert is_valid is False
    assert "size" in error.lower() or "zero" in error.lower()


def test_validate_hwpx_file_not_readable(tmp_path: Path) -> None:
    """REQ-001: Should fail for non-readable file."""
    # Arrange
    hwpx_file = tmp_path / "unreadable.hwpx"
    hwpx_file.write_bytes(b"some content")
    hwpx_file.chmod(0o000)

    # Act
    is_valid, error = validate_hwpx_file(hwpx_file)

    # Assert
    assert is_valid is False
    # Restore permissions for cleanup
    hwpx_file.chmod(0o644)


def test_validate_hwpx_file_invalid_zip(tmp_path: Path) -> None:
    """REQ-001: Should fail for invalid ZIP structure."""
    # Arrange
    hwpx_file = tmp_path / "invalid.hwpx"
    hwpx_file.write_bytes(b"not a valid zip file")

    # Act
    is_valid, error = validate_hwpx_file(hwpx_file)

    # Assert
    assert is_valid is False
    assert "zip" in error.lower() or "invalid" in error.lower()


def test_validate_hwpx_file_nonexistent(tmp_path: Path) -> None:
    """REQ-001: Should fail for non-existent file."""
    # Arrange
    hwpx_file = tmp_path / "nonexistent.hwpx"

    # Act
    is_valid, error = validate_hwpx_file(hwpx_file)

    # Assert
    assert is_valid is False
    assert "not found" in error.lower() or "exist" in error.lower()


# ============================================================================
# Backup Tests (REQ-005)
# ============================================================================


def test_create_backup_existing_file(tmp_path: Path) -> None:
    """REQ-005: Should create timestamped backup of existing file."""
    # Arrange
    output_file = tmp_path / "output.json"
    output_file.write_text('{"test": "data"}')

    # Act
    backup_path = create_backup(output_file)

    # Assert
    assert backup_path is not None
    assert backup_path.exists()
    assert backup_path.read_text() == '{"test": "data"}'
    # Check timestamp format in filename
    assert backup_path.suffix == ".json"
    # Backup filename should contain original stem and timestamp
    assert "output" in backup_path.stem


def test_create_backup_non_existing(tmp_path: Path) -> None:
    """REQ-005: Should return None for non-existing file."""
    # Arrange
    output_file = tmp_path / "nonexistent.json"

    # Act
    backup_path = create_backup(output_file)

    # Assert
    assert backup_path is None


def test_create_backup_preserves_content(tmp_path: Path) -> None:
    """REQ-005: Should preserve original file content in backup."""
    # Arrange
    original_content = '{"docs": [{"title": "Test"}]}'
    output_file = tmp_path / "output.json"
    output_file.write_text(original_content)

    # Act
    backup_path = create_backup(output_file)

    # Assert
    assert backup_path is not None
    assert backup_path.read_text() == original_content


# ============================================================================
# CLI Tests
# ============================================================================


def test_cli_help_option() -> None:
    """Should display help message and return 0."""
    # Act
    result = main(["--help"])

    # Assert - help returns 0 (success)
    assert result == 0


def test_cli_dry_run_option(tmp_path: Path) -> None:
    """Should not create output files in dry-run mode."""
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    hwpx_file = input_dir / "test.hwpx"
    with zipfile.ZipFile(hwpx_file, "w") as zf:
        zf.writestr("Content/content.xml", "<xml>test</xml>")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Act
    result = main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    # Assert
    assert result == 0
    # No output files should be created
    assert len(list(output_dir.glob("*.json"))) == 0


def test_cli_verbose_option(tmp_path: Path) -> None:
    """Should produce verbose output when enabled."""
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    hwpx_file = input_dir / "test.hwpx"
    with zipfile.ZipFile(hwpx_file, "w") as zf:
        zf.writestr("Content/content.xml", "<xml>test</xml>")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Act
    with patch("builtins.print") as mock_print:
        result = main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--verbose",
                "--dry-run",
            ]
        )

    # Assert
    assert result == 0


def test_cli_custom_input_output_dirs(tmp_path: Path) -> None:
    """Should accept custom input and output directories."""
    # Arrange
    input_dir = tmp_path / "custom_input"
    input_dir.mkdir()
    hwpx_file = input_dir / "test.hwpx"
    with zipfile.ZipFile(hwpx_file, "w") as zf:
        zf.writestr("Content/content.xml", "<xml>test</xml>")

    output_dir = tmp_path / "custom_output"
    output_dir.mkdir()

    # Act
    result = main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    # Assert
    assert result == 0


def test_cli_empty_input_directory(tmp_path: Path) -> None:
    """Should handle empty input directory gracefully."""
    # Arrange
    input_dir = tmp_path / "empty_input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Act
    result = main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    # Assert
    assert result == 0  # Should succeed with no files to process
