"""CLI command for re-parsing HWPX regulation files.

This module provides a command-line interface for:
- Discovering HWPX files in input directory
- Validating file integrity
- Creating backups before overwriting
- Processing files with dual output (standard + RAG optimized)
- Generating quality reports

Usage:
    uv run reparse-hwpx --input-dir data/input --output-dir data/output
    uv run reparse-hwpx --help
    uv run reparse-hwpx --dry-run --verbose
"""
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from src.enhance_for_rag import enhance_json
from src.parsing.hwpx_direct_parser import HWPXDirectParser


# ============================================================================
# File Discovery Functions
# ============================================================================


def discover_hwpx_files(input_dir: Path) -> List[Path]:
    """Discover all HWPX files in input directory, sorted alphabetically.

    Args:
        input_dir: Path to directory containing HWPX files.

    Returns:
        List of Path objects for HWPX files, sorted alphabetically by name.
        Returns empty list if directory is empty or contains no HWPX files.
    """
    if not input_dir.exists():
        return []

    hwpx_files = sorted(input_dir.glob("*.hwpx"))
    return hwpx_files


# ============================================================================
# File Validation Functions
# ============================================================================


def validate_hwpx_file(file_path: Path) -> Tuple[bool, str]:
    """Validate HWPX file integrity.

    Checks:
    1. File exists
    2. File size > 0
    3. File is readable
    4. File is a valid ZIP archive (HWPX is ZIP-based)

    Args:
        file_path: Path to the HWPX file to validate.

    Returns:
        Tuple of (is_valid, error_message).
        If valid: (True, "")
        If invalid: (False, "error description")
    """
    # Check file exists
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    # Check file size > 0
    if file_path.stat().st_size == 0:
        return False, f"File has zero size: {file_path}"

    # Check file is readable
    try:
        with open(file_path, "rb") as f:
            # Read first few bytes to verify readability
            f.read(4)
    except PermissionError:
        return False, f"File is not readable: {file_path}"
    except IOError as e:
        return False, f"Cannot read file: {e}"

    # Check valid ZIP structure (HWPX is ZIP-based)
    if not zipfile.is_zipfile(file_path):
        return False, f"Invalid ZIP structure (not a valid HWPX): {file_path}"

    return True, ""


# ============================================================================
# Backup Functions
# ============================================================================


def create_backup(file_path: Path) -> Optional[Path]:
    """Create a timestamped backup of an existing file.

    Args:
        file_path: Path to the file to backup.

    Returns:
        Path to the backup file if created, None if file doesn't exist.
    """
    if not file_path.exists():
        return None

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create backup filename
    stem = file_path.stem
    suffix = file_path.suffix
    parent = file_path.parent
    backup_name = f"{stem}_backup_{timestamp}{suffix}"
    backup_path = parent / backup_name

    # Copy file to backup
    shutil.copy2(file_path, backup_path)

    return backup_path


# ============================================================================
# Processing Functions
# ============================================================================


def process_hwpx_file(
    hwpx_path: Path,
    output_dir: Path,
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Process a single HWPX file and generate outputs.

    Uses HWPXDirectParser for actual parsing and enhance_json for RAG optimization.

    Args:
        hwpx_path: Path to the HWPX file.
        output_dir: Directory for output files.
        verbose: Whether to print verbose output.
        dry_run: If True, parse but do not write output files.

    Returns:
        Dictionary with processing results including:
        - input_file: Path to input HWPX file
        - output_files: Dict with 'standard' and 'rag' output paths (empty if dry_run)
        - quality_report: Chunk statistics and metrics
    """
    start_time = datetime.now()

    if verbose:
        click.echo(f"  Parsing HWPX: {hwpx_path.name}")

    # Step 1: Parse HWPX file using HWPXDirectParser
    parser = HWPXDirectParser()
    parsed_data = parser.parse_file(hwpx_path)

    # Step 2: Prepare standard output (raw parsed data)
    standard_data = {
        "metadata": parsed_data.get("metadata", {}),
        "toc": parsed_data.get("toc", []),
        "docs": parsed_data.get("docs", []),
        "parsing_method": "hwpx_direct",
    }

    # Step 3: Create RAG optimized output with chunk splitting
    rag_data = enhance_json(json.loads(json.dumps(standard_data)))

    # Step 4: Generate output file paths
    stem = hwpx_path.stem
    standard_path = output_dir / f"{stem}_hwpx_direct.json"
    rag_path = output_dir / f"{stem}_hwpx_direct_rag.json"

    # Step 5: Write output files (skip if dry_run)
    output_files: Dict[str, str] = {}
    if not dry_run:
        with open(standard_path, "w", encoding="utf-8") as f:
            json.dump(standard_data, f, ensure_ascii=False, indent=2)

        with open(rag_path, "w", encoding="utf-8") as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)

        output_files = {
            "standard": str(standard_path),
            "rag": str(rag_path),
        }

    # Step 6: Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds()

    # Step 7: Generate quality report from first document (if available)
    docs = rag_data.get("docs", [])
    quality_report = {
        "input_file": str(hwpx_path),
        "document_count": len(docs),
        "processing_time_seconds": processing_time,
        "generated_at": datetime.now().isoformat(),
    }

    # Add chunk statistics if docs exist
    if docs:
        # Count total chunks across all documents
        total_chunks = 0
        chunk_types: Dict[str, int] = {}

        def count_chunks_recursive(nodes: List[Dict[str, Any]]) -> int:
            """Recursively count chunks and their types."""
            count = 0
            for node in nodes:
                chunk_type = node.get("type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                count += 1
                children = node.get("children", [])
                if children:
                    count += count_chunks_recursive(children)
            return count

        for doc in docs:
            total_chunks += count_chunks_recursive(doc.get("content", []))

        chunk_types["total"] = total_chunks
        quality_report["chunk_statistics"] = chunk_types

        # Calculate max hierarchy depth
        max_depth = 0
        for doc in docs:
            for node in doc.get("content", []):
                depth = _calculate_depth(node)
                max_depth = max(max_depth, depth)
        quality_report["max_hierarchy_depth"] = max_depth

    if verbose and not dry_run:
        click.echo(f"  Standard output: {standard_path.name}")
        click.echo(f"  RAG output: {rag_path.name}")
        click.echo(f"  Documents: {len(docs)}, Total chunks: {quality_report.get('chunk_statistics', {}).get('total', 0)}")

    return {
        "input_file": str(hwpx_path),
        "output_files": output_files,
        "quality_report": quality_report,
    }


def _calculate_depth(node: Dict[str, Any]) -> int:
    """Calculate maximum depth of a node hierarchy.

    Args:
        node: Node dictionary with optional 'children' key.

    Returns:
        Maximum depth (1 for leaf nodes).
    """
    children = node.get("children", [])
    if not children:
        return 1
    return 1 + max(_calculate_depth(child) for child in children)


# ============================================================================
# CLI Command
# ============================================================================


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("data/input"),
    help="Directory containing HWPX files to process",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("data/output"),
    help="Directory for output JSON files",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Process files without creating output",
)
def cli(
    input_dir: Path,
    output_dir: Path,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Re-parse HWPX regulation files with quality analysis.

    Discovers HWPX files in the input directory, validates them,
    and generates dual output (standard + RAG optimized) with quality reports.
    """
    # Discover HWPX files
    hwpx_files = discover_hwpx_files(input_dir)

    if verbose:
        click.echo(f"Found {len(hwpx_files)} HWPX file(s) in {input_dir}")

    if not hwpx_files:
        click.echo(f"No HWPX files found in {input_dir}")
        return

    # Create output directory if not in dry-run mode
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    results = []
    for hwpx_path in hwpx_files:
        # Validate file
        is_valid, error = validate_hwpx_file(hwpx_path)
        if not is_valid:
            click.echo(f"Skipping invalid file: {hwpx_path.name} - {error}")
            continue

        if verbose:
            click.echo(f"Processing: {hwpx_path.name}")

        # Process file
        result = process_hwpx_file(hwpx_path, output_dir, verbose, dry_run)
        results.append(result)

    if verbose:
        click.echo(f"\nProcessed {len(results)} file(s)")


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success).
    """
    try:
        cli(args)
        return 0
    except SystemExit as e:
        # Click raises SystemExit for --help and errors
        if e.code == 0:
            return 0
        raise


if __name__ == "__main__":
    sys.exit(main())
