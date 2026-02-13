import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

# Suppress Transformers/PyTorch warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

if TYPE_CHECKING:
    from rich.console import Console

from dotenv import load_dotenv

from .cache_manager import CacheManager
from .converter import HwpToMarkdownReader
from .enhance_for_rag import enhance_json
from .formatter import RegulationFormatter
from .llm_client import LLMClient
from .metadata_extractor import MetadataExtractor
from .parsing.html_table_converter import replace_markdown_tables_with_html
from .parsing.hwpx_direct_parser import HWPXDirectParser
from .preprocessor import Preprocessor
from .collect_files import collect_hwp_files

PIPELINE_SIGNATURE_VERSION = "v5"
OUTPUT_SCHEMA_VERSION = "v4"


def _extract_source_metadata(file_name: str) -> Dict[str, Optional[str]]:
    """
    파일명에서 규정집 일련번호와 발행일을 추출합니다.

    예시: "규정집9-343(20250909).hwpx"
      -> {"source_serial": "9-343", "source_date": "2025-09-09"}

    Args:
        file_name: HWPX 파일명

    Returns:
        source_serial과 source_date를 담은 딕셔너리
    """
    result: Dict[str, Optional[str]] = {"source_serial": None, "source_date": None}

    # 일련번호: "규정집" 뒤의 "N-NNN" 패턴
    serial_match = re.search(r"규정집(\d+-\d+)", file_name)
    if serial_match:
        result["source_serial"] = serial_match.group(1)

    # 발행일: 괄호 안의 8자리 숫자 (YYYYMMDD)
    date_match = re.search(r"\((\d{8})\)", file_name)
    if date_match:
        raw = date_match.group(1)
        result["source_date"] = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    return result


def _resolve_preprocessor_rules_path() -> Path:
    rules_path = os.getenv("PREPROCESSOR_RULES_PATH")
    if rules_path:
        return Path(rules_path)
    return Path("data/config/preprocessor_rules.json")


def _compute_rules_hash(
    path: Path, cache_manager: CacheManager, console=None, verbose: bool = False
) -> str:
    if not path.exists():
        return "missing"
    try:
        return cache_manager.compute_file_hash(path)
    except Exception as e:
        if verbose and console:
            console.print(f"[yellow]규칙 파일 해시 계산 실패: {e}[/yellow]")
        return "error"


def _build_pipeline_signature(rules_hash: str, llm_signature: str) -> str:
    return f"{PIPELINE_SIGNATURE_VERSION}|rules:{rules_hash}|llm:{llm_signature}"


# --- Helper dataclasses for pipeline context ---


@dataclass
class FilePaths:
    """Output file paths for a single HWP file."""

    raw_md: Path
    raw_html: Path
    json_out: Path
    metadata: Path


@dataclass
class CacheState:
    """Cache state for a single file."""

    hwp_hash: Optional[str] = None
    raw_md_hash: Optional[str] = None
    cached_hwp_hash: Optional[str] = None
    cached_raw_md_hash: Optional[str] = None
    cached_pipeline_signature: Optional[str] = None
    cached_final_json_hash: Optional[str] = None
    cached_metadata_hash: Optional[str] = None
    cache_hit: bool = False
    raw_md_cache_hit: bool = True


@dataclass
class PipelineContext:
    """Shared context for pipeline execution."""

    cache_manager: CacheManager
    preprocessor: "Preprocessor"
    formatter: "RegulationFormatter"
    metadata_extractor: "MetadataExtractor"
    pipeline_signature: str
    args: argparse.Namespace
    console: "Console"


# --- Helper functions for run_pipeline ---


def _collect_hwp_files(input_path: Path, console: "Console") -> Optional[List[Path]]:
    """Collect HWPX files from input path.

    Note: This system only accepts .hwpx files (HWP XML format).
    If you have .hwp files, you must convert them to .hwpx first.
    """
    if not input_path.exists():
        console.print(f"[red]입력 경로가 존재하지 않습니다: {input_path}[/red]")
        return None

    files = []
    if input_path.is_file():
        files.append(input_path)
    elif input_path.is_dir():
        # Only accept .hwpx files (HWP XML format)
        files.extend(sorted(input_path.rglob("*.hwpx")))

    if not files:
        # Check if .hwp files exist (old format - not supported)
        hwp_legacy = list(input_path.rglob("*.hwp")) if input_path.is_dir() else []
        if hwp_legacy:
            console.print("[red].hwp 파일은 지원되지 않습니다.[/red]")
            console.print("[yellow]HWP → HWPX 변환 방법:[/yellow]")
            console.print("  1. 한글(Hwp)에서 파일 열기")
            console.print("  2. 파일 > 다른 이름으로 저장 > HWPX 파일 형식 선택")
            console.print("  3. 저장된 .hwpx 파일을 사용하세요")
            console.print("")
            console.print("[red]※ 이 시스템은 .hwpx 파일만 지원합니다.[/red]")
            return None
        else:
            console.print("[red]처리할 HWPX 파일이 없습니다.[/red]")
            console.print("[yellow]지원 형식: .hwpx (HWP XML)[/yellow]")
            return None

    return files


def _initialize_llm(
    args: argparse.Namespace, console: "Console"
) -> tuple[Optional["LLMClient"], str]:
    """Initialize LLM client if requested."""
    if not args.use_llm:
        return None, "disabled"

    provider_name = args.provider if args.provider else "openai"
    try:
        llm_client = LLMClient(
            provider=provider_name,
            model=args.model,
            base_url=args.base_url,
        )
        return llm_client, llm_client.cache_namespace()
    except Exception as e:
        if args.allow_llm_fallback:
            console.print(
                f"[yellow]LLM 초기화 실패: {e} - LLM 비활성화하고 계속 진행합니다.[/yellow]"
            )
            return None, "disabled"
        raise


def _get_file_paths(file: Path, output_dir: Path, input_path: Path) -> FilePaths:
    """Determine output paths for a file."""
    if input_path.is_dir():
        rel_path = file.relative_to(input_path)
        file_output_dir = output_dir / rel_path.parent
    else:
        file_output_dir = output_dir
    file_output_dir.mkdir(parents=True, exist_ok=True)

    return FilePaths(
        raw_md=file_output_dir / f"{file.stem}_raw.md",
        raw_html=file_output_dir / f"{file.stem}_raw.xhtml",
        json_out=file_output_dir / f"{file.stem}.json",
        metadata=file_output_dir / f"{file.stem}_metadata.json",
    )


def _load_cache_state(
    file: Path,
    paths: FilePaths,
    cache_manager: CacheManager,
    args: argparse.Namespace,
    console: "Console",
) -> CacheState:
    """Load and compute cache state for a file."""
    state = CacheState()

    file_state = cache_manager.get_file_state(str(file)) if cache_manager else None
    if file_state:
        state.cached_hwp_hash = file_state.get("hwp_hash")
        state.cached_raw_md_hash = file_state.get("raw_md_hash")
        state.cached_pipeline_signature = file_state.get("pipeline_signature")
        state.cached_final_json_hash = file_state.get("final_json_hash")
        state.cached_metadata_hash = file_state.get("metadata_hash")

    if cache_manager:
        try:
            state.hwp_hash = cache_manager.compute_file_hash(file)
            if state.cached_hwp_hash:
                state.cache_hit = state.cached_hwp_hash == state.hwp_hash
        except Exception as e:
            if args.verbose:
                console.print(f"[yellow]HWP 해시 계산 실패: {e}[/yellow]")

        if paths.raw_md.exists():
            try:
                state.raw_md_hash = cache_manager.compute_file_hash(paths.raw_md)
                if state.cached_raw_md_hash:
                    state.raw_md_cache_hit = (
                        state.cached_raw_md_hash == state.raw_md_hash
                    )
            except Exception as e:
                state.raw_md_cache_hit = False
                if args.verbose:
                    console.print(f"[yellow]RAW MD 해시 계산 실패: {e}[/yellow]")

    return state


def _check_full_cache_hit(
    paths: FilePaths,
    cache_state: CacheState,
    pipeline_signature: str,
    cache_manager: CacheManager,
    force: bool,
) -> bool:
    """Check if all outputs are cached and valid."""
    if force:
        return False

    required_files_exist = (
        paths.raw_md.exists() and paths.json_out.exists() and paths.metadata.exists()
    )
    if not required_files_exist:
        return False

    if not cache_state.cache_hit or not cache_state.raw_md_cache_hit:
        return False

    if cache_state.cached_pipeline_signature != pipeline_signature:
        return False

    def output_hash_matches(path: Path, cached_hash: Optional[str]) -> bool:
        if not cache_manager or not cached_hash or not path.exists():
            return False
        try:
            return cache_manager.compute_file_hash(path) == cached_hash
        except Exception:
            return False

    if not output_hash_matches(paths.json_out, cache_state.cached_final_json_hash):
        return False
    if not output_hash_matches(paths.metadata, cache_state.cached_metadata_hash):
        return False

    return True


def _log_cache_miss_reasons(
    paths: FilePaths,
    cache_state: CacheState,
    pipeline_signature: str,
    cache_manager: CacheManager,
    console: "Console",
) -> None:
    """Log reasons for cache miss in verbose mode."""
    reasons = []
    if not paths.raw_md.exists():
        reasons.append("raw_md 없음")
    if not paths.json_out.exists():
        reasons.append("json 없음")
    if not paths.metadata.exists():
        reasons.append("metadata 없음")
    if not cache_state.cache_hit:
        reasons.append("HWP 해시 불일치")
    if not cache_state.raw_md_cache_hit:
        reasons.append("raw_md 해시 불일치")
    if cache_state.cached_pipeline_signature != pipeline_signature:
        reasons.append("파이프라인 시그니처 변경")

    required_files_exist = (
        paths.raw_md.exists() and paths.json_out.exists() and paths.metadata.exists()
    )
    if required_files_exist:

        def output_hash_matches(path: Path, cached_hash: Optional[str]) -> bool:
            if not cache_manager or not cached_hash or not path.exists():
                return False
            try:
                return cache_manager.compute_file_hash(path) == cached_hash
            except Exception:
                return False

        if not output_hash_matches(paths.json_out, cache_state.cached_final_json_hash):
            reasons.append("json 해시 불일치")
        if not output_hash_matches(paths.metadata, cache_state.cached_metadata_hash):
            reasons.append("metadata 해시 불일치")

    if reasons:
        console.print(f"[dim]캐시 미스: {', '.join(reasons)}[/dim]")


def _convert_hwp_to_markdown(
    file: Path,
    paths: FilePaths,
    cache_state: CacheState,
    ctx: PipelineContext,
    status_callback: Optional[Callable],
) -> tuple[str, Optional[str], Optional[str]]:
    """Step 1: Convert HWP to Markdown (or load from cache)."""
    can_reuse_raw_md = (
        paths.raw_md.exists()
        and not ctx.args.force
        and cache_state.cache_hit
        and cache_state.raw_md_cache_hit
    )

    if can_reuse_raw_md:
        with open(paths.raw_md, "r", encoding="utf-8") as f:
            raw_md = f.read()
        html_content = None
        if paths.raw_html.exists():
            with open(paths.raw_html, "r", encoding="utf-8") as f:
                html_content = f.read()
        if ctx.cache_manager and cache_state.hwp_hash:
            hwp_hash_val = cache_state.hwp_hash
            raw_md_hash_val = cache_state.raw_md_hash
            if hwp_hash_val or raw_md_hash_val:
                ctx.cache_manager.update_file_state(
                    str(file),
                    hwp_hash=hwp_hash_val,
                    raw_md_hash=raw_md_hash_val,
                )
        return raw_md, html_content, cache_state.raw_md_hash

    # Need to convert from HWPX
    # Direct HWPX parsing (ZIP+XML) - no hwp5html dependency required
    reader = HwpToMarkdownReader(keep_html=False)
    docs = reader.load_data(
        file, status_callback=status_callback, verbose=ctx.args.verbose
    )
    raw_md = docs[0].text
    html_content = docs[0].metadata.get("html_content")

    with open(paths.raw_md, "w", encoding="utf-8") as f:
        f.write(raw_md)
    if html_content:
        with open(paths.raw_html, "w", encoding="utf-8") as f:
            f.write(html_content)

    raw_md_hash = None
    if ctx.cache_manager and cache_state.hwp_hash:
        raw_md_hash = ctx.cache_manager.compute_text_hash(raw_md)
        ctx.cache_manager.update_file_state(
            str(file),
            hwp_hash=cache_state.hwp_hash,
            raw_md_hash=raw_md_hash,
        )

    return raw_md, html_content, raw_md_hash


def _extract_and_save_metadata(
    clean_md: str,
    file: Path,
    paths: FilePaths,
    ctx: PipelineContext,
) -> tuple[Dict[str, Any], str]:
    """Step 3: Extract metadata and save to file."""
    extracted_metadata = ctx.metadata_extractor.extract(clean_md)
    metadata_payload = {"file_name": file.name, **extracted_metadata}
    metadata_text = json.dumps(metadata_payload, ensure_ascii=False, indent=2)
    with open(paths.metadata, "w", encoding="utf-8") as f:
        f.write(metadata_text)
    return extracted_metadata, metadata_text


def _format_documents(
    clean_md: str,
    html_content: Optional[str],
    file: Path,
    extracted_metadata: Dict[str, Any],
    ctx: PipelineContext,
    status_callback: Optional[Callable],
) -> List[Dict[str, Any]]:
    """Step 4: Format markdown into structured documents."""
    if html_content:
        clean_md = replace_markdown_tables_with_html(clean_md, html_content)

    final_docs = ctx.formatter.parse(
        clean_md,
        html_content=html_content,
        verbose_callback=status_callback,
        extracted_metadata=extracted_metadata,
        source_file_name=file.name,
    )

    # Backfill metadata
    scan_date = time.strftime("%Y-%m-%d")
    missing_rule_code = 0
    missing_page_range = 0
    for doc in final_docs:
        metadata = doc.setdefault("metadata", {})
        metadata.setdefault("scan_date", scan_date)
        metadata.setdefault("file_name", file.name)
        metadata.setdefault("rule_code", None)
        metadata.setdefault("page_range", None)
        if not metadata.get("rule_code"):
            missing_rule_code += 1
        if not metadata.get("page_range"):
            missing_page_range += 1

    if ctx.args.verbose:
        ctx.console.print(
            f"[dim]메타데이터 누락: rule_code {missing_rule_code}/{len(final_docs)}, "
            f"page_range {missing_page_range}/{len(final_docs)}[/dim]"
        )

    return final_docs


def _save_final_json(
    file: Path,
    paths: FilePaths,
    final_docs: List[Dict[str, Any]],
    extracted_metadata: Dict[str, Any],
    metadata_text: str,
    raw_md_hash: Optional[str],
    raw_md: str,
    cache_state: CacheState,
    ctx: PipelineContext,
) -> None:
    """Step 5: Save final JSON output."""
    source_meta = _extract_source_metadata(file.name)
    final_json = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline_signature": ctx.pipeline_signature,
        "file_name": file.name,
        "source_serial": source_meta["source_serial"],
        "source_date": source_meta["source_date"],
        "toc": extracted_metadata.get("toc") or [],
        "index_by_alpha": extracted_metadata.get("index_by_alpha") or [],
        "index_by_dept": extracted_metadata.get("index_by_dept") or {},
        "docs": final_docs,
    }

    if ctx.args.enhance_rag:
        final_json = enhance_json(final_json)
        if ctx.args.verbose:
            ctx.console.print("[dim]RAG 최적화 적용 완료[/dim]")

    final_json_text = json.dumps(final_json, ensure_ascii=False, indent=2)
    with open(paths.json_out, "w", encoding="utf-8") as f:
        f.write(final_json_text)

    if ctx.cache_manager:
        if raw_md_hash is None:
            raw_md_hash = ctx.cache_manager.compute_text_hash(raw_md)
        hwp_hash_val = cache_state.hwp_hash
        if hwp_hash_val or raw_md_hash:
            ctx.cache_manager.update_file_state(
                str(file),
                hwp_hash=hwp_hash_val or "",
                raw_md_hash=raw_md_hash or "",
                pipeline_signature=ctx.pipeline_signature or "",
                final_json_hash=ctx.cache_manager.compute_text_hash(final_json_text),
                metadata_hash=ctx.cache_manager.compute_text_hash(metadata_text),
            )


def _process_single_file(
    file: Path,
    input_path: Path,
    output_dir: Path,
    ctx: PipelineContext,
    progress: Any,
    total_task: Any,
    steps_per_file: int,
) -> bool:
    """Process a single HWP file through the pipeline."""
    status_callback = ctx.console.print if ctx.args.verbose else None
    paths = _get_file_paths(file, output_dir, input_path)

    cache_state = _load_cache_state(
        file, paths, ctx.cache_manager, ctx.args, ctx.console
    )

    full_cache_hit = _check_full_cache_hit(
        paths,
        cache_state,
        ctx.pipeline_signature,
        ctx.cache_manager,
        ctx.args.force,
    )

    if ctx.args.verbose and not full_cache_hit and not ctx.args.force:
        _log_cache_miss_reasons(
            paths, cache_state, ctx.pipeline_signature, ctx.cache_manager, ctx.console
        )

    if full_cache_hit:
        if ctx.args.verbose:
            ctx.console.print(
                f"[dim]캐시 적중: {file.name} (변환/전처리/포맷팅 생략)[/dim]"
            )
        progress.advance(total_task, steps_per_file)
        if ctx.cache_manager and (cache_state.hwp_hash or cache_state.raw_md_hash):
            hwp_hash_val = cache_state.hwp_hash
            raw_md_hash_val = cache_state.raw_md_hash
            if hwp_hash_val or raw_md_hash_val:
                ctx.cache_manager.update_file_state(
                    str(file),
                    hwp_hash=hwp_hash_val or "",
                    raw_md_hash=raw_md_hash_val or "",
                    pipeline_signature=ctx.pipeline_signature or "",
                )
        return True  # Success (cached)

    # Step 1: HWP -> MD
    raw_md, html_content, raw_md_hash = _convert_hwp_to_markdown(
        file, paths, cache_state, ctx, status_callback
    )
    progress.advance(total_task, 1)

    # Step 2: Preprocess
    clean_md = ctx.preprocessor.clean(raw_md, verbose_callback=status_callback)
    progress.advance(total_task, 1)

    # Step 3: Metadata extraction
    extracted_metadata, metadata_text = _extract_and_save_metadata(
        clean_md, file, paths, ctx
    )
    progress.advance(total_task, 1)

    # Step 4: Formatting
    final_docs = _format_documents(
        clean_md, html_content, file, extracted_metadata, ctx, status_callback
    )
    progress.advance(total_task, 1)

    # Step 5: Save JSON
    _save_final_json(
        file,
        paths,
        final_docs,
        extracted_metadata,
        metadata_text,
        raw_md_hash,
        raw_md,
        cache_state,
        ctx,
    )
    progress.advance(total_task, 1)

    return True  # Success


def _run_hwpx_direct_pipeline(
    file: Path,
    output_dir: Path,
    ctx: PipelineContext,
    progress: Any,
    total_task: Any,
    use_hwpx: bool = True,  # NEW: Parameter to control HWPX parser usage
) -> bool:
    """Run HWPX direct parsing pipeline.

    Args:
        file: HWPX file path.
        output_dir: Output directory.
        ctx: Pipeline context.
        progress: Progress bar.
        total_task: Total task tracker.

    Returns:
        True for success, False for failure.
    """
    status_callback = ctx.console.print if ctx.args.verbose else None

    # Determine output paths
    if output_dir.is_dir():
        file_output_dir = output_dir
    else:
        file_output_dir = output_dir
    file_output_dir.mkdir(parents=True, exist_ok=True)

    json_out = file_output_dir / f"{file.stem}.json"

    if ctx.args.verbose:
        ctx.console.print(f"[cyan]HWPX 직접 파싱: {file.name}[/cyan]")

    try:
        # Initialize HWPX direct parser (use_hwpx passed from collect_hwp_files)
        parser = HWPXDirectParser(
            status_callback=lambda msg: ctx.console.print(msg) if ctx.args.verbose else None
        )

        # Parse HWPX file
        start_time = time.time()
        result = parser.parse_file(file)
        parsing_time = time.time() - start_time

        # Add schema version and metadata
        source_meta = _extract_source_metadata(file.name)
        final_json = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pipeline_signature": ctx.pipeline_signature + "|hwpx_direct",
            "file_name": file.name,
            "source_serial": source_meta["source_serial"],
            "source_date": source_meta["source_date"],
            "parsing_method": "hwpx_direct",
            "parsing_time_seconds": parsing_time,
            **result,
        }

        # Apply RAG enhancement if requested
        if ctx.args.enhance_rag:
            final_json = enhance_json(final_json)
            if ctx.args.verbose:
                ctx.console.print("[dim]RAG 최적화 적용 완료[/dim]")

        # Save JSON output
        final_json_text = json.dumps(final_json, ensure_ascii=False, indent=2)
        with open(json_out, "w", encoding="utf-8") as f:
            f.write(final_json_text)

        # Update cache
        if ctx.cache_manager:
            ctx.cache_manager.update_file_state(
                str(file),
                hwp_hash=ctx.cache_manager.compute_file_hash(file),
                pipeline_signature=ctx.pipeline_signature + "|hwpx_direct",
                final_json_hash=ctx.cache_manager.compute_text_hash(final_json_text),
            )

        # Report statistics
        metadata = final_json.get("metadata", {})
        stats = metadata.get("parsing_statistics", {})
        total_regs = stats.get("total_regulations", len(final_json.get("docs", [])))
        success_rate = stats.get("success_rate", 100.0)

        ctx.console.print(
            f"[green]✓ {file.name}[/green]: "
            f"{total_regs} 규정 파싱 완료 ({success_rate:.1f}%) "
            f"({parsing_time:.2f}초)"
        )

        # Advance progress (direct parsing is faster - count as 2 steps)
        progress.advance(total_task, 2)

        return True

    except Exception as e:
        ctx.console.print(f"[red]HWPX 직접 파싱 실패 ({file.name}): {e}[/red]")
        import traceback
        if ctx.args.verbose:
            ctx.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False


def run_pipeline(args: argparse.Namespace, console: Optional["Console"] = None) -> int:
    """Run the HWP to JSON conversion pipeline.

    Args:
        args: Command line arguments namespace.
        console: Optional Rich console for output.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    if not console:
        from rich.console import Console

        console = Console()

    load_dotenv()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_output = (Path.cwd() / "output").resolve()
    if output_dir.resolve() == legacy_output:
        console.print(
            "[yellow]경고: 'output/'은 레거시 경로입니다. 'data/output' 사용을 권장합니다.[/yellow]"
        )

    # Collect files
    files = collect_hwp_files(input_path, console, use_hwpx=args.hwpx)
    if files is None:
        return 1

    # Initialize components
    cache_manager = CacheManager(cache_dir=args.cache_dir)
    rules_path = _resolve_preprocessor_rules_path()
    rules_hash = _compute_rules_hash(
        rules_path, cache_manager, console=console, verbose=args.verbose
    )

    try:
        llm_client, llm_signature = _initialize_llm(args, console)
    except Exception as e:
        console.print(f"[red]LLM 초기화 실패: {e}[/red]")
        return 1

    pipeline_signature = _build_pipeline_signature(rules_hash, llm_signature)

    ctx = PipelineContext(
        cache_manager=cache_manager,
        preprocessor=Preprocessor(llm_client=llm_client, cache_manager=cache_manager),
        formatter=RegulationFormatter(),
        metadata_extractor=MetadataExtractor(),
        pipeline_signature=pipeline_signature,
        args=args,
        console=console,
    )

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    console.rule("[bold blue]처리 시작[/bold blue]")

    had_errors = False
    STEPS_PER_FILE = 5

    # Adjust steps for HWPX direct mode (faster - fewer steps)
    if args.hwpx:
        STEPS_PER_FILE = 2  # Parse + Save (no conversion/preprocessing needed)

    with Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        BarColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        total_task = progress.add_task(
            "[green]전체 진행률[/green]", total=len(files) * STEPS_PER_FILE
        )

        for file in files:
            try:
                if args.hwpx:
                    # Use HWPX direct parser
                    success = _run_hwpx_direct_pipeline(
                        file,
                        output_dir,
                        ctx,
                        progress,
                        total_task,
                        use_hwpx=ctx.args.hwpx,  # Pass hwpx flag to control parser
                    )
                    if not success:
                        had_errors = True
                else:
                    # Use legacy converter pipeline
                    _process_single_file(
                        file,
                        input_path,
                        output_dir,
                        ctx,
                        progress,
                        total_task,
                        STEPS_PER_FILE,
                    )
            except Exception as e:
                console.print(f"[red]Error processing {file.name}: {e}[/red]")
                current = progress.tasks[total_task].completed
                expected = (files.index(file) + 1) * STEPS_PER_FILE
                remaining = expected - current
                if remaining > 0:
                    progress.advance(total_task, remaining)
                had_errors = True
            finally:
                if cache_manager:
                    cache_manager.save_all()

    console.rule("[bold blue]모든 작업 완료[/bold blue]")
    return 1 if had_errors else 0


def main():
    load_dotenv()

    providers = ["openai", "gemini", "openrouter", "ollama", "lmstudio", "local", "mlx"]
    default_provider = os.getenv("LLM_PROVIDER") or "openai"
    if default_provider not in providers:
        default_provider = "openai"
    default_model = os.getenv("LLM_MODEL") or None
    default_base_url = os.getenv("LLM_BASE_URL") or None

    parser = argparse.ArgumentParser(description="Regulation Management Pipeline")
    parser.add_argument(
        "input_path", type=str, help="Path to input HWP file or directory"
    )
    parser.add_argument("--output_dir", type=str, default="data/output")
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=providers,
    )
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--base_url", type=str, default=default_base_url)
    parser.add_argument(
        "--allow_llm_fallback",
        action="store_true",
        help="Allow regex-only fallback when LLM initialization fails",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-enhance-rag",
        action="store_false",
        dest="enhance_rag",
        help="Disable RAG optimization (parent_path, full_text, keywords, etc.)",
    )
    parser.add_argument(
        "--hwpx",
        action="store_true",
        default=True,
        help="Use HWPX direct parser (default: enabled for better accuracy). Disable with --no-hwpx for legacy parser.",
    )
    parser.add_argument(
        "--no-hwpx",
        action="store_false",
        dest="hwpx",
        help="Disable HWPX direct parser and use legacy converter.",
    )
    parser.set_defaults(enhance_rag=True)

    if len(sys.argv) == 1:
        from .interactive import run_interactive

        args = run_interactive()
    else:
        args = parser.parse_args()

    try:
        status = run_pipeline(args)
        if status != 0:
            sys.exit(status)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
