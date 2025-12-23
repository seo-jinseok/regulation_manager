import os
# Suppress Transformers/PyTorch warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
import argparse
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from .converter import HwpToMarkdownReader
from .preprocessor import Preprocessor
from .formatter import RegulationFormatter
from .llm_client import LLMClient
from .metadata_extractor import MetadataExtractor
from .cache_manager import CacheManager

PIPELINE_SIGNATURE_VERSION = "v1"

def _resolve_preprocessor_rules_path() -> Path:
    rules_path = os.getenv("PREPROCESSOR_RULES_PATH")
    if rules_path:
        return Path(rules_path)
    return Path("data/config/preprocessor_rules.json")

def _compute_rules_hash(path: Path, cache_manager: CacheManager, console=None, verbose: bool = False) -> str:
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

def run_pipeline(args, console=None):
    if not console:
        from rich.console import Console
        console = Console()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_output = (Path.cwd() / "output").resolve()
    if output_dir.resolve() == legacy_output:
        console.print("[yellow]경고: 'output/'은 레거시 경로입니다. 'data/output' 사용을 권장합니다.[/yellow]")
    
    if not input_path.exists():
        console.print(f"[red]입력 경로가 존재하지 않습니다: {input_path}[/red]")
        return 1
    
    files = []
    if input_path.is_file():
        files.append(input_path)
    elif input_path.is_dir():
        files.extend(sorted(input_path.rglob("*.hwp")))
    
    if not files:
        console.print("[red]처리할 HWP 파일이 없습니다.[/red]")
        return 1

    # Initialize components
    cache_manager = CacheManager(cache_dir=args.cache_dir)
    rules_path = _resolve_preprocessor_rules_path()
    rules_hash = _compute_rules_hash(rules_path, cache_manager, console=console, verbose=args.verbose)

    llm_client = None
    llm_signature = "disabled"
    if args.use_llm:
        provider_name = args.provider if args.provider else "openai"
        try:
            llm_client = LLMClient(
                provider=provider_name,
                model=args.model,
                base_url=args.base_url,
            )
            llm_signature = llm_client.cache_namespace()
        except Exception as e:
            if args.allow_llm_fallback:
                console.print(f"[yellow]LLM 초기화 실패: {e} - LLM 비활성화하고 계속 진행합니다.[/yellow]")
                llm_client = None
                llm_signature = "disabled"
            else:
                console.print(f"[red]LLM 초기화 실패: {e}[/red]")
                return 1

    pipeline_signature = _build_pipeline_signature(rules_hash, llm_signature)

    preprocessor = Preprocessor(llm_client=llm_client, cache_manager=cache_manager)
    formatter = RegulationFormatter()
    metadata_extractor = MetadataExtractor()

    from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
    
    console.rule("[bold blue]처리 시작[/bold blue]")
    
    had_errors = False
    with Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        BarColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        STEPS_PER_FILE = 5
        TOTAL_STEPS = len(files) * STEPS_PER_FILE
        total_task = progress.add_task("[green]전체 진행률[/green]", total=TOTAL_STEPS)
        
        for file in files:
            try:
                status_callback = console.print if args.verbose else None
                if input_path.is_dir():
                    rel_path = file.relative_to(input_path)
                    file_output_dir = output_dir / rel_path.parent
                else:
                    file_output_dir = output_dir
                file_output_dir.mkdir(parents=True, exist_ok=True)

                raw_md_path = file_output_dir / f"{file.stem}_raw.md"
                raw_html_path = file_output_dir / f"{file.stem}_raw.xhtml"
                json_path = file_output_dir / f"{file.stem}.json"
                metadata_path = file_output_dir / f"{file.stem}_metadata.json"
                html_content = None

                file_state = cache_manager.get_file_state(str(file)) if cache_manager else None
                cached_hwp_hash = file_state.get("hwp_hash") if file_state else None
                cached_raw_md_hash = file_state.get("raw_md_hash") if file_state else None
                cached_pipeline_signature = file_state.get("pipeline_signature") if file_state else None
                cached_final_json_hash = file_state.get("final_json_hash") if file_state else None
                cached_metadata_hash = file_state.get("metadata_hash") if file_state else None
                hwp_hash = None
                raw_md_hash = None
                cache_hit = False
                raw_md_cache_hit = True
                if cache_manager:
                    try:
                        hwp_hash = cache_manager.compute_file_hash(file)
                        if cached_hwp_hash:
                            cache_hit = cached_hwp_hash == hwp_hash
                    except Exception as e:
                        if args.verbose:
                            console.print(f"[yellow]HWP 해시 계산 실패: {e}[/yellow]")
                    if raw_md_path.exists():
                        try:
                            raw_md_hash = cache_manager.compute_file_hash(raw_md_path)
                            if cached_raw_md_hash:
                                raw_md_cache_hit = cached_raw_md_hash == raw_md_hash
                        except Exception as e:
                            raw_md_cache_hit = False
                            if args.verbose:
                                console.print(f"[yellow]RAW MD 해시 계산 실패: {e}[/yellow]")

                def output_hash_matches(path: Path, cached_hash: str) -> bool:
                    if not cache_manager or not cached_hash or not path.exists():
                        return False
                    try:
                        return cache_manager.compute_file_hash(path) == cached_hash
                    except Exception:
                        return False

                full_cache_hit = (
                    not args.force
                    and raw_md_path.exists()
                    and json_path.exists()
                    and metadata_path.exists()
                    and cache_hit
                    and raw_md_cache_hit
                    and cached_pipeline_signature == pipeline_signature
                    and output_hash_matches(json_path, cached_final_json_hash)
                    and output_hash_matches(metadata_path, cached_metadata_hash)
                )
                if full_cache_hit:
                    if args.verbose:
                        console.print(f"[dim]캐시 적중: {file.name} (변환/전처리/포맷팅 생략)[/dim]")
                    progress.advance(total_task, STEPS_PER_FILE)
                    if cache_manager and hwp_hash:
                        cache_manager.update_file_state(
                            str(file),
                            hwp_hash=hwp_hash,
                            raw_md_hash=raw_md_hash,
                            pipeline_signature=pipeline_signature,
                        )
                    continue
                
                # 1. HWP -> MD
                if not args.force and raw_md_path.exists() and cache_hit and raw_md_cache_hit:
                    with open(raw_md_path, "r", encoding="utf-8") as f:
                        raw_md = f.read()
                    if raw_html_path.exists():
                        with open(raw_html_path, "r", encoding="utf-8") as f:
                            html_content = f.read()
                    if cache_manager and hwp_hash:
                        cache_manager.update_file_state(str(file), hwp_hash=hwp_hash, raw_md_hash=raw_md_hash)
                else:
                    reader = HwpToMarkdownReader(keep_html=False)
                    docs = reader.load_data(file, status_callback=status_callback, verbose=args.verbose)
                    raw_md = docs[0].text
                    html_content = docs[0].metadata.get("html_content")
                    with open(raw_md_path, "w", encoding="utf-8") as f:
                        f.write(raw_md)
                    if html_content:
                        with open(raw_html_path, "w", encoding="utf-8") as f:
                            f.write(html_content)
                    if cache_manager and hwp_hash:
                        raw_md_hash = cache_manager.compute_text_hash(raw_md)
                        cache_manager.update_file_state(
                            str(file),
                            hwp_hash=hwp_hash,
                            raw_md_hash=raw_md_hash,
                        )
                
                # ... (rest of simple logic) ...
                # Preprocess
                clean_md = preprocessor.clean(raw_md, verbose_callback=status_callback)

                extracted_metadata = metadata_extractor.extract(clean_md)
                metadata_payload = {"file_name": file.name, **extracted_metadata}
                metadata_text = json.dumps(metadata_payload, ensure_ascii=False, indent=2)
                with open(metadata_path, "w", encoding="utf-8") as f:
                    f.write(metadata_text)
                
                # Format
                final_docs = formatter.parse(clean_md, html_content=html_content, verbose_callback=status_callback)
                
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
                if args.verbose:
                    console.print(
                        f"[dim]메타데이터 누락: rule_code {missing_rule_code}/{len(final_docs)}, "
                        f"page_range {missing_page_range}/{len(final_docs)}[/dim]"
                    )
                
                # Save
                final_json = {"file_name": file.name, "docs": final_docs}
                final_json_text = json.dumps(final_json, ensure_ascii=False, indent=2)
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(final_json_text)

                if cache_manager:
                    if raw_md_hash is None:
                        raw_md_hash = cache_manager.compute_text_hash(raw_md)
                    cache_manager.update_file_state(
                        str(file),
                        hwp_hash=hwp_hash,
                        raw_md_hash=raw_md_hash,
                        pipeline_signature=pipeline_signature,
                        final_json_hash=cache_manager.compute_text_hash(final_json_text),
                        metadata_hash=cache_manager.compute_text_hash(metadata_text),
                    )
                
                progress.advance(total_task, STEPS_PER_FILE)
                
            except Exception as e:
                console.print(f"[red]Error processing {file.name}: {e}[/red]")
                progress.advance(total_task, STEPS_PER_FILE)
                had_errors = True
            finally:
                if cache_manager:
                    cache_manager.save_all()

    console.rule("[bold blue]모든 작업 완료[/bold blue]")
    return 1 if had_errors else 0

def main():
    parser = argparse.ArgumentParser(description="Regulation Management Pipeline")
    parser.add_argument("input_path", type=str, help="Path to input HWP file or directory")
    parser.add_argument("--output_dir", type=str, default="data/output")
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument(
        "--allow_llm_fallback",
        action="store_true",
        help="Allow regex-only fallback when LLM initialization fails",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--verbose", action="store_true")
    
    if len(sys.argv) == 1:
        from .interactive import run_interactive
        args = run_interactive()
    else:
        args = parser.parse_args()
    
    load_dotenv()
    try:
        status = run_pipeline(args)
        if status != 0:
            sys.exit(status)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)

if __name__ == "__main__":
    main()
