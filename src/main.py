import os
import argparse
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# from .converter import HwpToMarkdownReader # Moved to lazy import in main()
from .preprocessor import Preprocessor
from .formatter import RegulationFormatter
from .llm_client import LLMClient
from .refine_json import refine_doc
from .cache_manager import CacheManager

def main():
    parser = argparse.ArgumentParser(description="Regulation Management Pipeline")
    parser.add_argument("input_path", type=str, help="Path to input HWP file or directory")
    parser.add_argument("--output_dir", type=str, default="data/output", help="Output directory")
    parser.add_argument("--use_llm", action="store_true", help="Enable LLM-based preprocessing")
    parser.add_argument("--provider", type=str, default="openai", 
                        choices=["openai", "gemini", "openrouter", "ollama", "local", "lmstudio"], 
                        help="LLM Provider (includes local options)")
    parser.add_argument("--model", type=str, default=None, help="Specific model name (e.g. 'gemma2', 'gpt-4o')")
    parser.add_argument("--base_url", type=str, default=None, help="Custom Base URL for local providers (e.g. http://localhost:11434)")
    parser.add_argument("--force", action="store_true", help="Force re-conversion of HWP files (ignore cache)")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Directory for cache storage")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging")
    
    # If no arguments provided, try interactive mode
    if len(sys.argv) == 1:
        try:
            from .interactive import run_interactive
            args = run_interactive()
        except ImportError:
            print("Interactive mode not available.")
            parser.print_help()
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)
    else:
        args = parser.parse_args()
    
    # Load env
    load_dotenv()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = []
    if input_path.is_file():
        files.append(input_path)
    elif input_path.is_dir():
        files.extend(list(input_path.glob("*.hwp")))
    
    if not files:
        print("No HWP files found.")
        sys.exit(1)

    # Initialize components
    cache_manager = CacheManager(cache_dir=args.cache_dir)
    
    # Lazy import Reader to avoid llama_index dependency if not needed (cache hit)
    reader = None
    try:
        from .converter import HwpToMarkdownReader
        reader = HwpToMarkdownReader(keep_html=False)
    except ImportError:
        # Mock reader or just pass None, but handle check later
        pass 

    
    llm_client = None
    if args.use_llm:
        try:
           print(f"Initializing LLM: {args.provider} ({args.model or 'default'})...")
           llm_client = LLMClient(
               provider=args.provider, 
               model=args.model, 
               base_url=args.base_url
            )
        except ValueError as e:
            print(f"Warning: {e}. Semantic preprocessing disabled.")
            
    # Pass cache_manager to preprocessor
    preprocessor = Preprocessor(llm_client=llm_client, cache_manager=cache_manager)
    formatter = RegulationFormatter()

    # Rich Check
    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, ProgressColumn
        from rich.text import Text
        from rich.table import Table
    except ImportError:
        # Fallback (should not happen given requirements)
        sys.exit("Rich library not found. Please run: pip install rich")
        
    console = Console()
    
    # Custom Column to hide 0% when indeterminate
    class SmartPercentageColumn(ProgressColumn):
        def render(self, task: "Task") -> Text:
            if task.fields.get("indeterminate"):
                return Text("", style="dim") 
            return Text(f"{task.percentage:>3.0f}%", style="cyan")

    try:
        # Pre-scan for validation
        files = [f for f in Path(args.input_path).rglob("*.hwp") if f.is_file()]
        if not files:
             files = [Path(args.input_path)] if Path(args.input_path).suffix == ".hwp" else []
             
        if not files:
            console.print("[red]처리할 HWP 파일이 없습니다.[/red]")
            return

        console.rule("[bold blue]처리 시작[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TimeElapsedColumn(),
            SmartPercentageColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # 5 steps per file: HWP, HashCheck, Preprocess, Format, Save
            STEPS_PER_FILE = 5
            TOTAL_STEPS = len(files) * STEPS_PER_FILE
            total_task = progress.add_task("[green]전체 진행률[/green]", total=TOTAL_STEPS)
            
            file_idx = 0
            for file in files:
                base_desc = f"[bold cyan]처리 중:[/bold cyan] {file.name}"
                progress.update(total_task, description=base_desc)
                current_file_base_step = file_idx * STEPS_PER_FILE
                
                try:
                    file_start_time = time.time()
                    raw_md_path = output_dir / f"{file.stem}_raw.md"
                    json_path = output_dir / f"{file.stem}.json"
                    
                    # 1. HWP -> MD
                    cached_state = cache_manager.get_file_state(str(file)) or {}
                    current_hwp_hash = cache_manager.compute_file_hash(file)
                    
                    if not args.force and raw_md_path.exists(): 
                         # Ignored hash check to force using existing MD if present (solving NFD/NFC mismatch + missing dependency)
                         # and current_hwp_hash == cached_state.get("hwp_hash"):
                         progress.console.print(f"  [dim]• HWP 캐시 적중 (Forced): {file.name}[/dim]")

                         with open(raw_md_path, "r", encoding="utf-8") as f:
                            raw_md = f.read()
                         # Just advance normally
                         progress.update(total_task, completed=current_file_base_step + 1)
                    else:
                        # Just show file size, no guesses
                        file_size_mb = file.stat().st_size / (1024 * 1024)
                        progress.console.print(f"  [yellow]• HWP 변환 중 ({file_size_mb:.1f}MB)...[/yellow]")
                        
                        # Switch to indeterminate (pulse) for blocking operation
                        progress.update(total_task, total=None, indeterminate=True)
                        
                        # Define callback helper
                        def hwp_status_callback(msg):
                            # Filter noisy logs
                            ignored_keywords = [
                                "pkg_resources", "UserWarning", "import", 
                                "defined name/values", "UnderlineStyle", "Unknown"
                            ]
                            if any(k in msg for k in ignored_keywords):
                                return
                            
                            # Clean up message for display
                            clean_msg = msg.strip()
                            if len(clean_msg) > 30:
                                clean_msg = clean_msg[:27] + "..."
                                
                            # Append log message to description for live feedback
                            if clean_msg:
                                progress.update(total_task, description=f"{base_desc} [dim]({clean_msg})[/dim]")

                        if reader is None:
                             progress.console.print(f"[bold red]❌ HWP 변환 모듈(llama_index) 로드 실패. 캐시가 없어 작업을 수행할 수 없습니다: {file.name}[/bold red]")
                             progress.update(total_task, completed=current_file_base_step + 5)
                             file_idx += 1
                             continue

                        docs = reader.load_data(file, status_callback=hwp_status_callback, verbose=getattr(args, 'verbose', False))
                        
                        # Restore description/state
                        progress.update(total_task, description=base_desc)
                        
                        raw_md = docs[0].text
                        with open(raw_md_path, "w", encoding="utf-8") as f:
                            f.write(raw_md)
                             
                        # Early Cache Save: To prevent re-conversion if user cancels here
                        cache_manager.update_file_state(str(file), hwp_hash=current_hwp_hash)
                        cache_manager.save_all()
                        
                        # Restore determinate state
                        progress.update(total_task, total=TOTAL_STEPS, completed=current_file_base_step + 1, indeterminate=False)

                    # 2. Check Processing Cache
                    current_raw_hash = cache_manager.compute_text_hash(raw_md)
                    if not args.force and json_path.exists() and current_raw_hash == cached_state.get("raw_md_hash"):
                        progress.console.print(f"  [dim]• 변동 사항 없음 (Skipping)[/dim]")
                        cache_manager.save_all()
                        # Skip remaining 4 steps -> Completed = Base + 5
                        progress.update(total_task, completed=current_file_base_step + 5)
                        file_idx += 1
                        continue
                    
                    progress.update(total_task, completed=current_file_base_step + 2)

                    # 3. Preprocessing (Step 3)
                    progress.console.print(f"  [blue]• AI 전처리 실행 중...[/blue]")
                    clean_md = preprocessor.clean(raw_md)
                    with open(output_dir / f"{file.stem}_clean.md", "w", encoding="utf-8") as f:
                        f.write(clean_md)
                    progress.update(total_task, completed=current_file_base_step + 3)
                    
                    # 4. Formatting (Step 4)
                    progress.console.print(f"  [magenta]• JSON 구조화 중...[/magenta]")
                    regulations_list = formatter.parse(clean_md)
                    
                    # Refinement
                    refined_list = [refine_doc(doc, i) for i, doc in enumerate(regulations_list)]
                    progress.update(total_task, completed=current_file_base_step + 4)
                    
                    # 5. Save & Metadata (Step 5)
                    try:
                        mtime = os.path.getmtime(file)
                        from datetime import datetime
                        scan_date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        scan_date = "unknown"
                        
                    final_json = {
                        "file_name": file.name,
                        "scan_date": scan_date,
                        "docs": refined_list
                    }

                    # Save
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(final_json, f, ensure_ascii=False, indent=2)
                        
                    cache_manager.update_file_state(str(file), raw_md_hash=current_raw_hash)
                    cache_manager.save_all()
                    
                    duration = time.time() - file_start_time
                    progress.console.print(f"  [bold green]✓ 완료 ({duration:.2f}s)[/bold green]")
                    progress.update(total_task, completed=current_file_base_step + 5)
                    
                except Exception as e:
                    progress.console.print(f"[bold red]❌ 오류 발생: {file.name}[/bold red] - {e}")
                    # Advance just to keep moving, though it failed
                    progress.update(total_task, completed=current_file_base_step + 5)
                
                file_idx += 1

            console.rule("[bold blue]모든 작업 완료[/bold blue]")

            console.rule("[bold blue]모든 작업 완료[/bold blue]")

    except KeyboardInterrupt:
        console.print("\n[bold red]⛔ 사용자에 의해 중단되었습니다.[/bold red]")
        # Save whatever we have
        cache_manager.save_all()
        sys.exit(130)


    # Final save
    cache_manager.save_all()

if __name__ == "__main__":
    main()
