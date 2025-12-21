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
from .refine_json import refine_doc
from .cache_manager import CacheManager

def run_pipeline(args, console=None):
    if not console:
        from rich.console import Console
        console = Console()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = []
    if input_path.is_file():
        files.append(input_path)
    elif input_path.is_dir():
        files.extend(list(input_path.glob("*.hwp")))
    
    if not files:
        console.print("[red]처리할 HWP 파일이 없습니다.[/red]")
        return

    # Initialize components
    cache_manager = CacheManager(cache_dir=args.cache_dir)
    
    llm_client = None
    if args.use_llm:
        provider_name = args.provider if args.provider else "openai"
        try:
           llm_client = LLMClient(
               provider=provider_name, 
               model=args.model, 
               base_url=args.base_url
            )
        except Exception as e:
            console.print(f"Warning: {e}")
            
    preprocessor = Preprocessor(llm_client=llm_client, cache_manager=cache_manager)
    formatter = RegulationFormatter()

    from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
    
    console.rule("[bold blue]처리 시작[/bold blue]")
    
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
                raw_md_path = output_dir / f"{file.stem}_raw.md"
                json_path = output_dir / f"{file.stem}.json"
                
                # 1. HWP -> MD
                if not args.force and raw_md_path.exists(): 
                     with open(raw_md_path, "r", encoding="utf-8") as f:
                        raw_md = f.read()
                else:
                    reader = HwpToMarkdownReader(keep_html=False)
                    docs = reader.load_data(file)
                    raw_md = docs[0].text
                    with open(raw_md_path, "w", encoding="utf-8") as f:
                        f.write(raw_md)
                
                # ... (rest of simple logic) ...
                # Preprocess
                clean_md = preprocessor.clean(raw_md)
                
                # Format
                final_docs = formatter.parse(clean_md)
                
                # Save
                final_json = {"docs": final_docs}
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(final_json, f, ensure_ascii=False, indent=2)
                
                progress.advance(total_task, STEPS_PER_FILE)
                
            except Exception as e:
                console.print(f"[red]Error processing {file.name}: {e}[/red]")
                progress.advance(total_task, STEPS_PER_FILE)

    console.rule("[bold blue]모든 작업 완료[/bold blue]")

def main():
    parser = argparse.ArgumentParser(description="Regulation Management Pipeline")
    parser.add_argument("input_path", type=str, help="Path to input HWP file or directory")
    parser.add_argument("--output_dir", type=str, default="data/output")
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
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
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)

if __name__ == "__main__":
    main()