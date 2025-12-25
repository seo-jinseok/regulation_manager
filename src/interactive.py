import os
import sys
from pathlib import Path
from typing import List, Any, Optional
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.layout import Layout

console = Console()

class InteractiveWizard:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()

    def _get_default_input_dir(self) -> Optional[Path]:
        base = self.root_dir
        data_input_dir = base / "data" / "input"
        legacy_dir = None
        
        # Try exact match first
        if (base / "규정").exists():
            legacy_dir = base / "규정"
        else:
            # Try glob for unicode safety (Mac NFD)
            for path in base.iterdir():
                if path.is_dir() and (path.name == "규정" or path.name == "규정"):
                    legacy_dir = path
                    break
        
        candidates = []
        if data_input_dir.exists():
            candidates.append(data_input_dir)
        if legacy_dir:
            candidates.append(legacy_dir)
        
        for cand in candidates:
            if any(cand.rglob("*.hwp")):
                return cand
        
        if candidates:
            return candidates[0]
        
        return None

    def run(self) -> Any:
        console.clear()
        welcome_panel = Panel(
            "[bold cyan]규정 관리 프로그램 (Regulation Manager)[/bold cyan]\n"
            "[white]규정 HWP 파일을 마크다운/JSON으로 변환하고 관리합니다.[/white]",
            title="[bold green]환영합니다[/bold green]",
            subtitle="v1.0",
            style="bold blue",
            expand=False
        )
        print(welcome_panel)
        print("")

        # 1. Scan for Files in input folder (prefer data/input, fallback to 규정)
        target_dir = self._get_default_input_dir()
        
        if not target_dir or not target_dir.exists():
            # Default to data/input for creation prompt
            target_dir = self.root_dir / "data" / "input"
            console.print(f"[bold red](!) 필수 폴더 없음:[/bold red] {target_dir}")
            console.print("    [yellow]'data/input'[/yellow] 또는 [yellow]'규정'[/yellow] 폴더를 생성하고 .hwp 파일을 넣어주세요.")
            
            import questionary
            if questionary.confirm(f"    지금 '{target_dir}' 폴더를 생성하시겠습니까?").ask():
                target_dir.mkdir(parents=True, exist_ok=True)
                console.print(f"    [green]>> 생성 완료.[/green] 파일을 이동시킨 후 다시 실행해주세요.")
            sys.exit(0)
            
        console.print(f"[blue]>> 입력 폴더 감지됨:[/blue] {target_dir}")
            
        with console.status("[bold green]파일 검색 중...[/bold green]", spinner="dots"):
            hwp_files = sorted(list(target_dir.rglob("*.hwp")))
        
        if not hwp_files:
            console.print(f"[bold red](!) 파일 없음:[/bold red] '{target_dir.name}' 폴더에 .hwp 파일이 없습니다.")
            sys.exit(0)

        selected_path = self._select_file_or_folder(hwp_files, target_dir)
        
        # 2. Configuration
        config = self._configure_options()
        config.input_path = str(selected_path)
        
        console.print(Panel("[bold green]설정 완료![/bold green] 처리를 시작합니다...", style="green"))
        return config

    def _select_file_or_folder(self, files: List[Path], base_dir: Path) -> Path:
        """Rich Table Selection"""
        # Default output dir assumption for status check
        default_output_dir = self.root_dir / "data" / "output"
        
        table = Table(title="[bold]처리할 대상 선택[/bold]", show_header=True, header_style="bold magenta")
        table.add_column("No.", style="cyan", justify="right")
        table.add_column("상태", justify="center")
        table.add_column("파일명", style="white")
        table.add_column("경로", style="dim")

        table.add_row("0", "", "[bold yellow]전체 일괄 처리[/bold yellow]", str(base_dir.relative_to(self.root_dir)))
        
        selection_map = {}
        idx = 1
        default_choice = 0 # Default to "All" if everything is converted
        
        for f in files:
            rel_path = f.relative_to(base_dir)
            
            # Check status
            json_path = default_output_dir / f"{f.stem}.json"
            if json_path.exists():
                status = "[green]✓ 변환됨[/green]"
            else:
                status = "[dim]-[/dim]"
                # If we haven't found an unconverted file yet, set this as default
                if default_choice == 0:
                    default_choice = idx
                
            table.add_row(
                str(idx), 
                status, 
                f.stem, 
                str(rel_path.parent) if str(rel_path.parent) != "." else ""
            )
            selection_map[idx] = f
            idx += 1

        # Rich table is nice for info, but for selection we want questionary
        # Let's show the table first as information
        print(table)
        print("")

        import questionary
        from questionary import Choice

        choices = []
        choices.append(Choice(title="0. 전체 일괄 처리 (All)", value=base_dir)) # Return base dir for all

        idx = 1
        default_choice = None
        
        for f in files:
            rel_path = f.relative_to(base_dir)
            json_path = default_output_dir / f"{f.stem}.json"
            is_converted = json_path.exists()
            status_icon = "✓" if is_converted else "-"
            
            # Format: "1. filename.hwp (✓)"
            title = f"{idx}. {f.name} ({status_icon})"
            choice_obj = Choice(title=title, value=f)
            choices.append(choice_obj)
            
            # Set default pointer to first unconverted file
            if default_choice is None and not is_converted:
                default_choice = choice_obj
            
            idx += 1

        selected = questionary.select(
            "처리할 파일을 선택하세요 (화살표 키로 이동, 엔터로 선택):",
            choices=choices,
            default=default_choice,
            style=questionary.Style([
                ('qmark', 'fg:cyan bold'),
                ('question', 'bold'),
                ('answer', 'fg:green bold'),
                ('pointer', 'fg:cyan bold'),
                ('highlighted', 'fg:cyan bold'),
                ('selected', 'fg:green'),
                ('separator', 'fg:grey'),
                ('instruction', 'fg:grey'),
                ('text', ''),
                ('disabled', 'fg:grey italic')
            ])
        ).ask()

        if selected == base_dir:
            console.print("[blue]>> 전체 처리를 시작합니다.[/blue]")
            return base_dir
        else:
            if selected:
                console.print(f"[blue]>> 선택됨:[/blue] {selected.name}")
                return selected
            else:
                 # Cancelled (Ctrl+C)
                 sys.exit(0)

    def _configure_options(self) -> Any:
        import questionary
        
        class Config:
            pass
        config = Config()
        
        print("\n[bold]설정 옵션[/bold]")

        # Output Directory (Fixed to 'data/output')
        config.output_dir = "data/output"
        
        # LLM Settings
        config.use_llm = questionary.confirm("LLM(AI)을 사용하여 텍스트 품질을 보정하시겠습니까?", default=False).ask()
        
        config.provider = "openai"
        config.model = None
        config.base_url = None
        config.allow_llm_fallback = False
        
        # Debug Options
        config.verbose = questionary.confirm("상세 로그를 보시겠습니까? (디버깅용)", default=False).ask()
        
        if config.use_llm:
            default_provider = os.getenv("LLM_PROVIDER") or "openai"
            default_model = os.getenv("LLM_MODEL") or ""
            default_base_url_env = os.getenv("LLM_BASE_URL") or ""
            providers = ["openai", "gemini", "openrouter", "ollama", "local", "lmstudio", "mlx"]
            config.provider = questionary.select(
                "LLM 제공자를 선택하세요:",
                choices=providers,
                default=default_provider
            ).ask()
            
            if config.provider in ["local", "ollama", "lmstudio", "mlx"]:
                if default_base_url_env:
                    default_url = default_base_url_env
                elif config.provider == "ollama":
                    default_url = "http://localhost:11434"
                elif config.provider == "mlx":
                    default_url = "http://localhost:8080"
                else:
                    default_url = "http://localhost:1234"
                config.base_url = questionary.text("Base URL:", default=default_url).ask()
            
            config.model = questionary.text(
                "모델 이름 (선택사항, 엔터로 건너뛰기):",
                default=default_model,
            ).ask()
            if not config.model:
                 config.model = None

        # Force Re-conversion
        config.force = questionary.confirm("기존 캐시 무시 (강제 변환)?", default=False).ask()
        config.cache_dir = ".cache"
        
        # RAG enhancement (default enabled)
        config.enhance_rag = True
        
        return config

def run_interactive() -> Any:
    try:
        wizard = InteractiveWizard()
        return wizard.run()
    except KeyboardInterrupt:
        sys.exit(0)
