from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, ProgressColumn
from rich.text import Text

class SmartPercentageColumn(ProgressColumn):
    def render(self, task):
        if task.total is None:
            return Text("SKIP", style="dim") # Explicit text to verify visibility
        return Text(f"{task.percentage:>3.0f}%", style="cyan")

def test_progress_column_handles_indeterminate():
    console = Console()

    with Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        SmartPercentageColumn(),
        BarColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Testing...", total=100)
        
        # Phase 1: Normal
        progress.update(task, completed=10)
        
        # Phase 2: Indeterminate
        progress.update(task, total=None)
        
        # Phase 3: Restore
        progress.update(task, total=100, completed=50)
