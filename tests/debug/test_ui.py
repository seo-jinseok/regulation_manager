from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, ProgressColumn
from rich.text import Text
import time

class SmartPercentageColumn(ProgressColumn):
    def render(self, task):
        if task.total is None:
            return Text("SKIP", style="dim") # Explicit text to verify visibility
        return Text(f"{task.percentage:>3.0f}%", style="cyan")

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
    progress.console.print("Phase 1: Normal 10%")
    progress.update(task, completed=10)
    time.sleep(1)
    
    # Phase 2: Indeterminate
    progress.console.print("Phase 2: Indeterminate (Should show SKIP)")
    progress.update(task, total=None)
    time.sleep(2)
    
    # Phase 3: Restore
    progress.console.print("Phase 3: Restore 50%")
    progress.update(task, total=100, completed=50)
    time.sleep(1)
