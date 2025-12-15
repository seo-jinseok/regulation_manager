import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
try:
    from markdownify import markdownify as md
except ImportError:
    md = None

class HwpToMarkdownReader(BaseReader):
    """
    Custom LlamaIndex Reader for HWP files.
    Uses hwp5html (via CLI or library) to convert HWP to HTML, 
    then converts HTML to Markdown to preserve tables and structure.
    """

    def __init__(self, keep_html: bool = False):
        self.keep_html = keep_html

    def load_data(self, file: Path, extra_info: Optional[dict] = None, status_callback=None, verbose: bool = False) -> List[Document]:
        """
        Load data from HWP file.
        status_callback: Optional function that receives a string status update (e.g. log line).
        """
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")

        # 1. Convert HWP to HTML
        # Resolve hwp5html path
        import shutil
        import sys
        
        # Create a temp directory for HTML output
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Try using the 'hwp5html' executable directly, which is more reliable in uv/venv.
                # If that fails, fallback to python module? No, just stick to executable if in venv.
                cmd = ["hwp5html", "--output", tmp_dir, str(file)]
                print(f"DEBUG: Executing command: {cmd}")
                
                # Stream output for user feedback (Suppressed for clean CLI)
                # print(f"    [hwp5html] Starting conversion for {file.name}...")
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    bufsize=1,
                    env=os.environ.copy()
                )
                
                # Start monitoring thread
                import threading
                import time
                
                stop_monitor = threading.Event()
                
                def monitor_output_size():
                    while not stop_monitor.is_set():
                        try:
                            # Calculate total size of tmp_dir
                            total_size = sum(f.stat().st_size for f in Path(tmp_dir).rglob('*') if f.is_file())
                            size_mb = total_size / (1024 * 1024)
                            if size_mb > 0 and status_callback:
                                status_callback(f"[dim]변환 데이터 생성 중... ({size_mb:.1f}MB)[/dim]")
                        except Exception:
                            pass
                        time.sleep(2)
                
                monitor_thread = threading.Thread(target=monitor_output_size, daemon=True)
                monitor_thread.start()

                # Print output in real-time
                captured_output = []
                if process.stdout:
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            # Capture but don't print unless error
                            captured_output.append(line)
                            
                            # Filter noisy logs unless verbose
                            if not verbose:
                                ignored = ["pkg_resources", "UserWarning", "import", "defined name/values", "UnderlineStyle", "Unknown"]
                                if any(k in line for k in ignored):
                                    continue

                            # Call status callback if provided
                            if status_callback:
                                status_callback(f"[dim]{line}[/dim]")
                
                process.wait()
                stop_monitor.set()
                monitor_thread.join(timeout=1)
                
                process.wait()
                
                if process.returncode != 0:
                    error_msg = "\n".join(captured_output)
                    raise RuntimeError(f"hwp5html failed with code {process.returncode}:\n{error_msg}")

                # Find the HTML file (usually index.html or named after file)
                # hwp5html usually creates 'index.html' inside the output dir
                html_path = Path(tmp_dir) / "index.xhtml" # hwp5html often produces xhtml or index.html
                if not html_path.exists():
                     # try index.html
                    html_path = Path(tmp_dir) / "index.html"
                
                if not html_path.exists():
                    # fallback: find any .html/.xhtml
                    candidates = list(Path(tmp_dir).glob("*.html")) + list(Path(tmp_dir).glob("*.xhtml"))
                    if candidates:
                        html_path = candidates[0]
                    else:
                        raise FileNotFoundError("No HTML output found in hwp5html directory")

                # 2. Read HTML content
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                # 3. Convert HTML to Markdown
                if md:
                    # Use markdownify to convert HTML to MD, preserving tables
                    markdown_content = md(html_content, heading_style="ATX")
                else:
                    markdown_content = html_content  # Fallback

                # 4. Create Document
                return [Document(
                    text=markdown_content,
                    metadata={
                        "file_name": file.name,
                        "source": str(file),
                        "html_content": html_content,  # Store raw HTML for table preservation
                        **(extra_info or {})
                    }
                )]
            except Exception as e:
                raise e


if __name__ == "__main__":
    # Test block
    import sys
    if len(sys.argv) > 1:
        reader = HwpToMarkdownReader(keep_html=True)
        docs = reader.load_data(Path(sys.argv[1]))
        print(docs[0].text[:500])
