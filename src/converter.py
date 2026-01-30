import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

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

    def load_data(
        self,
        file: Path,
        extra_info: Optional[dict] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        verbose: bool = False,
    ) -> List[Document]:
        """
        Load data from HWP file.

        Args:
            file: Path to the HWP file.
            extra_info: Optional extra metadata to include.
            status_callback: Optional function that receives status updates.
            verbose: Whether to print verbose output.

        Returns:
            List of Document objects containing the converted content.
        """
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")

        # 1. Convert HWP to HTML
        # Resolve hwp5html path

        # Create a temp directory for HTML output
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Use --html flag to skip bindata (image) extraction, which significantly improves performance.
                # When --html is used, the output should be a file path, not a directory.
                html_path = Path(tmp_dir) / "index.xhtml"
                cmd = ["hwp5html", "--html", "--output", str(html_path), str(file)]

                if verbose:
                    logger.info(f"Converting {file.name}...")

                # Stream output for user feedback (Suppressed for clean CLI)
                # print(f"    [hwp5html] Starting conversion for {file.name}...")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy(),
                )

                # Start monitoring thread
                import threading
                import time

                stop_monitor = threading.Event()

                def monitor_output_size():
                    last_reported_size = -1
                    while not stop_monitor.is_set():
                        try:
                            # Calculate total size of tmp_dir
                            total_size = sum(
                                f.stat().st_size
                                for f in Path(tmp_dir).rglob("*")
                                if f.is_file()
                            )
                            size_mb = total_size / (1024 * 1024)
                            if (
                                size_mb > 0
                                and size_mb != last_reported_size
                                and status_callback
                            ):
                                status_callback(
                                    f"[dim]변환 데이터 생성 중... ({size_mb:.1f}MB)[/dim]"
                                )
                                last_reported_size = size_mb
                        except (OSError, PermissionError):
                            # File access errors during monitoring are non-critical
                            pass
                        time.sleep(2)

                monitor_thread = threading.Thread(
                    target=monitor_output_size, daemon=True
                )
                monitor_thread.start()

                # Print output in real-time
                captured_output = []
                if process.stdout:
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            always_ignored = [
                                "undefined UnderlineStyle value",
                                "defined name/values",
                                "pkg_resources is deprecated as an API",
                                "The pkg_resources package is slated for removal",
                                "import pkg_resources",
                            ]
                            if any(k in line for k in always_ignored):
                                continue
                            # Capture but don't print unless error
                            captured_output.append(line)

                            # Filter noisy logs unless verbose
                            if not verbose:
                                ignored = [
                                    "pkg_resources",
                                    "UserWarning",
                                    "import",
                                    "defined name/values",
                                    "UnderlineStyle",
                                    "Unknown",
                                ]
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
                    raise RuntimeError(
                        f"hwp5html failed with code {process.returncode}:\n{error_msg}"
                    )

                # Find the HTML file (usually index.html or named after file)
                # hwp5html usually creates 'index.html' inside the output dir
                html_path = (
                    Path(tmp_dir) / "index.xhtml"
                )  # hwp5html often produces xhtml or index.html
                if not html_path.exists():
                    # try index.html
                    html_path = Path(tmp_dir) / "index.html"

                if not html_path.exists():
                    # fallback: find any .html/.xhtml
                    candidates = list(Path(tmp_dir).glob("*.html")) + list(
                        Path(tmp_dir).glob("*.xhtml")
                    )
                    if candidates:
                        html_path = candidates[0]
                    else:
                        raise FileNotFoundError(
                            "No HTML output found in hwp5html directory"
                        )

                # 2. Read HTML content
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                # 3. Convert HTML to Markdown
                if md:
                    try:
                        from bs4 import BeautifulSoup

                        from src.parsing.html_table_converter import (
                            convert_html_tables_to_markdown,
                        )

                        soup = BeautifulSoup(html_content, "html.parser")
                        tables = soup.find_all("table")

                        replaced_tables_map = {}

                        # Replace tables with unique placeholders
                        for idx, table in enumerate(tables):
                            # Use a UUID-like placeholder intended to survive markdownify
                            placeholder = f"__TABLE_PLACEHOLDER_{idx}__"
                            # Convert this specific table to robust markdown (handles rowspan)
                            # Note: convert_html_tables_to_markdown expects full HTML string but works with table string
                            # It returns a list of tables found. We expect exactly 1.

                            # We can just use the internal logic or pass the string representation of the table
                            table_html = str(table)
                            robust_md_list = convert_html_tables_to_markdown(table_html)

                            if robust_md_list:
                                robust_md = robust_md_list[0]
                                replaced_tables_map[placeholder] = robust_md
                                # Replace in DOM with pure text placeholder
                                table.replace_with(soup.new_string(placeholder))

                        # Convert the modified HTML (with placeholders) to Markdown
                        # markdownify will treat placeholders as normal text
                        modified_html = str(soup)
                        markdown_content = md(modified_html, heading_style="ATX")

                        # Restore placeholders with robust Markdown
                        # Iterate by length desc to avoid partial matches (though unlikely with this placeholder)
                        for ph, table_md in replaced_tables_map.items():
                            markdown_content = markdown_content.replace(ph, table_md)
                            # Also replace escaped version (markdownify escapes underscores as \_)
                            escaped_ph = ph.replace("_", "\\_")
                            markdown_content = markdown_content.replace(
                                escaped_ph, table_md
                            )

                    except Exception as e:
                        # Fallback to standard markdownify if anything fails (e.g. import error)
                        if verbose:
                            logger.warning(
                                f"Table robust conversion failed: {e}. Falling back to default."
                            )
                        markdown_content = md(html_content, heading_style="ATX")
                else:
                    markdown_content = html_content  # Fallback

                # 4. Create Document
                return [
                    Document(
                        text=markdown_content,
                        metadata={
                            "file_name": file.name,
                            "source": str(file),
                            "html_content": html_content,  # Store raw HTML for table preservation
                            **(extra_info or {}),
                        },
                    )
                ]
            except Exception as e:
                raise e


if __name__ == "__main__":
    # Test block
    import sys

    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) > 1:
        reader = HwpToMarkdownReader(keep_html=True)
        docs = reader.load_data(Path(sys.argv[1]))
        logger.debug(f"Converted text preview: {docs[0].text[:500]}")
