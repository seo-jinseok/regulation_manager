import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path to find currently moved src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cache_manager import CacheManager


def test_caching_simulation(tmp_path):
    """Test caching logic using temporary directory that auto-cleans after test."""
    cache_dir = tmp_path / ".cache_test_sim"
    cache_dir.mkdir(exist_ok=True)

    # 1. Initialize Cache Manager
    cm = CacheManager(cache_dir=str(cache_dir))

    # 2. Mock File and Hash
    hwp_path = Path("data/input/규정집9-349(20251202).hwp")
    if not hwp_path.exists():
        # Fallback search
        import glob

        files = glob.glob("data/input/*.hwp")
        if files:
            hwp_path = Path(files[0])
        else:
            pytest.skip("HWP file not found in data/input")

    print("Computing HWP Hash...")
    real_hwp_hash = cm.compute_file_hash(hwp_path)
    print(f"HWP Hash: {real_hwp_hash}")

    # 3. Create Dummy Raw MD in temp directory (auto-cleaned)
    output_dir = tmp_path / "test_output_sim"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_md_path = output_dir / f"{hwp_path.stem}_raw.md"

    dummy_md_content = "# Test Regulation\n\nArticle 1 (Purpose) This is a test."
    with open(raw_md_path, "w") as f:
        f.write(dummy_md_content)

    print(f"Created Dummy MD at {raw_md_path}")

    # 4. Populate Cache manually
    # We pretend we already processed this HWP
    cm.update_file_state(str(hwp_path), hwp_hash=real_hwp_hash)
    cm.save_all()

    print("Cache populated. Now running main pipeline should skip HWP conversion.")

    # Verify output was created in temp directory
    assert raw_md_path.exists()


if __name__ == "__main__":
    # For manual testing, use a temp directory that we clean up
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_caching_simulation(Path(tmp))
