import sys
import os
import json
from pathlib import Path

# Add parent directory to path to find currently moved src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache_manager import CacheManager
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_caching_simulation():
    cache_dir = Path(".cache_test_sim")
    cache_dir.mkdir(exist_ok=True)
    
    # 1. Initialize Cache Manager
    cm = CacheManager(cache_dir=str(cache_dir))
    
    # 2. Mock File and Hash
    # 2. Mock File and Hash
    # Assuming file was moved to data/input
    hwp_path = Path("data/input/규정집9-349(20251202).hwp")
    if not hwp_path.exists():
        # Fallback search
        import glob
        files = glob.glob("data/input/*.hwp")
        if files:
            hwp_path = Path(files[0])
        else:
            print(f"File not found in data/input: {hwp_path}")
            return

    print("Computing HWP Hash...")
    real_hwp_hash = cm.compute_file_hash(hwp_path)
    print(f"HWP Hash: {real_hwp_hash}")
    
    # 3. Create Dummy Raw MD
    output_dir = Path("data/test_output_sim")
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

if __name__ == "__main__":
    test_caching_simulation()
