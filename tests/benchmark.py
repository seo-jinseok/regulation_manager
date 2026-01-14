import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cache_manager import CacheManager
from src.preprocessor import Preprocessor


# Mock LLM to avoid actual API costs/latency for the benchmark,
# but simulate latency to make the speedup obvious.
class SimulatedLLM:
    def __init__(self, latency=0.5):
        self.call_count = 0
        self.latency = latency

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        time.sleep(self.latency)  # Simulate network/processing time
        # Extract content to return valid-ish string
        try:
            core_text = prompt.split("[텍스트 시작]")[1].split("[텍스트 끝]")[0].strip()
            return core_text
        except Exception:
            return "Processed Text"


def run_benchmark():
    # 1. Load Real Data
    raw_md_path = Path("data/output/규정집9-349(20251202)_raw.md")
    if not raw_md_path.exists():
        # Fallback to the other file if this one doesn't exist
        raw_md_path = Path("data/output/규정집9-343(20250909)_raw.md")

    if not raw_md_path.exists():
        print("Error: No raw markdown file found in data/output/")
        return

    print(f"Loading data from {raw_md_path.name}...")
    with open(raw_md_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # 2. Extract a "Middle Chunk" (e.g., 50 Articles) to verify user request
    # We'll split by "제" to roughly find articles
    split_text = full_text.split("제")
    start_idx = len(split_text) // 2
    # Take ~50 chunks from the middle
    subset_chunks = ["제" + chunk for chunk in split_text[start_idx : start_idx + 50]]
    subset_text = "".join(subset_chunks)

    print(
        f"Extracted partial text: {len(subset_text)} chars ({len(subset_chunks)} potential articles)"
    )

    # 3. Setup Components
    cache_dir = ".cache_benchmark"
    # Clean cache
    import shutil

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    cache_manager = CacheManager(cache_dir=cache_dir)
    # Simulate 0.1s latency per call (fast LLM)
    llm = SimulatedLLM(latency=0.1)
    preprocessor = Preprocessor(llm_client=llm, cache_manager=cache_manager)

    # 4. Run 1: Cold Cache
    print("\n[Run 1] Initial Processing (Cold Cache)...")
    start_time = time.time()
    preprocessor.clean(subset_text)
    duration_v1 = time.time() - start_time
    calls_v1 = llm.call_count
    print(f"Done in {duration_v1:.2f}s. LLM Calls: {calls_v1}")

    # Save cache so Run 2 can see it!
    cache_manager.save_all()

    # 5. Modify Data (Simulate 0.1% change)
    # We will modify the text of the 10th chunk slightly.
    # Note: subset_chunks logic in this script (split by "제") implies imperfect articles,
    # but as long as the text string changes, the hash will change for *that* specific unit
    # detected by preprocessor.

    # Let's just modify the subset_text directly to be sure
    # Find "제" and modify nearby
    modified_text = subset_text.replace("제5조", "제5조(수정됨)")
    if modified_text == subset_text:
        # Fallback if 제5조 not found
        modified_text = subset_text + " "

    print(f"Text Modified? {modified_text != subset_text}")

    # 6. Run 2: Warm Cache
    print("\n[Run 2] Incremental Processing (Warm Cache)...")
    llm.call_count = 0  # Reset
    cache_manager = CacheManager(
        cache_dir=cache_dir
    )  # Reload cache from disk to be sure
    preprocessor = Preprocessor(llm_client=llm, cache_manager=cache_manager)

    start_time = time.time()
    preprocessor.clean(modified_text)
    duration_v2 = time.time() - start_time
    calls_v2 = llm.call_count

    print(f"Done in {duration_v2:.2f}s. LLM Calls: {calls_v2}")

    # 7. Summary
    print("\n--- Benchmark Results ---")
    print(f"Content Size: {len(subset_text)} chars")
    print(f"Cold Cache Time: {duration_v1:.2f}s ({calls_v1} calls)")
    print(f"Warm Cache Time: {duration_v2:.2f}s ({calls_v2} calls)")
    if duration_v1 > 0:
        speedup = duration_v1 / duration_v2 if duration_v2 > 0 else 0
        print(f"Speedup: {speedup:.1f}x")

    if calls_v1 > calls_v2 and calls_v2 <= 2:
        print("\nSUCCESS: Optimization verified. Only changed parts were re-processed.")
    else:
        print("\nWARNING: Optimization results unexpected. Check logic.")


if __name__ == "__main__":
    run_benchmark()


def run_search_benchmark():
    """
    Benchmark search performance: cold start vs warm start.
    
    Tests:
    1. First search latency (cold - no cached index, no warmup)
    2. Warmup latency (time to pre-initialize)
    3. Search after warmup
    """
    print("\n=== Search Performance Benchmark ===\n")
    
    # Check if DB exists
    db_path = Path("data/chroma_db")
    if not db_path.exists():
        print("Error: ChromaDB not found at data/chroma_db")
        print("Run 'uv run regulation sync' first")
        return
    
    # 1. Cold Start (no cache, no warmup)
    print("[1] Cold Start (no BM25 cache, no warmup)...")
    
    # Clear any existing BM25 cache
    cache_path = Path("data/bm25_index.pkl")
    if cache_path.exists():
        cache_path.unlink()
    
    # Clear reranker
    try:
        from src.rag.infrastructure.reranker import clear_reranker
        clear_reranker()
    except ImportError:
        pass
    
    from src.rag.infrastructure.chroma_store import ChromaVectorStore
    from src.rag.application.search_usecase import SearchUseCase
    
    store = ChromaVectorStore(persist_directory=str(db_path))
    usecase = SearchUseCase(store=store, use_reranker=True, enable_warmup=False)
    
    test_query = "휴직 신청 절차"
    
    start = time.time()
    results = usecase.search(test_query, top_k=5)
    cold_time = time.time() - start
    print(f"  First search: {cold_time:.3f}s ({len(results)} results)")
    
    # 2. Second search (warm components)
    start = time.time()
    results = usecase.search(test_query, top_k=5)
    warm_time = time.time() - start
    print(f"  Second search: {warm_time:.3f}s")
    
    # 3. With BM25 cache enabled
    print("\n[2] With BM25 Index Cache...")
    os.environ["BM25_INDEX_CACHE_PATH"] = str(cache_path)
    
    # Force config reload
    from src.rag.config import reset_config
    reset_config()
    
    # First time - builds and saves cache
    store = ChromaVectorStore(persist_directory=str(db_path))
    usecase = SearchUseCase(store=store, use_reranker=True, enable_warmup=False)
    
    start = time.time()
    results = usecase.search(test_query, top_k=5)
    build_cache_time = time.time() - start
    print(f"  Build + Save cache: {build_cache_time:.3f}s")
    
    # Second time - loads from cache
    store2 = ChromaVectorStore(persist_directory=str(db_path))
    usecase2 = SearchUseCase(store=store2, use_reranker=False, enable_warmup=False)  # skip reranker for fair comparison
    
    start = time.time()
    results = usecase2.search(test_query, top_k=5)
    load_cache_time = time.time() - start
    print(f"  Load from cache: {load_cache_time:.3f}s")
    
    # 4. Background Warmup
    print("\n[3] Background Warmup Simulation...")
    
    # Clear and reset
    clear_reranker()
    reset_config()
    
    store3 = ChromaVectorStore(persist_directory=str(db_path))
    
    start = time.time()
    usecase3 = SearchUseCase(store=store3, use_reranker=True, enable_warmup=True)
    init_time = time.time() - start
    print(f"  Init with warmup thread: {init_time:.3f}s (returns immediately)")
    
    # Wait for warmup to complete
    print("  Waiting 3s for warmup thread...")
    time.sleep(3)
    
    start = time.time()
    results = usecase3.search(test_query, top_k=5)
    post_warmup_time = time.time() - start
    print(f"  Search after warmup: {post_warmup_time:.3f}s")
    
    # Summary
    print("\n--- Summary ---")
    print(f"Cold start first search: {cold_time:.3f}s")
    print(f"With BM25 cache load:    {load_cache_time:.3f}s")
    print(f"After background warmup: {post_warmup_time:.3f}s")
    
    if cold_time > 0:
        print(f"\nSpeedup with cache: {cold_time / max(load_cache_time, 0.001):.1f}x")
    
    # Cleanup
    os.environ.pop("BM25_INDEX_CACHE_PATH", None)
    reset_config()
    if cache_path.exists():
        cache_path.unlink()
