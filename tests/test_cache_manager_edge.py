import unittest
import os
import shutil
from src.cache_manager import CacheManager

class TestCacheManagerEdge(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tmp_cache_test"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cache_save_load_empty(self):
        cache = CacheManager(cache_dir=self.test_dir)
        cache.save_all()
        
        # New instance loading same dir
        cache2 = CacheManager(cache_dir=self.test_dir)
        self.assertEqual(cache2.llm_cache, {})

    def test_cache_llm_response(self):
        cache = CacheManager(cache_dir=self.test_dir)
        cache.cache_llm_response("input text", "output response")
        self.assertEqual(cache.get_cached_llm_response("input text"), "output response")
        
        cache.save_all()
        cache3 = CacheManager(cache_dir=self.test_dir)
        self.assertEqual(cache3.get_cached_llm_response("input text"), "output response")

if __name__ == "__main__":
    unittest.main()
