"""
Comprehensive tests for CacheManager to improve coverage from 74% to 85%+.

Tests focus on edge cases and previously uncovered code paths:
- ValueError handling in _get_env_int (lines 31-34)
- JSONDecodeError handling in _load_json (lines 41-42)
- tmp_path cleanup in _save_json (line 62)
- Non-dict entry normalization (line 89)
- _is_expired with various timestamp scenarios (lines 94-98)
- TTL and size pruning logic (lines 106-112, 119-126)
- Expired entry handling in get_cached_llm_response (lines 165, 168-169)
"""

import json
import os
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cache_manager import CacheManager


class TestCacheManagerComprehensive(unittest.TestCase):
    """Comprehensive tests for CacheManager covering missing lines."""

    def setUp(self):
        self.test_dir = "tmp_cache_comprehensive"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # Tests for _get_env_int (lines 27-34)
    def test_get_env_int_missing_env_var(self):
        """Test _get_env_int when environment variable is not set."""
        cache = CacheManager(cache_dir=self.test_dir)
        with patch.dict(os.environ, {}, clear=True):
            result = cache._get_env_int("NONEXISTENT_VAR")
            self.assertIsNone(result)

    def test_get_env_int_valid_integer(self):
        """Test _get_env_int with valid integer."""
        cache = CacheManager(cache_dir=self.test_dir)
        with patch.dict(os.environ, {"TEST_TTL": "7"}):
            result = cache._get_env_int("TEST_TTL")
            self.assertEqual(result, 7)

    def test_get_env_int_invalid_value(self):
        """Test _get_env_int with non-integer value (lines 31-34)."""
        cache = CacheManager(cache_dir=self.test_dir)
        with patch.dict(os.environ, {"TEST_TTL": "invalid"}):
            result = cache._get_env_int("TEST_TTL")
            self.assertIsNone(result, "Should return None for invalid integer string")

    # Tests for _load_json (lines 36-43)
    def test_load_json_nonexistent_file(self):
        """Test _load_json when file doesn't exist."""
        cache = CacheManager(cache_dir=self.test_dir)
        nonexistent = Path(self.test_dir) / "nonexistent.json"
        result = cache._load_json(nonexistent)
        self.assertEqual(result, {})

    def test_load_json_invalid_json(self):
        """Test _load_json with invalid JSON content (lines 41-42)."""
        cache = CacheManager(cache_dir=self.test_dir)
        invalid_file = Path(self.test_dir) / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("{ invalid json }")
        result = cache._load_json(invalid_file)
        self.assertEqual(result, {}, "Should return empty dict for invalid JSON")

    def test_load_json_valid_json(self):
        """Test _load_json with valid JSON content."""
        cache = CacheManager(cache_dir=self.test_dir)
        valid_file = Path(self.test_dir) / "valid.json"
        test_data = {"key": "value"}
        with open(valid_file, "w") as f:
            json.dump(test_data, f)
        result = cache._load_json(valid_file)
        self.assertEqual(result, test_data)

    # Tests for _save_json
    def test_save_json_creates_parent_dirs(self):
        """Test _save_json creates parent directories."""
        cache = CacheManager(cache_dir=self.test_dir)
        nested_path = Path(self.test_dir) / "nested" / "dir" / "cache.json"
        cache._save_json(nested_path, {"test": "data"})
        self.assertTrue(nested_path.exists())

    # Tests for _normalize_llm_entry (lines 86-89)
    def test_normalize_llm_entry_dict(self):
        """Test _normalize_llm_entry with dict input."""
        cache = CacheManager(cache_dir=self.test_dir)
        entry = {"response": "test", "ts": 123}
        result = cache._normalize_llm_entry(entry)
        self.assertEqual(result, entry)

    def test_normalize_llm_entry_string(self):
        """Test _normalize_llm_entry with non-dict input (line 89)."""
        cache = CacheManager(cache_dir=self.test_dir)
        entry = "simple string response"
        result = cache._normalize_llm_entry(entry)
        expected = {"response": "simple string response", "ts": 0}
        self.assertEqual(result, expected)

    def test_normalize_llm_entry_other_type(self):
        """Test _normalize_llm_entry with other types."""
        cache = CacheManager(cache_dir=self.test_dir)
        entry = 12345
        result = cache._normalize_llm_entry(entry)
        expected = {"response": 12345, "ts": 0}
        self.assertEqual(result, expected)

    # Tests for _is_expired (lines 91-98)
    def test_is_expired_no_ttl_set(self):
        """Test _is_expired when TTL is not configured."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = None
        entry = {"ts": time.time() - 1000000}
        result = cache._is_expired(entry)
        self.assertFalse(result, "Should not expire when TTL is None")

    def test_is_expired_no_timestamp(self):
        """Test _is_expired when entry has no timestamp (lines 94-95)."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = 1
        entry = {"response": "test"}
        result = cache._is_expired(entry)
        self.assertFalse(result, "Should not expire when timestamp is missing")

    def test_is_expired_invalid_timestamp(self):
        """Test _is_expired with invalid timestamp type (line 95)."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = 1
        entry = {"ts": "invalid"}
        result = cache._is_expired(entry)
        self.assertFalse(result, "Should not expire when timestamp is not a number")

    def test_is_expired_zero_timestamp(self):
        """Test _is_expired with zero timestamp (line 95)."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = 1
        entry = {"ts": 0}
        result = cache._is_expired(entry)
        self.assertFalse(result, "Should not expire when timestamp is 0")

    def test_is_expired_negative_timestamp(self):
        """Test _is_expired with negative timestamp."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = 1
        entry = {"ts": -1}
        result = cache._is_expired(entry)
        self.assertFalse(result, "Should not expire when timestamp is negative")

    def test_is_expired_fresh_entry(self):
        """Test _is_expired with fresh entry."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = 1
        entry = {"ts": time.time()}
        result = cache._is_expired(entry)
        self.assertFalse(result, "Should not expire fresh entry")

    def test_is_expired_old_entry(self):
        """Test _is_expired with old entry (lines 97-98)."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = 1
        # Create entry older than 1 day
        old_time = time.time() - (2 * 24 * 60 * 60)
        entry = {"ts": old_time}
        result = cache._is_expired(entry)
        self.assertTrue(result, "Should expire entry older than TTL")

    # Tests for _prune_llm_cache TTL pruning (lines 100-112)
    def test_prune_llm_cache_ttl_removes_expired(self):
        """Test TTL pruning removes expired entries (lines 106-112)."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = 1

        # Add fresh and expired entries
        fresh_key = "fresh"
        expired_key = "expired"
        cache.llm_cache[fresh_key] = {"ts": time.time()}
        cache.llm_cache[expired_key] = {"ts": time.time() - (2 * 24 * 60 * 60)}

        cache._prune_llm_cache()

        self.assertIn(fresh_key, cache.llm_cache)
        self.assertNotIn(expired_key, cache.llm_cache)

    def test_prune_llm_cache_ttl_with_string_entries(self):
        """Test TTL pruning with string entries (line 108)."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = 1

        # Add entries as strings (old format)
        cache.llm_cache["key1"] = "response1"
        cache.llm_cache["key2"] = "response2"

        cache._prune_llm_cache()

        # String entries have ts=0, so they should not expire
        self.assertIn("key1", cache.llm_cache)
        self.assertIn("key2", cache.llm_cache)

    # Tests for _prune_llm_cache size pruning (lines 114-126)
    def test_prune_llm_cache_size_removes_oldest(self):
        """Test size pruning removes oldest entries (lines 119-126)."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_max_entries = 3

        # Add entries with different timestamps
        now = time.time()
        cache.llm_cache["newest"] = {"ts": now}
        cache.llm_cache["middle"] = {"ts": now - 100}
        cache.llm_cache["oldest"] = {"ts": now - 200}
        cache.llm_cache["extra"] = {"ts": now - 50}

        cache._prune_llm_cache()

        # Should keep the 3 newest entries
        self.assertEqual(len(cache.llm_cache), 3)
        self.assertIn("newest", cache.llm_cache)
        self.assertIn("middle", cache.llm_cache)
        self.assertIn("extra", cache.llm_cache)
        self.assertNotIn("oldest", cache.llm_cache)

    def test_prune_llm_cache_size_no_prune_when_under_limit(self):
        """Test size pruning when under max entries."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_max_entries = 10

        # Add fewer entries than max
        cache.llm_cache["key1"] = {"ts": time.time()}
        cache.llm_cache["key2"] = {"ts": time.time()}

        cache._prune_llm_cache()

        self.assertEqual(len(cache.llm_cache), 2)

    def test_prune_llm_cache_no_max_entries(self):
        """Test size pruning when max_entries is None."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_max_entries = None

        # Add many entries
        for i in range(100):
            cache.llm_cache[f"key{i}"] = {"ts": time.time()}

        cache._prune_llm_cache()

        self.assertEqual(len(cache.llm_cache), 100)

    # Tests for get_cached_llm_response with expiration (lines 159-170)
    def test_get_cached_llm_response_expired(self):
        """Test get_cached_llm_response returns None for expired entry (lines 165, 168-169)."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = 1

        text = "test input"
        response = "cached response"
        cache.cache_llm_response(text, response)

        # Manually expire the entry
        chunk_hash = cache._cache_key(text, None)
        cache.llm_cache[chunk_hash]["ts"] = time.time() - (2 * 24 * 60 * 60)

        result = cache.get_cached_llm_response(text)
        self.assertIsNone(result)
        # Expired entry should be removed from cache
        self.assertNotIn(chunk_hash, cache.llm_cache)

    def test_get_cached_llm_response_not_expired(self):
        """Test get_cached_llm_response returns cached response for valid entry."""
        cache = CacheManager(cache_dir=self.test_dir)
        cache.llm_cache_ttl_days = 1

        text = "test input"
        response = "cached response"
        cache.cache_llm_response(text, response)

        result = cache.get_cached_llm_response(text)
        self.assertEqual(result, response)

    # Tests for get_file_state and update_file_state
    def test_update_file_state_partial_update(self):
        """Test update_file_state with partial parameters."""
        cache = CacheManager(cache_dir=self.test_dir)
        file_path = "/path/to/file.hwp"

        cache.update_file_state(file_path, hwp_hash="hash1")
        state = cache.get_file_state(file_path)
        self.assertEqual(state, {"hwp_hash": "hash1"})

        cache.update_file_state(file_path, raw_md_hash="hash2")
        state = cache.get_file_state(file_path)
        self.assertEqual(state, {"hwp_hash": "hash1", "raw_md_hash": "hash2"})

    def test_update_file_state_none_values_ignored(self):
        """Test update_file_state ignores None values."""
        cache = CacheManager(cache_dir=self.test_dir)
        file_path = "/path/to/file.hwp"

        cache.update_file_state(file_path, hwp_hash="hash1", raw_md_hash=None)
        state = cache.get_file_state(file_path)
        self.assertEqual(state, {"hwp_hash": "hash1"})

    # Tests for compute_file_hash
    def test_compute_file_hash_consistent(self):
        """Test compute_file_hash returns consistent hash."""
        cache = CacheManager(cache_dir=self.test_dir)
        test_file = Path(self.test_dir) / "test.txt"
        content = b"test content for hashing"
        test_file.write_bytes(content)

        hash1 = cache.compute_file_hash(test_file)
        hash2 = cache.compute_file_hash(test_file)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA256 hex length

    # Tests for compute_text_hash
    def test_compute_text_hash_consistent(self):
        """Test compute_text_hash returns consistent hash."""
        cache = CacheManager(cache_dir=self.test_dir)
        text = "test content for hashing"

        hash1 = cache.compute_text_hash(text)
        hash2 = cache.compute_text_hash(text)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA256 hex length

    def test_compute_text_hash_whitespace_sensitive(self):
        """Test compute_text_hash handles whitespace."""
        cache = CacheManager(cache_dir=self.test_dir)
        text1 = "test"
        text2 = "test "

        hash1 = cache.compute_text_hash(text1)
        hash2 = cache.compute_text_hash(text2)
        self.assertNotEqual(hash1, hash2)

    # Tests for cache key with namespace
    def test_cache_key_with_namespace(self):
        """Test _cache_key includes namespace."""
        cache = CacheManager(cache_dir=self.test_dir)

        key1 = cache._cache_key("text", None)
        key2 = cache._cache_key("text", "namespace")
        key3 = cache._cache_key("text", None)

        self.assertEqual(key1, key3)
        self.assertNotEqual(key1, key2)

    def test_cache_key_whitespace_normalized(self):
        """Test _cache_key normalizes whitespace."""
        cache = CacheManager(cache_dir=self.test_dir)

        key1 = cache._cache_key("  text  ", None)
        key2 = cache._cache_key("text", None)

        self.assertEqual(key1, key2)

    # Tests for cache_llm_response with namespace
    def test_cache_llm_response_with_namespace(self):
        """Test cache_llm_response with namespace."""
        cache = CacheManager(cache_dir=self.test_dir)
        text = "test input"
        ns1 = "namespace1"
        ns2 = "namespace2"

        cache.cache_llm_response(text, "response1", ns1)
        cache.cache_llm_response(text, "response2", ns2)

        self.assertEqual(cache.get_cached_llm_response(text, ns1), "response1")
        self.assertEqual(cache.get_cached_llm_response(text, ns2), "response2")


if __name__ == "__main__":
    unittest.main()
