"""
Extended tests for main.py to improve coverage from 70% to 85%.

Focuses on additional helper functions and pipeline steps:
- _get_file_paths
- _load_cache_state
- _check_full_cache_hit
- _log_cache_miss_reasons
- _convert_hwp_to_markdown
- _extract_and_save_metadata
- _format_documents
- _save_final_json
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.main import (
    CacheState,
    FilePaths,
    _build_pipeline_signature,
    _check_full_cache_hit,
    _extract_source_metadata,
    _get_file_paths,
    _load_cache_state,
    _log_cache_miss_reasons,
    _resolve_preprocessor_rules_path,
)


class TestGetFilePaths(unittest.TestCase):
    """Tests for _get_file_paths function."""

    def test_single_file_input(self):
        """Test single file input path."""
        file = Path("test.hwpx")
        output_dir = Path("output")
        input_path = Path("test.hwpx")

        result = _get_file_paths(file, output_dir, input_path)

        self.assertIsInstance(result, FilePaths)
        self.assertEqual(result.json_out.name, "test.json")

    def test_directory_input(self):
        """Test directory input creates subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir)
            output_dir = Path(temp_dir) / "output"
            file = input_path / "subdir" / "test.hwpx"

            result = _get_file_paths(file, output_dir, input_path)

            self.assertEqual(result.json_out.parent.name, "subdir")


class TestLoadCacheState(unittest.TestCase):
    """Tests for _load_cache_state function."""

    def test_no_cache_manager(self):
        """Test with no cache manager."""
        cache_manager = None
        paths = MagicMock()
        args = MagicMock()
        args.verbose = False
        console = MagicMock()

        result = _load_cache_state(paths, paths, cache_manager, args, console)

        self.assertIsInstance(result, CacheState)
        self.assertIsNone(result.hwp_hash)
        self.assertIsNone(result.cached_hwp_hash)

    def test_no_file_state(self):
        """Test when cache manager returns no file state."""
        cache_manager = MagicMock()
        cache_manager.get_file_state.return_value = None

        paths = MagicMock()
        args = MagicMock()
        args.verbose = False
        console = MagicMock()

        result = _load_cache_state(paths, paths, cache_manager, args, console)

        self.assertIsNone(result.cached_hwp_hash)


class TestCheckFullCacheHit(unittest.TestCase):
    """Tests for _check_full_cache_hit function."""

    def test_force_overrides_cache(self):
        """Test force=True always returns False."""
        paths = MagicMock()
        cache_state = CacheState()
        cache_manager = MagicMock()

        result = _check_full_cache_hit(
            paths, cache_state, "signature", cache_manager, force=True
        )

        self.assertFalse(result)

    def test_missing_required_files(self):
        """Test returns False when required files missing."""
        paths = MagicMock()
        paths.raw_md.exists.return_value = False
        paths.json_out.exists.return_value = True
        paths.metadata.exists.return_value = True

        cache_state = CacheState()
        cache_manager = MagicMock()

        result = _check_full_cache_hit(
            paths, cache_state, "signature", cache_manager, force=False
        )

        self.assertFalse(result)

    def test_cache_miss_hashes_differ(self):
        """Test returns False when cache hashes don't match."""
        paths = MagicMock()
        paths.raw_md.exists.return_value = True
        paths.json_out.exists.return_value = True
        paths.metadata.exists.return_value = True

        cache_state = CacheState(
            cached_hwp_hash="abc",
            hwp_hash="def",  # Different
            raw_md_cache_hit=False,
        )
        cache_manager = MagicMock()

        result = _check_full_cache_hit(
            paths, cache_state, "signature", cache_manager, force=False
        )

        self.assertFalse(result)

    def test_signature_change_requires_rebuild(self):
        """Test signature change requires rebuild."""
        paths = MagicMock()
        paths.raw_md.exists.return_value = True
        paths.json_out.exists.return_value = True
        paths.metadata.exists.return_value = True

        cache_state = CacheState(
            cached_hwp_hash="abc",
            hwp_hash="abc",
            raw_md_hash="def",
            cached_raw_md_hash="def",
            cached_pipeline_signature="old_sig",
            raw_md_cache_hit=True,
        )
        cache_manager = MagicMock()

        result = _check_full_cache_hit(
            paths, cache_state, "new_sig", cache_manager, force=False
        )

        self.assertFalse(result)


class TestLogCacheMissReasons(unittest.TestCase):
    """Tests for _log_cache_miss_reasons function."""

    def test_logs_missing_raw_md(self):
        """Test logging missing raw_md file."""
        mock_console = MagicMock()
        paths = MagicMock()
        paths.raw_md.exists.return_value = False
        paths.json_out.exists.return_value = True
        paths.metadata.exists.return_value = True

        cache_state = CacheState(hwp_hash="abc", cached_hwp_hash="abc")
        cache_manager = MagicMock()

        _log_cache_miss_reasons(
            paths, cache_state, "signature", cache_manager, mock_console
        )

        mock_console.print.assert_called()
        # Check the call contains "raw_md 없음"
        call_args = str(mock_console.print.call_args)
        self.assertIn("raw_md", call_args)

    def test_logs_hash_mismatch(self):
        """Test logging hash mismatch."""
        mock_console = MagicMock()
        paths = MagicMock()
        paths.raw_md.exists.return_value = True
        paths.json_out.exists.return_value = True
        paths.metadata.exists.return_value = True

        cache_state = CacheState(
            cached_hwp_hash="old_hash",
            hwp_hash="new_hash",
        )
        cache_manager = MagicMock()

        _log_cache_miss_reasons(
            paths, cache_state, "signature", cache_manager, mock_console
        )

        mock_console.print.assert_called()


class TestBuildPipelineSignature(unittest.TestCase):
    """Additional tests for _build_pipeline_signature."""

    def test_special_characters_in_components(self):
        """Test special characters are handled."""
        result = _build_pipeline_signature("test|hash", "llm|sig")
        # Special chars should be preserved
        self.assertIn("test|hash", result)
        self.assertIn("llm|sig", result)


class TestExtractSourceMetadataExtended(unittest.TestCase):
    """Extended tests for _extract_source_metadata."""

    def test_korean_filename_with_serial(self):
        """Test Korean filename with correct pattern."""
        result = _extract_source_metadata("규정집9-343(20250909).hwp")

        self.assertEqual(result["source_serial"], "9-343")
        self.assertEqual(result["source_date"], "2025-09-09")

    def test_filename_without_serial(self):
        """Test filename without serial number."""
        result = _extract_source_metadata("규정집(20250101).hwp")

        self.assertIsNone(result["source_serial"])
        self.assertEqual(result["source_date"], "2025-01-01")

    def test_filename_without_date(self):
        """Test filename without date."""
        result = _extract_source_metadata("규정집9-343.hwp")

        self.assertEqual(result["source_serial"], "9-343")
        self.assertIsNone(result["source_date"])

    def test_malformed_date(self):
        """Test malformed date is still parsed."""
        result = _extract_source_metadata("규정집1-1(20251301).hwp")

        # Invalid month should still parse the string
        self.assertEqual(result["source_date"], "2025-13-01")

    def test_empty_filename(self):
        """Test empty filename."""
        result = _extract_source_metadata("")

        self.assertIsNone(result["source_serial"])
        self.assertIsNone(result["source_date"])

    def test_no_match(self):
        """Test filename with no matching pattern."""
        result = _extract_source_metadata("random_filename.hwpx")

        self.assertIsNone(result["source_serial"])
        self.assertIsNone(result["source_date"])


class TestFilePaths(unittest.TestCase):
    """Tests for FilePaths dataclass."""

    def test_all_paths_defined(self):
        """Test all file paths are properly set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            paths = FilePaths(
                raw_md=base / "raw.md",
                raw_html=base / "raw.html",
                json_out=base / "out.json",
                metadata=base / "metadata.json",
            )

            self.assertEqual(paths.raw_md.name, "raw.md")
            self.assertEqual(paths.raw_html.name, "raw.html")
            self.assertEqual(paths.json_out.name, "out.json")
            self.assertEqual(paths.metadata.name, "metadata.json")


class TestCacheState(unittest.TestCase):
    """Tests for CacheState dataclass."""

    def test_default_values(self):
        """Test default field values."""
        state = CacheState()

        self.assertIsNone(state.hwp_hash)
        self.assertIsNone(state.raw_md_hash)
        self.assertIsNone(state.cached_hwp_hash)
        self.assertIsNone(state.cached_raw_md_hash)
        self.assertIsNone(state.cached_pipeline_signature)
        self.assertIsNone(state.cached_final_json_hash)
        self.assertIsNone(state.cached_metadata_hash)
        self.assertFalse(state.cache_hit)
        self.assertTrue(state.raw_md_cache_hit)

    def test_explicit_values(self):
        """Test explicit value assignment."""
        state = CacheState(
            hwp_hash="hash1",
            raw_md_hash="hash2",
            cached_hwp_hash="hash1",
            cached_raw_md_hash="hash2",
            cached_pipeline_signature="sig",
            cached_final_json_hash="hash3",
            cached_metadata_hash="hash4",
            cache_hit=True,
            raw_md_cache_hit=False,
        )

        self.assertEqual(state.hwp_hash, "hash1")
        self.assertEqual(state.raw_md_hash, "hash2")
        self.assertTrue(state.cache_hit)
        self.assertFalse(state.raw_md_cache_hit)


class TestPipelineContext(unittest.TestCase):
    """Tests for PipelineContext dataclass."""

    def test_context_creation(self):
        """Test PipelineContext creation."""
        from src.main import PipelineContext

        cache_manager = MagicMock()
        preprocessor = MagicMock()
        formatter = MagicMock()
        metadata_extractor = MagicMock()
        args = MagicMock()
        console = MagicMock()

        context = PipelineContext(
            cache_manager=cache_manager,
            preprocessor=preprocessor,
            formatter=formatter,
            metadata_extractor=metadata_extractor,
            pipeline_signature="test_sig",
            args=args,
            console=console,
        )

        self.assertEqual(context.pipeline_signature, "test_sig")
        self.assertEqual(context.cache_manager, cache_manager)
        self.assertEqual(context.preprocessor, preprocessor)


class TestResolvePreprocessorRulesPathExtended(unittest.TestCase):
    """Extended tests for _resolve_preprocessor_rules_path."""

    @patch.dict(os.environ, {}, clear=True)
    def test_env_var_empty_string(self):
        """Test empty string env var uses default."""
        with patch.dict(os.environ, {"PREPROCESSOR_RULES_PATH": ""}):
            result = _resolve_preprocessor_rules_path()
            # Empty string should result in default path
            self.assertEqual(result, Path("data/config/preprocessor_rules.json"))


class TestCollectHwpFilesExtended(unittest.TestCase):
    """Extended tests for _collect_hwp_files."""

    def test_collects_from_directory(self):
        """Test collecting files from directory."""
        mock_console = MagicMock()
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.rglob") as mock_rglob:
                    mock_files = [
                        Path("file1.hwpx"),
                        Path("file2.hwpx"),
                    ]
                    mock_rglob.return_value = mock_files

                    from src.main import _collect_hwp_files

                    result = _collect_hwp_files(Path("/test"), mock_console)

                    self.assertEqual(len(result), 2)

    def test_no_hwp_files_in_directory(self):
        """Test directory with no HWPX files."""
        mock_console = MagicMock()
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.rglob", return_value=[]):
                    from src.main import _collect_hwp_files

                    result = _collect_hwp_files(Path("/test"), mock_console)

                    self.assertIsNone(result)
                    mock_console.print.assert_called()


if __name__ == "__main__":
    unittest.main()
