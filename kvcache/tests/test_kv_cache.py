import os
import sys
import unittest
import tempfile
import shutil
import time
from pathlib import Path

# Add root directory to sys.path to ensure src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import (
    canonicalize_messages,
    sha256_text,
    build_prefix_hashes,
    compress_file,
    decompress_file,
    KVCacheStorage
)


class TestKVCacheHelpers(unittest.TestCase):

    def test_canonicalize_messages(self):
        messages = [
            {"role": "  system  ", "content": "You are  a   coding assistant. "},
            {"role": "user", "content": "Write a\nPython HTTP server."}
        ]
        # Spaces should be stripped and normalized
        expected = '[{"content":"You are a coding assistant.","role":"system"},{"content":"Write a Python HTTP server.","role":"user"}]'
        self.assertEqual(canonicalize_messages(messages), expected)

    def test_sha256_text(self):
        text = "hello"
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        self.assertEqual(sha256_text(text), expected)

    def test_build_prefix_hashes(self):
        tokens = [str(i) for i in range(600)]
        prefixes = build_prefix_hashes(tokens, step=256)
        
        # 600 tokens should yield prefixes for:
        # i = 256
        # i = 512
        self.assertEqual(len(prefixes), 2)
        self.assertEqual(prefixes[0]["tokens"], 256)
        self.assertEqual(prefixes[1]["tokens"], 512)
        
        # Check shorter list
        self.assertEqual(len(build_prefix_hashes(tokens[:200], step=256)), 0)


class TestKVCacheStorage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_kvcache.db")
        self.storage_dir = os.path.join(self.temp_dir.name, "test_kv_storage")
        self.storage = KVCacheStorage(db_path=self.db_path, storage_dir=self.storage_dir)

        # Create a dummy bin file representing a KV cache file
        self.dummy_bin = os.path.join(self.temp_dir.name, "dummy_slot.bin")
        with open(self.dummy_bin, "wb") as f:
            f.write(b"dummy binary data for kv cache" * 100)

    def tearDown(self):
        # Close connection to release sqlite file handle
        if hasattr(self.storage, "conn") and self.storage.conn:
            self.storage.conn.close()
        self.temp_dir.cleanup()

    def test_initialization(self):
        self.assertTrue(os.path.exists(self.db_path))
        self.assertTrue(os.path.exists(self.storage_dir))

    def test_save_cache_empty_prefix(self):
        messages = [{"role": "user", "content": "hello"}]
        # Tokenized prompt is too short to generate prefixes (length < 256)
        tokenized_prompt = ["hello"]
        result = self.storage.save_cache(
            model_name="test-model",
            messages=messages,
            kv_binary_path=self.dummy_bin,
            tokenized_prompt=tokenized_prompt
        )
        self.assertIn("prompt_hash", result)
        self.assertEqual(result["entries"], [])

    def test_save_cache_and_match(self):
        messages = [{"role": "user", "content": "long query"}]
        # Let's create more than 512 tokens to hit 2 prefixes
        tokenized_prompt = ["token"] * 550

        # Save cache
        result = self.storage.save_cache(
            model_name="test-model",
            messages=messages,
            kv_binary_path=self.dummy_bin,
            tokenized_prompt=tokenized_prompt
        )
        self.assertEqual(len(result["entries"]), 2)
        
        # Verify files are stored
        for entry in result["entries"]:
            self.assertTrue(os.path.exists(entry["path"]))

        # Find match with exactly matching prompt tokens
        match = self.storage.find_best_match(
            model_name="test-model",
            tokenized_prompt=tokenized_prompt
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["prefix_tokens"], 512)  # Should match longest prefix
        self.assertEqual(match["hit_count"], 0)  # hit count before increment was 0

        # Run find_best_match again and check hit_count increment
        match_again = self.storage.find_best_match(
            model_name="test-model",
            tokenized_prompt=tokenized_prompt
        )
        self.assertEqual(match_again["hit_count"], 1)

        # Test restoring cache
        output_restore = os.path.join(self.temp_dir.name, "restored.bin")
        self.storage.restore_cache(match_again["kv_path"], output_restore)
        self.assertTrue(os.path.exists(output_restore))
        with open(output_restore, "rb") as f:
            restored_content = f.read()
        with open(self.dummy_bin, "rb") as f:
            original_content = f.read()
        self.assertEqual(restored_content, original_content)

    def test_cleanup_old_cache(self):
        messages = [{"role": "user", "content": "cleanup test"}]
        tokenized_prompt = ["token"] * 300

        result = self.storage.save_cache(
            model_name="test-model",
            messages=messages,
            kv_binary_path=self.dummy_bin,
            tokenized_prompt=tokenized_prompt
        )
        self.assertEqual(len(result["entries"]), 1)
        cache_path = result["entries"][0]["path"]
        self.assertTrue(os.path.exists(cache_path))

        # Test cleanup with max_age_seconds=-1 (everything is expired)
        self.storage.cleanup_old_cache(max_age_seconds=-1)
        self.assertFalse(os.path.exists(cache_path))

        # Database should be empty
        stats = self.storage.stats()
        self.assertEqual(stats["entries"], 0)

    def test_stats(self):
        messages = [{"role": "user", "content": "stats test"}]
        tokenized_prompt = ["token"] * 300

        self.storage.save_cache(
            model_name="test-model",
            messages=messages,
            kv_binary_path=self.dummy_bin,
            tokenized_prompt=tokenized_prompt
        )
        stats_before = self.storage.stats()
        self.assertEqual(stats_before["entries"], 1)
        self.assertEqual(stats_before["total_hits"], 0)

        # Match once to increase hit count
        self.storage.find_best_match(
            model_name="test-model",
            tokenized_prompt=tokenized_prompt
        )

        stats_after = self.storage.stats()
        self.assertEqual(stats_after["total_hits"], 1)


if __name__ == "__main__":
    unittest.main()
