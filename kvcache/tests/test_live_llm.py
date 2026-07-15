import os
import sys
import json
import unittest
import tempfile
import urllib.request
import urllib.error
import subprocess
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import (
    canonicalize_messages,
    KVCacheStorage
)


class TestLiveLLMKVCache(unittest.TestCase):

    def setUp(self):
        self.base_url = "http://localhost:19001"
        self.model_name = "sonnet"
        
        # Verify the model is running and reachable
        try:
            req = urllib.request.urlopen(f"{self.base_url}/v1/models")
            models_data = json.loads(req.read().decode('utf-8'))
            model_ids = [m["id"] for m in models_data.get("data", [])]
            if self.model_name not in model_ids:
                self.skipTest(f"Model '{self.model_name}' is not available on the server. Available: {model_ids}")
        except Exception as e:
            self.skipTest(f"Failed to connect to local LLM server at {self.base_url}: {e}")

        # Setup temporary workspace for storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "live_kvcache.db")
        self.storage_dir = os.path.join(self.temp_dir.name, "live_kv_storage")
        self.storage = KVCacheStorage(db_path=self.db_path, storage_dir=self.storage_dir)

    def tearDown(self):
        if hasattr(self.storage, "conn") and self.storage.conn:
            self.storage.conn.close()
        self.temp_dir.cleanup()

        # Clean up files created inside the container slots directory
        for fn in ["live_test_slot.bin", "live_test_slot_restored.bin"]:
            try:
                subprocess.run(
                    ["docker", "exec", "llm", "rm", "-f", f"/app/slots/{fn}"],
                    capture_output=True, check=False
                )
            except Exception:
                pass

    def test_end_to_end_live_kv_cache(self):
        # 1. Trigger chat completion to populate KV cache on slot 0
        messages = [
            {"role": "system", "content": "You are a coding assistant. Help the user write code in Python."},
            {"role": "user", "content": "Write a hello world function."}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 10,
            "temperature": 0.0
        }
        
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            response = urllib.request.urlopen(req)
            result = json.loads(response.read().decode('utf-8'))
            print("Chat Completion Response:", result["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            self.fail(f"Chat completion failed: {e.code} {e.reason} - {e.read().decode('utf-8')}")

        # 2. Call the slots save API to save the slot 0 state to container file 'live_test_slot.bin'
        save_url = f"{self.base_url}/upstream/{self.model_name}/slots/0?action=save"
        save_payload = {"filename": "live_test_slot.bin"}
        save_req = urllib.request.Request(
            save_url,
            data=json.dumps(save_payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            save_res = urllib.request.urlopen(save_req)
            save_data = json.loads(save_res.read().decode('utf-8'))
            print("Save Slot Response:", save_data)
            self.assertEqual(save_data.get("filename"), "live_test_slot.bin")
        except urllib.error.HTTPError as e:
            self.fail(f"Save slot API failed: {e.code} {e.reason} - {e.read().decode('utf-8')}")

        # 3. Retrieve the saved slot file from the docker container to the host
        host_bin_path = os.path.join(self.temp_dir.name, "live_test_slot.bin")
        try:
            raw_bytes = subprocess.check_output([
                "docker", "exec", "llm", "cat", "/app/slots/live_test_slot.bin"
            ])
            self.assertGreater(len(raw_bytes), 0, "Saved slot file size should be greater than 0")
            with open(host_bin_path, "wb") as f:
                f.write(raw_bytes)
        except subprocess.CalledProcessError as e:
            self.fail(f"Failed to retrieve slot file from container: {e}")

        # 4. Save the cache using KVCacheStorage (this compresses the binary to zst and records it in sqlite)
        # We split the canonicalized messages into a token list to mock tokens
        tokenized_prompt = canonicalize_messages(messages).split()
        
        # We need enough tokens to trigger rolling prefix hash generation (step=256 by default).
        # Let's adjust step dynamically or mock build_prefix_hashes to test it with shorter prompt,
        # or pad the tokenized prompt so that it triggers.
        # Let's pad it to 300 tokens to generate at least one prefix.
        if len(tokenized_prompt) < 256:
            tokenized_prompt += ["padding"] * (260 - len(tokenized_prompt))

        save_result = self.storage.save_cache(
            model_name=self.model_name,
            messages=messages,
            kv_binary_path=host_bin_path,
            tokenized_prompt=tokenized_prompt
        )
        
        self.assertEqual(len(save_result["entries"]), 1)
        self.assertTrue(os.path.exists(save_result["entries"][0]["path"]))
        print("KVCacheStorage save_cache entries:", save_result["entries"])

        # 5. Verify find_best_match finds the cache
        match = self.storage.find_best_match(
            model_name=self.model_name,
            tokenized_prompt=tokenized_prompt
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["prefix_tokens"], 256)

        # 6. Restore the cache from KVCacheStorage to a temporary file on the host
        restored_host_path = os.path.join(self.temp_dir.name, "live_test_slot_restored.bin")
        self.storage.restore_cache(match["kv_path"], restored_host_path)
        self.assertTrue(os.path.exists(restored_host_path))

        # 7. Write the restored file back into the container as '/app/slots/live_test_slot_restored.bin'
        try:
            with open(restored_host_path, "rb") as f:
                restored_bytes = f.read()
            
            p = subprocess.Popen(
                ["docker", "exec", "-i", "llm", "sh", "-c", "cat > /app/slots/live_test_slot_restored.bin"],
                stdin=subprocess.PIPE
            )
            p.communicate(restored_bytes)
            self.assertEqual(p.returncode, 0, "Failed to write restored file back to container")
        except Exception as e:
            self.fail(f"Error copying restored file back to container: {e}")

        # 8. Restore the slot state in the local LLM using the restored filename
        restore_url = f"{self.base_url}/upstream/{self.model_name}/slots/0?action=restore"
        restore_payload = {"filename": "live_test_slot_restored.bin"}
        restore_req = urllib.request.Request(
            restore_url,
            data=json.dumps(restore_payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            restore_res = urllib.request.urlopen(restore_req)
            restore_data = json.loads(restore_res.read().decode('utf-8'))
            print("Restore Slot Response:", restore_data)
            self.assertEqual(restore_data.get("id_slot"), 0)
        except urllib.error.HTTPError as e:
            self.fail(f"Restore slot API failed: {e.code} {e.reason} - {e.read().decode('utf-8')}")


if __name__ == "__main__":
    unittest.main()
