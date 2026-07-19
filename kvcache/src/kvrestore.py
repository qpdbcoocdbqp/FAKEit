import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import KVCacheStorage

# ============================================================
# Llama-server KV Cache Verification Test
# ============================================================

def send_chat_request(base_url: str, model_name: str, messages: list) -> tuple:
    """
    Sends a chat completion request and returns (response_json, duration_seconds)
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 1,  # Set to 1 to isolate prefill (prompt evaluation) time from generation time
        "temperature": 0.0
    }
    
    url = f"{base_url}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as res:
            duration = time.time() - start_time
            resp_data = json.loads(res.read().decode('utf-8'))
            return resp_data, duration
    except urllib.error.HTTPError as e:
        print(f"    [-] Request failed: {e.code} {e.reason} - {e.read().decode('utf-8')}")
        raise e

def unload_model(base_url: str, model_name: str):
    """
    Sends a POST request to unload the model from llama-server to purge cache from memory.
    """
    url = f"{base_url}/api/models/unload/{model_name}"
    
    print(f"[*] Unloading model '{model_name}' to purge memory cache...")
    req = urllib.request.Request(
        url,
        data=b"",  # Empty body to specify POST request
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as res:
            res_content = res.read().decode('utf-8')
            try:
                resp = json.loads(res_content) if res_content.strip() else {}
            except json.JSONDecodeError:
                resp = res_content
            print(f"    [+] Unloaded successfully via: {url} -> {resp}")
            return True
    except Exception as e:
        print(f"    [-] Unload failed: {e}")
        return False

def ensure_model_loaded(base_url: str, model_name: str):
    """
    Sends a tiny request to force llama-server to reload the model if it was unloaded,
    ensuring model loading time is not included in subsequent test metrics.
    """
    print(f"[*] Ensuring model '{model_name}' is loaded (pre-heating)...")
    messages = [{"role": "user", "content": "."}]
    try:
        # This will block until the model is fully loaded and processed
        send_chat_request(base_url, model_name, messages)
        print("    [+] Model is loaded and ready.")
    except Exception as e:
        print(f"    [-] Pre-heating failed: {e}")

def test_kv_cache_efficacy(
    base_url: str = "http://localhost:19001",
    model_name: str = "luna",
    slot_id: int = 0,
    slots_dir_on_server: str = "./tmp/luna"
):
    print(f"[*] Target LLM Server: {base_url}")
    print(f"[*] Target Model Name: {model_name}")
    print(f"[*] Slots Directory: {slots_dir_on_server}")
    
    # Ensure model is initially loaded
    ensure_model_loaded(base_url, model_name)
    
    # 1. Prepare a very long prompt to make the prefill time noticeable
    long_prompt = (
        "You are a helpful and extremely detail-oriented coding assistant. "
        "Your goal is to provide clean, robust, and well-documented code examples. "
        "When writing Python code, make sure to use type hints, write docstrings, "
        "and handle potential exceptions. Always follow clean code principles. "
    ) * 40  # Generating ~500+ tokens
    
    messages = [
        {"role": "system", "content": long_prompt},
        {"role": "user", "content": "Write a hello world function."}
    ]
    
    tokenized_prompt = long_prompt.split() # For hashing
    
    print("\n[*] --- Step 1: Warmup & Save KV Cache ---")
    # Send request to populate the KV cache in the slot
    print("[*] Sending initial request to warm up KV cache...")
    _, warm_duration = send_chat_request(base_url, model_name, messages)
    print(f"    [+] Initial request (warmup) took: {warm_duration:.4f} seconds")
    
    # Save the slot state to disk
    filename = "verify_slot.bin"
    save_url = f"{base_url}/upstream/{model_name}/slots/{slot_id}?action=save"
    save_payload = {"filename": filename}
    save_req = urllib.request.Request(
        save_url,
        data=json.dumps(save_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    with urllib.request.urlopen(save_req) as res:
        print(f"    [+] Saved slot state to file: {json.loads(res.read().decode('utf-8'))}")
        
    # Compress and save via KVCacheStorage
    local_saved_file = Path(slots_dir_on_server) / filename
    storage = KVCacheStorage(db_path="kvcache.db", storage_dir="./kv_storage")
    
    save_result = storage.save_cache(
        model_name=model_name,
        messages=messages,
        kv_binary_path=str(local_saved_file),
        tokenized_prompt=tokenized_prompt
    )
    print(f"    [+] Compressed cache successfully stored in SQLite & disk.")

    print("\n[*] --- Step 2: Cold Run (Unloaded Model / Pure Cold) ---")
    # Unload the model to completely drop memory/KV cache
    unload_model(base_url, model_name)
    
    # Reload model BEFORE starting the timer to exclude model load time
    ensure_model_loaded(base_url, model_name)
    
    # Now, request the original long prompt. This is a COLD run because the model was unloaded.
    print("[*] Sending original prompt (COLD run - should recalculate)...")
    _, cold_duration = send_chat_request(base_url, model_name, messages)
    print(f"    [+] Cold run took: {cold_duration:.4f} seconds")

    print("\n[*] --- Step 3: Hot Run (Restored Cache) ---")
    # Find match and restore the file to the slot directory
    match = storage.find_best_match(model_name=model_name, tokenized_prompt=tokenized_prompt)
    if not match:
        print("    [-] Error: Match not found in database!")
        return
        
    restored_filename = "verify_slot_restored.bin"
    local_restored_file = Path(slots_dir_on_server) / restored_filename
    storage.restore_cache(match["kv_path"], str(local_restored_file))
    
    # Unload model again to make sure the in-memory cache is empty before restoring from file
    unload_model(base_url, model_name)
    
    # Trigger restore action via API (this will auto-load the model if unloaded, without polluting the slot)
    restore_url = f"{base_url}/upstream/{model_name}/slots/{slot_id}?action=restore"
    restore_payload = {"filename": restored_filename}
    restore_req = urllib.request.Request(
        restore_url,
        data=json.dumps(restore_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(restore_req) as res:
        print(f"    [+] Slot restored response: {json.loads(res.read().decode('utf-8'))}")
        
    # Send the original long prompt again. This is a HOT run because we restored the KV cache.
    print("[*] Sending original prompt (HOT run - should use restored cache)...")
    _, hot_duration = send_chat_request(base_url, model_name, messages)
    print(f"    [+] Hot run took: {hot_duration:.4f} seconds")
    
    # Analyze and assert performance difference
    speedup = cold_duration / hot_duration if hot_duration > 0 else 0
    print("\n[*] --- Performance Analysis ---")
    print(f"    [=] Cold Run Duration: {cold_duration:.4f}s")
    print(f"    [=] Hot Run Duration:  {hot_duration:.4f}s")
    print(f"    [=] Speedup Factor:    {speedup:.2f}x")
    
    if cold_duration - hot_duration > 0.1 and speedup > 1.5:
        print("\n    [SUCCESS] KV Cache Restore is working and significantly improved performance!")
    else:
        print("\n    [WARNING] Performance difference was not significant. Verify your model's server configuration.")

if __name__ == "__main__":
    test_kv_cache_efficacy(
        base_url="http://localhost:19001",
        model_name="sonnet",
        slot_id=0,
        slots_dir_on_server="./tmp/sonnet"
    )
