"""
generate_kvcache_test_traffic.py

Generates real KV-cache STORE/RETRIEVE traffic against your vllm-serve
(LMCacheMPConnector-backed) endpoint, so you have something concrete to
look for in:
  - lmcache-server container logs        (docker logs -f lmcache-server)
  - lmcache-server /status endpoint       (http://localhost:18000/status)
  - Prometheus (http://localhost:9091)    -> lmcache_mp.* metrics
  - Tempo traces via Grafana              (http://localhost:3000)

Strategy: build one long shared prefix from a small HF dataset (long
enough to span multiple 256-token chunks, since that's your configured
chunk_size), then send it as a prompt multiple times.
  - 1st call  -> prefix is new -> vLLM computes it -> LMCache STORE
  - 2nd/3rd call -> same prefix -> LMCache should RETRIEVE (cache hit)
    -> noticeably lower time-to-first-token on the repeat calls

Usage:
    pip install datasets requests
    python3 generate_kvcache_test_traffic.py
"""

import time

import requests
from datasets import load_dataset

VLLM_URL = "http://localhost:18001/v1/completions"
MODEL = "Qwen/Qwen3-0.6B"

# Roughly this many characters -> comfortably over 256+ tokens for most
# tokenizers (~4 chars/token rule of thumb), enough to span multiple
# LMCache chunks.
PREFIX_CHAR_TARGET = 4000


def build_shared_prefix() -> str:
    """Pull a small slice of wikitext and concatenate into one long block."""
    print("[+] Loading a small slice of wikitext-2-raw-v1 from Hugging Face...")
    # ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:200]")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:200]")


    chunks = []
    total_len = 0
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        chunks.append(text)
        total_len += len(text)
        if total_len >= PREFIX_CHAR_TARGET:
            break

    prefix = " ".join(chunks)[:PREFIX_CHAR_TARGET]
    print(f"[+] Built shared prefix: {len(prefix)} chars (~{len(prefix)//4} tokens estimate)")
    return prefix


def send_completion(prompt: str, max_tokens: int = 24) -> float:
    """POST one completion request, return wall-clock latency in seconds."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    resp = requests.post(VLLM_URL, json=payload, timeout=120)
    dt = time.perf_counter() - t0
    resp.raise_for_status()
    text = resp.json()["choices"][0]["text"]
    return dt, text


def main():
    prefix = build_shared_prefix()

    questions = [
        "\n\nQ: Summarize the above in one sentence.\nA:",
        "\n\nQ: What is the main topic discussed above?\nA:",
        "\n\nQ: List one key fact from the text above.\nA:",
    ]

    print("\n[+] Sending requests that share the SAME long prefix "
          "(different short suffix each time so vLLM still generates "
          "something new, but the prefix should hit cache after run 1).\n")

    latencies = []
    for i, q in enumerate(questions):
        prompt = prefix + q
        dt, text = send_completion(prompt)
        latencies.append(dt)
        print(f"run {i} ({'first (expect STORE)' if i == 0 else 'repeat (expect RETRIEVE hit)'}): "
              f"{dt:.3f}s  -> {text.strip()[:80]!r}")

    print("\n[+] Latency summary:")
    for i, dt in enumerate(latencies):
        print(f"    run {i}: {dt:.3f}s")
    if len(latencies) > 1 and latencies[0] > 0:
        speedup = latencies[0] / latencies[1]
        print(f"\n[+] run0 -> run1 speedup: {speedup:.2f}x "
              f"(a repeat call noticeably faster than run 0 is a good "
              f"sign the shared prefix hit the KV cache)")

    print("\n[+] Now go check:")
    print("    docker logs --since 2m lmcache-server | grep -Ei 'stored|retrieved'")
    print("    curl -s http://localhost:18000/status | python3 -m json.tool")
    print("    http://localhost:9091  (Prometheus, search lmcache_mp)")
    print("    http://localhost:3000  (Grafana, look for new traces in Tempo datasource)")


if __name__ == "__main__":
    main()
