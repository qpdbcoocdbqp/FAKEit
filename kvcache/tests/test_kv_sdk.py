"""
kv_sdk_example.py

Minimal example of LMCache's Python SDK (lmcache.sdk.kvcache), adapted to
this deployment's docker-compose setup:

    vLLM (OpenAI API)   -> http://localhost:18001   (served model: Qwen/Qwen3-0.6B)
    LMCache HTTP        -> http://localhost:18000
    LMCache ZMQ (mq)    -> tcp://localhost:6555

What it demonstrates (same idea as LMCache's own e2e_kv_edit.ipynb, trimmed
down):
    1. Send a prompt to vLLM -> normal LMCacheMPConnector path stores its KV.
    2. retrieve() that KV into an in-memory tensor via the SDK.
    3. store() the SAME KV tensor under a DIFFERENT token-id prefix.
    4. Send the new (target) token ids to vLLM -> it should hit the
       remapped KV instead of recomputing, and produce the same output
       for the shared suffix.

Requirements:
    pip install lmcache transformers httpx
    (must be run where `import lmcache` resolves to the SAME version as
    your lmcache-server container -- see the earlier mgmt-client script
    for why version skew breaks this.)
"""

import json
import time
from dataclasses import dataclass
from itertools import islice

import httpx
from transformers import AutoTokenizer

import lmcache.sdk.kvcache as lmc_sdk

# ── Config: matches this deployment's docker-compose.yml ──────────────
MODEL_NAME = "Qwen/Qwen3-0.6B"
VLLM_URL = "http://localhost:18001"
LMCACHE_HTTP_URL = "http://localhost:18000"
LMCACHE_MQ_URL = "tcp://localhost:6555"
CHUNK_SIZE = 256          # confirmed earlier via GET_CHUNK_SIZE -> 256
FAKE_PREFIX_TOKENS = 32   # must be < CHUNK_SIZE
MAX_TOKENS = 32
TIMEOUT = 60.0

SOURCE_PARAGRAPH = (
    "A systems researcher is studying how an inference cache changes the "
    "latency profile of a long language model prompt. The notes discuss "
    "attention keys, attention values, memory tiers, token chunks, and the "
    "careful measurement of cold and warm requests."
)


@dataclass(frozen=True)
class CompletionResult:
    text: str
    elapsed_seconds: float


def post_completion(prompt: list[int]) -> CompletionResult:
    """One non-streaming completion call against vLLM's OpenAI-compatible API."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "min_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    resp = httpx.post(f"{VLLM_URL}/v1/completions", json=payload, timeout=TIMEOUT)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    text = resp.json()["choices"][0]["text"]
    return CompletionResult(text=text, elapsed_seconds=elapsed)


def build_prompts(tokenizer, min_prompt_tokens: int):
    """Build equal-length source/target token-id lists that differ only in
    their first FAKE_PREFIX_TOKENS ids, sharing an identical suffix."""
    min_cache = max(1, min_prompt_tokens - FAKE_PREFIX_TOKENS)
    cache_tokens = ((min_cache + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE

    special = {int(t) for t in tokenizer.all_special_ids}
    candidates = (t for t in range(1000, int(tokenizer.vocab_size)) if t not in special)
    source_lead, target_lead = islice(candidates, 2)

    suffix_len = cache_tokens - FAKE_PREFIX_TOKENS
    text = SOURCE_PARAGRAPH
    suffix = tokenizer.encode(text, add_special_tokens=False)
    while len(suffix) < suffix_len:
        text = f"{text}\n\n{SOURCE_PARAGRAPH}"
        suffix = tokenizer.encode(text, add_special_tokens=False)
    suffix = [int(t) for t in suffix[-suffix_len:]]

    source = [source_lead] * FAKE_PREFIX_TOKENS + suffix
    target = [target_lead] * FAKE_PREFIX_TOKENS + suffix
    return source, target, cache_tokens


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    source_tokens, target_tokens, cache_tokens = build_prompts(
        tokenizer, min_prompt_tokens=CHUNK_SIZE * 2
    )

    print("== connect: open one SDK context, reuse it for retrieve/store ==")
    ctx = lmc_sdk.connect(
        url=LMCACHE_MQ_URL,
        http_url=LMCACHE_HTTP_URL,
        model_name=MODEL_NAME,
        timeout=TIMEOUT,
    )

    try:
        print("== step 1: source inference stores KV under source token IDs ==")
        source_completion = post_completion(source_tokens)

        print("== step 2: retrieve source KV into memory via the SDK ==")
        retrieved_kv = lmc_sdk.retrieve(ctx=ctx, tokens=source_tokens)
        if retrieved_kv is None:
            raise RuntimeError("retrieve() missed -- source KV wasn't cached yet")
        hit_tokens = int(retrieved_kv.shape[2])
        if hit_tokens < cache_tokens:
            raise RuntimeError(f"retrieved fewer tokens than expected: {hit_tokens} < {cache_tokens}")

        target_prefix = target_tokens[:hit_tokens]
        source_prefix = source_tokens[:hit_tokens]
        if source_prefix == target_prefix:
            raise RuntimeError("source/target prefixes unexpectedly identical")

        print("== step 3: store the SAME KV under the DIFFERENT target prefix ==")
        stored = lmc_sdk.store(ctx=ctx, kv=retrieved_kv, tokens=target_prefix)
        if not stored:
            raise RuntimeError("store() returned False -- target prefix already cached?")

        print("== step 4: target inference should reuse the remapped KV ==")
        target_completion = post_completion(target_tokens)

        result = {
            "cache_tokens": cache_tokens,
            "retrieved_shape": tuple(retrieved_kv.shape),
            "retrieved_dtype": str(retrieved_kv.dtype),
            "outputs_match": source_completion.text == target_completion.text,
            "source_latency_s": round(source_completion.elapsed_seconds, 3),
            "target_latency_s": round(target_completion.elapsed_seconds, 3),
            "source_preview": " ".join(source_completion.text.split()[:20]),
            "target_preview": " ".join(target_completion.text.split()[:20]),
        }
        print(json.dumps(result, indent=2))

    finally:
        print("== close: release the SDK context ==")
        lmc_sdk.close(ctx)


if __name__ == "__main__":
    main()
