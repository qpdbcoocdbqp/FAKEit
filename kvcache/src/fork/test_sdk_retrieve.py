import httpx
from transformers import AutoTokenizer
from src.fork.lmcache.sdk.kvcache import connect, retrieve


# create input tokens
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

text = "The quick brown fox jumps over the lazy dog. " * 60
input_ids = tokenizer.encode(text)
print("total tokens:", len(input_ids))

chunk_size = 256
n_chunks = len(input_ids) // chunk_size
tokens = input_ids[: n_chunks * chunk_size]
print("aligned tokens:", len(tokens))

# create KV cache with vllm
resp = httpx.post(
    "http://localhost:18001/v1/completions",
    json={
        "model": model_name,
        "prompt": tokens,
        "max_tokens": 8,
        "temperature": 0,
    },
    timeout=60,
)
print(resp.status_code, resp.json()["choices"][0]["text"] if resp.status_code == 200 else resp.text)

# retrieve KV cache with lmcache SDK
lmcache_url = "http://localhost:18000"
lmcache_mq_url = "tcp://localhost:6555"
timeout = 60

ctx = connect(
    url=lmcache_mq_url,
    http_url=lmcache_url,
    model_name=model_name,
    timeout=timeout,
)

result = retrieve(ctx, tokens)
print("retrieve result:", None if result is None else result.shape)
