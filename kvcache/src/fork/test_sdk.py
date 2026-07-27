# SPDX-License-Identifier: Apache-2.0
"""
SDK for retrieving and storing KV cache tensors.
"""

import torch
from src.fork.lmcache.sdk.kvcache import connect, store, retrieve

model_name = "Qwen/Qwen3-0.6B"
lmcache_url = "http://localhost:18000"
lmcache_mq_url = "tcp://localhost:6555"
timeout = 60

ctx = connect(
    url=lmcache_mq_url,
    http_url=lmcache_url,
    model_name=model_name,
    timeout=timeout,
)

# build store tensors
# and length of token mod chunk_size is 0
# kv shape: [2, L, T, D]
chunk_size = ctx.chunk_size
num_layers = len(ctx._kv_caches)
D = list(ctx._kv_caches.values())[0].shape[-1]
print(f"chunk_size={chunk_size}, num_layers={num_layers}, hidden_dim={D}")
tokens = list(range(chunk_size))
kv = torch.randn(2, num_layers, chunk_size, D, dtype=torch.float16)

store_status = store(ctx, kv, tokens)
print("store result:", store_status)

result = retrieve(ctx, tokens)
print("retrieve result:", None if result is None else result.shape)

if result is not None:
    matched = torch.allclose(result.cpu().float(), kv.float(), atol=1e-2)
    print("data matches:", matched)
