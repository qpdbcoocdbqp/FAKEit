# KV cache storage - Llama.cpp


### Reference

- [Tutorial: KV cache reuse with llama-server #13606](https://github.com/ggml-org/llama.cpp/discussions/13606)

- [`LMCache`](https://github.com/LMCache/LMCache)

```

docker pull lmcache/vllm-openai

docker run --rm -it \
--runtime nvidia --gpus all \
--entrypoint '' \
lmcache/vllm-openai:latest \
python3 -c 'import lmcache.c_ops'

MSYS_NO_PATHCONV=1 docker run --rm -it \
--runtime nvidia --gpus all \
-v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
-e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
-p 8000:8000 \
--ipc=host \
lmcache/vllm-openai:latest \
Qwen/Qwen3-0.6B \
--gpu-memory-utilization 0.3 \
--max-model-len 512 \
--max-num-seqs 1 \
--kv-transfer-config \
'{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

```

```bash
uv pip install lmcache
python lmcache_dev.py
python lmcache_dev.py --skip-engine

```

Next extensions：

- Redis metadata backend
- PostgreSQL
- S3 / MinIO storage
- partial KV delta
- async save
- FastAPI service
- llama-server middleware
- distributed cache coordinator
- GPU-aware scheduler
- multi-model namespace
- RAG document cache
- token-level trie index
- semantic prefix matching
- branch checkpoint graph
- mmap zero-copy restore