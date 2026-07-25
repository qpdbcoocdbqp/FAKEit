### wsl

* Model hub: `/mnt/c/Users/qpdbc/.cache/huggingface`

* Install SGLang in host

```bash
uv pip install https://github.com/LMCache/LMCache/releases/download/v0.5.2rc1/lmcache-0.5.2rc1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
uv pip install -U sglang==0.5.16 --prerelease=allow

```

* Start sglang

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 python3 -m sglang.launch_server \
--model-path /mnt/c/Users/qpdbc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
--mem-fraction-static 0.4 \
--context-length 3096 \
--max-running-requests 1 \
--cuda-graph-max-bs-decode 16 \
--enable-lmcache \
--lmcache-config-file ./lmcache_config.yaml

```

* lmcache server standalone

```bash
docker pull lmcache/standalone:v0.5.2-cu130

docker run --rm -it \
--gpus all --ipc=host --pid=host \
--shm-size=4g --ulimit memlock=-1 --ulimit stack=6710886 \
--name lmcache-server \
-p 18080:8080 \
-p 6555:6555 \
-v $(pwd)/lmcache-data:/data \
lmcache/standalone:v0.5.2-cu130 \
CUDA_LAUNCH_BLOCKING=1 lmcache server \
--host 0.0.0.0 \
--port 6555 \
--http-host 0.0.0.0 \
--http-port 8080 \
--l1-size-gb 4 \
--eviction-policy LRU \
--l2-adapter '{"type":"fs","base_path":"./lmcache-data"}' \
--enable-extra-logging

docker logs lmcache-server |tail -n 10
docker rm -f lmcache-server

```

* Test request

```bash
curl http://127.0.0.1:30000/generate \
-H "Content-Type: application/json" \
-d '{
 "text":"Explain KV cache in LLM inference",
 "sampling_params":{
   "temperature":0.0,
   "max_new_tokens":128
 }
}'
```