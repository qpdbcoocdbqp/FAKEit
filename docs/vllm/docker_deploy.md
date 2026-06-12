# vLLM with Docker

## Deploy

```bash
# <!-- 2026-06-10 -->
docker pull vllm/vllm-openai:latest-ubuntu2404

# version check, vllm  0.22.1
docker run --rm -it \
--gpus=all \
--name infos \
--entrypoint '' \
vllm/vllm-openai:latest-ubuntu2404 bash -c 'nvidia-smi && nvcc --version && vllm --version'

# $LLAMA_SWAP=<your llama-swap path>
docker run -itd \
--gpus=all \
-p 18001:8080 \
-v "/$LLAMA_SWAP:/app" \
-v "/$HOME/.cache/huggingface/hub:/models" \
--name vllm \
--entrypoint '' \
vllm/vllm-openai:latest-ubuntu2404 \
bash -c '/app/llama-swap -config /app/vllm.yaml -listen 0.0.0.0:8080'

```

## llama-swap

* `config.yaml`

```yaml
# llama-swap
healthCheckTimeout: 180  # use longer healthCheckTimeout for vLLM 
logToStdout: "both"      # proxy and upstream

models:
  qwen3.5-0.8b:
    cmd: |
        vllm serve /models/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
        --port ${PORT}
        --served-model-name qwen3.5-0.8b
        --gpu-memory-utilization 0.5
        --max_num_seqs 1
        --max-model-len 512
    checkEndpoint: /v1/models

```