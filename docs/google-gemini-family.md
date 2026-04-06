# Google Gemini family

## Prerequisites

* **Models**

    | ModelCard                                                                                                         | Type                |
    | ----------------------------------------------------------------------------------------------------------------- | ------------------- |
    | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)                               | Any to Any          |
    | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)                                   | Sentence Similarity |
    | [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)                               | Text Generation     |
    | [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)                                           | Text Generation     |
    | [google/gemma-3-270m-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized) | Text Generation     |
    | [google/t5gemma-2-270m-270m](https://huggingface.co/google/t5gemma-2-270m-270m)                                   | Image-Text-to-Text  |
    | [google/gemma-3-4b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-unquantized)     | Image-Text-to-Text  |
    | [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it)                                           | Image-Text-to-Text  |

* **Huggingface**

    ```sh
    hf auth login
    # embedding model
    hf download google/embeddinggemma-300m
    # text model
    hf download google/functiongemma-270m-it
    hf download google/gemma-3-270m-it
    hf download google/gemma-3-270m-it-qat-q4_0-unquantized
    # image-text model    
    hf download google/t5gemma-2-270m-270m
    hf download google/gemma-3-4b-it-qat-q4_0-unquantized
    hf download google/gemma-3n-E2B-it
    # any to any
    hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-e4b-it-Q4_K_M.gguf mmproj-gemma-4-e4b-it-f16.gguf
    ```

* **Python**

    ```sh
    uv venv --python 3.13
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    uv pip install -q git+https://github.com/huggingface/transformers.git
    uv pip install -U  rich accelerate sentence-transformers bitsandbytes timm
    ```

## Import Submodules

* `Ocean Pearl`

    ```sh
    git submodule add -b main --force https://github.com/qpdbcoocdbqp/Ocean-Pearl.git ./submodules/ocean_pearl
    ```

## Run

```sh
cd FAKEit
python -m script.google-gemini-family
```

## Reference

* [Gemma 4 Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
  * [docker - vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags)

    ```bash
    docker pull vllm/vllm-openai:v0.19.0-cu130-ubuntu2404
    docker run -it --gpus=all \
    -v '$HOME/.cache/huggingface:/root/.cache/huggingface' \
    --entrypoint '' \
    -u 0 \
    docker.io/vllm/vllm-openai:gemma4-cu130 bash

    ls //root/.cache/huggingface
    vllm serve \
        --model google/gemma-4-E2B-it \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --max-num-seqs 1 \
        --gpu-memory-utilization 0.7 \
        --cpu-offload-gb 4 \
        --kv-offloading-size 4 \
        --kv-offloading-backend native \
        --trust_remote_code \
        --enforce-eager \
        --host 0.0.0.0 --port 8000
    # TBD OOM (8GB VRAM device)
    ```
