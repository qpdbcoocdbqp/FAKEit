
```sh
hf download unsloth/Z-Image-Turbo-GGUF z-image-turbo-Q3_K_M.gguf
hf download BennyDaBall/Qwen3-4b-Z-Image-Engineer-V2.5 Z-Engineer-2.5-Q4_K_M.gguf
hf download tensorart/stable-diffusion-3.5-medium-turbo sd3.5m_turbo-Q4_K_M.gguf

docker run -it --gpus all --runtime=nvidia \
-p 11001:8000 \
-v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
--name dev \
lmsysorg/sglang:v0.5.8-cu130-runtime bash
```
