# SGLang Diffusion 

* [**SGLang Diffusion**](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)
* [**support model**](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/support_matrix.md)

```sh
hf download unsloth/Z-Image-Turbo-GGUF z-image-turbo-Q3_K_M.gguf
hf download BennyDaBall/Qwen3-4b-Z-Image-Engineer-V2.5 Z-Engineer-2.5-Q4_K_M.gguf
hf download tensorart/stable-diffusion-3.5-medium-turbo sd3.5m_turbo-Q4_K_M.gguf
hf download unsloth/FLUX.2-klein-4B-GGUF flux-2-klein-4b-Q3_K_M.gguf

docker run -it --gpus all --runtime=nvidia \
-p 11001:8000 \
-v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
--name dev \
lmsysorg/sglang:v0.5.8-cu130-runtime bash
```

* **sglang run SD-turbo in Docker**
 
```sh
# download fp16 model
hf download stabilityai/sd-turbo --include "*fp16.safetensors" --include "*.json" --include "*.txt"

# rename models
cd ~/.cache/huggingface/hub/models--stabilityai--sd-turbo/snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2
ln -s ./vae/diffusion_pytorch_model.fp16.safetensors ./vae/diffusion_pytorch_model.safetensors
ln -s ./unet/diffusion_pytorch_model.fp16.safetensors ./unet/diffusion_pytorch_model.safetensors
ln -s ./text_encoder/model.fp16.safetensors ./text_encoder/model.safetensors

# start container
docker run -it --gpus all \
--shm-size 16g \
-v ~/.cache/huggingface:/root/.cache/huggingface \
lmsysorg/sglang:dev \
bash

# in container
sglang generate \
--model-path /root/.cache/huggingface/hub/models--stabilityai--sd-turbo/snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2 \
--dit-precision fp16 \
--vae-precision fp16 \
--text-encoder-precisions fp16 \
--prompt "A logo With Bold Large text: SGL Diffusion" \
--num-inference-steps 9 \
--guidance-scale 0.0 \
--save-output

```