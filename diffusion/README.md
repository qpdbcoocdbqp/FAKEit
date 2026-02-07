# SGLang Diffusion 

## Reference

* [**SGLang Diffusion**](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)
  * [**support model**](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/support_matrix.md)
* [**huggingface/diffusers**](https://github.com/huggingface/diffusers)
* [**vladmandic/sdnext**](https://github.com/vladmandic/sdnext)

## Model

| Model                                                                                                       | Transformer (Model Size)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [stable-diffusion-3.5-medium-turbo](https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo)     | (1.79 GB) [sd3.5m_turbo-Q4_K_M.gguf]([sd3.5m_turbo-Q4_K_M.gguf](https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo/blob/main/sd3.5m_turbo-Q4_K_M.gguf))                                                                                                                                                                                                                                                                                                                                                                                                |
| [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)                                            | (4.19 GB) [z-image-turbo-Q3_K_M.gguf](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/blob/main/z-image-turbo-Q3_K_M.gguf)                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)                                 | (2.46 GB) [flux-2-klein-4b-nvfp4.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4/blob/main/flux-2-klein-4b-nvfp4.safetensors)<br>(1.83 GB) [flux-2-klein-4b-Q2_K.gguf](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/blob/main/flux-2-klein-4b-Q2_K.gguf)<br>(2.12 GB) [flux-2-klein-4b-Q3_K_M.gguf](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/blob/main/flux-2-klein-4b-Q3_K_M.gguf)<br>(2.6 GB) [flux-2-klein-4b-Q4_K_M.gguf](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/blob/main/flux-2-klein-4b-Q4_K_M.gguf) |
| [Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic) | (2.47 GB) [diffusion_pytorch_model.safetensors](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic/blob/main/transformer/diffusion_pytorch_model.safetensors)                                                                                                                                                                                                                                                                                                                                                                                             |

## Run service

* **SGLang run SD-turbo in Docker**

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

* Python `diffuser` and `sdnq` examples

  * setup

    ```sh
    uv venv --python 3.12
    uv pip install git+https://github.com/huggingface/diffusers
    uv pip install sdnq
    ```

  * [FLUX.2-klein-4B](examples/flux.py)
    * ![FLUX.2-klein-4B](examples/imgs/flux-klein-4b-gguf.png)
  * [FLUX.2-klein-4B-SDNQ-4bit-dynamic](examples/flux-sdnq.py)
    * ![FLUX.2-klein-4B-SDNQ-4bit-dynamic](examples/imgs/flux-klein-sdnq-4bit-dynamic.png)
  * [Z-Image-Turbo](examples/zimage.py)
    * ![Z-Image-Turbo](examples/imgs/zimage-turbo-gguf.png)
  * [stable-diffusion-3.5-medium-turbo](examples/sd3-5.py)
    * ![stable-diffusion-3.5-medium-turbo](examples/imgs/sd3.5-medium-turbo.png)
