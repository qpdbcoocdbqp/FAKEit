<!--
* [Wan 2.1 [GGUF] Text-to-Video & Image-to-Video in ComfyUI Tutorial + Workflow - Low VRAM GPU](https/:/youtube.com/watch?v=-JE1tt_guGE&start=0)
* [Uncensored WAN 2.2 14B Rapid v10 in ComfyUI | All-In-One Workflow](https://www.youtube.com/watch?v=3mYfMvRJkeU)
-->

## Reference

* **[Extension]**

  * [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
  * [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
  * [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main)

## Models

* **`comfyUI/models/unet`**

    * [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/blob/main/Wan2.2-TI2V-5B-Q4_K_M.gguf)
    * [befox/WAN2.2-14B-Rapid-AllInOne-GGUF](https://huggingface.co/befox/WAN2.2-14B-Rapid-AllInOne-GGUF/blob/main/Mega-v12/wan2.2-rapid-mega-aio-nsfw-v12.1-Q4_K.gguf)

    ```sh
    # QuantStack/Wan2.2-TI2V-5B-GGUF
    hf download QuantStack/Wan2.2-TI2V-5B-GGUF Wan2.2-TI2V-5B-Q4_K_M.gguf
    # befox/WAN2.2-14B-Rapid-AllInOne-GGUF 
    hf download befox/WAN2.2-14B-Rapid-AllInOne-GGUF Mega-v12/wan2.2-rapid-mega-aio-nsfw-v12.1-Q4_K.gguf
    ```

* **`comfyUI/models/text_encoders`**

    * [city96/t5-v1_1-xxl-encoder-gguf](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/blob/main/t5-v1_1-xxl-encoder-Q4_K_M.gguf)
    * [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf/blob/main/umt5-xxl-encoder-Q4_K_M.gguf)

    ```sh
    # city96/t5-v1_1-xxl-encoder-gguf
    hf download city96/t5-v1_1-xxl-encoder-gguf t5-v1_1-xxl-encoder-Q4_K_M.gguf
    # city96/umt5-xxl-encoder-gguf
    hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q4_K_M.gguf
    ```

* **`comfyUI/models/vae`**

    * [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/vae)

    ```sh
    # wan2.2_vae
    hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/vae/wan2.2_vae.safetensors
    # wan_2.1_vae
    hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/vae/wan_2.1_vae.safetensors
    ```

* **`comfyUI/models/clip_vision`**

    * [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/clip_vision/clip_vision_h.safetensors)

    ```sh
    # clip_vision
    hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/clip_vision/clip_vision_h.safetensors
    ```

## Workflows

* [befox/WAN2.2-14B-Rapid-AllInOne-GGUF/example-workflows](https://huggingface.co/befox/WAN2.2-14B-Rapid-AllInOne-GGUF/tree/main/example-workflows)
  * I2V: image to video
  * T2V: text to video

# ComfyUI

```sh
cd ComfyUI
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r manager_requirements.txt
uv pip install gguf triton-windows sageattention
python -m main --enable-manager
```

