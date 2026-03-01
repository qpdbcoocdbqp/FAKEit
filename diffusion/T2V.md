## Reference

* **[Extension]**

  * [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
  * [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
  * [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main)
  * [Phr00t/WAN2.2-14B-Rapid-AllInOne](https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne)

## Models

* **`comfyUI/models/unet`**

    * [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/blob/main/Wan2.2-TI2V-5B-Q4_K_M.gguf)
    * [befox/WAN2.2-14B-Rapid-AllInOne-GGUF](https://huggingface.co/befox/WAN2.2-14B-Rapid-AllInOne-GGUF/blob/main/Mega-v12/wan2.2-rapid-mega-aio-nsfw-v12.1-Q4_K.gguf)
    * [Kijai/WanVideo_comfy_GGUF](https://huggingface.co/Kijai/WanVideo_comfy_GGUF/blob/main/Wan22Animate/Wan2_2_Animate_14B_Q4_K_M.gguf)

    ```sh
    # QuantStack/Wan2.2-TI2V-5B-GGUF
    hf download QuantStack/Wan2.2-TI2V-5B-GGUF Wan2.2-TI2V-5B-Q4_K_M.gguf
    # befox/WAN2.2-14B-Rapid-AllInOne-GGUF 
    hf download befox/WAN2.2-14B-Rapid-AllInOne-GGUF Mega-v12/wan2.2-rapid-mega-aio-nsfw-v12.1-Q4_K.gguf
    # Kijai/WanVideo_comfy_GGUF
    hf download Kijai/WanVideo_comfy_GGUF Wan22Animate/Wan2_2_Animate_14B_Q4_K_M.gguf
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

* **`comfyUI/models/lora`**

    * [WanAnimate - relight](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors)
    * [lightx2v- I2V](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Lightx2v)
    * [Wan22_Lightx2v](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/LoRAs/Wan22_Lightx2v)
    ```sh
    # relight
    hf download Kijai/WanVideo_comfy LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors
    hf download Kijai/WanVideo_comfy LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16_resized_from_128_to_dynamic_22.safetensors
    # lightx2v - I2V
    hf download Kijai/WanVideo_comfy Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank16_bf16.safetensors
    hf download Kijai/WanVideo_comfy Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors
    # Wan22 Lightx2v
    hf download Kijai/WanVideo_comfy LoRAs/Wan22_Lightx2v/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors
    hf download Kijai/WanVideo_comfy LoRAs/Wan22_Lightx2v/Wan_2_2_I2V_A14B_HIGH_lightx2v_MoE_distill_lora_rank_64_bf16.safetensors
    ```

* **`comfyUI/models/detection`**

    * [yolov10m](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/blob/main/process_checkpoint/det/yolov10m.onnx)
    * [JunkyByte/easy_ViTPose](https://huggingface.co/JunkyByte/easy_ViTPose/blob/main/onnx/wholebody/vitpose-l-wholebody.onnx)

    ```sh
    hf download Wan-AI/Wan2.2-Animate-14B process_checkpoint/det/yolov10m.onnx
    hf download JunkyByte/easy_ViTPose onnx/wholebody/vitpose-l-wholebody.onnx
    ```

## Workflows

* [befox/WAN2.2-14B-Rapid-AllInOne-GGUF/example-workflows](https://huggingface.co/befox/WAN2.2-14B-Rapid-AllInOne-GGUF/tree/main/example-workflows)
  * [Image to video](comfy-workflows/wan2.2-i2v-rapid-aio-gguf-example.json): image to video
  * [Text to video](comfy-workflows/wan2.2-t2v-rapid-aio-gguf-example.json): text to video
  * [First last frame to video](comfy-workflows/wan2.2-flf2v-rapid-aio-gguf-example.json)

* [kijai/ComfyUI-WanVideoWrapper/example-workflows](https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/example_workflows)
  * Character animation and replacement
    * [example-1](comfy-workflows/wanvideo_WanAnimate_example_01.json)
    * [example-2](comfy-workflows/wanvideo_WanAnimate_preprocess_example_02.json)
      * seperate 2 components from `example-2`.
      * [video-masking](comfy-workflows/video-masking.json)
      * [character-replace](comfy-workflows/character-replace.json)

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

