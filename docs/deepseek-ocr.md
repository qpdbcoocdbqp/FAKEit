# DeepseekOCR

## Prerequisites

* **Models**

    | ModelCard                                                                                     | Type               |
    | --------------------------------------------------------------------------------------------- | ------------------ |
    | [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)                   | Image-Text-to-Text |
    | [Jalea96/DeepSeek-OCR-bnb-4bit-NF4](https://huggingface.co/Jalea96/DeepSeek-OCR-bnb-4bit-NF4) | Image-Text-to-Text |

* **Huggingface**

    ```sh
    hf auth login
    hf download deepseek-ai/DeepSeek-OCR
    hf download Jalea96/DeepSeek-OCR-bnb-4bit-NF4
    ```

* **Python**

    ```sh
    uv venv --python 3.13
    uv pip install einops addict easydict ninja packaging matplotlib

    # For testing case, `flash-attn` is not necessary.
    # when `flash-attn` is failed to install, use `eager` attention_mode to load model.
    uv pip install flash-attn --no-build-isolation

    uv pip install -U 'transformers==4.46.3'
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    ```

* **Get test image**

    ```sh

    wget -O script/dev-bee.jpg https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg

    ```

## Import Submodules

* `Ocean Pearl`

    ```sh
    git submodule add -b main --force https://github.com/qpdbcoocdbqp/Ocean-Pearl.git ./submodules/ocean_pearl
    ```

## Run

```sh
cd FAKEit
python -m script.deepseek-ocr
```
