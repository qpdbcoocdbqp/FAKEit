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
