# Google Gemini family

## Prerequisites

* **Models**

    | ModelCard                                                                                                         | Type                |
    | ----------------------------------------------------------------------------------------------------------------- | ------------------- |
    |[yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF gemma4-coding-Q4_K_M.gguf]()| Image-Text-to-Text  |
    |[sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4](https://huggingface.co/sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4)| Image-Text-to-Text  |
    | [unsloth/gemma-4-12B-it-qat-GGUF:gemma-4-12B-it-qat-UD-Q4_K_XL.gguf](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) | Image-Text-to-Text  |
    | [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)                 | Image-Text-to-Text  |
    | [unsloth/gemma-4-E4B-it-qat-GGUF:gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF) |  Any to Any |
    | [google/gemma-4-E4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)                 | Any to Any          |
    | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)                               | Any to Any          |
    | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)                                   | Sentence Similarity |
    | [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)                               | Text Generation     |
    | [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)                                           | Text Generation     |
    | [google/gemma-3-270m-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized) | Text Generation     |
    | [google/t5gemma-2-270m-270m](https://huggingface.co/google/t5gemma-2-270m-270m)                                   | Image-Text-to-Text  |
    | [google/gemma-3-4b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-unquantized)     | Image-Text-to-Text  |
    | [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it)                                           | Image-Text-to-Text  |

    | ModelCard                                                                                                         | Type                | TPS |
    | ----------------------------------------------------------------------------------------------------------------- | ------------------- | --- |
    | [unsloth/gemma-4-12B-it-qat-GGUF:mtp-gemma-4-12B-it.gguf](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) | Draft model | 32.23 t/s -> 72.25 t/s|
    | [unsloth/gemma-4-E4B-it-qat-GGUF:mtp-gemma-4-E4B-it](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF) | Draft model |79.34 t/s -> 71.05 ~ 153.90 t/s |



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
    hf download yuxinlu1/gemma-4-12B-it-Claude-4.6-4.8-Opus-GGUF gemma4-opus48-Q4_K_M.gguf
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

* Gemma 4 - [llama.cpp](https://github.com/ggml-org/llama.cpp)

    ```bash
    ./llama-server
    --host 127.0.0.1
    --port 9006
    --model /models/models--ggml-org--gemma-4-E4B-it-GGUF/snapshots/6b352c53e1d2e4bb974d9f8cafcf85887c224219/gemma-4-e4b-it-Q4_K_M.gguf
    --mmproj /models/models--ggml-org--gemma-4-E4B-it-GGUF/snapshots/6b352c53e1d2e4bb974d9f8cafcf85887c224219/mmproj-gemma-4-e4b-it-f16.gguf
    --threads 1 --parallel 1 --ubatch-size 512
    --ctx-size 16384 -ctk q4_0 -ctv q4_0
    -ngl 43  -fa on --no-mmproj-offload
    --cache-ram 0
    --reasoning on
    --jinja
    ```
* Use MTP draft

    ```bash
    ./llama-server
    --host 127.0.0.1
    --port 9006
    --model /models/models--unsloth--gemma-4-E4B-it-qat-GGUF/snapshots/bbcd9d849c2541ecc2af7ef64b3c3c2c7aa14e96/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
    --mmproj /models/models--unsloth--gemma-4-E4B-it-qat-GGUF/snapshots/bbcd9d849c2541ecc2af7ef64b3c3c2c7aa14e96/mmproj-BF16.gguf
    --model-draft /models/models--unsloth--gemma-4-E4B-it-qat-GGUF/snapshots/bbcd9d849c2541ecc2af7ef64b3c3c2c7aa14e96/mtp-gemma-4-E4B-it.gguf
    --spec-type draft-mtp --spec-draft-n-max 4
    --threads 2 --parallel 1 -fa off --n-gpu-layers 49
    --spec-type draft-mtp
    --spec-draft-n-max 2
    --reasoning on
    --cache-ram 0
    --jinja
    ```