
# Google Gemini family

## Prerequisites

* **Models**

    | ModelCard                                                                                                         | Type                |
    | ----------------------------------------------------------------------------------------------------------------- | ------------------- |
    | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)                                   | Sentence Similarity |
    | [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)                               | Text Generation     |
    | [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)                                           | Text Generation     |
    | [google/gemma-3-270m-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized) | Text Generation     |
    | [google/t5gemma-2-270m-270m](https://huggingface.co/google/t5gemma-2-270m-270m)                                   | Image-Text-to-Text  |

* **Huggingface**

    ```sh
    hf auth login
    hf download google/embeddinggemma-300m
    hf download google/functiongemma-270m-it
    hf download google/gemma-3-270m-it
    hf download google/gemma-3-270m-it-qat-q4_0-unquantized
    hf download google/t5gemma-2-270m-270m
    ```

* **Python**

    ```sh
    uv venv --python 3.12
    uv pip install -q git+https://github.com/huggingface/transformers.git
    uv pip install -U sentence-transformers
    uv pip install -U bitsandbytes>=0.46.1
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
