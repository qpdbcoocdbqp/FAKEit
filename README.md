# FAKEit
Start SGLang service. Playing with [FAKEit](https://www.youtube.com/watch?v=a_iU8YeH944).

* **About FAKEit**

> FAKEit·SawanoHiroyuki[nZk]
>
> SawanoHiroyuki[nZk]:Laco「FAKEit」Music Video Fate/strange Fake -Whispers of Dawn- ver.

## Reference

* [sgl-project/sglang](https://github.com/sgl-project/sglang)

## SGLang

* **Setup**

  ```sh
  uv venv --python 3.12
  uv pip install openai requests
  ```

* **Docker**

  ```sh
  docker pull lmsysorg/sglang:latest-runtime

  # Set your HF_TOKEN=<token> in .env file
  cat .env.example

  # simple case
  docker compose --project-directory . --env-file .env -f sgl/simple-docker-compose.yaml up -d sglang-chat
  docker compose --project-directory . --env-file .env -f sgl/simple-docker-compose.yaml up -d sglang-embedding
  docker compose --project-directory . --env-file .env -f sgl/simple-docker-compose.yaml up -d sglang-reranker
  docker compose --project-directory . --env-file .env -f sgl/simple-docker-compose.yaml up -d sglang-multimodal
  docker compose --project-directory . --env-file .env -f sgl/simple-docker-compose.yaml up -d sglang-function

  # prefill-decode case
  docker compose \
  --project-directory . \
  --env-file .env \
  -f sgl/pd-docker-compose.yaml \
  up -d
  ```

  * **Troubleshoot: GPU OOM**
    * `--mem-fraction-static`: GPU memory usage percentage.
    * `--cpu-offload-gb`: offload to CPU memory.


### Service in docker compose

* `sgl/simple-docker-compose.yaml`: simple case

  <details> <summary> Service </summary>

    | service           |        port | model                                         |
    | :---------------- | ----------: | --------------------------------------------- |
    | sglang-chat       | 30000:30000 | `google/gemma-3-270m-it-qat-q4_0-unquantized` |
    | sglang-embedding  | 30001:30000 | `Qwen/Qwen3-Embedding-0.6B`                   |
    | sglang-reranker   | 30002:30000 | `BAAI/bge-reranker-v2-m3`                     |
    | sglang-multimodal | 30003:30000 | `Qwen/Qwen3-VL-2B-Instruct`                   |
    | sglang-function   | 30004:30000 | `google/functiongemma-270m-it`                |

  </details>


* `sgl/pd-docker-compose.yaml`: prefill-decode case

  <details> <summary> Service </summary>

    | service             |       port | model                                         |
    | :------------------ | ---------: | --------------------------------------------- |
    | sglang-chat-prefill |    -:30000 | `google/gemma-3-270m-it-qat-q4_0-unquantized` |
    | sglang-chat-decode  |    -:30000 | `google/gemma-3-270m-it-qat-q4_0-unquantize`  |
    | sglang-chat-router  | 30000:8000 | -                                             |
    
  </details>