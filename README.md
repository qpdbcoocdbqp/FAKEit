# FAKEit
Start SGLang service. Playing with [FAKEit](https://www.youtube.com/watch?v=a_iU8YeH944).

* **About FAKEit**

> FAKEit·SawanoHiroyuki[nZk]
>
> SawanoHiroyuki[nZk]:Laco「FAKEit」Music Video Fate/strange Fake -Whispers of Dawn- ver.

## Reference

* [sgl-project/sglang](https://github.com/sgl-project/sglang)
* [DockerHub/lmsysorg](https://hub.docker.com/u/lmsysorg)

## SGLang

* **Setup**

  ```sh
  uv venv --python 3.12
  uv pip install openai requests
  ```
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

    | service             |        port | model                                         |
    | :------------------ | ----------: | --------------------------------------------- |
    | sglang-chat-prefill |     -:30000 | `google/gemma-3-270m-it-qat-q4_0-unquantized` |
    | sglang-chat-decode  |     -:30000 | `google/gemma-3-270m-it-qat-q4_0-unquantized` |
    | sglang-chat-router  | 30000:30000 | -                                             |
    
  </details>

* `sgl/quantization-docker-compose.yaml`: qunatization case

  * Here use `Qwen/Qwen3-0.6B` to be the quantization based model.

  <details> <summary> Service </summary>

    | service         |        port | quantization          |
    | :-------------- | ----------: | --------------------- |
    | modelopt-fp8    |     -:30000 | `nvidia-modelopt/fp8` |
    | modelopt-fp4    |     -:30000 | `nvidia-modelopt/fp4` |
    | autoround-w4a16 |     -:30000 | `auto-round/W4A16`    |
    | autoround-awq   |     -:30000 | `auto-round/AWQ`      |
    | autoround-gptq  |     -:30000 | `auto-round/GPTQ`     |
    | sglang-router   | 30000:30000 | -                     |
    
  </details>

* `sgl/sd-docker-compose.yaml`: speculative decoding eagle-3 case

  * Here use `Qwen/Qwen3-0.6B` to be the target model.
  * Train a `qwen3-0.6b-eagle3-sharegpt` draft model
  * This is just for testing. Not finished in training process. Accurray of draft model is `~0.3`.

  <details> <summary> Service </summary>

    | service       |        port | model                                             |
    | :------------ | ----------: | ------------------------------------------------- |
    | native-model  |     -:30000 | `Qwen/Qwen3-0.6B`                                 |
    | eagle3-model  |     -:30000 | `Qwen/Qwen3-0.6B`<br>`qwen3-0.6b-eagle3-sharegpt` |
    | sglang-router | 30000:30000 | -                                                 |
    
  </details>


### Start SGLang server with Docker

* **Stable version (cuda 12)**

  ```sh
  # stable version (cuda 12)
  docker pull lmsysorg/sglang:latest-runtime
  ```
  
* <details> <summary> Nightly version (cuda 13) </summary>

  ```sh
  docker pull lmsysorg/sglang:v0.5.6.post2-cu130-runtime
  docker build -t lmsysorg/sglang:v0.5.6.post2-cu130-runtime-pd -f sgl/pd-dockerfile .
  ```

</details>

* **Set your HF_TOKEN=<token> in .env file**

  ```sh
  # Use `.env` SGLANG_TAG and SGLANG_PD_TAG to control image tag.
  # Default tag is `latest-runtime`
  cat .env.example
  ```

* <details> <summary> Simple case </summary>

  ```sh
  docker compose --project-directory . --env-file .env -f sgl/simple-docker-compose.yaml up -d sglang-chat

  docker compose --project-directory . --env-file .env -f sgl/simple-docker-compose.yaml up -d sglang-embedding

  docker compose --project-directory . --env-file .env -f sgl/simple-docker-compose.yaml up -d sglang-reranker

  docker compose --project-directory . --env-file .env -f sgl/simple-docker-compose.yaml up -d sglang-multimodal

  docker compose --project-directory . --env-file .env -f sgl/simple-docker-compose.yaml up -d sglang-function
  ```

</details>

* <details> <summary> prefill-decode case </summary>

  ```sh
    docker compose --project-directory . --env-file .env -f sgl/pd-docker-compose.yaml up -d
  ```

</details>

* <details> <summary> qunatization case </summary>

  ```sh
    docker compose --project-directory . --env-file .env -f sgl/quantization-docker-compose.yaml up -d
  ```

</details>

* <details> <summary> speculative decoding case </summary>

  ```sh
    docker compose --project-directory . --env-file .env -f sgl/sd-docker-compose.yaml up -d
  ```

</details>

* **Troubleshoot: GPU OOM**
  * `--mem-fraction-static`: GPU memory usage percentage.
  * `--cpu-offload-gb`: offload to CPU memory.

## Google Gemini Family

* **Reference**

    * [Google Gemini Family](docs/google-Gemini-Family.md)