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
  docker compose \
  --project-directory . \
  --env-file .env \
  -f sgl/docker-compose.yaml \
  up -d
  ```

* **Sample model**

  | service           | port  | model                                         |
  | ----------------- | ----- | --------------------------------------------- |
  | sglang-chat       | 30000 | `google/gemma-3-270m-it-qat-q4_0-unquantized` |
  | sglang-embedding  | 30001 | `Qwen/Qwen3-Embedding-0.6B`                   |
  | sglang-reranker   | 30002 | `BAAI/bge-reranker-v2-m3`                     |
  | sglang-multimodal | 30003 | `Qwen/Qwen3-VL-2B-Instruct`                   |
