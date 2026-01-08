## SGLang Model Gateway

### start runtime

```sh
docker run -itd --gpus all --runtime=nvidia \
-p 30000:30000 \
-v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
--name dev \
lmsysorg/sglang:v0.5.7-runtime bash
```

### Option 1: start grpc model server with http router

```sh
docker exec -it dev python3 -m sglang_router.launch_server \
--grpc-mode \
--host 0.0.0.0 --port 30000 \
--model-path google/gemma-3-4b-it-qat-q4_0-unquantized \
--tokenizer-path google/gemma-3-4b-it-qat-q4_0-unquantized \
--chat-template gemma-it \
--dtype bfloat16 \
--cpu-offload-gb 8 \
--mem-fraction-static 0.4 \
--context-length 2048 \
--max-total-tokens 2048 \
--max-prefill-tokens 512 \
--trust-remote-code \
--disable-cuda-graph \
--log-level info
```

### Option 2: start SEPARATED

* start **grpc model server**

    ```sh
    docker exec -it dev python3 -m sglang.launch_server \
    --model-path google/gemma-3-4b-it-qat-q4_0-unquantized \
    --tokenizer-path google/gemma-3-4b-it-qat-q4_0-unquantized \
    --chat-template gemma-it \
    --grpc-mode \
    --attention-backend flashinfer \
    --dtype bfloat16 \
    --cpu-offload-gb 8 \
    --mem-fraction-static 0.4 \
    --context-length 2048 \
    --max-total-tokens 2048 \
    --max-prefill-tokens 512 \
    --trust-remote-code \
    --log-level info \
    --host 0.0.0.0 \
    --port 20000
    ```

* start **http router for model server**

    ```sh
    docker exec -it dev python3 -m sglang_router.launch_router \
    --worker-urls grpc://localhost:20000 \
    --model-path google/gemma-3-4b-it-qat-q4_0-unquantized \
    --tokenizer-path google/gemma-3-4b-it-qat-q4_0-unquantized \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000
    ```

### Test case `google/gemma-3-270m-it` 

* **something is wrong when grpc-mode is running.**
* `/v1/chat/completions` stop at the first token. It got eos_token (`106`).

### Test case `google/gemma-3-4b-it-qat-q4_0-unquantized`

* When grpc-mode is running, `Repetition Loop` issue will happen. Even set high `presence_penalty=1.1`, `frequency_penalty=1.1`.

    ```sh
    # test code for google/gemma-3-4b-it-qat-q4_0-unquantized
    python -m test.test_grpc_streaming_oai_client
    ```