# Qwen

* [Qwen 3.5](https://huggingface.co/collections/Qwen/qwen35)

    ```sh
    hf download bartowski/Qwen_Qwen3.5-4B-GGUF Qwen_Qwen3.5-4B-Q4_K_M.gguf
    hf download bartowski/Qwen_Qwen3.5-4B-GGUF mmproj-Qwen_Qwen3.5-4B-bf16.gguf
    ```

* `llama.cpp` inference

    * Run inference server

        ```sh
        docker run -d --gpus all --runtime=nvidia \
        -p 30000:8080 \
        -v $HOME/.cache/huggingface/hub/:/models \
        --name models \
        ghcr.io/mostlygeek/llama-swap:v197-cuda13-b8202
        ```

    * Run models

      * `--chat-template-file`: add [qwen35_nonthinking.jinja](chat-template/qwen35_nonthinking.jinja) for stucture output scenario.

        ```sh
        ./llama-server
        --host 127.0.0.1
        --port 8994
        --model /models/models--bartowski--Qwen_Qwen3.5-4B-GGUF/snapshots/b16ef105f32d7852fd9a5c190c6c346874f70a6e/Qwen_Qwen3.5-4B-Q4_K_M.gguf
        -ngl 33 -fa on
        --mmproj /models/models--bartowski--Qwen_Qwen3.5-4B-GGUF/snapshots/b16ef105f32d7852fd9a5c190c6c346874f70a6e/mmproj-Qwen_Qwen3.5-4B-bf16.gguf
        --ctx-size 16384 -ub 512 -b 512
        --cache-ram 0
        --jinja
        # --chat-template-file /app/chat_template/qwen35_nonthinking.jinja
        ```
