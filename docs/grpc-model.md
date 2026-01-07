## SGLang Model Gateway

### start runtime

```sh
docker run -itd --gpus all --runtime=nvidia \
-p 30000:30000 \
-v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
--name dev \
lmsysorg/sglang:v0.5.7-runtime bash
```

### start grpc model server

```sh
docker exec -it dev python3 -m sglang.launch_server \
--model-path google/gemma-3-270m-it \
--tokenizer-path google/gemma-3-270m-it \
--grpc-mode \
--attention-backend flashinfer \
--dtype bfloat16 \
--context-length 2048 \
--mem-fraction-static 0.1 \
--max-total-tokens 2048 \
--max-prefill-tokens 512 \
--trust-remote-code \
--log-level info \
--host 0.0.0.0 \
--port 20000
```

### start http router for model server
```sh
docker exec -it dev python3 -m sglang_router.launch_router \
--worker-urls grpc://localhost:20000 \
--model-path google/gemma-3-270m-it \
--tokenizer-path google/gemma-3-270m-it \
--policy cache_aware \
--host 0.0.0.0 \
--port 30000
```

### Test case `google/gemma-3-270m-it` 

* **something is wrong when grpc-mode is running.**
* `/v1/chat/completions` stop when it got token `106`.


```py
import requests

resp = requests.get("http://localhost:30000/health")
print(resp.ok)

# >>> True
resp = requests.get("http://localhost:30000/workers")
print(resp.text)

# >>> {"workers":[{"id":"ef84103e-752c-46ca-b179-ed6ff19435b3","url":"grpc://localhost:20000","model_id":"google/gemma-3-270m-it","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"gRPC","runtime_type":"sglang","metadata":{"id2label_json":"{\"0\": \"LABEL_0\", \"1\": \"LABEL_1\"}","architectures":"[\"Gemma3ForCausalLM\"]","tokenizer_path":"google/gemma-3-270m-it","weight_version":"default","max_context_length":"2048","model_path":"google/gemma-3-270m-it","served_model_name":"google/gemma-3-270m-it","vocab_size":"128256","bos_token_id":"1","is_generation":"true","model_type":"gemma3_text","max_req_input_len":"2042","num_labels":"2"}}],"total":1,"stats":{"prefill_count":0,"decode_count":0,"regular_count":1}}

text_data = {
    "model": "google/gemma-3-270m-it",
    "messages":[
        {
            "role": "user",
            "content": "Why sky is blue?",
        }
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": False
}

response = requests.post("http://localhost:30000/v1/chat/completions", json=text_data)
print(response.json())
# >>> {'id': 'chatcmpl-026c3001-be9b-40b2-aa58-87da365bfe73', 'object': 'chat.completion', 'created': 1767795236, 'model': 'google/gemma-3-270m-it', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'reasoning_content': None}, 'finish_reason': 'stop', 'matched_stop': 106}], 'usage': {'prompt_tokens': 13, 'completion_tokens': 1, 'total_tokens': 14}, 'system_fingerprint': 'default'}

```

