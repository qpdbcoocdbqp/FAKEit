## Setup

* **SpecForge**

```sh
# os
sudo apt-get update
sudo apt-get install libnuma-dev python3.12-dev

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
cd source
git clone https://github.com/sgl-project/SpecForge.git

# python
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r SpecForge/requirements.txt
uv pip install -v ./SpecForge --prerelease=allow
uv pip install torch-c-dlpack-ext

# envs
echo $CUDA_HOME

# get dataset
python SpecForge/scripts/prepare_data.py --dataset sharegpt

```

* **EAGLE3 Training**

  * target model: `Qwen/Qwen3-0.6B`
  * draft model config: `qwen3-0.6b-eagle3.json`

    <details> <summary>qwen3-0.6b-eagle3.json</summary>

    ```json
    {
        "architectures": [
            "LlamaForCausalLMEagle3"
        ],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 40960,
        "max_window_layers": 36,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 1,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": null,
        "rope_theta": 1000000,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936,
        "draft_vocab_size": 32000
    }

    ```


    </details>

  * train data: `sharegpt_train_8192.jsonl`

