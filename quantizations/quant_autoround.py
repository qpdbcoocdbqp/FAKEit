from auto_round import AutoRound
from transformers import AutoModelForCausalLM, AutoTokenizer


# * Scheme examples: "W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4" (no real kernels), "GGUF:Q4_K_M", etc.
# * format = 'auto_round'(default), 'auto_gptq', 'auto_awq'

# --- simple ---
scheme = "W4A16"
model_name = "Qwen/Qwen3-0.6B"

autoround = AutoRound(model=model_name, scheme=scheme)
autoround.quantize_and_save(f"./quantizations/models--autoround-{scheme.lower()}", format="auto_round")

del scheme, model_name, autoround

# --- awq & gptq ---
model_name = "Qwen/Qwen3-0.6B"
for save_format in ["awq", "gptq"]:
    output_dir = f"./quantizations/models--autoround-{save_format}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
    autoround = AutoRound(
        model, tokenizer,
        bits=4, group_size=128, sym=True,
        iters=200, enable_torch_compile=True
        )
    autoround.quantize_and_save(
        output_dir, 
        format=f"auto_{save_format}", 
        use_safetensors=True
    )

    del save_format, output_dir, tokenizer, model, autoround
