from auto_round import AutoRound

# Scheme examples: "W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4" (no real kernels), "GGUF:Q4_K_M", etc.
scheme = "W4A16"
format = "auto_round"
model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

autoround = AutoRound(
    model_path, scheme=scheme
    )

autoround.quantize_and_save(f"./quantizations/models--Qwen--Qwen3-0.6B-4bit", format=format)
