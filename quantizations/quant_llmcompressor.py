from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
# from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from datasets import load_dataset


model_path = "Qwen/Qwen3-0.6B"
output_dir = "./quantizations/models--Qwen--Qwen3-0.6B-llmcompressor-4bit"
dataset = load_dataset('NeelNanda/pile-10k')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", device_map="auto")
recipe = [
#     SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]

# Apply quantization
oneshot(model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=dataset,
        max_seq_length=128,
        num_calibration_samples=16,
        output_dir=output_dir
        )

