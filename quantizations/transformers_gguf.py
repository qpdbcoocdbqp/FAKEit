import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GgufConfig
from transformers.integrations.gguf.kernels import get_gguf_kernel

print("Support gguf model transformers-community/ggml-quantization:", get_gguf_kernel())

model_id = "bartowski/Qwen_Qwen3.5-0.8B-GGUF"
filename = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"

gguf_config = GgufConfig(dequantize=False)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    gguf_file=filename,
    device_map="auto",
    quantization_config=gguf_config,
    dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)

input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)
  
output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
