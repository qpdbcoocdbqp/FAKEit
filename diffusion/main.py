import sglang as sgl


model_path = "/models/models--BennyDaBall--Qwen3-4b-Z-Image-Engineer-V2.5/snapshots/3a2fd8ec37df842c1ae3521739fd90df7b7f83d3/Z-Engineer-2.5-Q4_K_M.gguf"
model_path.lower()

engine = sgl.Engine(model_path=model_path, random_seed = 42)


prompt = "Today is a sunny day and I like"
sampling_params = {"temperature": 0, "max_new_tokens": 256}
outputs = engine.generate(prompt, sampling_params)["text"]
print(outputs)
engine.shutdown()


import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, GGUFQuantizationConfig


transformer = SD3Transformer2DModel.from_single_file(
    "/models//models--tensorart--stable-diffusion-3.5-medium-turbo/snapshots/e89195b004b3fb37c1ca7c6be91ff0114b7ccf38/sd3.5m_turbo-Q4_K_M.gguf",
    config="/models/models--tensorart--stable-diffusion-3.5-medium-turbo\\snapshots\\e89195b004b3fb37c1ca7c6be91ff0114b7ccf38",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

# Load the pipeline using the transformer
# Note: We use the official SD3.5-medium-turbo repo to grab the VAE and Text Encoders
pipe = StableDiffusion3Pipeline.from_pretrained(
    "tensorart/stable-diffusion-3.5-medium-turbo", # Or the turbo repo
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

pipe.to("cuda")

# 4. Generate
image = pipe("A cyberpunk city reflected in a puddle, neon lights, 8k resolution").images[0]
image.save("sd3.5-medium-turbo-output.png")
