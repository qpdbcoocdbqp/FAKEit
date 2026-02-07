import os
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, GGUFQuantizationConfig

# set the path to the GGUF file and the config file
gguf_path = os.path.expanduser("~/.cache/huggingface/hub/models--tensorart--stable-diffusion-3.5-medium-turbo/snapshots/e89195b004b3fb37c1ca7c6be91ff0114b7ccf38/sd3.5m_turbo-Q4_K_M.gguf")
config_path = os.path.expanduser("~/.cache/huggingface/hub/models--tensorart--stable-diffusion-3.5-medium-turbo/snapshots/e89195b004b3fb37c1ca7c6be91ff0114b7ccf38")

# load transformer model
transformer = SD3Transformer2DModel.from_single_file(
    gguf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config=config_path
    )

# load the pipeline with the transformer and enable model CPU offload
pipeline = StableDiffusion3Pipeline.from_pretrained(
    "tensorart/stable-diffusion-3.5-medium-turbo",
    transformer=transformer,
    torch_dtype=torch.bfloat16
    ).to("cuda")
pipeline.enable_model_cpu_offload()

# generate an image with the pipeline
# experiment with different prompts, image sizes, guidance scales, and number of inference steps to see how it affects the output image
prompt = "A cyberpunk city reflected in a puddle, neon lights, 8k resolution"
image = pipeline(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

# save the image
image.save("sd3.5-medium-turbo.png")
