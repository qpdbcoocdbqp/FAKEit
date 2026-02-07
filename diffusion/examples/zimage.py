import os
import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig

# set the path to the GGUF file
gguf_path = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Z-Image-Turbo-GGUF/snapshots/6c80814333b7b6a70a2e5b469a7c6437ce65de0f/z-image-turbo-Q3_K_M.gguf")

# load transformer model
transformer = ZImageTransformer2DModel.from_single_file(
    gguf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    )

# load the pipeline with the transformer and enable model CPU offload
pipeline = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
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
    num_inference_steps=9,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

# save the image
image.save("zimage-turbo-gguf.png")
