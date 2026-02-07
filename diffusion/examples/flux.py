import os
import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, GGUFQuantizationConfig

# set the path to the GGUF file and the config file
gguf_path = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--FLUX.2-klein-4B-GGUF/snapshots/0084d1df98e2e2137fe776d55170bc4792ec1d66/flux-2-klein-4b-Q4_0.gguf")
config_path = os.path.expanduser("~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/5e67da950fce4a097bc150c22958a05716994cea/transformer")

# load transformer model
transformer = Flux2Transformer2DModel.from_single_file(
    gguf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config=config_path
    )

# load the pipeline with the transformer and enable model CPU offload
pipeline = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
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
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

# save the image
image.save("flux-klein-4b-gguf.png")
