# FLUX.2-klein-4B

import os
import torch
from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig
# Flux2KleinPipeline

gguf_path = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--FLUX.2-klein-4B-GGUF/snapshots/0084d1df98e2e2137fe776d55170bc4792ec1d66/flux-2-klein-4b-Q4_0.gguf")
config_path = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--FLUX.2-klein-4B-GGUF/snapshots/0084d1df98e2e2137fe776d55170bc4792ec1d66")

## Fail OOM
transformer = Flux2Transformer2DModel.from_single_file(
    gguf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config=config_path,
    device_map=None,
    low_cpu_mem_usage=True
    )

# import fail, no support yet. 
from diffusers import Flux2KleinPipeline
pipeline = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
