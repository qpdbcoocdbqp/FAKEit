import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import AutoModel, AutoTokenizer
import torch
# from transformers.models.deepseek_v2 import DeepseekV2Config, DeepseekV2Model, DeepseekV2ForCausalLM
from script.DeepSeekOCR.modeling_deepseekocr import DeepseekOCRForCausalLM


model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)

# ds_config = DeepseekV2Config()

model = DeepseekOCRForCausalLM.from_pretrained(
    model_name, _attn_implementation='eager',
    device_map="auto",
    offload_buffers=True,
    # offload_folder="./script/offloads",
    trust_remote_code=True,
    use_safetensors=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16
    )

model = model.eval()
type(model)

# org = model = AutoModel.from_pretrained(
#     model_name, _attn_implementation='eager',
#     trust_remote_code=True, use_safetensors=True, local_files_only=True,
#     device_map="cpu", offload_buffers=True,
#     torch_dtype=torch.bfloat16
#     )

# model = org.eval()

prompt = "<image>\nDescribe details in the image."
image_file = "script/bee.jpg"
output_path = "script/output"

# Gundam: base_size = 1024, image_size = 640, crop_mode = True
res = model.infer(
    tokenizer, prompt=prompt, image_file=image_file, output_path=output_path,
    base_size=1024, image_size = 640,
    crop_mode=True, save_results=True, test_compress=True, eval_mode=True
    )

print(res)

nano_res = model.infer(
    tokenizer, prompt=prompt, image_file=image_file, output_path=output_path,
    base_size=64, image_size=64,
    crop_mode=False, save_results=True, test_compress=True, eval_mode=True
    )
print(nano_res)

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
