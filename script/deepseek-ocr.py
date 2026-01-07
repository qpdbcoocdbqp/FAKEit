import os
import torch
from PIL import Image
from script.utils import console, ThemeConsole


theme_console = ThemeConsole()
color_markers = theme_console.get_random_markers()
console.print(*color_markers)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable device-side assertions for better error messages

# Gundam: base_size = 1024, image_size = 640, crop_mode = True
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

def test_base_infer(tokenizer, model, prompt: str, image_file: str):
    try:
        base_result = model.infer(
            tokenizer=tokenizer,
            prompt=prompt,
            image_file=image_file,
            base_size=1024,
            image_size=1024,
            crop_mode=False,
            eval_mode=True,
            )
        console.print(color_markers[1], base_result)
    except Exception as error:
        console.print(color_markers[3], error)
    pass

# Text Embeddings
def test_text_embeddings(tokenizer, model, text: str):
    try:
        with torch.inference_mode():
            text_embeddings = model.get_text_embeddings(
                tokenizer(text, return_tensors="pt").input_ids
                )
        console.print(color_markers[1], f"Text embeddings shape: {text_embeddings.shape}")
    except Exception as error:
        console.print(color_markers[3], error)
    pass

# Image Embeddings
def test_image_embeddings(model, image_file: str, spatial_crop: bool=False):
    try:
        console.print(color_markers[1], f"Image opened: {image_file}")
        pil_image = Image.open(image_file).convert("RGB")

        with torch.inference_mode():
            # Preprocess image
            image_data = model.image_processor.process_image(pil_image, crop_mode=spatial_crop)
            # Prepare tensors for DeepSeek-OCR format
            images_ori = torch.stack([image_data['images_list'][0]], dim=0).to(model.device, model.dtype)
            if image_data['images_crop_list']:
                images_crop = torch.stack(image_data['images_crop_list'], dim=0).to(model.device, model.dtype)
            else:
                # Default zero tensor for crops if spatial_crop is False
                crop_size = model.processing_config.image_size
                images_crop = torch.zeros((1, 3, crop_size, crop_size), device=model.device, dtype=model.dtype)

            images_tensor = [(images_crop, images_ori)]
            images_spatial_crop = torch.tensor([image_data['crop_ratio']], dtype=torch.long, device=model.device)
            image_embeddings = model.get_image_embeddings(images_tensor, images_spatial_crop)
        console.print(color_markers[1], f"Image items: {len(image_embeddings)}")
        console.print(color_markers[1], f"Image embeddings shape: {image_embeddings[0].shape}")
    except Exception as error:
        console.print(color_markers[3], error)
    pass

# Multimodal Embeddings
def test_multimodal_embeddings(tokenizer, model, prompt: str, image_file: str, spatial_crop: bool=False):
    try:
        console.print(color_markers[1], f"Image opened: {image_file}")
        pil_image = Image.open(image_file).convert("RGB")

        with torch.inference_mode():
            image_data = model.image_processor.process_image(pil_image, crop_mode=spatial_crop)
            images_ori = torch.stack([image_data['images_list'][0]], dim=0).to(model.device, model.dtype)
            if image_data['images_crop_list']:
                images_crop = torch.stack(image_data['images_crop_list'], dim=0).to(model.device, model.dtype)
            else:
                # Default zero tensor for crops if spatial_crop is False
                crop_size = model.processing_config.image_size
                images_crop = torch.zeros((1, 3, crop_size, crop_size), device=model.device, dtype=model.dtype)
            images_tensor = [(images_crop, images_ori)]

            # Use text_processor to create the correct sequence with image tokens
            input_data = model.text_processor.create_input_sequence(
                tokenizer, prompt, [image_data]
                )
            # Move and prepare inputs for the model
            input_ids = input_data['input_ids'].unsqueeze(0).to(model.device)
            images_seq_mask = input_data['images_seq_mask'].unsqueeze(0).to(model.device)
            images_spatial_crop_input = input_data['images_spatial_crop'].to(model.device)
            
            # get_multimodal_embeddings expects: input_ids, images, images_seq_mask, images_spatial_crop
            multimodal_embeddings = model.get_multimodal_embeddings(
                input_ids=input_ids,
                images=images_tensor,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop_input
            )
        console.print(color_markers[1], f"Multimodal embeddings calculated, shape: {multimodal_embeddings.shape}")
    except Exception as error:
        console.print(color_markers[3], error)
    pass


# --- main ---
if __name__ == "__main__":
    import gc
    from transformers import AutoTokenizer
    from script.DeepSeekOCR.modeling_deepseekocr_refactored import DeepseekOCRForCausalLM


    # prepare inputs
    text = "Describe details in the image."
    prompt = "<image>\nDescribe details in the image."
    image_file = "script/dev-bee.jpg"
    spatial_crop = False

    for model_name in ["deepseek-ai/DeepSeek-OCR", "Jalea96/DeepSeek-OCR-bnb-4bit-NF4"]:
        console.print(color_markers[1], f"Load model {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        model = DeepseekOCRForCausalLM.from_pretrained(
            model_name,
            _attn_implementation='eager',
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            local_files_only=True
            )
        model = model.eval()

        # test embeddings
        test_text_embeddings(tokenizer, model, text=text)
        test_image_embeddings(model, image_file=image_file, spatial_crop=spatial_crop)
        test_multimodal_embeddings(tokenizer, model, prompt=prompt, image_file=image_file, spatial_crop=spatial_crop)

        # test base infer
        # base_size=1024
        # image_size=1024
        # crop_mode=False
        # eval_mode=True
        test_base_infer(tokenizer, model, prompt=prompt, image_file=image_file)

        model = None
        gc.collect()
        torch.cuda.empty_cache()
        console.print(color_markers[1], "VRAM released")