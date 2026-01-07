from .modeling_deepseekv2 import DeepseekV2Model, DeepseekV2ForCausalLM
from .configuration_deepseek_v2 import DeepseekV2Config
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers.cache_utils import Cache
from PIL import Image, ImageOps, ImageDraw, ImageFont
from io import BytesIO
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import os
from .deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from addict import Dict as AdictDict
from transformers import TextStreamer
from .conversation import get_conv_template
from abc import ABC
import math
import re
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """Configuration for image and text processing"""
    patch_size: int = 16
    downsample_ratio: int = 4
    image_token: str = '<image>'
    image_token_id: int = 128815
    bos_id: int = 0
    eos_id: int = 1
    base_size: int = 1024
    image_size: int = 640
    min_crop_num: int = 2
    max_crop_num: int = 9
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)


class ImageProcessor:
    """Modular image preprocessing component"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """Create image transformation pipeline"""
        transform_pipelines = [transforms.ToTensor()]
        
        if self.config.mean is not None and self.config.std is not None:
            normalize = transforms.Normalize(mean=self.config.mean, std=self.config.std)
            transform_pipelines.append(normalize)
        
        return transforms.Compose(transform_pipelines)
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and correct image orientation"""
        try:
            image = Image.open(image_path)
            corrected_image = ImageOps.exif_transpose(image)
            return corrected_image.convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            try:
                return Image.open(image_path).convert("RGB")
            except:
                return None
    
    def find_closest_aspect_ratio(self, aspect_ratio: float, target_ratios: List[Tuple[int, int]], 
                                 width: int, height: int) -> Tuple[int, int]:
        """Find the closest aspect ratio from target ratios"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * self.config.image_size * self.config.image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        
        return best_ratio
    
    def dynamic_preprocess(self, image: Image.Image, use_thumbnail: bool = False) -> Tuple[List[Image.Image], Tuple[int, int]]:
        """Dynamically preprocess image into patches"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(self.config.min_crop_num, self.config.max_crop_num + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= self.config.max_crop_num and i * j >= self.config.min_crop_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find closest aspect ratio
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height
        )
        
        # Calculate target dimensions
        target_width = self.config.image_size * target_aspect_ratio[0]
        target_height = self.config.image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize and split image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // self.config.image_size)) * self.config.image_size,
                (i // (target_width // self.config.image_size)) * self.config.image_size,
                ((i % (target_width // self.config.image_size)) + 1) * self.config.image_size,
                ((i // (target_width // self.config.image_size)) + 1) * self.config.image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self.config.image_size, self.config.image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images, target_aspect_ratio
    
    def process_image(self, image: Image.Image, crop_mode: bool = True) -> Dict[str, Any]:
        """Process a single image and return processed data"""
        w, h = image.size
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
        
        processed_data = {
            'original_size': (w, h),
            'ratio': ratio,
            'images_list': [],
            'images_crop_list': [],
            'crop_ratio': [1, 1],
            'valid_img_tokens': 0
        }
        
        if crop_mode and (image.size[0] > 640 or image.size[1] > 640):
            # Dynamic preprocessing for large images
            images_crop_raw, crop_ratio = self.dynamic_preprocess(image)
            processed_data['crop_ratio'] = crop_ratio
            
            # Process local views
            for crop_img in images_crop_raw:
                crop_tensor = self.transform(crop_img)
                processed_data['images_crop_list'].append(crop_tensor)
            
            if self.config.image_size == 640:
                processed_data['valid_img_tokens'] += len(images_crop_raw) * 100
        
        # Process global view
        global_view = ImageOps.pad(
            image, 
            (self.config.base_size, self.config.base_size),
            color=tuple(int(x * 255) for x in self.config.mean)
        )
        
        global_tensor = self.transform(global_view)
        processed_data['images_list'].append(global_tensor)
        
        # Calculate valid image tokens for global view
        if self.config.base_size == 1024:
            processed_data['valid_img_tokens'] += int(256 * ratio)
        elif self.config.base_size == 1280:
            processed_data['valid_img_tokens'] += int(400 * ratio)
        elif self.config.base_size == 640:
            processed_data['valid_img_tokens'] += int(100 * 1)
        elif self.config.base_size == 512:
            processed_data['valid_img_tokens'] += int(64 * 1)
        
        return processed_data


class TextProcessor:
    """Modular text preprocessing component"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def text_encode(self, tokenizer, text: str, bos: bool = True, eos: bool = False) -> List[int]:
        """Encode text to token IDs"""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if bos:
            tokens = [self.config.bos_id] + tokens
        if eos:
            tokens = tokens + [self.config.eos_id]
        return tokens
    
    def format_messages(self, conversations: List[Dict[str, str]], 
                       sft_format: str = "deepseek", system_prompt: str = "") -> str:
        """Format conversation messages"""
        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        for message in conversations:
            conv.append_message(message["role"], message["content"].strip())
        return conv.get_prompt().strip()
    
    def load_pil_images(self, conversations: List[Dict[str, str]], 
                       image_processor: ImageProcessor) -> List[Image.Image]:
        """Load PIL images from conversation"""
        pil_images = []
        for message in conversations:
            if "images" not in message:
                continue
            for image_path in message["images"]:
                pil_img = image_processor.load_image(image_path)
                if pil_img is not None:
                    pil_images.append(pil_img)
        return pil_images
    
    def create_input_sequence(self, tokenizer, prompt: str, images_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Create input sequence with text and image tokens"""
        text_splits = prompt.split(self.config.image_token)
        tokenized_str = []
        images_seq_mask = []
        images_spatial_crop = []
        
        # Check if image token ID is valid
        vocab_size = tokenizer.vocab_size
        if self.config.image_token_id >= vocab_size:
            print(f"Warning: image_token_id {self.config.image_token_id} >= vocab_size {vocab_size}")
            # Use a safe token ID (like unk_token_id)
            safe_image_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
            print(f"Using safe_image_token_id: {safe_image_token_id}")
        else:
            safe_image_token_id = self.config.image_token_id
        
        for i, (text_sep, img_data) in enumerate(zip(text_splits[:-1], images_data)):
            # Add text tokens
            tokenized_sep = self.text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)
            
            # Add image tokens
            crop_ratio = img_data['crop_ratio']
            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])
            
            num_queries = math.ceil((self.config.image_size // self.config.patch_size) / self.config.downsample_ratio)
            num_queries_base = math.ceil((self.config.base_size // self.config.patch_size) / self.config.downsample_ratio)

            # Create image token sequence
            tokenized_image = ([safe_image_token_id] * num_queries_base + [safe_image_token_id]) * num_queries_base
            tokenized_image += [safe_image_token_id]
            
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([safe_image_token_id] * (num_queries * width_crop_num) + [safe_image_token_id]) * (num_queries * height_crop_num)
                        
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
        
        # Add final text split
        tokenized_sep = self.text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)
        
        # Add BOS token
        tokenized_str = [self.config.bos_id] + tokenized_str
        images_seq_mask = [False] + images_seq_mask
        
        return {
            'input_ids': torch.LongTensor(tokenized_str),
            'images_seq_mask': torch.tensor(images_seq_mask, dtype=torch.bool),
            'images_spatial_crop': torch.tensor(images_spatial_crop, dtype=torch.long)
        }


class OutputProcessor:
    """Modular output post-processing component"""
    
    def __init__(self):
        pass
    
    def re_match(self, text: str) -> Tuple[List, List, List]:
        """Extract reference patterns from text"""
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        matches_image = []
        matches_other = []
        for a_match in matches:
            if '<|ref|>image<|/ref|>' in a_match[0]:
                matches_image.append(a_match[0])
            else:
                matches_other.append(a_match[0])
        
        return matches, matches_image, matches_other
    
    def extract_coordinates_and_label(self, ref_text: Tuple, image_width: int, image_height: int) -> Optional[Tuple]:
        """Extract coordinates and labels from reference text"""
        try:
            label_type = ref_text[1]
            cor_list = eval(ref_text[2])
            return (label_type, cor_list)
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            return None
    
    def draw_bounding_boxes(self, image: Image.Image, refs: List, output_path: str) -> Image.Image:
        """Draw bounding boxes on image"""
        image_width, image_height = image.size
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()
        
        img_idx = 0
        
        for i, ref in enumerate(refs):
            try:
                result = self.extract_coordinates_and_label(ref, image_width, image_height)
                if result:
                    label_type, points_list = result
                    color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                    color_a = color + (20,)
                    
                    for points in points_list:
                        x1, y1, x2, y2 = points
                        
                        # Normalize coordinates
                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)
                        
                        if label_type == 'image':
                            try:
                                cropped = image.crop((x1, y1, x2, y2))
                                os.makedirs(f"{output_path}/images", exist_ok=True)
                                cropped.save(f"{output_path}/images/{img_idx}.jpg")
                                img_idx += 1
                            except Exception as e:
                                print(f"Error saving cropped image: {e}")
                        
                        # Draw bounding box
                        try:
                            width = 4 if label_type == 'title' else 2
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                            
                            # Draw label
                            text_x = x1
                            text_y = max(0, y1 - 15)
                            text_bbox = draw.textbbox((0, 0), label_type, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            
                            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                         fill=(255, 255, 255, 30))
                            draw.text((text_x, text_y), label_type, font=font, fill=color)
                        except Exception as e:
                            print(f"Error drawing bounding box: {e}")
            except Exception as e:
                print(f"Error processing reference: {e}")
                continue
        
        img_draw.paste(overlay, (0, 0), overlay)
        return img_draw
    
    def process_output(self, output_text: str, original_image: Image.Image, 
                      output_path: str) -> Dict[str, Any]:
        """Process model output and generate results"""
        # Clean output text
        stop_str = '<｜end▁of▁sentence｜>'
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)]
        output_text = output_text.strip()
        
        # Extract references
        matches_ref, matches_images, matches_other = self.re_match(output_text)
        
        # Process image with references
        result_image = self.draw_bounding_boxes(original_image, matches_ref, output_path)
        
        # Replace image references with markdown
        processed_text = output_text
        for idx, a_match_image in enumerate(matches_images):
            processed_text = processed_text.replace(a_match_image, f'![](images/{idx}.jpg)\n')
        
        # Clean other references
        for a_match_other in matches_other:
            processed_text = processed_text.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
        
        # Save results
        os.makedirs(output_path, exist_ok=True)
        with open(f'{output_path}/result.mmd', 'w', encoding='utf-8') as f:
            f.write(processed_text)
        
        result_image.save(f"{output_path}/result_with_boxes.jpg")
        
        return {
            'processed_text': processed_text,
            'result_image': result_image,
            'matches': matches_ref,
            'output_path': output_path
        }


class NoEOSTextStreamer(TextStreamer):
    """Custom text streamer without EOS tokens"""
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")


class DeepseekOCRConfig(DeepseekV2Config):
    model_type = "DeepseekOCR"


class DeepseekOCRModel(DeepseekV2Model):
    config_class = DeepseekOCRConfig

    def __init__(self, config: DeepseekV2Config):
        super(DeepseekOCRModel, self).__init__(config)

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        n_embed = 1280
        self.projector = MlpProjector(AdictDict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        
    def _ensure_vision_models_dtype(self):
        """Ensure vision models have the same dtype as the main model"""
        model_dtype = next(self.parameters()).dtype
        if hasattr(self, 'sam_model') and self.sam_model is not None:
            self.sam_model = self.sam_model.to(dtype=model_dtype)
        if hasattr(self, 'vision_model') and self.vision_model is not None:
            self.vision_model = self.vision_model.to(dtype=model_dtype)

    def get_text_embeddings(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Get text embeddings directly"""
        return self.get_input_embeddings()(input_ids)
    
    def get_image_embeddings(self, images: torch.FloatTensor, 
                           images_spatial_crop: torch.LongTensor) -> torch.FloatTensor:
        """Get image embeddings directly"""
        sam_model = getattr(self, 'sam_model', None)
        vision_model = getattr(self, 'vision_model', None)
        
        if sam_model is None or vision_model is None:
            raise ValueError("Vision models not initialized")
        
        # Ensure vision models have correct dtype
        self._ensure_vision_models_dtype()
        
        # Get model dtype for consistent processing
        model_dtype = next(self.parameters()).dtype
        
        batch_embeddings = []
        
        for image, crop_shape in zip(images, images_spatial_crop):
            patches = image[0].to(dtype=model_dtype)  # Convert to model dtype
            image_ori = image[1].to(dtype=model_dtype)  # Convert to model dtype
            
            with torch.no_grad():
                if torch.sum(patches).item() != 0:
                    # Process local features
                    local_features_1 = sam_model(patches)
                    local_features_2 = vision_model(patches, local_features_1)
                    local_features = torch.cat((local_features_2[:, 1:], 
                                              local_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                    local_features = self.projector(local_features)
                    
                    # Process global features
                    global_features_1 = sam_model(image_ori)
                    global_features_2 = vision_model(image_ori, global_features_1)
                    global_features = torch.cat((global_features_2[:, 1:], 
                                               global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                    global_features = self.projector(global_features)
                    
                    # Reshape and combine features
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)
                    
                    _, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2 ** 0.5)
                    
                    width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]
                    
                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat([global_features, 
                                               self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1)
                    global_features = global_features.view(-1, n_dim)
                    
                    local_features = local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
                    local_features = local_features.permute(0, 2, 1, 3, 4).reshape(height_crop_num*h2, width_crop_num*w2, n_dim2)
                    local_features = torch.cat([local_features, 
                                              self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1)
                    local_features = local_features.view(-1, n_dim2)
                    
                    combined_features = torch.cat([local_features, global_features, 
                                                 self.view_seperator[None, :]], dim=0)
                else:
                    # Only global features
                    global_features_1 = sam_model(image_ori)
                    global_features_2 = vision_model(image_ori, global_features_1)
                    global_features = torch.cat((global_features_2[:, 1:], 
                                               global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                    global_features = self.projector(global_features)
                    
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)
                    
                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat([global_features, 
                                               self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1)
                    global_features = global_features.view(-1, n_dim)
                    
                    combined_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
                
                batch_embeddings.append(combined_features)
        
        return batch_embeddings
    
    def get_multimodal_embeddings(self, input_ids: torch.LongTensor, 
                                 images: Optional[torch.FloatTensor] = None,
                                 images_seq_mask: Optional[torch.BoolTensor] = None,
                                 images_spatial_crop: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """Get combined text and image embeddings"""
        inputs_embeds = self.get_text_embeddings(input_ids)
        
        if images is not None and images_seq_mask is not None and torch.sum(images[0][1]).item() != 0:
            image_embeddings = self.get_image_embeddings(images, images_spatial_crop)
            
            for idx, img_embeds in enumerate(image_embeddings):
                if img_embeds is not None:
                    mask = images_seq_mask[idx]  # Shape: [seq_len]
                    
                    if inputs_embeds[idx].shape[0] == mask.shape[0]:
                        mask_positions = torch.where(mask)[0]
                        
                        if len(mask_positions) == img_embeds.shape[0]:
                            inputs_embeds[idx][mask_positions] = img_embeds
                        else:
                            print(f"Warning: mask positions ({len(mask_positions)}) != img_embeds size ({img_embeds.shape[0]})")
                            if len(mask_positions) > img_embeds.shape[0]:
                                inputs_embeds[idx][mask_positions[:img_embeds.shape[0]]] = img_embeds
                            else:
                                inputs_embeds[idx][mask_positions] = img_embeds[:len(mask_positions)]
                    else:
                        pass
        
        return inputs_embeds

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
                images: Optional[torch.FloatTensor] = None, images_seq_mask: Optional[torch.FloatTensor] = None,
                images_spatial_crop: Optional[torch.FloatTensor] = None, return_dict: Optional[bool] = None,
                ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.get_multimodal_embeddings(
                input_ids, images, images_seq_mask, images_spatial_crop
            )

        return super(DeepseekOCRModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids=position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class DeepseekOCRForCausalLM(DeepseekV2ForCausalLM):
    config_class = DeepseekOCRConfig

    def __init__(self, config):
        super(DeepseekV2ForCausalLM, self).__init__(config)
        self.model = DeepseekOCRModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize processors
        self.processing_config = ProcessingConfig()
        self.image_processor = ImageProcessor(self.processing_config)
        self.text_processor = TextProcessor(self.processing_config)
        # self.output_processor = OutputProcessor()
        
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_text_embeddings(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Direct access to text embeddings"""
        return self.model.get_text_embeddings(input_ids)
    
    def get_image_embeddings(self, images: torch.FloatTensor, 
                           images_spatial_crop: torch.LongTensor) -> torch.FloatTensor:
        """Direct access to image embeddings"""
        return self.model.get_image_embeddings(images, images_spatial_crop)
    
    def get_multimodal_embeddings(self, input_ids: torch.LongTensor, 
                                 images: Optional[torch.FloatTensor] = None,
                                 images_seq_mask: Optional[torch.BoolTensor] = None,
                                 images_spatial_crop: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """Direct access to multimodal embeddings"""
        return self.model.get_multimodal_embeddings(input_ids, images, images_seq_mask, images_spatial_crop)

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, images: Optional[torch.FloatTensor] = None,
                images_seq_mask: Optional[torch.FloatTensor] = None, images_spatial_crop: Optional[torch.FloatTensor] = None,
                return_dict: Optional[bool] = None) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            position_ids=position_ids, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            images=images, images_seq_mask=images_seq_mask, images_spatial_crop=images_spatial_crop,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, 
                                    inputs_embeds=None, **kwargs):
        """Prepare inputs for generation with device-aware processing"""
        # Standard generation input preparation
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (max_cache_length is not None and attention_mask is not None and 
                cache_length + input_ids.shape[1] > max_cache_length):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], 
                                    device=position_ids.device)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images", None),
            "images_seq_mask": kwargs.get("images_seq_mask", None),
            "images_spatial_crop": kwargs.get("images_spatial_crop", None),
        })
        return model_inputs

    def infer(self, tokenizer, prompt: str = '', image_file: str = '', 
             base_size: int = 1024, image_size: int = 640, crop_mode: bool = True, 
             eval_mode: bool = False, **generation_kwargs) -> str:
        """
        Modular inference method with device-aware processing
        
        Args:
            tokenizer: Tokenizer for text processing
            prompt: Text prompt
            image_file: Path to image file
            base_size: Base image size for global view
            image_size: Image size for local views
            crop_mode: Whether to use dynamic cropping
            eval_mode: Evaluation mode flag
            **generation_kwargs: Additional generation parameters
        """
        # Update processing config
        self.processing_config.base_size = base_size
        self.processing_config.image_size = image_size
        
        # Create conversation
        if prompt and image_file:
            conversation = [
                {"role": "<|User|>", "content": prompt, "images": [image_file]},
                {"role": "<|Assistant|>", "content": ""},
            ]
        elif prompt:
            conversation = [
                {"role": "<|User|>", "content": prompt},
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            raise ValueError("Prompt cannot be empty")
        
        # Format prompt
        formatted_prompt = self.text_processor.format_messages(conversation, sft_format='plain')
        
        # Load and process images
        images = self.text_processor.load_pil_images(conversation, self.image_processor)
        images_data = []
        
        if images:
            for image in images:
                img_data = self.image_processor.process_image(image, crop_mode)
                images_data.append(img_data)
        
        # Create input sequence
        input_data = self.text_processor.create_input_sequence(
            tokenizer, formatted_prompt, images_data
        )
        
        # Move input data to model device
        model_device = self.model.device
        input_data['input_ids'] = input_data['input_ids'].to(device=model_device)
        input_data['images_seq_mask'] = input_data['images_seq_mask'].to(device=model_device)
        input_data['images_spatial_crop'] = input_data['images_spatial_crop'].to(device=model_device)

        # Prepare image tensors
        if images_data:
            images_ori = torch.stack([data['images_list'][0] for data in images_data], dim=0)
            
            if any(data['images_crop_list'] for data in images_data):
                all_crops = []
                for data in images_data:
                    all_crops.extend(data['images_crop_list'])
                if all_crops:
                    images_crop = torch.stack(all_crops, dim=0)
                else:
                    images_crop = torch.zeros((1, 3, base_size, base_size), device=self.model.device, dtype=next(self.model.parameters()).dtype)
            else:
                images_crop = torch.zeros((1, 3, base_size, base_size), device=self.model.device, dtype=next(self.model.parameters()).dtype)
            
            # Move tensors to model device and dtype
            model_device = self.model.device
            model_dtype = next(self.model.parameters()).dtype
            
            images_ori = images_ori.to(device=model_device, dtype=model_dtype)
            images_crop = images_crop.to(device=model_device, dtype=model_dtype)
            
            images_tensor = [(images_crop, images_ori)]
        else:
            model_device = self.model.device
            model_dtype = next(self.model.parameters()).dtype
            images_tensor = [(torch.zeros((1, 3, base_size, base_size), device=model_device, dtype=model_dtype),
                            torch.zeros((1, 3, image_size, image_size), device=model_device, dtype=model_dtype))]
        
        # Generation parameters
        gen_kwargs = {
            'max_new_tokens': 8192,
            'use_cache': True,
            'do_sample': eval_mode,
            'temperature': 0.0 if not eval_mode else 1.0,
            'eos_token_id': tokenizer.eos_token_id,
            'no_repeat_ngram_size': 20 if not eval_mode else 35,
        }
        gen_kwargs.update(generation_kwargs)
        
        # Generate
        if not eval_mode:
            streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            gen_kwargs['streamer'] = streamer
        
        with torch.no_grad():
            output_ids = self.generate(
                input_data['input_ids'].unsqueeze(0),
                images=images_tensor,
                images_seq_mask=input_data['images_seq_mask'].unsqueeze(0),  # Keep as bool
                images_spatial_crop=input_data['images_spatial_crop'],  # Keep as long
                **gen_kwargs
            )

        # Decode output
        output_text = tokenizer.decode(output_ids[0, input_data['input_ids'].shape[0]:], skip_special_tokens=True)

        return output_text.strip()