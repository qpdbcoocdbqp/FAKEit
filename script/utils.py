import re
import json
import gc
import requests
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from rich.console import Console
from PIL import Image


console = Console()

class GeminiFamily:
    EMBEDDING_MODEL_IDS = ["google/embeddinggemma-300m"]
    TEXT_MODEL_IDS = [
        "google/functiongemma-270m-it",
        "google/gemma-3-270m-it",
        "google/gemma-3-270m-it-qat-q4_0-unquantized"
    ]
    IMAGE_TEXT_MODEL_IDS = [
        "google/t5gemma-2-270m-270m",
        "google/gemma-3-4b-it-qat-q4_0-unquantized",
        "google/gemma-3n-e2b-it"
    ]

    def __init__(self,
                 local_files_only: bool = True,
                 enable_embedding: bool = False,
                 enable_text_generate: bool = False,
                 enable_image_text_generate: bool = False,
                 model_id: str = None,
                 markers: List[str] = ["[DEBUG]", "[INFO]", "[WARNING]", "[ERROR]", "[MESSAGE]"]):
        self.local_files_only = local_files_only
        self.markers = markers
        self.embedding_model = None
        self.text_processor = None
        self.text_model = None
        self.image_text_processor = None
        self.image_text_model = None

        if enable_embedding:
            self._init_embedding_model()
        if enable_text_generate:
            self.text_processor, self.text_model = self._load_hf_model(
                model_id, self.TEXT_MODEL_IDS
            )
        if enable_image_text_generate:
            self.image_text_processor, self.image_text_model = self._load_hf_model(
                model_id, self.IMAGE_TEXT_MODEL_IDS
            )

    def _get_model_id(self, model_id: str, family: List[str]) -> str:
        if model_id:
            for m in family:
                if model_id in m:
                    return m
        return family[-1]

    def _load_hf_model(self, model_id: str, family: List[str]):
        try:
            load_id = self._get_model_id(model_id, family)
            console.print(self.markers[1], f"Model loaded {load_id}")
            
            processor = AutoProcessor.from_pretrained(
                load_id, 
                device_map="auto", 
                use_fast=True, 
                local_files_only=self.local_files_only
            )
            
            # Special handling for Seq2Seq models if needed, defaults to CausalLM
            model_cls = AutoModelForSeq2SeqLM if "t5gemma" in load_id else AutoModelForCausalLM
            model = model_cls.from_pretrained(
                load_id, 
                dtype="auto", 
                device_map="auto", 
                local_files_only=self.local_files_only
            ).eval()
            
            return processor, model
        except Exception as error_msg:
            console.print(self.markers[3], f"Error loading {model_id}: {error_msg}")
            return None, None

    def _init_embedding_model(self):
        try:
            mid = self.EMBEDDING_MODEL_IDS[0]
            console.print(self.markers[1], f"Model loaded {mid}")
            self.embedding_model = SentenceTransformer(
                mid, 
                device="cuda" if torch.cuda.is_available() else "cpu", 
                local_files_only=self.local_files_only
            ).eval()
        except Exception as error_msg:
            console.print(self.markers[3], f"Error: {error_msg}")

    def similarity(self, query: str, documents: List[str], dim: int = 768, normalize_embeddings: bool = True):
        with torch.inference_mode():
            q_emb = self.embedding_model.encode_query(query, truncate_dim=dim, normalize_embeddings=normalize_embeddings)
            d_emb = self.embedding_model.encode_document(documents, truncate_dim=dim, normalize_embeddings=normalize_embeddings)
            sims = self.embedding_model.similarity(q_emb, d_emb)
        console.print(self.markers[1], f"Embeddings shape: query: {q_emb.shape}, document: {d_emb.shape}")
        console.print(self.markers[1], f"Similarities: {sims}")
        return sims, q_emb, d_emb

    def encode(self, content: str, prompt: str = None, dim: int = 768, normalize_embeddings: bool = True):
        return self.embedding_model.encode(
            content, prompt=prompt, truncate_dim=dim, normalize_embeddings=normalize_embeddings
        )

    def parse_tool_calls(self, content):
        pattern = r'<start_function_call>call:(.*?)\{(.*?)\}<end_function_call>'
        tool_calls = re.findall(pattern, content)
        if not tool_calls:
            return {"tool_calls": []}
        
        result = {"tool_calls": []}
        for name, args_raw in tool_calls:
            try:
                # Basic string cleanup to form valid JSON
                args_json = "{" + args_raw.replace('<escape>', '"') + "}"
                args_json = re.sub(r'(\w+):', r'"\1":', args_json)
                args = json.loads(args_json)
            except json.JSONDecodeError:
                args = args_raw # Fallback
                
            result["tool_calls"].append({
                "type": "function", 
                "function": {"name": name, "arguments": args}
            })
        return result

    def _get_active_model(self):
        # Prefer text model, fallback to image model
        return (self.text_processor or self.image_text_processor, 
                self.text_model or self.image_text_model)

    def generate(self, messages: List[Dict], schema: Dict = None, max_new_tokens: int = 128):
        processor, model = self._get_active_model()
        if not model: return None, None

        with torch.inference_mode():
            # Prepare kwargs for chat template
            template_kwargs = {
                "add_generation_prompt": True, 
                "return_dict": True, 
                "return_tensors": "pt"
            }
            if schema:
                template_kwargs["tools"] = [schema]

            inputs = processor.apply_chat_template(
                messages, **template_kwargs
            ).to(model.device)
            
            outputs = model.generate(
                **inputs,
                pad_token_id=processor.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True
            )
            # Decode response, skipping input prompt
            response = processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
        if schema:
            return self.parse_tool_calls(response), outputs
            
        return response, outputs

    def generate_image_text(self, messages: List[Dict] = None, prompts: str = None, image_url: str = None, max_new_tokens: int = 128):
        if not self.image_text_model: return None, None
        
        image = Image.open(requests.get(image_url, stream=True).raw) if image_url else None
        processor = self.image_text_processor
        
        with torch.inference_mode():
            if messages:
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                )
            else:
                inputs = processor(text=prompts, images=image, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            
            inputs = inputs.to(self.image_text_model.device, dtype=self.image_text_model.dtype)
            # Handle potential dtype mismatch for image/text
            # if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            #     inputs.pixel_values = inputs.pixel_values.to(dtype=self.image_text_model.dtype)

            outputs = self.image_text_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True
            )
            # Image models sometimes don't echo prompt or handle it differently, keeping batch_decode
            response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
        return response, outputs

    def release_vram(self):
        """Explicitly release memory and CUDA cache."""
        self.embedding_model = None
        self.text_model = None
        self.text_processor = None
        self.image_text_model = None
        self.image_text_processor = None
        
        gc.collect()
        torch.cuda.empty_cache()
        console.print(self.markers[1], "VRAM released")
