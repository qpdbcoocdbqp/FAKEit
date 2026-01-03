import re
import sys
import os
import random
from pathlib import Path
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

class ThemeConsole:
    def __init__(self, markers: List[str] = None):
        self.markers = markers or ["[DEBUG]", "[INFO]", "[WARNING]", "[ERROR]", "[MESSAGE]"]
        self._setup_path()
        try:
            from ocean_pearl.src.theme import prettyderby
            self.theme = prettyderby()
        except ImportError:
            self.theme = {}

    def _setup_path(self):
        try:
            # Assumes utils.py is in script/ folder, and submodules is in parent/submodules
            base_dir = Path(__file__).resolve().parent.parent
            module_path = os.path.join(base_dir, "submodules")
            if module_path not in sys.path:
                sys.path.append(module_path)
        except NameError:
            pass

    def get_random_markers(self) -> List[str]:
        if not self.theme:
            return self.markers
        
        characters = list(self.theme.keys())
        if not characters:
            return self.markers
            
        character = random.choice(characters)
        colors = self.theme.get(character, [])
        
        colored_markers = []
        for i, mark in enumerate(self.markers):
            if i < len(colors):
                hex_color = colors[i].get("hex", "")
                if hex_color:
                    colored_markers.append(f"[{hex_color}]{mark}[/{hex_color}]")
                else:
                    colored_markers.append(mark)
            else:
                colored_markers.append(mark)
        console.print(f"[ThemeConsole]: {character}", *colored_markers)
        return colored_markers

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
                offload_folder="offload",
                offload_buffers=True,
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

    def generate(self, messages: List[Dict] = None, prompts: str = None, image_url: str = None, schema: Dict = None, max_new_tokens: int = 128):
        # 1. Determine Mode and Load Function
        has_inline_image = False
        if messages:
            has_inline_image = any(
                part.get("type") == "image"
                for msg in messages
                if isinstance(msg.get("content"), list)
                for part in msg["content"]
            )
            
        is_image = has_inline_image or bool(image_url)
        if is_image:
            if not self.image_text_model: return None, None
            processor, model = self.image_text_processor, self.image_text_model
        else:
            processor, model = self._get_active_model()
            if not model: return None, None

        eos_token_id = getattr(processor, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = processor.tokenizer.eos_token_id
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": eos_token_id,
            "do_sample": True
            }
        
        with torch.inference_mode():
            # 2. Prepare Inputs
            template_kwargs = {"add_generation_prompt": True, "tokenize": True, "return_dict": True, "return_tensors": "pt"}
            if schema:
                template_kwargs.update({"tools": [schema]})
            if messages:
                inputs = processor.apply_chat_template(messages, **template_kwargs)
            else:
                template_kwargs.update({"text": prompts})
                if is_image:
                    image = Image.open(requests.get(image_url, stream=True).raw) if image_url else None
                    template_kwargs.update({"images": image})
                inputs = processor(**template_kwargs)
            inputs = inputs.to(model.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

            # 3. Generate
            outputs = model.generate(**inputs, **gen_kwargs)

            # 4. Decode
            decode_tokens = outputs[0]
            if not is_image:
                decode_tokens = decode_tokens[inputs["input_ids"].shape[1]:]
            
            response = processor.decode(decode_tokens, skip_special_tokens=True)

        # 5. Post-processing
        if schema and not is_image:
            return self.parse_tool_calls(response), outputs
            
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
