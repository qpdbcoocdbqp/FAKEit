import re
import json
import requests
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from rich.console import Console
from PIL import Image


console = Console()

class GeminiFamily():
    embedding_model_ids = [
        "google/embeddinggemma-300m"
        ]

    text_model_ids = [
        "google/functiongemma-270m-it",
        "google/gemma-3-270m-it",
        "google/gemma-3-270m-it-qat-q4_0-unquantized"
        ]

    image_text_model_ids = [
        "google/t5gemma-2-270m-270m"
        ]
    
    def __init__(self,
                 local_files_only: bool=True,
                 enable_embedding: bool=False,
                 enable_text_generate: bool=False,
                 enable_image_text_generate: bool=False,
                 text_model_id: str=None,
                 markers: List[str]=["[DEBUG]", "[INFO]", "[WARNING]", "[ERROR]", "[MESSAGE]"]
                 ):
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
            self._init_text_model(model_id=text_model_id)
        if enable_image_text_generate:
            self._init_image_text_model()
        pass

    def _match_model(self, model_id: str, family: List[str]) -> List[str]:
        return list(filter(lambda x: model_id in x, family))

    def _init_embedding_model(self, model_id: str=None) -> None:
        try:
            console.print(self.markers[1], f"Model loaded {GeminiFamily.embedding_model_ids[0]}")
            self.embedding_model = SentenceTransformer(GeminiFamily.embedding_model_ids[0], device="cuda" if torch.cuda.is_available() else "cpu", local_files_only=self.local_files_only)
            self.embedding_model = self.embedding_model.eval()
        except Exception as error_msg:
            console.print(self.markers[3], f"Error: {error_msg}")
        pass

    def _init_text_model(self, model_id: str=None) -> None:
        try:
            load_model_id = GeminiFamily.text_model_ids[-1]
            if model_id:
                load_model_id = self._match_model(model_id=model_id, family=GeminiFamily.text_model_ids)
                if load_model_id:
                    load_model_id = load_model_id[0]

            console.print(self.markers[1], f"Model loaded {load_model_id}")
            self.text_processor = AutoProcessor.from_pretrained(load_model_id, device_map="auto", local_files_only=self.local_files_only)
            self.text_model = AutoModelForCausalLM.from_pretrained(load_model_id, dtype="auto", device_map="auto", local_files_only=self.local_files_only)
            self.text_model = self.text_model.eval()
            del load_model_id
        except Exception as error_msg:
            console.print(self.markers[3], f"Error: {error_msg}")
        pass

    def _init_image_text_model(self, model_id: str=None) -> None:
        try:
            console.print(self.markers[1], f"Model loaded {GeminiFamily.image_text_model_ids[0]}")
            self.image_text_processor = AutoProcessor.from_pretrained(GeminiFamily.image_text_model_ids[0], device_map="auto", use_fast=True, local_files_only=self.local_files_only)
            self.image_text_model = AutoModelForSeq2SeqLM.from_pretrained(GeminiFamily.image_text_model_ids[0], dtype="auto", device_map="auto", local_files_only=self.local_files_only)
            self.image_text_model = self.image_text_model.eval()
        except Exception as error_msg:
            console.print(self.markers[3], f"Error: {error_msg}")
        pass

    def similarity(self, query: str, documents: List[str], dim: int=768, normalize_embeddings: bool=True):
        with torch.inference_mode():
            query_embeddings = self.embedding_model.encode_query(query, truncate_dim=dim, normalize_embeddings=normalize_embeddings)
            document_embeddings = self.embedding_model.encode_document(documents, truncate_dim=dim, normalize_embeddings=normalize_embeddings)
            similarities = self.embedding_model.similarity(query_embeddings, document_embeddings)
        console.print(self.markers[1], f"Embeddings shape: query: {query_embeddings.shape}, document: {document_embeddings.shape}" )
        console.print(self.markers[1], f"Similarities: {similarities}")
        return similarities, query_embeddings, document_embeddings

    def encode(self, content: str, prompt: str=None, dim: int=768, normalize_embeddings: bool=True):
        embedding = self.embedding_model.encode(
            content, prompt=prompt, truncate_dim=dim, normalize_embeddings=normalize_embeddings
            )
        return embedding

    def generate(self, messages: List[Dict], max_new_tokens: int=128):
        processor = self.text_processor if self.text_processor else self.image_text_processor
        model = self.text_model if self.text_model else self.image_text_model
        with torch.inference_mode():
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
                )
            outputs = model.generate(
                **inputs.to(model.device),
                pad_token_id=processor.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True
                )
            response = processor.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
                )
        del processor, model, inputs
        return response, outputs

    def parse_tool_calls(self, content):
        tool_calls = re.findall(r'<start_function_call>call:(.*?)\{(.*?)\}<end_function_call>', content)
        if len(tool_calls) == 0:
            del tool_calls
            return {"tool_calls": []}
        result = {"tool_calls": []}
        for name, args_raw in tool_calls:
            args_json = "{" + args_raw.replace('<escape>', '"') + "}"
            args_json = re.sub(r'(\w+):', r'"\1":', args_json)
            try:
                args = json.loads(args_json)
            except:
                args = args_raw
            result["tool_calls"].append({
                "type": "function",
                "function": {"name": name, "arguments": args}
            })
            del name, args_raw, args_json, args
        del tool_calls
        return result

    def generate_function(self, messages: List[Dict], schema: Dict, max_new_tokens: int=128):
        processor = self.text_processor if self.text_processor else self.image_text_processor
        model = self.text_model if self.text_model else self.image_text_model
        with torch.inference_mode():
            inputs = processor.apply_chat_template(
                messages,
                tools=[schema],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
                )
            outputs = model.generate(
                **inputs.to(model.device),
                pad_token_id=processor.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True
                )
            response = processor.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
                )
        tool_calls = self.parse_tool_calls(response)
        del processor, model
        return tool_calls, outputs

    def generate_image_text(self, prompts: str, image_url: str=None, max_new_tokens: int=128):
        image = Image.open(requests.get(image_url, stream=True).raw) if image_url else None
        with torch.inference_mode():
            inputs = self.image_text_processor(
                text=prompts,
                images=image,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
                )
            outputs = self.image_text_model.generate(
                **inputs.to(self.image_text_model.device),
                max_new_tokens=max_new_tokens,
                do_sample=True
                )
            response = self.image_text_processor.batch_decode(
                outputs,
                skip_special_tokens=True
                )[0]
        del image, inputs
        return response, outputs
