import base64
import mimetypes
import random
import requests
from pathlib import Path
from script.utils import GeminiFamily, console, ThemeConsole


theme_console = ThemeConsole()
color_markers = theme_console.get_random_markers()
console.print(*color_markers)


def test_similarity():
    GF = GeminiFamily(enable_embedding=True, markers=color_markers)
    truncate_dim = random.randint(128, 768)
    console.print(color_markers[1], f"Random embedding dimension (128-768): {truncate_dim}")
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
        ]
    console.print(color_markers[1], "Query:", query)
    console.print(color_markers[1], "Documents:", documents)
    similarity, q_emb, d_emb = GF.similarity(query, documents, dim=truncate_dim)
    GF.release_vram()
    pass

def test_multi_embed():
    URL = "http://localhost:19001/v1/embeddings"
    IMAGE_PATH = Path("test.png")

    mime_type = mimetypes.guess_type(IMAGE_PATH.name)[0]
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("請使用 JPEG、PNG 或 WebP 圖片")

    image_base64 = base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
    image_url = f"data:{mime_type};base64,{image_base64}"

    text_part = {
        "type": "text",
        "text": "title: none | text: EmbeddingGemma2 ",
    }
    image_part = {
        "type": "image_url",
        "image_url": {"url": image_url},
    }

    payload = {
        "model": "embeddinggemma-2",
        "input": [
            {
                "content": [text_part, image_part],
            }
        ],
        "encoding_format": "float",
    }

    response = requests.post(URL, json=payload, timeout=120)

    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}\n{response.text}")

    result = response.json()
    embedding = result["data"][0]["embedding"]

    console.print("Dimension：", len(embedding))
    console.print("Firt 8 numbers：", embedding[:8])
    console.print("Usage：", result.get("usage"))

def test_generate():
    models_id = ['google/functiongemma-270m-it', 'google/gemma-3-270m-it', 'google/gemma-3-270m-it-qat-q4_0-unquantized']
    fc_schema = {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Gets the current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco",
                    },
                },
                "required": ["location"],
            },
        }
    }
    fc_messages = [
        {
            "role": "system", # "developer" is only for `functiongemma`
            "content": "You are a model that can do function calling with the following functions"
        },
        {
            "role": "user", 
            "content": "What's the temperature in London?"
        }
    ]
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"},]
        }
    ]
    chat_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Write a poem on Hugging Face, the company"
        }
    ]
    for model_id in models_id:
        GF = GeminiFamily(model_id=model_id, enable_text_generate=True, markers=color_markers)
        console.print(GF.generate(messages=chat_messages, max_new_tokens=20))
        console.print(GF.generate(messages=messages, max_new_tokens=20))
        console.print(GF.generate(messages=fc_messages, schema=fc_schema, max_new_tokens=40))
        GF.release_vram()
    pass

def test_image_generate():
    prompts = "<start_of_image> in this image, there is"
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "in this image, there is"},
            {"type": "image", "image": image_url}
            ]}
    ]
    models_id = ["google/t5gemma-2-270m-270m" , "google/gemma-4-E4B-it"]
    for model_id in models_id:
        GF = GeminiFamily(model_id=model_id, enable_image_text_generate=True, markers=color_markers)
        if model_id == "google/t5gemma-2-270m-270m":
            console.print(GF.generate(prompts=prompts, image_url=image_url, max_new_tokens=40))
        else:
            console.print(GF.generate(messages=messages, max_new_tokens=40))
        GF.release_vram()

if __name__ == "__main__":
    test_similarity()
    test_multi_embed()
    test_generate()
    test_image_generate()
