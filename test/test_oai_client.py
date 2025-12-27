import os
import dotenv
import openai
import requests
from rich.console import Console


dotenv.load_dotenv()
console = Console()

def test_chat():
    sgl = openai.OpenAI(base_url=os.getenv("CHAT_BASE_URL"), api_key=os.getenv("API_KEY"))
    model_id = sgl.models.list().data[0].id
    response = sgl.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "why sky is blue?"}],
        max_tokens=512,
        temperature=1.5
        )
    output = {
        "model": model_id,
        "response": response.choices[0].message.content
        }
    console.print(f"{output}")
    pass

def test_embedding():
    sgl = openai.OpenAI(base_url=os.getenv("EMBED_BASE_URL"), api_key=os.getenv("API_KEY"))
    model_id = sgl.models.list().data[0].id
    for dim in [128, 256, 512, 1024]:
        response = sgl.embeddings.create(model=model_id, input="why sky is blue?", dimensions=dim)
        console.print("Embedding:", len(response.data[0].embedding), "Vector Sum", sum(response.data[0].embedding))
    pass

def test_rerank():
    sgl = openai.OpenAI(base_url=os.getenv("RERANK_BASE_URL"), api_key=os.getenv("API_KEY"))
    model_id = sgl.models.list().data[0].id
    payload = {
        "model": model_id,
        "query": "What is the capital of the United States?",
        "documents": [
            "Carson City is the capital city of the American state of Nevada.",
            "Washington, D.C. is the capital of the United States.",
            "Capital punishment has existed in the United States since before the country was founded."
        ],
        # "top_k": 1,
        # "is_cross_encoder_request": True
    }
    response = requests.post(os.getenv("RERANK_BASE_URL")+"/rerank", json=payload)
    response_json = response.json()
    console.print(response_json)

def test_multimodal():
    sgl = openai.OpenAI(base_url=os.getenv("MULTIMODAL_BASE_URL"), api_key=os.getenv("API_KEY"))
    model_id = sgl.models.list().data[0].id
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                        "image_url": {
                            "url": "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/39c0b46a-6553-4c35-8e05-51b1fa2dfee1/d6l4hnf-a55837eb-aca3-4123-a1b6-dac18cf8749b.jpg/v1/fill/w_1024,h_768,q_75,strp/lithium_flower_by_a_neverending_d6l4hnf-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NzY4IiwicGF0aCI6Ii9mLzM5YzBiNDZhLTY1NTMtNGMzNS04ZTA1LTUxYjFmYTJkZmVlMS9kNmw0aG5mLWE1NTgzN2ViLWFjYTMtNDEyMy1hMWI2LWRhYzE4Y2Y4NzQ5Yi5qcGciLCJ3aWR0aCI6Ijw9MTAyNCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.suGnwCuIZ4Sb99uO7pWtCDvQ69MGKWHxDyzMkYD_zjc"
                        },
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    response = sgl.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=512,
        temperature=1.5
        )
    response_json = response.json()
    console.print(response_json)

def test_function():
    sgl = openai.OpenAI(base_url=os.getenv("FUNCTION_BASE_URL"), api_key=os.getenv("API_KEY"))
    model_id = sgl.models.list().data[0].id
    response = sgl.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "why sky is blue?"}],
        max_tokens=512,
        temperature=1.5
        )
    response_json = response.json()
    console.print(response_json)


if __name__ == "__main__":
    test_chat()
    test_embedding()
    test_rerank()
    test_multimodal()
    test_function()
