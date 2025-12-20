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
        max_tokens=2048,
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
