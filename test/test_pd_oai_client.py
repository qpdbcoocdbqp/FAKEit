import os
import dotenv
import openai
from rich.console import Console


dotenv.load_dotenv()
console = Console()

def test_pd_chat():
    sgl = openai.OpenAI(base_url=os.getenv("PD_BASE_URL"), api_key=os.getenv("API_KEY"))
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


if __name__ == "__main__":
    test_pd_chat()
