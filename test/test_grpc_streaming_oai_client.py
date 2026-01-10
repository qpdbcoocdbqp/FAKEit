from openai import OpenAI
from rich.console import Console


console = Console()
client = OpenAI(base_url="http://localhost:30000/v1",api_key="***")

console.print(client.models.list())

for _ in range(5):
    stream = client.chat.completions.create(
        model="google/gemma-3-4b-it-qat-q4_0-unquantized",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain why sky is blue?"}
        ],
        temperature=0.8,
        presence_penalty=1.1,
        frequency_penalty=1.1,
        stop=["<eos>", "<end_of_turn>"],
        max_tokens=128,
        stream=True
        )
    
    print("="*20, "\n")
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
