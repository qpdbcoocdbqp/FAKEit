import os
import dotenv
import openai
import time
import json
from rich.console import Console


dotenv.load_dotenv()
console = Console()

def test_sd_chat():
    sgl = openai.OpenAI(base_url=os.getenv("SD_BASE_URL"), api_key=os.getenv("API_KEY"))
    infos = sgl.models.list().data
    console.print(
        "="*12,
        f"[bold yellow]Model information[/bold yellow]: {infos}",
        "="*12,
        sep="\n"
        )
    for info in infos:
        st = time.time()
        response = sgl.chat.completions.create(
            model=info.id,
            messages=[{"role": "user", "content": "why sky is blue?"}],
            max_tokens=512,
            temperature=1.5,
            )
        dt = time.time() - st
        console.print(
            f"[bold yellow]Model[/bold yellow]: [bold green]{info.id}[/bold green]",
            f"[bold yellow]Completion tokens[/bold yellow]: {response.usage.completion_tokens}",
            f"[bold yellow]E2E-TG Token per second[/bold yellow]: {response.usage.completion_tokens / dt:.2f}",
            "[bold yellow]Generate[/bold yellow]:",
            "-"*12,
            f"{response.choices[0].message.content[:20]}...",
            '-'*12,
            "="*12,
            sep="\n"
            )
    pass


if __name__ == "__main__":
    test_sd_chat()
