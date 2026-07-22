import time
import openai

client = openai.OpenAI(base_url="http://localhost:18001/v1", api_key="***")
print(client.models.list())

# Long shared prefix
prefix = (
    "You are a helpful assistant. Explain concepts clearly. "
    * 25   # was 50 — tune down if you still hit 400
)
messages = [
    {"role": "user", "content": prefix + " Say OK."},  # one message saves template tokens
]

def timed_call():
    t0 = time.perf_counter()
    client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )
    return time.perf_counter() - t0

cold = timed_call()   # run 1: stores KV
hot = timed_call()    # run 2: should reuse KV

print(f"cold: {cold:.3f}s  hot: {hot:.3f}s  speedup: {cold/hot:.2f}x")
