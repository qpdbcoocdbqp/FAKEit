import base64
from openai import OpenAI

client = OpenAI(api_key="***", base_url="http://localhost:30000/v1")
print(client.models.list())

img = client.images.generate(
    prompt="A cyberpunk city reflected in a puddle, neon lights, 8k resolution",
    size="1024x1024",
    n=1,
    response_format="b64_json"
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("sd-turbo.png", "wb") as f:
    f.write(image_bytes)
