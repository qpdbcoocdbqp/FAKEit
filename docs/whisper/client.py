from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18000/v1",
    api_key="***"
)
client.models.list()

with open("docs/whisper/audio.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="openai/whisper-large-v3-turbo",
        file=f,
        response_format="json"
    )

print(result.text)