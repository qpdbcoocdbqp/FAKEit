# For llama.cpp tag b8783, support for audio input was added to the Gemma-4-E2B-it model.
# Audio example: [PolyAI/minds14] https://huggingface.co/datasets/PolyAI/minds14

import openai
import base64

# Initialize the OpenAI client pointing to your local llama.cpp server
client = openai.OpenAI(
    base_url="http://localhost:9006/v1",
    api_key="not-needed"
    )

MODEL_ID = "google/gemma-4-E2B-it"

# Load and encode your audio file
audio_path = "tmp/audio_input.wav"
with open(audio_path, "rb") as audio_file:
    audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

# Send the request
response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe or analyze this audio:"},
                {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}}
            ]
        }
    ]
)

print(response.choices[0].message.content)
