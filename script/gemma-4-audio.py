# uv pip install -U transformers==5.5.0
# uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

torch.cuda.is_available()


MODEL_ID = "google/gemma-4-E2B-it"

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

for param in model.parameters():
    print(param.device)

# Prompt - add audio before text
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/journal1.wav"},
            {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)
