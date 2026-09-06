# uv pip install llama-cpp-python

import os
from llama_cpp import Llama


user_profile = os.environ["USERPROFILE"]

llm = Llama(
    model_path=f"{user_profile}/.cache/huggingface/hub/models--SandLogicTechnologies--translategemma-4b-it-GGUF/snapshots/cd39c30302f20fb8b788234d86a0a35a0d050619/translategemma-4b_Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False,
)

def translate_en_to_zh(text: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "zh-TW",
                        "text": text,
                    }
                ],
            }
        ],
        temperature=0,
        max_tokens=256,
    )

    return response["choices"][0]["message"]["content"].strip()

text = "We need to scale the inference service horizontally."

print(translate_en_to_zh(text))
