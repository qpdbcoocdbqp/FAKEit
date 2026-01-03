from script.utils import GeminiFamily, console, ThemeConsole
import random


theme_console = ThemeConsole()
color_markers = theme_console.get_random_markers()
console.print(*color_markers)


def test_similarity():
    GF = GeminiFamily(enable_embedding=True, markers=color_markers)
    truncate_dim = random.randint(128, 768)
    console.print(color_markers[1], f"Random embedding dimension (128-768): {truncate_dim}")
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
        ]
    console.print(color_markers[1], "Query:", query)
    console.print(color_markers[1], "Documents:", documents)
    similarity, q_emb, d_emb = GF.similarity(query, documents, dim=truncate_dim)
    GF.release_vram()
    pass

def test_generate():
    models_id = ['google/functiongemma-270m-it', 'google/gemma-3-270m-it', 'google/gemma-3-270m-it-qat-q4_0-unquantized']
    fc_schema = {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Gets the current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco",
                    },
                },
                "required": ["location"],
            },
        }
    }
    fc_messages = [
        {
            "role": "system", # "developer" is only for `functiongemma`
            "content": "You are a model that can do function calling with the following functions"
        },
        {
            "role": "user", 
            "content": "What's the temperature in London?"
        }
    ]
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"},]
        }
    ]
    chat_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Write a poem on Hugging Face, the company"
        }
    ]
    for model_id in models_id:
        GF = GeminiFamily(model_id=model_id, enable_text_generate=True, markers=color_markers)
        console.print(GF.generate(messages=chat_messages, max_new_tokens=20))
        console.print(GF.generate(messages=messages, max_new_tokens=20))
        console.print(GF.generate(messages=fc_messages, schema=fc_schema, max_new_tokens=40))
        GF.release_vram()
    pass

def test_image_generate():
    prompts = "<start_of_image> in this image, there is"
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "in this image, there is"},
            {"type": "image", "image": image_url}
            ]}
    ]
    models_id = ["google/t5gemma-2-270m-270m" , "google/gemma-3-4b-it-qat-q4_0-unquantized", "google/gemma-3n-e2b-it"]
    for model_id in models_id:
        GF = GeminiFamily(model_id=model_id, enable_image_text_generate=True, markers=color_markers)
        if model_id == "google/t5gemma-2-270m-270m":
            console.print(GF.generate(prompts=prompts, image_url=image_url, max_new_tokens=40))
        else:
            console.print(GF.generate(messages=messages, max_new_tokens=40))
        GF.release_vram()

if __name__ == "__main__":
    test_similarity()
    test_generate()
    test_image_generate()
