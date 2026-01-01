import json
import numpy as np 
import random
import os
import re
import requests
import sys
import torch
from pathlib import Path
from PIL import Image
from rich.console import Console
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from typing import List

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

sys.path.append(os.path.join(Path(BASE_DIR), "submodules", "ocean_pearl"))

from src.theme import prettyderby


def parse_tool_calls(content):
    tool_calls = re.findall(r'<start_function_call>call:(.*?)\{(.*?)\}<end_function_call>', content)
    if len(tool_calls) == 0:
        del tool_calls
        return {"tool_calls": []}
    result = {"tool_calls": []}
    for name, args_raw in tool_calls:
        args_json = "{" + args_raw.replace('<escape>', '"') + "}"
        args_json = re.sub(r'(\w+):', r'"\1":', args_json)
        try:
            args = json.loads(args_json)
        except:
            args = args_raw
        result["tool_calls"].append({
            "type": "function",
            "function": {"name": name, "arguments": args}
        })
        del name, args_raw, args_json, args
    del tool_calls
    return result

# --- set console logs ---
console = Console()
theme = prettyderby()
makrers = ["[DEBUG]", "[INFO]", "[WARNING]", "[ERROR]", "[MESSAGE]"]
characters = list(theme.keys())
random.shuffle(characters)
character = characters[0]
color_markers = [
    f"[{color.get("hex")}]{mark}[/{color.get("hex")}]"
    for mark, color in zip(makrers, theme.get(character))
    ]
console.print(color_markers[1], f"Use character theme: {character}")
console.print(color_markers[1], "torch version:", torch.__version__, "cuda available:", torch.cuda.is_available())

# --- models ---
models_ids = [
    "google/embeddinggemma-300m",
    "google/functiongemma-270m-it",
    "google/gemma-3-270m-it",
    "google/gemma-3-270m-it-qat-q4_0-unquantized",
    "google/t5gemma-2-270m-270m",
    ]

# --- main ---
for model_id in models_ids:
    console.print(color_markers[1], "Model id:", model_id)
    if "embeddinggemma" in model_id:
        try:
            console.print(color_markers[1], "Load embedding model")
            model = SentenceTransformer(model_id, device="cuda" if torch.cuda.is_available() else "cpu", local_files_only=True)
            model = model.eval()
            console.print(color_markers[1], "Model loaded")
            # Run inference with queries and documents
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
            # method 1: with encode
            # query_embeddings = model.encode(query, prompt_name="query", truncate_dim=truncate_dim, normalize_embeddings=True)
            # document_embeddings = np.array(map(lambda doc: model.encode(doc, prompt_name="document", truncate_dim=truncate_dim, normalize_embeddings=True), documents))
            # method 2: with encode_query and encode_document
            with torch.inference_mode():
                query_embeddings = model.encode_query(query, truncate_dim=truncate_dim, normalize_embeddings=True)
                document_embeddings = model.encode_document(documents, truncate_dim=truncate_dim, normalize_embeddings=True)
                del query, documents, truncate_dim
                # Compute similarities to determine a ranking
                similarities = model.similarity(query_embeddings, document_embeddings)
            console.print(color_markers[1], f"Embeddings shape: query: {query_embeddings.shape}, document: {document_embeddings.shape}" )
            console.print(color_markers[1], f"Similarities: {similarities}")
            del query_embeddings, document_embeddings
            del similarities, model
        except Exception as e:
            console.print(color_markers[3], f"Error: {e}")

    if "functiongemma" in model_id:
        try:
            console.print(color_markers[1], "Load function calling model")
            processor = AutoProcessor.from_pretrained(model_id, device_map="auto", local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto", local_files_only=True)
            model = model.eval()
            # function calling
            weather_function_schema = {
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
            # chat messages
            messages = [
                {
                    "role": "developer",
                    "content": "You are a model that can do function calling with the following functions"
                },
                {
                    "role": "user", 
                    "content": "What's the temperature in London?"
                }
            ]
            with torch.inference_mode():
                inputs = processor.apply_chat_template(messages, tools=[weather_function_schema], add_generation_prompt=True, return_dict=True, return_tensors="pt")
                out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
                response = processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            tool_calls = parse_tool_calls(response)
            del inputs, out, response
            console.print(color_markers[1], f"Input messages: {messages}")
            console.print(color_markers[1], f"Function schema: {weather_function_schema}")
            console.print(color_markers[1], f"Tool calls: {tool_calls}")
            del messages, weather_function_schema, tool_calls
            del model, processor
        except Exception as e:
            console.print(color_markers[3], f"Error: {e}")

    if "gemma-3-270m-it" in model_id:
        try:
            console.print(color_markers[1], "Load chat model")
            processor = AutoProcessor.from_pretrained(model_id, device_map="auto", local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto", local_files_only=True)
            model = model.eval()
            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."},]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"},]
                    },
                ],
            ]
            with torch.inference_mode():
                inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
                out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
                response = processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            console.print(color_markers[1], f"Input messages: {messages}")
            console.print(color_markers[1], f"Response: {response[:12]} {{skip}} ...")
            del messages, response
            del model, processor
        except Exception as e:
            console.print(color_markers[3], f"Error: {e}")

    if "t5gemma-2" in model_id:
        try:
            console.print(color_markers[1], "Load encoder-decoder model")
            processor = AutoProcessor.from_pretrained(model_id, device_map="auto", use_fast=True, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype="auto", device_map="auto", local_files_only=True)
            model = model.eval()
            # Text Sampling -- few-shot prompting
            prompts = """Translate the text from English to Chinese (zh_CN).

            English: Hook the vacuum up to it, it helps keep the dust down, and you can prep your concrete that way also. We prefer to do it with the hand grinders ourself, so either way will work pretty good. Usually if it's got an epoxy coating already, the hand grinders work a little better, but this thing works really good too. So after we do that, we fix all the cracks, fix all the divots with our patch repair material, and then we grind them smooth. And then we clean the concrete and get it ready for the first coating. This is going to be a 100% solids epoxy, so it goes on in its different procedures, different stages.
            Chinese (zh_CN): 把吸尘器连在上面，这样可以减少灰尘，你还可以顺便处理一下混凝土。我们更喜欢用手持式研磨机，两种方法的效果都不错。通常，如果表面已经有环氧树脂涂层的话，手持式研磨机的效果会更好一些，但这台机器的效果也很好。处理完之后，我们用修补材料把所有的裂缝和坑洼都补好，再把它们磨平。然后清洁混凝土表面，为涂刷第一层涂料做好准备。我们这里用的是 100% 固体环氧树脂，所以涂刷的时候会有不同的步骤和阶段。

            English: The Zoroastrian text, Vendidad, states that Yima built an underground city on the orders of the god Ahura Mazda, to protect his people from a catastrophic winter. Much like the account of Noah in the Bible, Yima was instructed to collect pairs of the best animals and people as well as the best seeds in order to reseed the Earth after the winter cataclysm. This was before the last Ice Age, 110,000 years ago.
            Chinese (zh_CN): 琐罗亚斯德教经典《万迪达德》中说，Yima 奉阿胡拉·马兹达神之命建造了一座地下城市，以保护他的人民躲避一场灾难性的寒冬。就像《圣经》中诺亚的故事一样，Yima 被指示收集最好的动物、人类以及最好的种子，以便在冬季灾难过后重新在地球上播种。这发生在最后一个冰河时代之前，也就是 11 万年前。

            English: Okay, so let me explain. All right, so the problem is, if you look inside there... You see the wood siding? There's the old siding, and it butts up to the shingles there. And then I put this over it. And what happens is the dirt collects there, to that flashing.
            Chinese (zh_CN): 好的，我来解释一下。问题是，你们看里面……看到木头墙板了吗？那是原来的墙板，紧挨着那边的瓦。然后我把这个盖在上面。结果灰尘就堆积在那儿，堆积到泛水板上。

            English: Hey guys, Thunder E here, and welcome to the video you've been waiting for. I am talking about gaming on the ASUS ROG Phone 5. Now, the ROG Phone series is well known for its gaming powers, but in this video, we're going to find out if the ROG Phone 5 is truly taking back the crown as the king of gaming phones.
            Chinese (zh_CN): 大家好，我是雷霆 E，欢迎收看大家期待已久的视频。今天要评测的是华硕 ROG Phone 5 的游戏性能。ROG Phone 系列手机一直以其强大的游戏性能而闻名，那么，ROG Phone 5 能否真正加冕“游戏手机之王”？我们拭目以待。

            English: It is December 1997, and the Imperial Sugar Company is acquiring a new production site at Port Wentworth from Savannah Foods and Industries Incorporated. There is nothing really of note here. It was doing what businesses do, and that is acquiring to expand. The site has been home to food production and processing since the early 1900s. Savannah Industries Incorporated began construction of granulated sugar production facilities at Port Wentworth during the 1910s, completing it in 1917.
            Chinese (zh_CN): 那是 1997 年 12 月，帝国糖业公司正从萨凡纳食品和工业有限公司手中收购位于温特沃斯港的一个新生产基地。这的确没什么值得注意的，它做的只是一家公司都会做的事情，那就是通过收购来扩张。该基地自 20 世纪初以来一直是食品生产和加工的场所。萨凡纳工业有限公司在 20 世纪 10 年代开始在温特沃斯港建造砂糖生产设施，并于 1917 年竣工。

            English: Time for the Scotty Kilmer channel. Does your car have faded paint on it? Then stay tuned, because today I'm going to show you how to polish off faded paint. And all it takes is a bucket of water, a polisher, and a bottle of this Meguiar's Ultimate Compound.
            Chinese (zh_CN):"""

            # 1=><eos>, 108=>\n\n
            with torch.inference_mode():
                inputs = processor(text=prompts, return_tensors="pt")
                out = model.generate(**inputs.to("cuda"), max_new_tokens=100, do_sample=True, eos_token_id=[1, 108])
                response = processor.batch_decode(out, skip_special_tokens=True)[0]
            console.print(color_markers[1], f"Input message: {prompts[:30]} {{skip}} ...")
            console.print(color_markers[1], f"Response: {response[:12]} {{skip}} ...")
            del prompts, inputs, out, response

            # Text Sampling -- UL2 Infilling
            prompts = """A large language model (LLM) is a language model trained with self-supervised machine learning on a vast amount of text,
            <unused6237>. The largest and most capable LLMs are generative
            pre-trained transformers (GPTs) and provide the core capabilities of modern chatbots. LLMs can be fine-tuned for specific tasks
            or guided by prompt engineering. These models <unused6236>"""

            with torch.inference_mode():
                inputs = processor(text=prompts, return_tensors="pt")
                out = model.generate(**inputs.to("cuda"), max_new_tokens=100, do_sample=True)
                response = processor.batch_decode(out, skip_special_tokens=True)[0]
            console.print(color_markers[1], f"Input message: {prompts[:30]} {{skip}} ...")
            console.print(color_markers[1], f"Response: {response[:12]} {{skip}} ...")
            del prompts, inputs, out, response

            # Image Understanding
            prompts = "<start_of_image> in this image, there is"
            url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
            image = Image.open(requests.get(url, stream=True).raw)

            with torch.inference_mode():
                inputs = processor(text=prompts, images=image, return_tensors="pt")
                out = model.generate(**inputs.to("cuda"), max_new_tokens=120, do_sample=True)
                response = processor.batch_decode(out, skip_special_tokens=True)[0]
            console.print(color_markers[1], f"Input message: {prompts[:30]} {{skip}} ...")
            console.print(color_markers[1], f"Response: {response[:12]} {{skip}} ...")
            del prompts, inputs, out, response
        except Exception as e:
            console.print(color_markers[3], f"Error: {e}")
