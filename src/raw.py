# uv pip install -q git+https://github.com/huggingface/transformers.git

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor

torch.__version__
torch.cuda.is_available()

model_id = "t5gemma-2-270m-270m"
# loading from huggingface
model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{model_id}", device_map="cuda")
processor = AutoProcessor.from_pretrained(f"google/{model_id}")
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

model_inputs = processor(text=prompts, return_tensors="pt").to("cuda")

# 1=><eos>, 108=>\n\n
generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=True, eos_token_id=[1, 108])

decoded = processor.batch_decode(generation)
decoded[0]

# Text Sampling -- UL2 Infilling
prompts = """A large language model (LLM) is a language model trained with self-supervised machine learning on a vast amount of text,
<unused6237>. The largest and most capable LLMs are generative
pre-trained transformers (GPTs) and provide the core capabilities of modern chatbots. LLMs can be fine-tuned for specific tasks
or guided by prompt engineering. These models <unused6236>"""

model_inputs = processor(text=prompts, return_tensors="pt").to("cuda")

generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)

decoded = processor.batch_decode(generation)
decoded[0]

# Image Understanding
import requests
from PIL import Image
from transformers import AutoProcessor

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<start_of_image> in this image, there is"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

generation = model.generate(**model_inputs, max_new_tokens=120, do_sample=True)

decoded = processor.batch_decode(generation)
decoded[0]