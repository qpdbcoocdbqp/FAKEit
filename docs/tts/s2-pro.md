# S2 Pro

* [HuggingFace](https://huggingface.co/rodrigomt/s2-pro-gguf)

## Setup (WSL)

* Download model

```bash
hf download rodrigomt/s2-pro-gguf s2-pro-q6_k.gguf tokenizer.json
```


* Clone and build s2.cpp

```bash
git clone --recurse-submodules https://github.com/rodrigomatta/s2.cpp.git
cd s2.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_CUDA=ON
cmake --build build --parallel 4
```

* Synthesize

```bash
./build/s2 \
  -m $HOME/.cache/huggingface/hub/models--rodrigomt--s2-pro-gguf/snapshots/a7320690b5585b03b20ed6484b55926f3015f48d/s2-pro-q6_k.gguf \
  -t $HOME/.cache/huggingface/hub/models--rodrigomt--s2-pro-gguf/snapshots/a7320690b5585b03b20ed6484b55926f3015f48d/tokenizer.json \
  -text "Hello, this is a test." \
  -c 0 \
  -o output.wav
```

* Synthesize Reference

```bash
./build/s2 \
  -m $HOME/.cache/huggingface/hub/models--rodrigomt--s2-pro-gguf/snapshots/a7320690b5585b03b20ed6484b55926f3015f48d/s2-pro-q6_k.gguf \
  -t $HOME/.cache/huggingface/hub/models--rodrigomt--s2-pro-gguf/snapshots/a7320690b5585b03b20ed6484b55926f3015f48d/tokenizer.json \
  -pa ../../docs/tts/resource/reference.wav \
  -pt "A sudden mistaken transfer might indicate a system crash, and upon investigation, it turns out the data was written by AI, while the only person responsible for overseeing it never reviewed it. So, who should bear the blame?" \
  -text "Text to synthesize in that voice." \
  -c 0 \
  -o ref_output.wav
```
