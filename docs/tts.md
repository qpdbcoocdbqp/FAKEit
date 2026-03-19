
### Reference

* [hexgrad/Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)

### Setup

```bash
# 1. Create a Python 3.12 virtual environment
uv venv --python 3.12

# 2. Activate it (PowerShell)
.venv\Scripts\Activate.ps1
# OR (CMD)
.venv\Scripts\activate.bat
# OR (Bash/Git Bash)
source .venv/Scripts/activate

# 3. Install the required packages
uv pip install kokoro>=0.8.2 "misaki[en,zh]>=0.8.2" soundfile pip

# 4. Run your script
python script/tts-kokoro.py

```

### Example

* [make_en.py]https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/samples/make_en.py)
* [make_zh.py](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/samples/make_zh.py)

