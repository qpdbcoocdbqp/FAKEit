# SGLang Omni 安裝與部署指南 (使用 uv)

本文件紀錄使用 [`uv`](https://docs.astral.sh/uv/) 在 Linux / WSL 環境中安裝最新版本 **SGLang Omni** 及 **Audio8-TTS 插件** 的完整流程。

---

## 1. 系統依賴與環境要求

| 依賴項 | 版本要求 | 說明 |
|---|---|---|
| **Python** | `3.12` (或 `>=3.10`) | 推薦使用 3.12 |
| **SGLang Omni** | `Latest (main 分支)` | 直接使用最新程式碼庫 |
| **PyTorch** | `>=2.5.0` (建議搭配對應 CUDA 版本) | 支援 BF16 |
| **Transformers** | `>=4.57.0, <5.0.0` | ⚠️ 注意勿升級至 5.x 避免 tokenizer 相容性問題 |

### 系統底層依賴 (Ubuntu / WSL)
```bash
sudo apt update && sudo apt install -y git build-essential ninja-build numactl libnuma-dev
```

---

## 2. 下載最新專案與建立虛擬環境 (`uv`)

### 步驟 2.1：Clone 最新 SGLang Omni 倉庫
```bash

# Clone 最新 main 分支
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni

```

### 步驟 2.2：使用 `uv` 建立虛擬環境
```bash
# 啟用虛擬環境
source ~/.venv/bin/activate
```

### 步驟 2.3：使用 `uv pip` 安裝最新 SGLang Omni
```bash
# 以可編輯模式 (Editable) 安裝最新 sglang-omni 及其所有依賴
uv pip install -v -e sglang-omni

# 安裝 / 鎖定相容的 transformers 版本
uv pip install "transformers>=4.57.0,<5.0.0"
```

---

## 3. 安裝 Audio8-TTS 插件 (Adapter)

回到 `Audio8_TTS` 倉庫目錄：

```bash
cd ./source/Audio8_TTS

# 修復 Windows 換行符 (CRLF -> LF)
sed -i 's/\r$//' ./sglang_omni/scripts/*.sh
```

### 步驟 3.1：安裝適配器
使用 Python 自動抓取套件實際路徑進行安裝（最穩健）：
```bash
SGLANG_OMNI_PACKAGE="$(python -c 'import sys, importlib.util, pathlib; sys.path.pop(0); s=importlib.util.find_spec("sglang_omni"); assert s and s.origin; print(pathlib.Path(s.origin).parent)')"
./sglang_omni/scripts/install_adapter.sh "${SGLANG_OMNI_PACKAGE}"
```

### 步驟 3.2：驗證安裝
```bash
export MODEL=$HOME/.cache/huggingface/hub/models--Audio8--Audio8-TTS-Preview-0.6b/snapshots/f9612f13a0ab40facf3d050fc908b9e6db05c2be

python3 ./sglang_omni/scripts/verify_install.py --model-path "${MODEL}"
```

---

## 4. 啟動推論服務 (Run Server)

### 方案 A：Hopper 架構 GPU（如 H20 / H100）
```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_OMNI_ROOT="${SGLANG_OMNI_ROOT}" \
MODEL="${MODEL}" \
AUDIO8_TTS_ENABLE_TORCH_COMPILE=1 \
AUDIO8_TTS_ATTENTION_BACKEND=fa3 \
HOST=0.0.0.0 \
PORT=8010 \
./sglang_omni/scripts/run_server.sh
```

### 方案 B：Blackwell / 消費級 GPU（如 RTX 5090 / 4090）
```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_OMNI_ROOT="${SGLANG_OMNI_ROOT}" \
MODEL="${MODEL}" \
AUDIO8_TTS_ATTENTION_BACKEND=flashinfer \
HOST=0.0.0.0 \
PORT=8010 \
./sglang_omni/scripts/run_server.sh
```

---

## 5. API 呼叫測試 (OpenAI 相容格式)

服務啟動後，使用 `curl` 進行測試：

```bash
curl -sS -X POST http://localhost:8010/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audio8/tts-0.6b",
    "input": "你好，這是 SGLang Omni 與 Audio8-TTS 服務測試。",
    "response_format": "wav",
    "max_new_tokens": 256,
    "temperature": 0.8
  }' --output test_output.wav
```

### 常用環境變數參數參考
- `AUDIO8_TTS_MEM_FRACTION_STATIC`：靜態顯存分配比例（預設 `0.2`，微調可設 `0.1`）。
- `AUDIO8_TTS_MAX_RUNNING_REQUESTS`：最大並發請求數（預設 `32`）。
- `AUDIO8_TTS_ENABLE_TORCH_COMPILE`：是否開啟 TorchInductor JIT 編譯加速（`0` 或 `1`）。
