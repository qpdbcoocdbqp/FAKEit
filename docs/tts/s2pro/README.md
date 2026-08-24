# s2.cpp Docker 推論伺服器

Fish Audio S2 Pro TTS 模型的本地化 Docker 部署方案，基於 [s2.cpp](https://github.com/rodrigomatta/s2.cpp)。

---

## 目錄結構

```
s2cpp-docker/
├── Dockerfile          # 兩段式 build (cuda:devel → cuda:runtime)
├── docker-compose.yml  # GPU 配置 + volume 掛載
├── download_model.sh   # 從 HuggingFace 下載 GGUF 模型
├── models/             # 放 .gguf 模型檔（需自行建立）
└── voices/             # 放 .s2voice 聲音 profile（可選）
```

---

## 快速開始

### 1. 前置需求

- Docker + Docker Compose v2
- NVIDIA GPU + CUDA 12.4+
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# 確認 GPU 可被 Docker 存取
docker run --rm --gpus all nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 nvidia-smi

cd docs/tts/s2pro
docker build -t s2cpp:latest .

```


### 2. 下載模型

```bash
chmod +x download_model.sh

# 推薦（6–9 GB VRAM）
./download_model.sh q6_k

# VRAM 較少時（≥ 6 GB）
./download_model.sh q4_k_m

# 近乎無損品質（需 ≥ 10 GB VRAM）
./download_model.sh q8_0
```

### 3. 啟動伺服器

```bash
docker compose up --build
```

首次啟動會編譯 s2.cpp（約 5–10 分鐘）。之後可加 `-d` 在背景執行：

```bash
docker compose up -d
```

---

## API 使用方式

伺服器啟動後監聽 `http://localhost:3030`。

### 基本合成

```bash
curl -X POST http://localhost:3030/generate \
  --form "text=你好，這是語音合成測試。" \
  -o output.wav
```

### 帶參數合成

```bash
curl -X POST http://localhost:3030/generate \
  --form "text=The quick brown fox jumps over the lazy dog." \
  --form 'params={"max_new_tokens":512,"temperature":0.58,"top_p":0.88,"top_k":40}' \
  -o output.wav
```

### 聲音克隆（Voice Cloning）

```bash
curl -X POST http://localhost:3030/generate \
  --form "reference=@reference.wav" \
  --form "reference_text=參考音訊的逐字稿內容。" \
  --form "text=用這個聲音合成這段文字。" \
  -o output_cloned.wav
```

### 即時串流（PCM16 低延遲）

```bash
curl -sN -X POST http://localhost:3030/generate \
  --form "text=即時串流語音輸出測試。" \
  --form 'params={
    "stream": true,
    "chunked": true,
    "output_format": "pcm_s16le",
    "segment_sentences": true,
    "stream_start_buffer_ms": 4000,
    "max_new_tokens": 512
  }' \
| ffplay -autoexit -nodisp -infbuf -f s16le -ar 44100 -ac 1 -
```

### 使用儲存的聲音 Profile

```bash
# 先把 .s2voice 檔案放到 ./voices/ 目錄
curl -X POST http://localhost:3030/generate \
  --form "voice=my_voice" \
  --form "text=用已儲存的聲音 profile 合成。" \
  -o output.wav
```

---

## 環境變數設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `MODEL_PATH` | `/app/models/s2-pro-q6_k.gguf` | GGUF 模型路徑 |
| `PORT` | `3030` | 伺服器 port |
| `GPU_LAYERS` | `-1` | GPU 層數；`-1` = 全部；`0` = 純 CPU |
| `THREADS` | `0` | CPU 執行緒數；`0` = 自動 |
| `LOG_LEVEL` | `info` | `error` / `warn` / `info` / `debug` |
| `EXTRA_ARGS` | — | 額外 CLI 參數，例如 `--codec-cpu` |

### VRAM 不足時的調整

在 `docker-compose.yml` 中設定：

```yaml
environment:
  GPU_LAYERS: 18       # 只把部分層放 GPU
  EXTRA_ARGS: --codec-cpu  # codec 留在 CPU 上
```

| 顯卡 VRAM | 建議模型 | GPU_LAYERS |
|-----------|----------|------------|
| ≥ 10 GB   | q8_0     | -1（全放）  |
| 8–9 GB    | q6_k     | -1（全放）  |
| 6–7 GB    | q4_k_m   | 18 + --codec-cpu |
| < 6 GB    | q4_k_m   | 10 + --codec-cpu |

---

## 常用指令

```bash
# 查看 log
docker compose logs -f

# 停止
docker compose down

# 進入容器
docker compose exec s2cpp /bin/bash

# 重新 build（更新 s2.cpp 版本後）
docker compose build --no-cache
docker compose up -d
```

---

## 授權

模型權重受 [Fish Audio Research License](https://github.com/rodrigomatta/s2.cpp/blob/main/LICENSE.md) 約束。
商業使用需向 Fish Audio 取得授權：business@fish.audio
