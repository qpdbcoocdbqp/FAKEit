# Uninstall Ollama

```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /etc/systemd/system/ollama.service
sudo rm -r $(which ollama | tr 'bin' 'lib')
sudo rm -r $(which ollama)
sudo userdel ollama
sudo groupdel ollama
sudo rm -r /usr/share/ollama
```

# Install Ollama (Ubuntu)

```bash
# For all
curl -fsSL https://ollama.com/install.sh | sh

# For linux amd64 + NVIDIA GPU
./docs/ollama/linux-install.sh
```

---

# Ollama Manual Installation Guide (Linux amd64 + NVIDIA GPU)

This guide is specifically designed for **x86_64 (amd64)** architecture and **NVIDIA GPU (e.g., A30, A100)** environments. It uses a user directory installation method (`~/.local`), which does not require root privileges.

## 1. Download Installation Package

For NVIDIA A30/A100 GPUs, please download the standard Linux amd64 version:

- **ZST File** (Requires `zstd` installation, smaller size): [https://ollama.com/download/ollama-linux-amd64.tar.zst](https://ollama.com/download/ollama-linux-amd64.tar.zst)

## 2. Extract and Install

Create directories and extract the files to `~/.local`:

```bash
mkdir -p ~/.local/bin
mkdir -p ~/.local/lib/ollama
tar -xzf ollama-linux-amd64.tar.zst -C ~/.local
ln -sf ~/.local/bin/ollama ~/.local/bin/ollama
```

* Prune unnecessary files

```bash
cd ~/.local/lib/ollama
rm libggml-cpu-icelake.so libggml-cpu-sse42.so libggml-cpu-alderlake.so libggml-cpu-sandybridge.so libggml-cpu-haswell.so libggml-cpu-skylakex.so
# Remove folders that are not cuda_v12
rm -r cuda_v13 mlx_cuda_v13 vulkan
```

## 3. Set Environment Variables

Add `~/.local/bin` to your `PATH`. Edit `~/.bashrc`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 4. Set up Systemd Service (User Level)

Create a user-level service to run Ollama in the background and start automatically with the system.

Create the file `~/.config/systemd/user/ollama.service`:

```ini
[Unit]
Description=Ollama Service (user)
After=network-online.target

[Service]
# %h represents the user's home directory
ExecStart=%h/.local/bin/ollama serve
Restart=always
RestartSec=3
Environment="PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="HOME=%h"

[Install]
WantedBy=default.target
```

**Start and enable the service**:
```bash
systemctl --user daemon-reload
systemctl --user enable --now ollama
```

## 5. Verification and GPU Detection

Run the following commands to confirm the version and verify if the GPU is correctly detected:

```bash
# Verify version
ollama --version

# Verify API service
curl http://127.0.0.1:11434

# Test run a model and confirm GPU usage (monitor with nvidia-smi in another terminal)
ollama run llama3
```

---
> [!IMPORTANT]
> **NVIDIA GPU A30/A100 Requirements**:
> - The system must have **NVIDIA Driver** pre-installed (requires root privileges for one-time installation).
> - This installation package includes built-in CUDA support libraries required for A30/A100, which will be automatically installed in `~/.local/lib/ollama`.
> - If `nvidia-smi` runs normally and detects the graphics card, Ollama will automatically utilize the GPU upon startup.
