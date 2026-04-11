#!/bin/sh
# Ollama Linux installer - NVIDIA GPU, no sudo required
# Installs entirely under ~/.local (user-local, no root needed)
#
# Binary  : ~/.local/bin/ollama
# Libs    : ~/.local/lib/ollama/
# Service : ~/.config/systemd/user/ollama.service  (user systemd)
#
# NOTE: CUDA / kernel driver installation ALWAYS needs root.
#       If nvidia-smi already works, this script will use the GPU automatically.
#       Otherwise, ask your sysadmin to install the NVIDIA driver first.

set -eu

red="$( (/usr/bin/tput bold || :; /usr/bin/tput setaf 1 || :) 2>&-)"
plain="$( (/usr/bin/tput sgr0 || :) 2>&-)"

status() { echo ">>> $*" >&2; }
error()  { echo "${red}ERROR:${plain} $*" >&2; exit 1; }
warning(){ echo "${red}WARNING:${plain} $*" >&2; }

TEMP_DIR=$(mktemp -d)
cleanup() { rm -rf "$TEMP_DIR"; }
trap cleanup EXIT

available() { command -v "$1" >/dev/null 2>&1; }
require() {
    local MISSING=''
    for TOOL in $*; do
        if ! available "$TOOL"; then
            MISSING="$MISSING $TOOL"
        fi
    done
    echo $MISSING
}

# ── OS ─────────────────────────────────────────
OS="$(uname -s)"
[ "$OS" = "Linux" ] || error 'This script is intended to run on Linux only.'

ARCH=$(uname -m)
case "$ARCH" in
    x86_64)        ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) error "Unsupported architecture: $ARCH" ;;
esac

VER_PARAM="${OLLAMA_VERSION:+?version=$OLLAMA_VERSION}"

IS_WSL2=false
KERN=$(uname -r)
case "$KERN" in
    *icrosoft*WSL2 | *icrosoft*wsl2) IS_WSL2=true ;;
    *icrosoft) error "Microsoft WSL1 is not supported. Please use WSL2." ;;
    *) ;;
esac

# ── Required tools check ──────────────────────────────────────────
NEEDS=$(require curl awk grep sed xargs)
if [ -n "$NEEDS" ]; then
    status "ERROR: The following tools are required but missing:"
    for NEED in $NEEDS; do
        echo "  - $NEED"
    done
    exit 1
fi

# ── Installation directory (all in $HOME, no root required) ───────────────────
BINDIR="$HOME/.local/bin"
OLLAMA_INSTALL_DIR="$HOME/.local"

mkdir -p "$BINDIR"
mkdir -p "$OLLAMA_INSTALL_DIR/lib/ollama"

# ── Download and extract (supports .tar.zst / .tgz automatic fallback) ────
download_and_extract() {
    local url_base="$1"
    local dest_dir="$2"
    local filename="$3"

    if curl --fail --silent --head --location "${url_base}/${filename}.tar.zst${VER_PARAM}" >/dev/null 2>&1; then
        if ! available zstd; then
            error "This version requires zstd for extraction. Please install it:
  - Debian/Ubuntu: sudo apt-get install zstd
  - RHEL/Fedora:   sudo dnf install zstd
  - Arch:          sudo pacman -S zstd"
        fi
        status "Downloading ${filename}.tar.zst"
        curl --fail --show-error --location --progress-bar \
            "${url_base}/${filename}.tar.zst${VER_PARAM}" | \
            zstd -d | tar -xf - -C "${dest_dir}"
        return 0
    fi

    status "Downloading ${filename}.tgz"
    curl --fail --show-error --location --progress-bar \
        "${url_base}/${filename}.tgz${VER_PARAM}" | \
        tar -xzf - -C "${dest_dir}"
}

# ── Install Ollama main program ────────────────────────────────────
if [ -d "$OLLAMA_INSTALL_DIR/lib/ollama" ]; then
    status "Cleaning up old lib at $OLLAMA_INSTALL_DIR/lib/ollama"
    rm -rf "$OLLAMA_INSTALL_DIR/lib/ollama"
    mkdir -p "$OLLAMA_INSTALL_DIR/lib/ollama"
fi

status "Installing ollama to $OLLAMA_INSTALL_DIR (no sudo)"
download_and_extract "https://ollama.com/download" "$OLLAMA_INSTALL_DIR" "ollama-linux-${ARCH}"

# Ensure binary is in BINDIR
if [ ! -f "$BINDIR/ollama" ]; then
    ln -sf "$OLLAMA_INSTALL_DIR/bin/ollama" "$BINDIR/ollama" 2>/dev/null || \
    ln -sf "$OLLAMA_INSTALL_DIR/ollama"     "$BINDIR/ollama"
fi

# ── PATH ─────────────────────────────────────────────
case ":$PATH:" in
    *":$BINDIR:"*) ;;
    *)
        warning "$BINDIR is not in your PATH."
        warning "Add the following line to your ~/.bashrc or ~/.profile:"
        warning "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        ;;
esac

install_success() {
    status 'The Ollama API is now available at 127.0.0.1:11434.'
    status "Install complete. Run: $BINDIR/ollama serve"
}
trap install_success EXIT

# ── user systemd service (no root required) ──────────────────────
configure_user_systemd() {
    local SERVICE_DIR="$HOME/.config/systemd/user"
    mkdir -p "$SERVICE_DIR"

    status "Creating user systemd service at $SERVICE_DIR/ollama.service"
    cat > "$SERVICE_DIR/ollama.service" <<EOF
[Unit]
Description=Ollama Service (user)
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama serve
Restart=always
RestartSec=3
Environment="PATH=$PATH"
Environment="HOME=$HOME"

[Install]
WantedBy=default.target
EOF

    # systemctl --user requires dbus and loginctl linger
    if systemctl --user is-system-running >/dev/null 2>&1 || \
       systemctl --user status >/dev/null 2>&1; then
        status "Enabling user ollama service..."
        systemctl --user daemon-reload
        systemctl --user enable ollama
        trap 'systemctl --user restart ollama' EXIT
    else
        warning "User systemd is not running. Start manually:"
        warning "  systemctl --user daemon-reload"
        warning "  systemctl --user enable --now ollama"
        if [ "$IS_WSL2" = true ]; then
            warning "WSL2 systemd tip: https://learn.microsoft.com/en-us/windows/wsl/systemd"
        fi
    fi
}

if available systemctl; then
    configure_user_systemd
fi

# ── WSL2: nvidia-smi GPU detection ────────────────────────────
if [ "$IS_WSL2" = true ]; then
    if available nvidia-smi && [ -n "$(nvidia-smi | grep -o 'CUDA Version: [0-9]*\.[0-9]*')" ]; then
        status "NVIDIA GPU detected via nvidia-smi."
    else
        warning "No CUDA-capable GPU detected in WSL2. Ollama will run in CPU-only mode."
    fi
    install_success
    exit 0
fi

# ── GPU detection ──────────────────────────────────────────────
check_gpu() {
    case $1 in
        lspci)
            case $2 in
                nvidia) available lspci && lspci -d '10de:' | grep -q 'NVIDIA' || return 1 ;;
            esac ;;
        lshw)
            # lshw -c display 不需要 root（部分資訊可能不全，但足夠偵測廠商）
            case $2 in
                nvidia) available lshw && lshw -c display -numeric 2>/dev/null | grep -q 'vendor: .* \[10DE\]' || return 1 ;;
            esac ;;
        nvidia-smi) available nvidia-smi || return 1 ;;
    esac
}

# nvidia-smi 已正常就直接結束
if check_gpu nvidia-smi; then
    if [ -n "$(nvidia-smi | grep -o 'CUDA Version: [0-9]*\.[0-9]*')" ]; then
        status "NVIDIA GPU with CUDA ready."
    else
        status "nvidia-smi found but CUDA version undetected. Driver may need update."
    fi
    install_success
    exit 0
fi

# 無 GPU 偵測工具
if ! available lspci && ! available lshw; then
    warning "Cannot detect GPU — lspci/lshw not found. Ollama will run in CPU-only mode."
    install_success
    exit 0
fi

# 無任何 GPU
if ! check_gpu lspci nvidia && ! check_gpu lshw nvidia; then
    warning "No NVIDIA GPU detected. Ollama will run in CPU-only mode."
    install_success
    exit 0
fi

# ── NVIDIA GPU detected, but nvidia-smi is not available ──────────────
# CUDA driver installation requires root; only prompt here, do not attempt installation
warning "NVIDIA GPU detected, but nvidia-smi is not available."
warning "The CUDA driver must be installed by a system administrator (requires root)."
warning "  Ubuntu/Debian : sudo apt-get install -y cuda-drivers"
warning "  RHEL/Fedora   : sudo dnf install -y cuda-drivers"
warning "  Manual guide  : https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
warning "After the driver is installed, re-run this script."

install_success
