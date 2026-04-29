# Offline Build and Installation Guide for vLLM on RHEL 8 (glibc 2.28) with NVIDIA A30

This guide provides step-by-step instructions for building and installing vLLM from source on an offline RHEL 8 system with glibc 2.28 and an NVIDIA A30 GPU.

## Overview

Building vLLM on RHEL 8 with glibc 2.28 requires source compilation because pre-compiled wheels need glibc >= 2.35 [1](#7-0) . The NVIDIA A30 uses CUDA, so we'll prepare all CUDA-specific dependencies for offline building.

---

## Phase 1: Online Preparation (Download All Dependencies)

Perform these steps on an internet-connected machine with the same architecture as your target RHEL 8 server.

### 1.1 Create Directory Structure

```bash
mkdir -p ./vllm-offline/{wheels,source,cuda,torch,cutlass,triton}
cd ./vllm-offline
```

### 1.2 Download vLLM Source Code

```bash
git clone https://github.com/vllm-project/vllm.git ./source/vllm
```

### 1.3 Download CUDA Toolkit

```bash
# Download CUDA 12.1 (compatible with A30)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
```

### 1.4 Download CUTLASS (Required for GPU builds)

```bash
git clone https://github.com/nvidia/cutlass.git ./cutlass
cd ./cutlass
git checkout v4.4.2  # Version required by vLLM
cd ..
```

### 1.5 Download Triton (Required for CUDA)

```bash
git clone https://github.com/openai/triton.git ./triton
cd ./triton
git checkout main  # Use latest stable commit
cd ..
```

### 1.6 Download PyTorch CUDA Wheels

```bash
uv pip download torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -d ./torch
```

### 1.7 Download Python Dependencies

```bash
# Build dependencies
uv pip download -r ./source/vllm/requirements/build/cuda.txt -d ./wheels/build

# Runtime dependencies
uv pip download -r ./source/vllm/requirements/cuda.txt -d ./wheels/runtime

# Common dependencies
uv pip download -r ./source/vllm/requirements/common.txt -d ./wheels/common
```

### 1.8 Download uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh -o install_uv.sh
```

---

## Phase 2: Transfer to Offline RHEL 8 Server

Copy the entire `vllm-offline` directory to your RHEL 8 server using your preferred method (USB drive, network transfer, etc.).

---

## Phase 3: System Preparation on RHEL 8

### 3.1 Install System Dependencies

```bash
# Enable EPEL repository
sudo dnf install -y epel-release

# Install development tools
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y gcc-c++ cmake make python3-devel python3-pip

# Install GPU-specific dependencies
sudo dnf install -y kernel-devel-$(uname -r) pciutils

# Install performance libraries
sudo dnf install -y libnuma-devel
```

### 3.2 Install CUDA Toolkit

```bash
# Make installer executable
chmod +x ./vllm-offline/cuda_12.1.0_530.30.02_linux.run

# Run installer (accept license agreements)
sudo ./vllm-offline/cuda_12.1.0_530.30.02_linux.run --silent --toolkit

# Set CUDA environment variables
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=${CUDA_HOME}/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
```

### 3.3 Install uv Package Manager

```bash
bash ./vllm-offline/install_uv.sh
source ~/.bashrc
```

---

## Phase 4: Build vLLM from Source

### 4.1 Create Python Environment

```bash
cd ./vllm-offline/source/vllm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 4.2 Install PyTorch from Local Wheels

```bash
uv pip install --no-index --find-links ../../torch torch torchvision torchaudio
```

### 4.3 Install Build Dependencies

```bash
uv pip install --no-index --find-links ../../wheels/build -r requirements/build/cuda.txt
```

### 4.4 Install Runtime Dependencies

```bash
uv pip install --no-index --find-links ../../wheels/runtime -r requirements/cuda.txt
uv pip install --no-index --find-links ../../wheels/common -r requirements/common.txt
```

### 4.5 Build Triton from Source

```bash
cd ../../triton
pip install -e .
cd ../source/vllm
```

### 4.6 Configure CUTLASS Path

```bash
export VLLM_CUTLASS_SRC_DIR=$(realpath ../../cutlass)
```

### 4.7 Build vLLM

```bash
# Limit parallel jobs to avoid resource exhaustion
export MAX_JOBS=4

# Build vLLM with CUDA support
uv pip install -e . --no-build-isolation
```

---

## Phase 5: Installation and Verification

### 5.1 Build Wheel Package (Optional)

```bash
# Create distributable wheel
uv build --wheel --no-build-isolation

# Install from wheel
uv pip install --no-index --find-links dist vllm-*.whl
```

### 5.2 Verify Installation

```bash
# Test vLLM import
python -c "import vllm; print(vllm.__version__)"

# Check GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
```

### 5.3 Test with A30 GPU

```bash
# Create a simple test script
cat > test_vllm.py << 'EOF'
from vllm import LLM, SamplingParams

# Test with a small model
llm = LLM("facebook/opt-125m", trust_remote_code=True)
sampling_params = SamplingParams(max_tokens=10)
outputs = llm.generate("Hello, world!", sampling_params)
for output in outputs:
    print(f"Generated: {output.outputs[0].text}")
EOF

# Run test
python test_vllm.py
```

---

## Troubleshooting

### Common Issues and Solutions

1. **CUTLASS not found**: Ensure `VLLM_CUTLASS_SRC_DIR` is set to absolute path [2](#7-1) 

2. **CUDA errors**: Verify CUDA toolkit installation and environment variables

3. **Memory issues**: Reduce `MAX_JOBS` if build fails due to memory [3](#7-2) 

4. **Missing dependencies**: Check that all wheels were downloaded for correct Python version and architecture

### Build Optimization

For faster subsequent builds, install ccache:

```bash
sudo dnf install -y ccache
export CCACHE_DIR=/tmp/ccache
export CCACHE_NOHASHDIR="true"
```

---

## Notes

- **glibc Compatibility**: This build is necessary because RHEL 8's glibc 2.28 is incompatible with vLLM's pre-compiled wheels which require glibc >= 2.35 [4](#7-3) 
- **NVIDIA A30**: The A30 GPU supports CUDA compute capability 8.0, which is compatible with vLLM's requirements
- **Performance**: Building from source may result in slightly different performance compared to official wheels
- **Updates**: To update vLLM, repeat the download and build process with the new version

Wiki pages you might want to explore:
- [Dependency Management (vllm-project/vllm)](/wiki/vllm-project/vllm#11.2)
- [Build Variants and Configuration (vllm-project/vllm)](/wiki/vllm-project/vllm#11.3)

Wiki pages you might want to explore:
- [Dependency Management (vllm-project/vllm)](/wiki/vllm-project/vllm#11.2)

### Citations

**File:** docs/getting_started/installation/gpu.cuda.inc.md (L180-189)
```markdown
##### Use the local cutlass for compilation

Currently, before starting the build process, vLLM fetches cutlass code from GitHub. However, there may be scenarios where you want to use a local version of cutlass instead.
To achieve this, you can set the environment variable VLLM_CUTLASS_SRC_DIR to point to your local cutlass directory.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_CUTLASS_SRC_DIR=/path/to/cutlass uv pip install -e . --torch-backend=auto
```
```

**File:** docs/getting_started/installation/gpu.cuda.inc.md (L193-202)
```markdown
To avoid your system being overloaded, you can limit the number of compilation jobs
to be run simultaneously, via the environment variable `MAX_JOBS`. For example:

```bash
export MAX_JOBS=6
uv pip install -e .
```

This is especially useful when you are building on less powerful machines. For example, when you use WSL it only [assigns 50% of the total memory by default](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings), so using `export MAX_JOBS=1` can avoid compiling multiple files simultaneously and running out of memory.
A side effect is a much slower build process.
```

**File:** docker/Dockerfile (L36-40)
```text
# Important: We build with an old version of Ubuntu to maintain broad
# compatibility with other Linux OSes. The main reason for this is that the
# glibc version is baked into the distro, and binaries built with one glibc
# version are not backwards compatible with OSes that use an earlier version.
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
```
