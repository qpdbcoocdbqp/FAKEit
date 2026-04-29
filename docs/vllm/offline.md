To build vLLM on an offline server, you need to pre-download all dependencies and resources. Here's what you should prepare:

## Essential Preparations

### 1. Python Dependencies
Download all required wheels and packages:
```bash
# For CUDA builds
uv pip download -r requirements/build/cuda.txt -d ./wheels/build
uv pip download -r requirements/cuda.txt -d ./wheels/runtime

# For CPU builds  
uv pip download -r requirements/build/cpu.txt -d ./wheels/build
uv pip download -r requirements/cpu.txt -d ./wheels/runtime
```

### 2. PyTorch Wheels
Download PyTorch wheels matching your CUDA/CPU configuration [1](#2-0) :
```bash
# CUDA example
uv pip download torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    -d ./wheels/torch

# CPU example
uv pip download torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -d ./wheels/torch
```

### 3. Source Code for Complex Dependencies
Some dependencies require building from source. Clone these repositories:
- **Triton** (for CUDA): `https://github.com/openai/triton.git`
- **Triton for ROCm**: `https://github.com/ROCm/triton.git` [2](#2-1) 
- **AITER** (for ROCm): `https://github.com/ROCm/aiter.git`
- **CUTLASS** (for CUDA): `https://github.com/NVIDIA/cutlass.git`

### 4. Compiler Toolchain
Ensure you have the required compilers installed [3](#2-2) :
```bash
# For x86/ARM
sudo apt-get install gcc-12 g++-12 libnuma-dev

# For s390x (RHEL)
dnf install gcc-toolset-14 gcc-toolset-14-binutils gcc-toolset-14-libatomic-devel
```

### 5. CUDA/ROCm Toolkit (if building for GPUs)
- Download CUDA Toolkit installer from NVIDIA's website
- For ROCm, download from AMD's website

## Offline Build Process

### Method 1: Using Local Wheel Index
Create a local wheel index and install from it:
```bash
# On the offline server
uv pip install --no-index --find-links ./wheels/build -r requirements/build/cuda.txt
uv pip install --no-index --find-links ./wheels/runtime -r requirements/cuda.txt
uv pip install --no-index --find-links ./wheels/torch torch torchvision torchaudio
```

### Method 2: Docker-based Offline Build
Build a Docker image with all dependencies pre-installed [4](#2-3) :
```dockerfile
# Copy pre-downloaded wheels
COPY ./wheels /tmp/wheels

# Install from local wheels
RUN uv pip install --no-index --find-links /tmp/wheels -r requirements.txt
```

### Method 3: Using Pre-compiled Wheels
If you only need Python changes, use pre-compiled wheels [5](#2-4) :
```bash
# Download the wheel beforehand
wget https://wheels.vllm.ai/nightly/vllm-*.whl

# Install offline
uv pip install --no-index --find-links . vllm-*.whl
```

## Platform-Specific Notes

### For s390x (IBM Z)
Build these dependencies from source first [6](#2-5) :
- `torchvision`
- `llvmlite` 
- `numba`
- `llguidance`
- `pyarrow`
- `opencv-headless`

### For ROCm
Install ROCm-specific components [2](#2-1) :
```bash
# Build Triton for ROCm from source
git clone https://github.com/ROCm/triton.git
cd triton
git checkout f9e5bf54
python setup.py install
```

## Build Optimization for Offline

### Use ccache
Install ccache to speed up repeated builds [7](#2-6) :
```bash
export CCACHE_DIR=/path/to/cache
export CCACHE_NOHASHDIR="true"
```

### Control Build Resources
Limit parallel jobs to avoid resource exhaustion [8](#2-7) :
```bash
export MAX_JOBS=4
```

## Notes
- The `uv` package manager is recommended for faster dependency resolution [9](#2-8) 
- For CPU builds on x86/ARM, ensure TCMalloc is available and set `LD_PRELOAD` [10](#2-9) 
- The Dockerfiles provide good reference for complete offline build setups [11](#2-10) 

Wiki pages you might want to explore:
- [Dependency Management (vllm-project/vllm)](/wiki/vllm-project/vllm#11.2)
- [Build Variants and Configuration (vllm-project/vllm)](/wiki/vllm-project/vllm#11.3)

### Citations

**File:** docs/getting_started/installation/gpu.cuda.inc.md (L89-104)
```markdown
#### Set up using Python-only build (without compilation) {#python-only-build}

If you only need to change Python code, you can build and install vLLM without compilation. Using `uv pip`'s [`--editable` flag](https://docs.astral.sh/uv/pip/packages/#editable-packages), changes you make to the code will be reflected when you run vLLM:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

This command will do the following:

1. Look for the current branch in your vLLM clone.
1. Identify the corresponding base commit in the main branch.
1. Download the pre-built wheel of the base commit.
1. Use its compiled libraries in the installation.
```

**File:** docs/getting_started/installation/gpu.cuda.inc.md (L141-149)
```markdown
    Building from source requires a lot of compilation. If you are building from source repeatedly, it's more efficient to cache the compilation results.

    For example, you can install [ccache](https://github.com/ccache/ccache) using `conda install ccache` or `apt install ccache` .
    As long as `which ccache` command can find the `ccache` binary, it will be used automatically by the build system. After the first build, subsequent builds will be much faster.

    When using `ccache` with `pip install -e .`, you should run `CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .`. This is because `pip` creates a new folder with a random name for each build, preventing `ccache` from recognizing that the same files are being built.

    [sccache](https://github.com/mozilla/sccache) works similarly to `ccache`, but has the capability to utilize caching in remote storage environments.
    The following environment variables can be set to configure the vLLM `sccache` remote: `SCCACHE_BUCKET=vllm-build-sccache SCCACHE_REGION=us-west-2 SCCACHE_S3_NO_CREDENTIALS=1`. We also recommend setting `SCCACHE_IDLE_TIMEOUT=0`.
```

**File:** docs/getting_started/installation/gpu.cuda.inc.md (L154-167)
```markdown
##### Use an existing PyTorch installation

There are scenarios where the PyTorch dependency cannot be easily installed with `uv`, for example, when building vLLM with non-default PyTorch builds (like nightly or a custom build).

To build vLLM using an existing PyTorch installation:

```bash
# install PyTorch first, either from PyPI or from source
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build/cuda.txt
uv pip install --no-build-isolation -e .
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

**File:** docs/getting_started/installation/gpu.rocm.inc.md (L162-176)
```markdown
1. Install [Triton for ROCm](https://github.com/ROCm/triton.git)

    Install ROCm's Triton following the instructions from [ROCm/triton](https://github.com/ROCm/triton.git)

    ```bash
    python3 -m pip install ninja cmake wheel pybind11
    pip uninstall -y triton
    git clone https://github.com/ROCm/triton.git
    cd triton
    # git checkout $TRITON_BRANCH
    git checkout f9e5bf54
    if [ ! -f setup.py ]; then cd python; fi
    python3 setup.py install
    cd ../..
    ```
```

**File:** docs/getting_started/installation/cpu.x86.inc.md (L71-77)
```markdown
Install recommended compiler. We recommend to use `gcc/g++ >= 12.3.0` as the default compiler to avoid potential problems. For example, on Ubuntu 22.4, you can run:

```bash
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```
```

**File:** docs/getting_started/installation/cpu.x86.inc.md (L133-147)
```markdown
!!! warning "set `LD_PRELOAD`"
    Before use vLLM CPU installed via wheels, make sure TCMalloc and Intel OpenMP are installed and added to `LD_PRELOAD`:
    ```bash
    # install TCMalloc, Intel OpenMP is installed with vLLM CPU
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    sudo find / -iname *libiomp5.so
    TC_PATH=...
    IOMP_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
    ```
```

**File:** docker/Dockerfile.cpu (L31-57)
```text
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -y \
    && apt-get install -y --no-install-recommends sudo ccache git curl wget ca-certificates \
    gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 jq lsof make xz-utils \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
ENV CCACHE_DIR=/root/.cache/ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV UV_HTTP_TIMEOUT=500

# Install Python dependencies
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE="copy"

# Copy requirements files for installation
```

**File:** docs/getting_started/installation/cpu.s390x.inc.md (L46-57)
```markdown
!!! tip
    Please build the following dependencies, `torchvision`, `llvmlite`, `numba`, `llguidance`, `pyarrow`, `opencv-headless` from source before building vLLM.

```bash
    uv pip install -v \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --torch-backend auto \
        -r requirements/build/cpu.txt \
        -r requirements/cpu.txt \
    VLLM_TARGET_DEVICE=cpu python setup.py bdist_wheel && \
        uv pip install dist/*.whl
```
```

**File:** docker/Dockerfile.ppc64le (L226-257)
```text
# this step installs vllm and populates uv cache
# with all the transitive dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    dnf install llvm15 llvm15-devel -y && \
    rpm -ivh --nodeps https://mirror.stream.centos.org/9-stream/CRB/ppc64le/os/Packages/protobuf-lite-devel-3.14.0-16.el9.ppc64le.rpm && \
    source /opt/rh/gcc-toolset-14/enable && \
    git clone https://github.com/huggingface/xet-core.git && cd xet-core/hf_xet/ && \
    uv pip install maturin && \
    uv build --wheel --out-dir /hf_wheels/

ENV CXXFLAGS="-fno-lto -Wno-error=free-nonheap-object" \
    CFLAGS="-fno-lto"
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=torch-builder,source=/torchwheels/,target=/torchwheels/,ro \
    --mount=type=bind,from=arrow-builder,source=/arrowwheels/,target=/arrowwheels/,ro \
    --mount=type=bind,from=cv-builder,source=/opencvwheels/,target=/opencvwheels/,ro \
    --mount=type=bind,from=numa-builder,source=/numactl/,target=/numactl/,rw \
    --mount=type=bind,from=numba-builder,source=/numbawheels/,target=/numbawheels/,ro \
    --mount=type=bind,src=.,dst=/src/,rw \
    source /opt/rh/gcc-toolset-14/enable && \
    export PATH=$PATH:/usr/lib64/llvm15/bin && \
    uv pip install /opencvwheels/*.whl /arrowwheels/*.whl /torchwheels/*.whl /numbawheels/*.whl && \
    sed -i -e 's/.*torch.*//g' /src/pyproject.toml /src/requirements/*.txt && \
    sed -i -e 's/.*sentencepiece.*//g' /src/pyproject.toml /src/requirements/*.txt && \
    uv pip install sentencepiece==0.2.0 pandas pythran nanobind pybind11 /hf_wheels/*.whl && \
    make -C /numactl install && \
    # sentencepiece.pc is in some pkgconfig inside uv cache
    export PKG_CONFIG_PATH=$(find / -type d -name "pkgconfig" 2>/dev/null | tr '\n' ':') && \
    nanobind_DIR=$(uv pip show nanobind | grep Location | sed 's/^Location: //;s/$/\/nanobind\/cmake/') && uv pip install -r /src/requirements/common.txt -r /src/requirements/cpu.txt -r /src/requirements/build/cuda.txt --no-build-isolation && \
    cd /src/ && \
    uv build --wheel --out-dir /vllmwheel/ --no-build-isolation && \
    uv pip install /vllmwheel/*.whl
```
