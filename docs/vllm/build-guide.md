# VLLM build on RHEL 8 (glibc 2.28)

## Prepare RHEL 8 environment

```bash
docker run -itd --name vllm-dev \
--gpus all \
-v "/$(pwd)/docs/vllm/volume:/build/vllm-offline" \
nvidia/cuda:12.9.1-devel-ubi8 bash
```

## Install `UV`, `Python` and build wheel

```bash
# install building tools
dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
dnf install -y ccache git gcc-toolset-12
source /opt/rh/gcc-toolset-12/enable

# UV
curl -LsSf https://astral.sh/uv/install.sh -o install_uv.sh
sh install_uv.sh
source $HOME/.local/bin/env

# install python
uv python install 3.12
cd $HOME
uv venv --python 3.12
source ~/.venv/bin/activate

# download vllm repository
git clone https://github.com/vllm-project/vllm.git /build/vllm-offline/source/vllm
# download external_projects
git clone https://github.com/nvidia/cutlass.git /build/vllm-offline/source/vllm/.deps/cutlass-src
git clone https://github.com/deepseek-ai/DeepGEMM.git /build/vllm-offline/source/vllm/.deps/deepgemm-src
git clone https://github.com/vllm-project/FlashMLA.git /build/vllm-offline/source/vllm/.deps/flashmla-src
git clone https://github.com/IST-DASLab/qutlass.git /build/vllm-offline/source/vllm/.deps/qutlass-src
git clone https://github.com/triton-lang/triton.git /build/vllm-offline/source/vllm/.deps/triton_kernels-src
git clone https://github.com/vllm-project/flash-attention.git /build/vllm-offline/source/vllm/.deps/vllm-flash-attn-src

# download package wheels
cd /build/vllm-offline/source/vllm
git checkout v0.20.0

python3 -m pip download -r /build/vllm-offline/source/vllm/requirements/build/cuda.txt -d ./wheels/build && \
python3 -m pip download -r /build/vllm-offline/source/vllm/requirements/cuda.txt -d ./wheels/runtime && \
python3 -m pip download -r /build/vllm-offline/source/vllm/requirements/common.txt -d ./wheels/common

# build vllm wheel
cd /build/vllm-offline/source/vllm/
uv pip install --no-index --find-links /build/vllm-offline/wheels/build -r /build/vllm-offline/source/vllm/requirements/build/cuda.txt
gcc --version

export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export CUDA_HOME=/usr/local/cuda-12.9
export VLLM_TARGET_DEVICE=cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=4
export VLLM_CUTLASS_SRC_DIR=/build/vllm-offline/source/vllm/.deps/cutlass-src
export DEEPGEMM_SRC_DIR=/build/vllm-offline/source/vllm/.deps/deepgemm-src
export FLASH_MLA_SRC_DIR=/build/vllm-offline/source/vllm/.deps/flashmla-src
export QUTLASS_SRC_DIR=/build/vllm-offline/source/vllm/.deps/qutlass-src
export TRITON_KERNELS_SRC_DIR=/build/vllm-offline/source/vllm/.deps/triton_kernels-src/python/triton_kernels
export VLLM_FLASH_ATTN_SRC_DIR=/build/vllm-offline/source/vllm/.deps/vllm-flash-attn-src
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export CCACHE_NOHASHDIR="true"
export VLLM_PYTHON_EXECUTABLE=$(which python3)

# optional: use installed torch
# python use_existing_torch.py

# The major building command will take a long time.
uv build --wheel --no-build-isolation --no-index \
--find-links /build/vllm-offline/wheels/build \
--find-links /build/vllm-offline/wheels/runtime \
--find-links /build/vllm-offline/wheels/common 2>&1 | tee build-wheel.log &

# install vllm from wheel
uv pip install --no-index --no-deps dist/vllm-*.cu129-cp312-cp312-linux_x86_64.whl

# import check
python3 -c "import vllm._C; print('vLLM extension loaded successfully')"
```

## Test LLM generate

```python
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.2) 
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r} Generated text: {output.outputs[0].text!r}")
```
