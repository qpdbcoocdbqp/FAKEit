# Try: using container's prebuild vllm, copy from image, this is wrong

## Step1: copy vllm from container image
```bash
docker run -it \
-v "/$(pwd)/volume:/build/vllm-offline" \
--entrypoint "" \
vllm/vllm-openai:v0.20.0-cu129 bash

tar -zcf /build/vllm-offline/pylib.tar.gz /usr/local/lib/python3.12/dist-packages
tar -zcf /build/vllm-offline/pybin.tar /usr/local/bin/

```

## Step2: restore to almalinux:8
```bash
docker run -it --rm --name vllm-dev \
--gpus all \
-v "/$(pwd)/volume:/build/vllm-offline" \
almalinux:8 bash

tar -zxf /build/vllm-offline/pybin.tar -C /usr/local/bin/
tar -zxf /build/vllm-offline/pylib.tar.gz -C /usr/local/lib/python3.12/
sh /build/vllm-offline/install_uv.sh
ln -s /root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 /usr/bin/python3
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.12/dist-packages

```
## Step3: run 
```bash
python -c "from vllm import LLM, SamplingParams; llm = LLM(model='facebook/opt-125m'); sampling_params = SamplingParams(max_tokens=10); outputs = llm.generate('Hello, world!', sampling_params); print(outputs[0].outputs[0].text)"

```
## Step4: Failed reason

```bash
ImportError: /usr/lib64/libc.so.6: version `GLIBC_2.34' not found (required by /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so)  
```

### Why it failed:
1. **Path Issue**: The libraries were extracted to `/usr/local/lib/python3.12/dist-packages`, but the `uv`-installed Python only looks in its own `site-packages` by default.
2. **GLIBC Compatibility**: The `vllm` binary/wheels you are using were compiled on a newer system (like Ubuntu 22.04 or RHEL 9). AlmaLinux 8's GLIBC is too old to run them.

## Clean container
```bash
docker stop vllm-dev
docker rm vllm-dev
```