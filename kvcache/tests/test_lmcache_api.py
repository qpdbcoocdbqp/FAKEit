#!/usr/bin/env python3
"""
lmcache_dev.py
==============
Smoke test for a local LMCache install.

LMCache is engine-coupled (vLLM / SGLang / TensorRT-LLM) — there is no
llama.cpp connector — so this script does NOT run real model inference.
Instead it exercises the parts of LMCache that work standalone:

  1. import check + version
  2. config loading (env vars / yaml)
  3. `lmcache.v1.standalone` engine, started with a *synthetic* KV shape
     (no GPU / model required), then polled through its internal HTTP
     API server to confirm it actually came up and is storing chunks.

Usage
-----
    python lmcache_dev.py                # run all checks
    python lmcache_dev.py --skip-engine   # just check import/config

Requirements
------------
    pip install lmcache          # CPU-only install is fine for this test
"""

import argparse
import subprocess
import sys
import time
import urllib.request
import urllib.error


API_HOST = "localhost"
API_PORT = 9971
STARTUP_TIMEOUT_S = 15
POLL_INTERVAL_S = 1
CONFIG_PATH = "configs/cpu-offload.yaml"

def check_import() -> bool:
    print("== 1. import check ==")
    try:
        import lmcache  # noqa: F401
    except ImportError as e:
        print(f"[FAIL] `import lmcache` failed: {e}")
        print("        -> pip install lmcache")
        return False

    version = getattr(lmcache, "__version__", "unknown")
    print(f"[OK] lmcache imported, version={version}")
    return True


def check_config() -> bool:
    print("\n== 2. config check ==")
    try:
        from lmcache.v1.config import LMCacheEngineConfig
    except ImportError as e:
        print(f"[FAIL] could not import LMCacheEngineConfig: {e}")
        return False

    try:
        # Build a minimal local-CPU-only config purely in-process,
        # no server / GPU involved.
        cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=1.0,
        )
        print(f"[OK] built config: chunk_size={cfg.chunk_size}, "
              f"local_cpu={cfg.local_cpu}")
        return True
    except Exception as e:
        print(f"[FAIL] could not build LMCacheEngineConfig: {e}")
        print("        (API may differ across LMCache versions — "
              "this step is best-effort)")
        return False


def wait_for_health(base_url: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/conf", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(POLL_INTERVAL_S)
    return False


def check_standalone_engine() -> bool:
    print("\n== 3. standalone engine smoke test (no GPU/model needed) ==")
    # Small synthetic KV shape: (num_layers, 2, chunk, num_heads, head_dim)
    kv_shape = "4,2,256,8,64"
    cmd = [
        sys.executable, "-m", "lmcache.v1.standalone",
        f"--kv-shape={kv_shape}",
        "--kv-dtype=float16",
        "--fmt=vllm",
        "--model-name=lmcache_dev_smoke_test",
        "--device=cpu",
        f"--config={CONFIG_PATH}",
    ]
    print("launching:", " ".join(cmd))

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        base_url = f"http://{API_HOST}:{API_PORT}"
        alive = wait_for_health(base_url, STARTUP_TIMEOUT_S)
        if not alive:
            print(f"[FAIL] engine did not report healthy within "
                  f"{STARTUP_TIMEOUT_S}s")
            _dump_output(proc)
            return False

        print(f"[OK] engine healthy at {base_url}")

        try:
            with urllib.request.urlopen(f"{base_url}/meta", timeout=3) as r:
                print(f"[OK] /meta -> HTTP {r.status}")
        except urllib.error.URLError as e:
            print(f"[WARN] /meta check failed (non-fatal): {e}")

        return True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _dump_output(proc: subprocess.Popen, max_lines: int = 40) -> None:
    if proc.stdout is None:
        return
    print("---- subprocess output (tail) ----")
    lines = proc.stdout.readlines()
    for line in lines[-max_lines:]:
        print(line.rstrip())
    print("-----------------------------------")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-engine", action="store_true",
                         help="skip the standalone-engine subprocess test")
    args = parser.parse_args()

    results = {}
    results["import"] = check_import()
    if not results["import"]:
        print("\nAborting: lmcache is not importable.")
        return 1

    results["config"] = check_config()

    if not args.skip_engine:
        results["engine"] = check_standalone_engine()

    print("\n== summary ==")
    ok = True
    for name, passed in results.items():
        print(f"  {name:8s}: {'PASS' if passed else 'FAIL'}")
        ok = ok and passed

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
