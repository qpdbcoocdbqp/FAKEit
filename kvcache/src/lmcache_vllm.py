import time
import openai

client = openai.OpenAI(base_url="http://localhost:18001/v1", api_key="***")
print(client.models.list())

# Long shared prefix
prefix = """
# A KV Cache Management Layer for Scalable LLM Inference

## Updates
- [2026/05] 🔥 Agentic workload benchmark on AMD MI300X ([blog](https://blog.lmcache.ai/en/2026/05/12/benchmarking-lmcache-for-multi-turn-agentic-workloads-on-amd-mi300x/)).
- [2026/04] 🔥 LMCache's new multiprocess (MP) architecture release ([blog](https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x/)).
- [2026/03] LMCache at GTC 2026 ([post](https://www.linkedin.com/posts/lmcache-lab_llm-opensource-nvidiagtc-activity-7442721875664826369-pMAu?utm_source=share&utm_medium=member_desktop&rcm=ACoAADkIIvQBTyG53kXXX70OZdE5rhpllYQqmIA)).
- [2026/01] LMCache multi-node P2P CPU memory sharing, from experimental feature to production ([blog](https://blog.lmcache.ai/en/2026/01/21/p2p-1/)).

<details>
<summary>More</summary>

- [2025/11] LMCache x CoreWeave accelerate efficient LLM inference for Cohere ([blog](https://blog.lmcache.ai/en/2025/10/29/breaking-the-memory-barrier-how-lmcache-and-coreweave-power-efficient-llm-inference-for-cohere/)).
- [2025/10] LMCache joins the PyTorch Foundation and Tensormesh unveiled ([blog](https://blog.lmcache.ai/en/2025/10/31/tensormesh-unveiled-and-lmcache-joins-the-pytorch-foundation/), [PyTorch](https://pytorch.org/blog/lmcache-joins-pytorch-ecosystem/)).
- [2025/09] NVIDIA Dynamo integrates LMCache, accelerating LLM inference ([blog](https://blog.lmcache.ai/en/2025/09/18/nvidia-dynamo-integrates-lmcache-accelerating-llm-inference/)).
- [2025/08] 🎉 LMCache hits 5,000+ GitHub stars ([blog](https://blog.lmcache.ai/en/2025/08/28/%f0%9f%8e%89-lmcache-hits-5000-github-stars-thank-you-community/)).
- [2025/08] LMCache supports gpt-oss (20B/120B) on day 1 ([blog](https://blog.lmcache.ai/en/2025/08/05/lmcache-supports-gpt-oss-20b-120b-on-day-1/)).
- [2025/07] Get faster LLM inference and cheaper responses with LMCache and Redis ([Redis blog](https://redis.io/blog/get-faster-llm-inference-and-cheaper-responses-with-lmcache-and-redis/)).
- [2025/07] LMCache extends its turbo-boost to multimodal models in vLLM V1 ([blog](https://blog.lmcache.ai/en/2025/07/03/lmcache-extends-its-turbo-boost-to-multimodal-models-in-vllm-v1/)).
- [2025/06] LLM Production Stack goes cross-hardware: AMD, Arm and Ascend ([blog](https://blog.lmcache.ai/en/2025/06/20/llm-production-stack-goes-cross-hardware-ascend-arm-and-amd-support-incoming/)).

</details>

## About

LMCache is a **KV cache management layer** for LLM inference. It turns KV cache from a temporary state into reusable *AI-native knowledge* that can be *stored* persistently, *reused* across multiple serving engines, *monitored* with an observability stack, and *transformed* for better generation quality. As a result, LMCache **reduces TTFT** (time-to-first-token) and **improves throughput**, especially for long-context agentic, multi-turn conversation, and knowledge-augmented workloads (e.g., RAG).

LMCache is **vendor-neutral**. It can be used as a KV cache layer for a range of mainstream open-source serving engines, inference frameworks, hardware vendors, storage systems, and infrastructure providers. The vendor neutrality allows users to freely switch between serving engines and storage vendors, while reusing the stored KV caches.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="asset/deployment_modes_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="asset/deployment_modes_light.png">
  <img alt="LMCache Deployment Modes" src="asset/deployment_modes_light.png">
</picture>
</p>

### Key features

- **Engine-independent deployment**: LMCache, as a standalone daemon process, manages KV cache independently from the inference engine process, so that KV cache will not be lost even if the inference engine crashes (i.e., no fate-sharing with engines).

- **Persistent, tiered KV cache offloading and reuse**: Move KV caches out of GPU memory into a tiered storage hierarchy spanning CPU memory, local storage, and remote backends, enabling reuse across requests, sessions, and engine instances to reduce repeated prefill computation and improve TTFT.

- **Production-level KV cache observability**: LMCache provides a rich set of KV cache observability metrics, including typical Kubernetes metrics (health monitoring, performance diagnostics), KV-cache-specific metrics (request-level and token-level prefix cache hits, lifecycle, request-level KV cache performance), management metrics (user-specific usage), and more.

- **Pluggable storage and transport backends**: Easily integrate remote storage and KV transfer backends through a unified interface, enabling KV cache offloading and sharing across storage providers. Through this interface, LMCache supports storage backends including CPU RAM, local disk (SSD), Redis/Valkey, Mooncake, InfiniStore, S3-compatible object storage, NIXL, and GDS.

- **Non-prefix KV reuse**: Extend KV reuse beyond prefix caching by reusing cached KV blocks at any position in the prompt. This leverages CacheBlend to selectively recompute tokens for quality recovery.

- **PD disaggregation and KV transfer**: Support KV cache transfer from prefill workers to decode workers over NVLink, RDMA, or TCP through transport layers such as NIXL.

- **Pluggable KV transformation**: A simple interface for researchers to write compression, token dropping, and custom serialization through a flexible SERDE interface.

LMCache is becoming an integral layer in the LLM inference *ecosystem*, with *community*-driven integration with serving engines, inference frameworks, hardware vendors, storage systems, and infrastructure providers:

<p align="center">
  <img src="asset/ecosystem.png" alt="LMCache ecosystem">
</p>

## Getting Started

To use LMCache, simply install `lmcache` from your package manager, e.g. pip:
```bash
pip install lmcache
```

For more setup options and examples, see:
- [Installation](https://docs.lmcache.ai/getting_started/installation.html)
- [Quickstart](https://docs.lmcache.ai/getting_started/quickstart.html)
- [LMCache Recipes](https://docs.lmcache.ai/recipes/index.html)
- [CLI Reference](https://docs.lmcache.ai/cli/index.html)
- [Benchmarking Guide](https://docs.lmcache.ai/getting_started/benchmarking.html)
- [Production Deployment](https://docs.lmcache.ai/mp/deployment.html)

## Contributing
We welcome and value contributions and collaborations. Join us in improving LMCache. Check out the [Contributing Guide](https://docs.lmcache.ai/developer_guide/contributing.html) or join our [Slack community](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3zxjao8h0-lRfBfnLqbALOtLsWn2ITxA) to get started.

## Adoption and Partnerships
LMCache has a growing community of developers, researchers, industry adopters, and partners building the next generation of efficient LLM inference systems.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/partner_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="asset/partner_light.png">
    <img alt="LMCache Adoption and Partnerships" src="asset/partner_light.png">
  </picture>
</p>

As an independent open-source project, LMCache is becoming the de-facto standard for KV Cache management in LLM inference. Its continued development and community work are supported in part by [Tensormesh](https://www.tensormesh.ai/).

"""

messages = [
    {"role": "user", "content": prefix + " Say OK."},  # one message saves template tokens
]

def timed_call():
    t0 = time.perf_counter()
    client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_tokens=1,
        temperature=0.0,
    )
    return time.perf_counter() - t0

cold = timed_call()   # run 1: stores KV
hot = timed_call()    # run 2: should reuse KV

print(f"cold: {cold:.3f}s  hot: {hot:.3f}s  speedup: {cold/hot:.2f}x")
