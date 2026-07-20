"""
docker compose up lmcache-server -d
docker cp ./src/lmcache_mgmt_client.py lmcache-server:/workspace/lmcache_mgmt_client.py

lmcache_mgmt_client.py

A real (not blind-probing) ZMQ client for LMCache's MP server, built on
top of the library's own MessageQueueClient + RequestType, rather than
hand-rolled msgspec encoding.

Why this approach instead of a from-scratch client:
  - The wire format (mq.py: MessageQueueClient.process_outbound_task /
    process_inbound) is `msgspec.msgpack`-encoded per-field, with strict
    type validation on decode. Getting this byte-exact by hand is fragile
    and version-dependent.
  - The RequestType -> payload/response class mapping is defined and
    *validated* inside the lmcache package itself (protocol.py ->
    protocols/base.py, engine.py, controller.py, debug.py). It is not
    meant to be reverse-engineered; it's a first-class part of the
    package's Python API.
  - Since your deployment's docker image (lmcache/vllm-openai) already
    has `lmcache` installed, the correct move is to import and use it
    directly, from a Python environment that has the same lmcache
    version as your server.

Scope of what this script can safely do:
  Only the ManagementModule request types are used here -- CLEAR,
  GET_CHUNK_SIZE, PING, NOOP -- because their handlers
  (lmcache/v1/multiprocess/modules/management.py) take/return plain
  Python types (int, bool, str, None) with no GPU IPC involved.

  LOOKUP / STORE / RETRIEVE and friends are NOT exposed here on purpose:
  - LOOKUP is fire-and-forget (returns None) and requires a real
    IPCCacheServerKey tied to a GPU context already registered on the
    server (model_name/world_size that match a running vLLM worker);
    calling it standalone just hits the "no GPU context found" error path.
  - STORE/RETRIEVE payloads carry DeviceIPCWrapper (CUDA IPC / shared-
    memory handles) -- there is no "tensor bytes over ZMQ" request type
    to call here even if you wanted the raw KV data.

Usage:
    # Run in an environment where `import lmcache` succeeds and the
    # version matches your lmcache-server container.
    python3 lmcache_mgmt_client.py --host tcp://localhost --port 6555 noop
    python3 lmcache_mgmt_client.py --host tcp://localhost --port 6555 ping
    python3 lmcache_mgmt_client.py --host tcp://localhost --port 6555 chunk-size
    python3 lmcache_mgmt_client.py --host tcp://localhost --port 6555 clear --yes
"""

import argparse
import sys

import zmq

try:
    from lmcache.v1.multiprocess.mq import MessageQueueClient
    from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
except ImportError as e:
    print(
        "[!] Could not import lmcache. This script must run in an "
        "environment with the SAME lmcache version as your server "
        "(e.g. inside the lmcache/vllm-openai container, or a venv "
        "with a matching `pip install lmcache`).\n"
        f"    Import error: {e}",
        file=sys.stderr,
    )
    sys.exit(1)


def call(client: MessageQueueClient, request_type: RequestType, payloads: list, timeout_s: float = 5.0):
    """Submit a request and block for the result."""
    future = client.submit_request(
        request_type,
        payloads,
        response_cls=get_response_class(request_type),
    )
    return future.result(timeout=timeout_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="tcp://localhost")
    ap.add_argument("--port", type=int, default=6555)
    ap.add_argument("--timeout", type=float, default=5.0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("noop", help="Health check -> expects 'OK'")
    p_ping = sub.add_parser("ping", help="Ping the server, optionally as a named worker instance")
    p_ping.add_argument("--instance-id", type=int, default=None)
    sub.add_parser("chunk-size", help="Query the server's configured KV chunk size")
    p_clear = sub.add_parser("clear", help="Clear ALL L1 (CPU) KV cache data -- destructive")
    p_clear.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")

    args = ap.parse_args()

    zmq_ctx = zmq.Context.instance()
    endpoint = f"{args.host}:{args.port}"
    client = MessageQueueClient(endpoint, zmq_ctx)

    try:
        if args.cmd == "noop":
            result = call(client, RequestType.NOOP, [])
            print(f"NOOP -> {result!r}")

        elif args.cmd == "ping":
            result = call(client, RequestType.PING, [args.instance_id])
            print(f"PING(instance_id={args.instance_id}) -> {result!r}")

        elif args.cmd == "chunk-size":
            result = call(client, RequestType.GET_CHUNK_SIZE, [])
            print(f"GET_CHUNK_SIZE -> {result!r}")

        elif args.cmd == "clear":
            if not args.yes:
                confirm = input(
                    "This clears ALL KV cache data currently held in L1 "
                    "(CPU) memory on the server. Type 'yes' to continue: "
                )
                if confirm.strip().lower() != "yes":
                    print("Aborted.")
                    return
            result = call(client, RequestType.CLEAR, [])
            print(f"CLEAR -> {result!r}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
