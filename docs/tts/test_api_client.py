"""Smoke test for the Audio8 HTTP API.

Usage:
    python test_api_client.py --reference-text "Exact words in the recording"
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def request_json(url: str, method: str = "GET", body: bytes | None = None,
                headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
            content_type = response.headers.get("Content-Type", "")
            return response.status, data, content_type
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc


def multipart(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----audio8-test-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"),
            b"\r\n",
        ])
    content_type = mimetypes.guess_type(file_path.name)[0] or "audio/wav"
    chunks.extend([
        f"--{boundary}\r\n".encode(),
        (f'Content-Disposition: form-data; name="{file_field}"; '
         f'filename="{file_path.name}"\r\n').encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        file_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Audio8 TTS API")
    parser.add_argument("--base-url", default="http://localhost:8024")
    parser.add_argument("--audio", type=Path, default=Path("resource/reference.wav"))
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--voice", default="api_test_voice")
    parser.add_argument("--text", default="Hello from the Audio8 API server.")
    parser.add_argument("--output", type=Path, default=Path("api_test_output.wav"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    if not args.audio.is_file():
        print(f"Reference audio not found: {args.audio}", file=sys.stderr)
        return 2

    status, body, _ = request_json(f"{base_url}/health")
    health = json.loads(body)
    assert status == 200 and health["status"] == "ok", health
    print(f"[ok] health: {health}")

    body, content_type = multipart(
        {"text": args.reference_text, "name": args.voice,
         "overwrite": str(args.overwrite).lower()},
        "audio", args.audio,
    )
    status, body, _ = request_json(
        f"{base_url}/register_voice", method="POST", body=body,
        headers={"Content-Type": content_type},
    )
    registration = json.loads(body)
    assert status == 200 and registration["name"] == args.voice, registration
    print(f"[ok] register_voice: {registration['name']}")

    payload = json.dumps({"text": args.text, "voice": args.voice}).encode("utf-8")
    status, body, content_type = request_json(
        f"{base_url}/synthesize_to_wav", method="POST", body=payload,
        headers={"Content-Type": "application/json"},
    )
    assert status == 200 and body[:4] == b"RIFF" and "audio/wav" in content_type, (
        status, content_type
    )
    args.output.write_bytes(body)
    print(f"[ok] synthesize_to_wav: {args.output} ({len(body)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
