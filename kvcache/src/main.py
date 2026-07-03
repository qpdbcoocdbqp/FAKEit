import os
import json
import time
import sqlite3
import hashlib
import shutil
import zstandard as zstd
from pathlib import Path
from typing import Optional, List, Dict, Any


# ============================================================
# Prompt Canonicalizer
# ============================================================

def canonicalize_messages(messages: List[Dict[str, str]]) -> str:
    """
    Normalize prompt messages to stable format.
    """

    normalized = []

    for msg in messages:
        normalized.append({
            "role": msg["role"].strip(),
            "content": " ".join(msg["content"].split())
        })

    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")
    )


# ============================================================
# Prefix Fingerprint
# ============================================================

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prefix_hashes(tokens: List[str], step=256):
    """
    Generate rolling prefix hashes.

    Example:
        token[0:256]
        token[0:512]
        token[0:768]
    """

    results = []

    for i in range(step, len(tokens) + 1, step):
        prefix = " ".join(tokens[:i])
        results.append({
            "tokens": i,
            "hash": sha256_text(prefix)
        })

    return results


# ============================================================
# Compression Helpers
# ============================================================

def compress_file(src: str, dst: str):
    cctx = zstd.ZstdCompressor(level=3)

    with open(src, "rb") as fin:
        with open(dst, "wb") as fout:
            cctx.copy_stream(fin, fout)


def decompress_file(src: str, dst: str):
    dctx = zstd.ZstdDecompressor()

    with open(src, "rb") as fin:
        with open(dst, "wb") as fout:
            dctx.copy_stream(fin, fout)


# ============================================================
# KV Cache Storage
# ============================================================

class KVCacheStorage:

    def __init__(
        self,
        db_path="kvcache.db",
        storage_dir="./kv_storage"
    ):
        self.db_path = db_path
        self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    # --------------------------------------------------------

    def _init_db(self):

        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS kv_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            prompt_hash TEXT,
            prefix_hash TEXT,
            prefix_tokens INTEGER,
            kv_path TEXT,
            created_at INTEGER,
            last_used INTEGER,
            hit_count INTEGER,
            size_bytes INTEGER
        )
        """)

        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_prefix_hash
        ON kv_cache(prefix_hash)
        """)

        self.conn.commit()

    # --------------------------------------------------------

    def save_cache(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        kv_binary_path: str,
        tokenized_prompt: List[str]
    ) -> Dict[str, Any]:

        canonical = canonicalize_messages(messages)

        prompt_hash = sha256_text(canonical)

        prefixes = build_prefix_hashes(tokenized_prompt)

        saved_entries = []

        for prefix in prefixes:

            prefix_hash = prefix["hash"]
            prefix_tokens = prefix["tokens"]

            cache_filename = (
                f"{model_name}_"
                f"{prefix_tokens}_"
                f"{prefix_hash}.kv.zst"
            )

            final_path = self.storage_dir / cache_filename

            compress_file(kv_binary_path, str(final_path))

            size_bytes = final_path.stat().st_size

            cursor = self.conn.cursor()

            cursor.execute("""
            INSERT INTO kv_cache (
                model_name,
                prompt_hash,
                prefix_hash,
                prefix_tokens,
                kv_path,
                created_at,
                last_used,
                hit_count,
                size_bytes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                prompt_hash,
                prefix_hash,
                prefix_tokens,
                str(final_path),
                int(time.time()),
                int(time.time()),
                0,
                size_bytes
            ))

            self.conn.commit()

            saved_entries.append({
                "prefix_tokens": prefix_tokens,
                "path": str(final_path)
            })

        return {
            "prompt_hash": prompt_hash,
            "entries": saved_entries
        }

    # --------------------------------------------------------

    def find_best_match(
        self,
        model_name: str,
        tokenized_prompt: List[str]
    ) -> Optional[Dict[str, Any]]:

        prefixes = build_prefix_hashes(tokenized_prompt)

        prefixes.reverse()

        cursor = self.conn.cursor()

        for prefix in prefixes:

            cursor.execute("""
            SELECT
                id,
                prefix_hash,
                prefix_tokens,
                kv_path,
                hit_count
            FROM kv_cache
            WHERE model_name = ?
            AND prefix_hash = ?
            ORDER BY prefix_tokens DESC
            LIMIT 1
            """, (
                model_name,
                prefix["hash"]
            ))

            row = cursor.fetchone()

            if row:

                cache_id = row[0]

                cursor.execute("""
                UPDATE kv_cache
                SET
                    hit_count = hit_count + 1,
                    last_used = ?
                WHERE id = ?
                """, (
                    int(time.time()),
                    cache_id
                ))

                self.conn.commit()

                return {
                    "cache_id": row[0],
                    "prefix_hash": row[1],
                    "prefix_tokens": row[2],
                    "kv_path": row[3],
                    "hit_count": row[4]
                }

        return None

    # --------------------------------------------------------

    def restore_cache(
        self,
        kv_path: str,
        output_path: str
    ):

        decompress_file(kv_path, output_path)

        return output_path

    # --------------------------------------------------------

    def cleanup_old_cache(
        self,
        max_age_seconds=86400
    ):

        cutoff = int(time.time()) - max_age_seconds

        cursor = self.conn.cursor()

        cursor.execute("""
        SELECT id, kv_path
        FROM kv_cache
        WHERE last_used < ?
        """, (cutoff,))

        rows = cursor.fetchall()

        for row in rows:

            cache_id = row[0]
            kv_path = row[1]

            try:
                os.remove(kv_path)
            except:
                pass

            cursor.execute("""
            DELETE FROM kv_cache
            WHERE id = ?
            """, (cache_id,))

        self.conn.commit()

    # --------------------------------------------------------

    def stats(self):

        cursor = self.conn.cursor()

        cursor.execute("""
        SELECT
            COUNT(*),
            SUM(size_bytes),
            SUM(hit_count)
        FROM kv_cache
        """)

        row = cursor.fetchone()

        return {
            "entries": row[0] or 0,
            "size_bytes": row[1] or 0,
            "total_hits": row[2] or 0
        }


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    storage = KVCacheStorage()

    messages = [
        {
            "role": "system",
            "content": "You are a coding assistant."
        },
        {
            "role": "user",
            "content": "Write a Python HTTP server."
        }
    ]

    tokenized_prompt = (
        canonicalize_messages(messages)
        .split()
    )

    # Example llama.cpp slot save output
    kv_binary = "slot1.bin"

    result = storage.save_cache(
        model_name="qwen3-14b",
        messages=messages,
        kv_binary_path=kv_binary,
        tokenized_prompt=tokenized_prompt
    )

    print("saved:", result)

    match = storage.find_best_match(
        model_name="qwen3-14b",
        tokenized_prompt=tokenized_prompt
    )

    print("match:", match)

    if match:

        restored = storage.restore_cache(
            match["kv_path"],
            "restored_slot.bin"
        )

        print("restored:", restored)

    print(storage.stats())
