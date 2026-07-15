import ctypes
import logging
import struct
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# GGML type -> numpy dtype mapping
# ============================================================

# Only pure floating-point types can be read directly via np.frombuffer;
# quantized types require dequantization.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 8
GGML_TYPE_BF16 = 30

GGML_DTYPE_MAP = {
    GGML_TYPE_F32: (np.float32, 4),
    GGML_TYPE_F16: (np.float16, 2),
    GGML_TYPE_BF16: (None, 2),  # bfloat16 needs manual conversion
}

GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1",
    8: "q8_0", 30: "bf16",
}

# File header magic. NOTE: verify this against the actual llama.cpp
# state_write() implementation for your version -- the byte order/value
# of the magic can differ between llama.cpp releases.
MAGIC = b"qsgg"


# ============================================================
# Data structures
# ============================================================

@dataclass
class KVCell:
    """Metadata for a single token's KV cache slot."""
    pos: int                  # RoPE position
    seq_ids: List[int]        # Sequence IDs this cell belongs to


@dataclass
class KVLayer:
    """K and V vectors for a single transformer layer."""
    layer_idx: int            # Layer order as stored in the file (0-based)
    cell_count: int

    k_type: int                # ggml type id
    k_row_size: int             # bytes per row
    k_data: bytes                # raw bytes

    v_type: int
    v_row_size: int             # bytes per row (v_trans=False) or element size
    v_data: bytes

    v_trans: bool                       # True if V is stored transposed
    n_embd_v_gqa: Optional[int]         # only set when v_trans=True

    def k_numpy(self) -> np.ndarray:
        """
        Reconstruct the K tensor -> shape: [cell_count, k_row_size / dtype_size].
        Quantized ggml types are returned as raw uint8 arrays.
        """
        return _ggml_to_numpy(self.k_data, self.k_type, self.cell_count)

    def v_numpy(self) -> np.ndarray:
        """
        Reconstruct the V tensor -> shape: [cell_count, n_embd_v_gqa].
        If v_trans=True, the data is un-transposed before being returned.
        """
        if not self.v_trans:
            return _ggml_to_numpy(self.v_data, self.v_type, self.cell_count)

        # Transposed layout: data is [n_embd_v_gqa, cell_count] -> transpose
        # back to [cell_count, n_embd_v_gqa].
        arr = _ggml_to_numpy(self.v_data, self.v_type, self.n_embd_v_gqa * self.cell_count)
        arr = arr.reshape(self.n_embd_v_gqa, self.cell_count)
        return arr.T


@dataclass
class KVStream:
    """Full data for one stream (usually only stream 0 is used)."""
    stream_idx: int
    cell_count: int
    cells: List[KVCell]        # metadata: pos + seq_ids
    v_trans: bool
    layers: List[KVLayer]


@dataclass
class KVCacheFile:
    """Parsed result of a slot*.bin file."""
    magic: bytes
    version: int
    n_tokens: int               # token count from the file header
    token_list: List[int]       # processed token ids
    n_stream: int
    streams: List[KVStream]


# ============================================================
# Helpers
# ============================================================

def _ggml_to_numpy(data: bytes, ggml_type: int, n_rows: int) -> np.ndarray:
    """
    Convert raw bytes to a numpy array.
    Pure floating-point types are read directly via frombuffer;
    quantized types are returned as raw uint8 arrays.
    """
    if ggml_type == GGML_TYPE_F32:
        return np.frombuffer(data, dtype=np.float32).reshape(n_rows, -1)
    elif ggml_type == GGML_TYPE_F16:
        return np.frombuffer(data, dtype=np.float16).reshape(n_rows, -1)
    elif ggml_type == GGML_TYPE_BF16:
        # bfloat16: reinterpret raw u16 bits as f32 (pad high 16 bits with zeros)
        raw = np.frombuffer(data, dtype=np.uint16)
        f32_bits = raw.astype(np.uint32) << 16
        return f32_bits.view(np.float32).reshape(n_rows, -1)
    else:
        # Quantized type (q4_0, q8_0, etc.) -> return raw uint8 bytes
        type_name = GGML_TYPE_NAMES.get(ggml_type, f"type{ggml_type}")
        logger.warning("quantized type '%s' - returning raw uint8 array", type_name)
        return np.frombuffer(data, dtype=np.uint8).reshape(n_rows, -1)


# ============================================================
# Main parser (pure binary parsing, no model required)
# ============================================================

def read_llama_kvcache(path: str, verbose: bool = True) -> KVCacheFile:
    """
    Parse a llama.cpp slot*.bin file and reconstruct the full KV tensor structure.

    Format reference: llama-kv-cache.cpp
      state_write() / state_write_meta() / state_write_data()

    Returns
    -------
    KVCacheFile
        .streams[s].cells[i]              -> KVCell(pos, seq_ids)
        .streams[s].layers[l].k_numpy()   -> np.ndarray [cell_count, k_dim]
        .streams[s].layers[l].v_numpy()   -> np.ndarray [cell_count, v_dim]
    """
    log = logger.info if verbose else logger.debug

    with open(path, "rb") as f:
        raw = f.read()

    buf = memoryview(raw)
    pos = 0

    def rd(fmt):
        nonlocal pos
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, buf, pos)
        pos += size
        return vals[0] if len(vals) == 1 else vals

    def rd_bytes(n):
        nonlocal pos
        data = bytes(buf[pos:pos + n])
        pos += n
        return data

    # ── File header ──────────────────────────────────────────
    # Format: magic(4) + version(4) + n_tokens(4)
    #         + token_list(n_tokens x i32)
    #         + state_write() payload
    magic = rd_bytes(4)
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r} (expected {MAGIC!r})")
    version = rd("<I")
    n_tokens = rd("<I")

    log("magic    : %s", magic)
    log("version  : %s", version)
    log("n_tokens : %s", n_tokens)
    log("file size: %s bytes", f"{len(raw):,}")

    # ── Token list ────────────────────────────────────────────
    token_list = list(struct.unpack_from(f"<{n_tokens}i", buf, pos))
    pos += n_tokens * 4
    log("token_list[:8]: %s", token_list[:8])

    # ── state_write() -- repeats for multiple KV cache blocks ──
    # Some models (e.g. Qwen2.5) have two KV caches: full attention + SWA.
    # Each block: n_stream(4B) -> per stream (cell_count -> meta -> data)
    all_streams: List[KVStream] = []
    block_idx = 0

    while pos < len(raw):
        n_stream = rd("<I")
        log("[KV block %d] n_stream=%d", block_idx, n_stream)

        for s in range(n_stream):
            cell_count = rd("<I")
            log("  [stream %d] cell_count = %d", s, cell_count)

            if cell_count == 0:
                all_streams.append(KVStream(s, 0, [], False, []))
                continue

            # ── state_write_meta(): per-cell metadata ─────────────
            cells = []
            for _ in range(cell_count):
                pos_val = rd("<i")     # int32 RoPE position
                n_seq_id = rd("<I")    # uint32
                seq_ids = [rd("<i") for _ in range(n_seq_id)]
                cells.append(KVCell(pos=pos_val, seq_ids=seq_ids))

            log("    pos range: %d ~ %d", cells[0].pos, cells[-1].pos)

            # ── state_write_data(): K then V tensors ──────────────
            v_trans_u32 = rd("<I")
            n_layer = rd("<I")
            v_trans = bool(v_trans_u32)

            log("    v_trans=%s, n_layer=%d", v_trans, n_layer)

            # ── Read all K tensors ────────────────────────────────
            k_blocks = []
            for _ in range(n_layer):
                k_type = rd("<i")      # int32 ggml type
                k_size_row = rd("<Q")  # uint64 bytes per row
                total = k_size_row * cell_count
                k_data = rd_bytes(total)
                k_blocks.append((k_type, k_size_row, k_data))

            type_name = GGML_TYPE_NAMES.get(k_blocks[0][0], f"type{k_blocks[0][0]}")
            log("    K: type=%s, row=%dB x %d layers", type_name, k_blocks[0][1], n_layer)

            # ── Read all V tensors ────────────────────────────────
            v_blocks = []
            if not v_trans:
                for _ in range(n_layer):
                    v_type = rd("<i")
                    v_size_row = rd("<Q")
                    total = v_size_row * cell_count
                    v_data = rd_bytes(total)
                    v_blocks.append((v_type, v_size_row, None, v_data))
            else:
                for _ in range(n_layer):
                    v_type = rd("<i")
                    v_size_el = rd("<I")       # uint32 element size
                    n_embd_v_gqa = rd("<I")    # uint32 embedding dim
                    total = v_size_el * n_embd_v_gqa * cell_count
                    v_data = rd_bytes(total)
                    v_blocks.append((v_type, v_size_el, n_embd_v_gqa, v_data))

            type_name = GGML_TYPE_NAMES.get(v_blocks[0][0], f"type{v_blocks[0][0]}")
            log("    V: type=%s x %d layers", type_name, n_layer)

            # ── Assemble KVLayer objects ───────────────────────────
            layers = []
            for l in range(n_layer):
                kt, kr, kd = k_blocks[l]
                vt, vr, ve, vd = v_blocks[l]
                layers.append(KVLayer(
                    layer_idx=l,
                    cell_count=cell_count,
                    k_type=kt,
                    k_row_size=kr,
                    k_data=kd,
                    v_type=vt,
                    v_row_size=vr,
                    v_data=vd,
                    v_trans=v_trans,
                    n_embd_v_gqa=ve,
                ))

            all_streams.append(KVStream(
                stream_idx=len(all_streams),
                cell_count=cell_count,
                cells=cells,
                v_trans=v_trans,
                layers=layers,
            ))

        block_idx += 1

    log("parsed %s / %s bytes", f"{pos:,}", f"{len(raw):,}")
    log("total KV blocks: %d, total streams: %d", block_idx, len(all_streams))

    return KVCacheFile(
        magic=magic,
        version=version,
        n_tokens=n_tokens,
        token_list=token_list,
        n_stream=len(all_streams),
        streams=all_streams,
    )


# ============================================================
# llama_state_seq_load_file -- load via llama-cpp-python low-level API
# ============================================================

def load_kvcache_into_ctx(
    model_path: str,
    bin_path: str,
    dest_seq_id: int = 0,
    n_ctx: int = 4096,
) -> dict:
    """
    Restore slot0.bin into a context using llama_state_seq_load_file.

    Parameters
    ----------
    model_path  : path to the GGUF model (must match the model that
                  produced the .bin file)
    bin_path    : path to slot0.bin
    dest_seq_id : destination sequence slot to restore into (usually 0)
    n_ctx       : context size (must be >= original token count)

    Returns
    -------
    dict containing n_token_count and the ctx object (usable for inference)
    """
    from llama_cpp import llama_cpp as _lib

    model = None
    ctx = None
    try:
        # --- initialize model & context ---
        model_params = _lib.llama_model_default_params()
        model = _lib.llama_load_model_from_file(model_path.encode(), model_params)
        if not model:
            raise RuntimeError(f"Failed to load model: {model_path}")

        ctx_params = _lib.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx = _lib.llama_new_context_with_model(model, ctx_params)
        if not ctx:
            raise RuntimeError("Failed to create context")

        # --- buffers to receive restored tokens ---
        tokens_out = (ctypes.c_int32 * n_ctx)()
        n_token_count_out = ctypes.c_size_t(0)

        # --- load the kv cache ---
        n_bytes = _lib.llama_state_seq_load_file(
            ctx,
            bin_path.encode(),
            dest_seq_id,
            tokens_out,
            n_ctx,
            ctypes.byref(n_token_count_out),
        )

        if n_bytes == 0:
            raise RuntimeError(f"llama_state_seq_load_file failed: {bin_path}")

        n_tokens = int(n_token_count_out.value)
        if n_tokens > n_ctx:
            raise RuntimeError(
                f"Reported token count ({n_tokens}) exceeds buffer capacity ({n_ctx})"
            )
        tokens = list(tokens_out[:n_tokens])

        logger.info("loaded   : %s", bin_path)
        logger.info("n_bytes  : %s", f"{n_bytes:,}")
        logger.info("n_tokens : %s", n_tokens)
        logger.info("tokens[:8]: %s", tokens[:8])

        return {
            "n_bytes": n_bytes,
            "n_tokens": n_tokens,
            "tokens": tokens,
            "ctx": ctx,
            "model": model,
        }
    except Exception:
        # Avoid leaking model/context on failure.
        if ctx:
            _lib.llama_free(ctx)
        if model:
            _lib.llama_free_model(model)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    bin_path = "tmp/slot0.bin"

    # ── Method 1: pure binary parsing, reconstruct tensor structure ──
    kv = read_llama_kvcache(bin_path)

    print("\n=== All KV streams ===")
    for i, s in enumerate(kv.streams):
        if not s.layers:
            continue
        l0 = s.layers[0]
        K = l0.k_numpy()
        V = l0.v_numpy()
        print(f"stream[{i}]: cells={s.cell_count}, layers={len(s.layers)}, "
              f"K={K.shape} {K.dtype}, V={V.shape} {V.dtype}")

    # First stream's layer 0 vectors
    s0 = kv.streams[0]
    K = s0.layers[0].k_numpy()
    V = s0.layers[0].v_numpy()
    print(f"\nstream[0] layer[0] K[0,:8] = {K[0, :8]}")
    print(f"stream[0] layer[0] V[0,:8] = {V[0, :8]}")

    # ── Method 2: restore into a context via llama-cpp-python ──
    # MODEL_PATH = "path/to/your/model.gguf"  # <- set actual path
    # ctx_result = load_kvcache_into_ctx(
    #     model_path=MODEL_PATH,
    #     bin_path=bin_path,
    #     dest_seq_id=0,
    #     n_ctx=4096,
    # )
