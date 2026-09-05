import warnings
import os
import sys
import time
import queue
import threading
import signal
import logging
import numpy as np
import soundcard as sc
from openai import OpenAI

warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK_SIZE  = 2048

# ==============================================================================
# Windows: Before importing faster_whisper, add nvidia pip wheel DLL paths
# so that ctranslate2 can locate libraries like cublas64_12.dll
# ==============================================================================
if sys.platform == "win32":
    try:
        import site
        for _sp in site.getsitepackages():
            _nvidia = os.path.join(_sp, "nvidia")
            if os.path.exists(_nvidia):
                for _sub in os.listdir(_nvidia):
                    _bin = os.path.join(_nvidia, _sub, "bin")
                    if os.path.exists(_bin):
                        os.add_dll_directory(_bin)
                        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


class HypothesisBuffer:
    """
    Maintains the common prefix of two consecutive inference results,
    acting as a buffer for confirmed committed words via Local Agreement.
    """

    def __init__(self):
        self.commited_in_buffer: list[tuple] = []   # Committed words not yet removed by chunk_at
        self.buffer:             list[tuple] = []   # Previous inference result
        self.new:                list[tuple] = []   # Current inference result

        self.last_commited_time = 0.0
        self.last_commited_word = None

    def insert(self, new: list[tuple], offset: float):
        """
        Adds current inference result (new) after adding offset,
        and removes prefixes overlapping with committed blocks (n-gram comparison up to 5-gram).
        """
        new = [(a + offset, b + offset, t) for a, b, t in new]
        # Only keep words after last committed time (allowing 0.1s timestamp error)
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if self.new:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    # Compare 1~5 gram, find longest matching tail and discard
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c    = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            for _ in range(i):
                                self.new.pop(0)
                            break

    def flush(self) -> list[tuple]:
        """
        Returns the longest common prefix between buffer and new (committed words).
        Also updates buffer = new (for next round comparison) and clears new.
        """
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            if not self.buffer:
                break
            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time: float):
        """Removes committed words within the audio range already trimmed by chunk_at"""
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self) -> list[tuple]:
        """Returns candidate words remaining in buffer that are not yet committed (tentative)"""
        return self.buffer


class FasterWhisperASR:
    """Official FasterWhisperASR backend with Windows CUDA hot fallback protection"""

    sep = ""  # faster-whisper attaches space to words directly, no extra separator needed

    def __init__(self, lan="en", modelsize="small", device="auto", compute_type="auto", beam_size=5):
        self.original_language = None if lan == "auto" else lan
        self.modelsize  = modelsize
        self.device     = device
        self.compute_type = compute_type
        self.beam_size  = beam_size
        self.transcribe_kargs: dict = {}
        self.model = self._load_model()

    def _load_model(self):
        from faster_whisper import WhisperModel

        dev = self.device
        if dev == "auto":
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"

        comp = self.compute_type
        if comp == "auto":
            comp = "float16" if dev == "cuda" else "int8"

        print(f"[Model] Loading faster-whisper ({self.modelsize}) — {dev} / {comp} ...")
        try:
            model = WhisperModel(self.modelsize, device=dev, compute_type=comp)
        except Exception as e:
            if dev == "cuda":
                print(f"[Warning] CUDA load failed ({e}), falling back to CPU ...")
                dev, comp = "cpu", "int8"
                model = WhisperModel(self.modelsize, device=dev, compute_type=comp)
            else:
                raise

        self.device       = dev
        self.compute_type = comp
        print(f"[Model] Loading completed ({dev} / {comp})")
        return model

    # ── Official Interface ───────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        try:
            segments, _ = self.model.transcribe(
                audio,
                language=self.original_language,
                initial_prompt=init_prompt or None,
                beam_size=self.beam_size,
                word_timestamps=True,
                condition_on_previous_text=True,   # Official: True (utilize prompt context)
                **self.transcribe_kargs,
            )
            return list(segments)
        except Exception as e:
            err = str(e).lower()
            if self.device == "cuda" and ("cublas" in err or "cuda" in err or "cudnn" in err):
                sys.stdout.write(f"\n[Warning] CUDA execution error detected ({e}), hot switching to CPU...\n")
                sys.stdout.flush()
                from faster_whisper import WhisperModel
                self.device       = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(self.modelsize, device="cpu", compute_type="int8")
                segments, _ = self.model.transcribe(
                    audio,
                    language=self.original_language,
                    initial_prompt=init_prompt or None,
                    beam_size=self.beam_size,
                    word_timestamps=True,
                    condition_on_previous_text=True,
                    **self.transcribe_kargs,
                )
                return list(segments)
            raise

    def ts_words(self, segments: list) -> list[tuple]:
        """Converts inference result to [(start, end, word), ...]"""
        out = []
        for seg in segments:
            if getattr(seg, "no_speech_prob", 0.0) > 0.9:   # Official threshold
                continue
            for w in seg.words:
                out.append((w.start, w.end, w.word))
        return out

    def segments_end_ts(self, segments: list) -> list[float]:
        """End timestamp for each segment, used by chunk_completed_segment"""
        return [s.end for s in segments]

    def use_vad(self):
        """Enable faster-whisper built-in VAD filter"""
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        """Switch to translation task (output English)"""
        self.transcribe_kargs["task"] = "translate"


class OnlineASRProcessor:
    """
    Official ufal/whisper_streaming OnlineASRProcessor,
    using Local Agreement strategy to output committed text in real-time without relying on speech pauses.
    """

    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr: FasterWhisperASR,
        buffer_trimming: tuple = ("segment", 15),
    ):
        """
        asr            : FasterWhisperASR instance
        buffer_trimming: ("segment"|"sentence", duration in seconds)
                         "segment"  -- Trims at Whisper segment boundary (default, better quality)
                         "sentence" -- Trims at sentence boundary (requires sentence segmenter)
        """
        self.asr = asr
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming
        self.init()

    def init(self, offset: float | None = None):
        """Resets processing state (called when starting a new paragraph)"""
        self.audio_buffer      = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer()
        self.buffer_time_offset = offset if offset is not None else 0.0
        if offset is not None:
            self.transcript_buffer.last_commited_time = offset
        self.commited: list[tuple] = []

    # ── Official Core Methods ─────────────────────────────────────────────────

    def insert_audio_chunk(self, audio: np.ndarray):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self) -> tuple[str, str]:
        """
        Returns (prompt, context):
          prompt  -- Last 200 characters of committed text scrolled out of audio window as initial inference prompt
          context -- Committed text still inside audio window (for debugging only)
        """
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        p = [t for _, _, t in self.commited[:k]]
        prompt_parts, length = [], 0
        while p and length < 200:
            x = p.pop(-1)
            length += len(x) + 1
            prompt_parts.append(x)

        non_prompt = self.commited[k:]
        return (
            self.asr.sep.join(prompt_parts[::-1]),
            self.asr.sep.join(t for _, _, t in non_prompt),
        )

    def process_iter(self) -> tuple:
        """
        Core iteration, processing current audio buffer.

        Returns: (beg, end, "Committed text") or (None, None, "")
        """
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt!r}")
        logger.debug(f"CONTEXT: {non_prompt!r}")
        logger.debug(
            f"transcribing {len(self.audio_buffer) / self.SAMPLING_RATE:.2f}s "
            f"from {self.buffer_time_offset:.2f}"
        )

        try:
            res  = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
            tsw  = self.asr.ts_words(res)
        except Exception as e:
            sys.stdout.write(f"\n[Inference Warning] {e}\n")
            sys.stdout.flush()
            return None, None, ""

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)

        logger.debug(f"COMPLETE: {self.to_flush(o)}")
        logger.debug(f"INCOMPLETE: {self.to_flush(self.transcript_buffer.complete())}")

        # Trim buffer according to strategy
        if o and self.buffer_trimming_way == "sentence":
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:
                self.chunk_completed_sentence()

        trim_sec = (
            self.buffer_trimming_sec
            if self.buffer_trimming_way == "segment"
            else 30
        )
        if len(self.audio_buffer) / self.SAMPLING_RATE > trim_sec:
            self.chunk_completed_segment(res)

        logger.debug(f"buffer length: {len(self.audio_buffer) / self.SAMPLING_RATE:.2f}s")
        return self.to_flush(o)

    def finish(self) -> tuple:
        """
        Called when stream ends, forcing flush of all remaining candidate words.
        Return format is identical to process_iter().
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"finish, non-committed: {f}")
        self.buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        return f

    # ── Buffer Trimming ──────────────────────────────────────────────────────

    def chunk_at(self, time: float):
        """Trims audio and hypothesis buffer at timestamp time"""
        self.transcript_buffer.pop_commited(time)
        cut = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut * self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def chunk_completed_segment(self, res: list):
        """
        Trims at second-to-last segment end timestamp,
        ensuring audio is not cut off mid-sentence (official strategy).
        """
        if not self.commited:
            return
        ends = self.asr.segments_end_ts(res)
        t = self.commited[-1][1]
        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                logger.debug(f"segment chunk at {e:.2f}s")
                self.chunk_at(e)
            else:
                logger.debug("last segment not within committed area")
        else:
            logger.debug("not enough segments to chunk")

    def chunk_completed_sentence(self):
        """Trims at sentence boundaries (requires sentence tokenizer)"""
        # This project primarily uses "segment" strategy, this method is preserved for extension
        pass

    # ── Auxiliary ─────────────────────────────────────────────────────────────

    def to_flush(self, words: list, sep: str | None = None, offset: float = 0) -> tuple:
        """
        Merges [(beg, end, word), ...] into a single output tuple.
        Returns (beg, end, "text") or (None, None, "")
        """
        if sep is None:
            sep = self.asr.sep
        if not words:
            return None, None, ""
        return (
            offset + words[0][0],
            offset + words[-1][1],
            sep.join(w[2] for w in words),
        )

    @property
    def tentative(self) -> tuple:
        """Candidate words in current buffer not yet confirmed (for real-time display)"""
        return self.to_flush(self.transcript_buffer.complete())


# ==============================================================================
# VACOnlineASRProcessor -- Official VAC (Voice Activity Controller) wrapper
#
# Differences from older utils.py:
#   - Older version only used RMS < 0.003 to skip silence, without Voice Activity Detection
#   - Official uses Silero VAD model to accurately detect speech start/end boundaries
#   - Stops accumulating audio during non-voice sections, drastically reducing unnecessary inferences
#   - Immediately calls finish() when voice end is detected, reducing latency
# ==============================================================================

class VACOnlineASRProcessor(OnlineASRProcessor):
    """
    Wraps OnlineASRProcessor with Silero VAD voice activity detection.
    Requires: pip install torch torchaudio
    """

    def __init__(self, online_chunk_size: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.online_chunk_size = online_chunk_size

        import torch
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        try:
            from silero_vad_iterator import FixedVADIterator
            self.vac = FixedVADIterator(vad_model)
        except ImportError:
            # utils contains (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
            VADIterator = utils[3]
            self.vac = VADIterator(vad_model)
        self._vac_init()

    def _vac_init(self):
        self.vac.reset_states()
        self.vac_audio_buffer        = np.array([], dtype=np.float32)
        self.vac_buffer_offset       = 0
        self.vac_status              = None   # "voice" | "nonvoice" | None
        self.vac_is_currently_final  = False
        self.vac_chunk_buffer_size   = 0

    def init(self, offset: float | None = None):
        super().init(offset)
        if hasattr(self, "vac"):   # super().__init__ calls init() first, before vac is established
            self._vac_init()

    def _clear_vac_buffer(self):
        self.vac_buffer_offset += len(self.vac_audio_buffer)
        self.vac_audio_buffer   = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio: np.ndarray):
        res = self.vac(audio)
        self.vac_audio_buffer = np.append(self.vac_audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0] - self.vac_buffer_offset
            if "start" in res and "end" not in res:
                self.vac_status = "voice"
                chunk = self.vac_audio_buffer[frame:]
                super().init(offset=(frame + self.vac_buffer_offset) / self.SAMPLING_RATE)
                super().insert_audio_chunk(chunk)
                self.vac_chunk_buffer_size += len(chunk)
                self._clear_vac_buffer()
            elif "end" in res and "start" not in res:
                self.vac_status = "nonvoice"
                chunk = self.vac_audio_buffer[:frame]
                super().insert_audio_chunk(chunk)
                self.vac_chunk_buffer_size += len(chunk)
                self.vac_is_currently_final = True
                self._clear_vac_buffer()
            else:
                beg   = res["start"] - self.vac_buffer_offset
                end_f = res["end"]   - self.vac_buffer_offset
                self.vac_status = "nonvoice"
                chunk = self.vac_audio_buffer[beg:end_f]
                super().init(offset=(beg + self.vac_buffer_offset) / self.SAMPLING_RATE)
                super().insert_audio_chunk(chunk)
                self.vac_chunk_buffer_size += len(chunk)
                self.vac_is_currently_final = True
                self._clear_vac_buffer()
        else:
            if self.vac_status == "voice":
                super().insert_audio_chunk(self.vac_audio_buffer)
                self.vac_chunk_buffer_size += len(self.vac_audio_buffer)
                self._clear_vac_buffer()
            else:
                # Retain the last 1 second to prevent losing speech onset detected during silence
                trim = max(0, len(self.vac_audio_buffer) - self.SAMPLING_RATE)
                self.vac_buffer_offset += trim
                self.vac_audio_buffer   = self.vac_audio_buffer[-self.SAMPLING_RATE:]

    def process_iter(self) -> tuple:
        if self.vac_is_currently_final:
            return self._finish_vac()
        if self.vac_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            self.vac_chunk_buffer_size = 0
            return super().process_iter()
        return None, None, ""

    def _finish_vac(self) -> tuple:
        ret = super().finish()
        self.vac_chunk_buffer_size  = 0
        self.vac_is_currently_final = False
        return ret


# ==============================================================================
# SubtitleTranslator -- llama.cpp OpenAI-compatible translation endpoint
# ==============================================================================

class SubtitleTranslator:
    def __init__(self, base_url="http://127.0.0.1:8080/v1", api_key="no-key", model_name="default"):
        self.client     = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self._warned    = False

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一個高精準的即時字幕翻譯員。"
                            "將使用者的英文語音字幕準確、流暢地翻譯成繁體中文（台灣習慣用詞）。"
                            "不要添加任何額外解釋、前綴、後綴或引號，直接輸出翻譯後的繁體中文。"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=256,
                timeout=5.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if not self._warned:
                print(f"\n[Note] Connection to llama.cpp server failed ({e}). Please ensure llama.cpp server is running.")
                self._warned = True
            return ""


# ==============================================================================
# Audio Loopback Recording
# ==============================================================================

def get_loopback_microphone():
    speaker = sc.default_speaker()
    if speaker is None:
        raise RuntimeError("Default speaker not found!")
    print(f"[Audio] Listening output device: {speaker.name}")
    try:
        return sc.get_microphone(id=str(speaker.name), include_loopback=True)
    except Exception:
        loopback_mics = [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
        if not loopback_mics:
            raise RuntimeError("No audio device supporting Loopback was found!")
        return loopback_mics[0]


def audio_recorder(audio_queue: queue.Queue, stop_event: threading.Event):
    loopback = get_loopback_microphone()
    with loopback.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE) as rec:
        while not stop_event.is_set():
            data = rec.record(numframes=BLOCK_SIZE)
            audio_queue.put(data[:, 0])


# ==============================================================================
# Load Configuration
# ==============================================================================

def load_env_config() -> dict:
    """Loads .env configuration (supports python-dotenv or manual reading)"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

    return {
        "model_size":       os.getenv("MODEL_SIZE",        "small"),
        "device":           os.getenv("DEVICE",            "auto"),
        "compute_type":     os.getenv("COMPUTE_TYPE",      "auto"),
        "beam_size":        int(os.getenv("BEAM_SIZE",     "5")),
        "stream_step":      float(os.getenv("STREAM_STEP", "0.6")),
        "buffer_trimming":  os.getenv("BUFFER_TRIMMING",   "segment"),
        "buffer_trim_sec":  float(os.getenv("BUFFER_TRIM_SEC", "15")),
        "use_vac":          os.getenv("USE_VAC",           "false").lower() in ("true", "1", "yes"),
        "vac_chunk_size":   float(os.getenv("VAC_CHUNK_SIZE", "0.04")),
        "enable_translate": os.getenv("ENABLE_TRANSLATE",  "false").lower() in ("true", "1", "yes"),
        "llama_url":        os.getenv("LLAMA_URL",         "http://127.0.0.1:8080/v1"),
        "llama_model":      os.getenv("LLAMA_MODEL",       "default"),
    }


# ==============================================================================
# Main Program
# ==============================================================================

def main():
    config = load_env_config()

    # Initialize ASR backend
    asr = FasterWhisperASR(
        lan=          "en",
        modelsize=    config["model_size"],
        device=       config["device"],
        compute_type= config["compute_type"],
        beam_size=    config["beam_size"],
    )

    # Create OnlineASRProcessor (or version with VAC)
    bt = (config["buffer_trimming"], config["buffer_trim_sec"])
    if config["use_vac"]:
        try:
            online = VACOnlineASRProcessor(
                config["vac_chunk_size"], asr, buffer_trimming=bt
            )
            print(f"[VAC] Silero VAD enabled (chunk_size={config['vac_chunk_size']}s)")
        except Exception as e:
            print(f"[VAC] Loading failed ({e}), falling back to standard mode")
            online = OnlineASRProcessor(asr, buffer_trimming=bt)
    else:
        online = OnlineASRProcessor(asr, buffer_trimming=bt)

    # Translator
    translator = None
    if config["enable_translate"]:
        translator = SubtitleTranslator(
            base_url=   config["llama_url"],
            model_name= config["llama_model"],
        )
        print(f"[Translation] Connecting to llama.cpp server: {config['llama_url']}")
    else:
        print("[Translation] ENABLE_TRANSLATE=false, outputting real-time English subtitles only")

    audio_queue = queue.Queue()
    stop_event  = threading.Event()

    record_thread = threading.Thread(
        target=audio_recorder, args=(audio_queue, stop_event), daemon=True
    )
    record_thread.start()

    print("\n" + "=" * 65)
    print(" [whisper_streaming] Real-time audio streaming recognition starting (.env config)")
    print(" Local Agreement strategy: No speech pauses needed, real-time rolling commit")
    print(" Press Ctrl+C to exit")
    print("=" * 65 + "\n")

    step_samples    = int(config["stream_step"] * SAMPLE_RATE)
    accum_samples   = 0
    committed_line  = ""

    try:
        while True:
            try:
                frame = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            online.insert_audio_chunk(frame)
            accum_samples += len(frame)

            if accum_samples >= step_samples:
                accum_samples = 0
                beg, end, committed = online.process_iter()

                if committed:
                    committed_line += (" " if committed_line else "") + committed
                    if len(committed_line.split()) >= 6:
                        ts = time.strftime("%H:%M:%S")
                        sys.stdout.write(f"\r\033[K[{ts}] [EN] {committed_line}\n")
                        sys.stdout.flush()
                        if translator:
                            chinese = translator.translate(committed_line)
                            if chinese:
                                print(f"[{ts}] [ZH] {chinese}")
                        committed_line = ""

                # Real-time echo (committed + tentative)
                _, _, tentative = online.tentative
                disp = (committed_line + " " + (tentative or "")).strip()
                if disp:
                    sys.stdout.write(f"\r\033[K[Realtime] {disp}")
                else:
                    rms  = float(np.sqrt(np.mean(frame ** 2)))
                    bars = min(10, int(rms * 200))
                    sys.stdout.write(f"\r\033[K[Listening {'█' * bars}{'░' * (10 - bars)}]")
                sys.stdout.flush()

    except (KeyboardInterrupt, SystemExit):
        print("\n[System] Interruption signal received, exiting...")
        # Force flush remaining candidate words
        beg, end, remaining = online.finish()
        if remaining:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [EN] {committed_line + ' ' + remaining}".strip())
    finally:
        stop_event.set()
        os._exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))
    logging.basicConfig(level=logging.WARNING)
    main()
