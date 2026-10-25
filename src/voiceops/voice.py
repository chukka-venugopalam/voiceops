"""Gnani STT/TTS wrapper. Only Gnani is used for speech (rule). Includes caching (saves credits), retry,
long-clip splitting (REST STT limit: 60 s), format conversion via ffmpeg when available, and a dry-run mode."""
import hashlib, json, os, pathlib, shutil, subprocess, tempfile, time, wave

STT_LANGS = os.environ.get("STT_LANGUAGES", "en-IN,hi-IN").split(",")
TTS_LANG = {"en": "en-IN", "hi": "hi-IN", "hi-en": "hi-en", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN", "ml": "ml-IN",
            "bn": "bn-IN", "gu": "gu-IN", "mr": "mr-IN", "pa": "pa-IN"}
CLIP_SECONDS = 50

def _sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

def _wav_data(path: str):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if f.read(12)[:4] != b"RIFF":
            return None
        f.seek(12)
        while True:
            header = f.read(8)
            if len(header) != 8:
                return None
            chunk_size = int.from_bytes(header[4:], "little")
            data_start = f.tell()
            if header[:4] == b"data":
                return data_start, min(chunk_size, size - data_start), chunk_size
            if chunk_size == 0xFFFFFFFF:
                return None
            f.seek(chunk_size + (chunk_size & 1), os.SEEK_CUR)

def _wav_frames(path: str, reader):
    frames = reader.getnframes()
    frame_size = reader.getnchannels() * reader.getsampwidth()
    data = _wav_data(path)
    if data and frame_size:
        frames = min(frames, data[1] // frame_size)
    return frames

def _normalize_wav(path: str):
    data = _wav_data(path)
    if not data or data[1] == data[2]:
        return
    file_size = os.path.getsize(path)
    if data[1] > 0xFFFFFFFF or file_size - 8 > 0xFFFFFFFF:
        raise ValueError(f"WAV is too large for RIFF normalization: {path}")
    with open(path, "r+b") as f:
        f.seek(data[0] - 4)
        f.write(data[1].to_bytes(4, "little"))
        f.seek(4)
        f.write((file_size - 8).to_bytes(4, "little"))

def write_silence(path, seconds=0.5, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(b"\x00\x00" * int(rate * seconds))

SUPPORTED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}      # accepted by the Gnani SDK

def _duration(path: str):
    try:
        with wave.open(path) as w: return _wav_frames(path, w) / w.getframerate()
    except Exception: pass
    if shutil.which("ffprobe"):
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True)
        try: return float(r.stdout.strip())
        except Exception: return None
    return None

def _to_wav(src: str, tmpdir: str) -> str:
    """Convert to mono 16 kHz wav (ffmpeg) when the format is unsupported (e.g. webm) or a non-wav clip needs splitting."""
    ext = os.path.splitext(src)[1].lower()
    dur = _duration(src)
    if ext in SUPPORTED_EXT and not (ext != ".wav" and dur and dur > CLIP_SECONDS + 5): return src
    if shutil.which("ffmpeg"):
        out = os.path.join(tmpdir, "conv.wav")
        r = subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", out], capture_output=True)
        if r.returncode == 0: return out
    return src

def _split_wav(path: str, tmpdir: str, seconds=CLIP_SECONDS):
    try:
        with wave.open(path) as w:
            n, rate = _wav_frames(path, w), w.getframerate()
            if n / rate <= seconds + 5 and n == w.getnframes(): return [path]
            chunks, step = [], int(rate * seconds)
            for i, start in enumerate(range(0, n, step)):
                w.setpos(start); frames = w.readframes(min(step, n - start))
                p = os.path.join(tmpdir, f"part{i}.wav")
                with wave.open(p, "wb") as o:
                    o.setnchannels(w.getnchannels()); o.setsampwidth(w.getsampwidth()); o.setframerate(rate); o.writeframes(frames)
                chunks.append(p)
            return chunks
    except Exception:
        return [path]

class Voice:
    def __init__(self, dry_run=False, cache_dir=".cache/voice", tts_model=None, tts_voice=None, retries=2):
        self.dry_run = dry_run
        self.cache = pathlib.Path(cache_dir)
        self.tts_model = tts_model or os.environ.get("TTS_MODEL", "timbre-v2.5")
        self.tts_voice = tts_voice or os.environ.get("TTS_VOICE", "Kaveri")
        self.retries = retries
        self.stt = self.tts = None
        self.stats = {"stt_calls": 0, "tts_calls": 0, "stt_cache_hits": 0, "tts_cache_hits": 0}
        if not dry_run:
            if not os.environ.get("GNANI_API_KEY"):
                raise RuntimeError("GNANI_API_KEY is not set (put it in .env). Use --dry-run to test without Gnani.")
            from gnani.stt import GnaniSTTClient
            from gnani.tts import GnaniTTSClient
            self.stt, self.tts = GnaniSTTClient(), GnaniTTSClient()

    # ---------- STT ----------
    def _stt_one(self, wav: str, lang: str) -> str:
        last = None
        for a in range(self.retries + 1):
            try:
                self.stats["stt_calls"] += 1
                fmt = os.environ.get("STT_FORMAT", "")          # "" / "verbatim" (default) or "transcribe" (ITN: digits for numbers/dates; en-IN and hi-IN only)
                kw = {"format": fmt} if fmt and lang in ("en-IN", "hi-IN") else {}
                r = self.stt.transcribe(wav, language_code=lang, **kw)
                return (r.get("transcript") or "").strip() if isinstance(r, dict) else str(r).strip()
            except Exception as e:
                last = e; time.sleep(1.0 + a)
        raise RuntimeError(f"Gnani STT failed: {last}")

    def transcribe(self, path: str, languages=None) -> dict:
        """Returns {'transcript', 'language_code', 'cached'}. Tries en-IN first (accepts Hindi mixed in), then hi-IN if empty."""
        path = str(path)
        if self.dry_run:
            side = pathlib.Path(path).with_suffix(".txt")
            text = side.read_text(encoding="utf-8").strip() if side.exists() else ""
            return {"transcript": text, "language_code": "dry-run", "cached": False}
        langs = languages or STT_LANGS
        raw = pathlib.Path(path).read_bytes()
        key = _sha(raw + ("|".join(langs) + os.environ.get("STT_FORMAT", "")).encode())
        cf = self.cache / "stt" / f"{key}.json"
        if cf.exists():
            self.stats["stt_cache_hits"] += 1
            return {**json.loads(cf.read_text(encoding="utf-8")), "cached": True}
        with tempfile.TemporaryDirectory() as td:
            wav = _to_wav(path, td)
            parts = _split_wav(wav, td)
            for lang in langs:
                text = " ".join(t for t in (self._stt_one(p, lang) for p in parts) if t).strip()
                if text:
                    out = {"transcript": text, "language_code": lang}
                    cf.parent.mkdir(parents=True, exist_ok=True); cf.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
                    return {**out, "cached": False}
        return {"transcript": "", "language_code": langs[-1], "cached": False}

    # ---------- TTS ----------
    def synthesize(self, text: str, lang: str, out_path: str) -> str:
        out_path = str(out_path); pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            write_silence(out_path); return out_path
        code = TTS_LANG.get(lang, "en-IN")
        key = _sha(json.dumps([text, code, self.tts_model, self.tts_voice, os.environ.get("TTS_SPEED", "")], ensure_ascii=False).encode())
        cf = self.cache / "tts" / f"{key}.wav"
        if cf.exists():
            self.stats["tts_cache_hits"] += 1; shutil.copyfile(cf, out_path); _normalize_wav(out_path); return out_path
        last = None
        for a in range(self.retries + 1):
            try:
                self.stats["tts_calls"] += 1
                kw = {"speed": float(os.environ["TTS_SPEED"])} if os.environ.get("TTS_SPEED") else {}      # 0.85 - 1.15, timbre-v2.5 only
                self.tts.synthesize(text, voice=self.tts_voice, model=self.tts_model, language=code, output_file=out_path, **kw)
                _normalize_wav(out_path)
                cf.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(out_path, cf)
                return out_path
            except Exception as e:
                last = e; time.sleep(1.0 + a)
        raise RuntimeError(f"Gnani TTS failed: {last}")
