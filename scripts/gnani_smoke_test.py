#!/usr/bin/env python3
"""One-shot Gnani STT/TTS diagnostic using the included synthetic demo.

No transcript text, API key, credential file, or raw provider response is printed.
Each test operation makes at most one provider request (Voice(retries=0)).
Run: python scripts/gnani_smoke_test.py
Optional: python scripts/gnani_smoke_test.py --audio path/to/speech.wav
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceops.config import load_env  # noqa: E402
from voiceops.voice import Voice  # noqa: E402


def redact(text: str, secret: str = "") -> str:
    """Remove known secret plus common credential spellings before diagnostic output."""
    out = str(text)
    if secret:
        out = out.replace(secret, "[REDACTED]")
    out = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[REDACTED]", out)
    out = re.sub(r"(?i)((?:api[_ -]?key|x-api-key-id|token)\s*[:=]\s*)\S+", r"\1[REDACTED]", out)
    return out[:700]


def exception_chain(exc: BaseException):
    """Yield root and wrapped exceptions without dumping arbitrary objects."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def prepare_test_wav(source: pathlib.Path, target: pathlib.Path) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFMPEG_NOT_FOUND: install ffmpeg or use a machine with ffmpeg on PATH")
    if not source.is_file():
        raise FileNotFoundError(f"AUDIO_FILE_NOT_FOUND: {source.name}")
    # Normalize all test inputs to common STT format. The default source is the bundled
    # prerecorded synthetic demo, not a real customer's recording.
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", "0", "-i", str(source),
           "-t", "12", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode or not target.is_file() or target.stat().st_size < 1000:
        # Do not echo ffmpeg's unrestricted output; it can contain local paths.
        raise RuntimeError(f"AUDIO_CONVERSION_FAILED: ffmpeg exit={proc.returncode}; check that the input has an audio stream")
    details = inspect_wav(target)
    if details["duration_seconds"] < 1.0 or details["rms"] < 100:
        raise RuntimeError("AUDIO_TEST_CLIP_INVALID: clip is too short or near-silent; use a clear synthetic speech WAV")
    return details


def inspect_wav(path: pathlib.Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
            raise ValueError("WAV_FORMAT_INVALID: expected mono 16-bit PCM at 16000 Hz")
        frames = wav.getnframes()
        raw = wav.readframes(frames)
        count = len(raw) // 2
        values = struct.unpack("<" + "h" * count, raw) if count else ()
        rms = math.sqrt(sum(value * value for value in values) / count) if count else 0.0
        peak = max((abs(value) for value in values), default=0)
        return {"sample_rate": wav.getframerate(), "channels": wav.getnchannels(),
                "duration_seconds": round(frames / wav.getframerate(), 2), "rms": round(rms, 1),
                "peak": peak, "bytes": len(raw)}


def print_exception(prefix: str, exc: BaseException, secret: str) -> None:
    chain = list(exception_chain(exc))
    for index, item in enumerate(chain):
        print(f"{prefix}_ERROR_{index}_TYPE={type(item).__module__}.{type(item).__name__}")
        # SDKs commonly expose these stable fields; never print arbitrary provider messages.
        for field in ("status_code", "http_status", "error_code", "request_id", "code"):
            value = getattr(item, field, None)
            if isinstance(value, (str, int)) and str(value).strip():
                print(f"{prefix}_ERROR_{index}_{field.upper()}={redact(str(value), secret)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=pathlib.Path, help="optional synthetic speech audio file; defaults to demo.mp4")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stt-only", action="store_true", help="run only one STT request")
    group.add_argument("--tts-only", action="store_true", help="run only one TTS request")
    args = parser.parse_args(argv)

    # Use the application's ordinary configuration loader. Never print or serialize env values.
    load_env()
    secret = os.environ.get("GNANI_API_KEY", "")
    if not secret:
        print("RESULT=FAIL\nERROR=GNANI_API_KEY_NOT_CONFIGURED (value was not read or printed by this script)")
        return 2
    print("GNANI_API_KEY_CONFIGURED=YES")

    try:
        sdk_version = importlib.metadata.version("gnani-vachana")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "UNKNOWN"
    print(f"GNANI_SDK_VERSION={sdk_version}")
    source = args.audio.resolve() if args.audio else ROOT / "demo.mp4"
    with tempfile.TemporaryDirectory(prefix="voiceops_gnani_smoke_") as temp_name:
        temp = pathlib.Path(temp_name)
        wav_path = temp / "stt_input.wav"
        if not args.tts_only:
            try:
                audio = prepare_test_wav(source, wav_path)
                print("STT_INPUT=prepared synthetic/test clip")
                print("STT_INPUT_FORMAT=mono PCM16 16000Hz")
                print(f"STT_INPUT_DURATION_SECONDS={audio['duration_seconds']}")
                print(f"STT_INPUT_RMS={audio['rms']}")
            except Exception as exc:
                print("STT_INPUT=FAIL")
                print_exception("STT_INPUT", exc, secret)
                return 2

        voice = None
        all_ok = True
        try:
            # No retries in this diagnostic: at most one STT and, unless --stt-only, one TTS request.
            voice = Voice(dry_run=False, cache_dir=str(temp / "cache"), retries=0)
            if not args.tts_only:
                try:
                    result = voice.transcribe(str(wav_path), languages=["en-IN"])
                    transcript = str(result.get("transcript", "")).strip()
                    if transcript:
                        print(f"STT_RESULT=PASS; nonempty transcript; characters={len(transcript)}; language={result.get('language_code', 'unknown')}")
                    else:
                        print("STT_RESULT=FAIL; provider returned an empty transcript for the supplied test clip")
                        all_ok = False
                except Exception as exc:
                    print("STT_RESULT=FAIL")
                    print_exception("STT", exc, secret)
                    all_ok = False

            if not args.stt_only:
                try:
                    tts_path = temp / "tts_output.wav"
                    voice.synthesize("This is a VoiceOps speech integration test.", "en", str(tts_path))
                    with wave.open(str(tts_path), "rb") as wav:
                        frames = wav.getnframes()
                        channels = wav.getnchannels()
                        sample_width = wav.getsampwidth()
                        duration = frames / wav.getframerate()
                        audio_bytes = wav.readframes(frames)
                        good = (frames > 0 and channels >= 1 and sample_width in (1, 2, 3, 4)
                                and len(audio_bytes) == frames * channels * sample_width)
                    if good and duration >= 0.25 and tts_path.stat().st_size > 1000:
                        print(f"TTS_RESULT=PASS; readable nonempty WAV; duration_seconds={duration:.2f}; bytes={tts_path.stat().st_size}")
                    else:
                        print("TTS_RESULT=FAIL; output WAV failed the nonempty/audio-format check")
                        all_ok = False
                except Exception as exc:
                    print("TTS_RESULT=FAIL")
                    print_exception("TTS", exc, secret)
                    all_ok = False
        except Exception as exc:
            print("CLIENT_INIT=FAIL")
            print_exception("CLIENT", exc, secret)
            return 2

        print(f"PROVIDER_REQUESTS_OBSERVED: STT={voice.stats['stt_calls']} TTS={voice.stats['tts_calls']}")
        print("CREDENTIAL_VALUES_PRINTED=NO")
        print("TRANSCRIPT_TEXT_PRINTED=NO")
        print("RESULT=" + ("PASS" if all_ok else "FAIL"))
        return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
