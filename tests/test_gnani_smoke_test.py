"""Offline-only tests for the Gnani smoke-test diagnostics; these never call the provider."""
import math
import pathlib
import struct
import sys
import wave

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.gnani_smoke_test as smoke_test
from scripts.gnani_smoke_test import inspect_wav, prepare_test_wav, print_exception, redact
from voiceops.voice import Voice


def make_speech_like_fixture(path: pathlib.Path, seconds: float = 2.0) -> None:
    rate = 16000
    count = int(rate * seconds)
    values = [int(7000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(count)]
    raw = struct.pack("<" + "h" * len(values), *values)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(raw)


def test_redact_removes_known_key_and_common_token_spellings():
    text = "bad request: API key=abc123 Authorization: Bearer token-xyz"
    got = redact(text, "abc123")
    assert "abc123" not in got
    assert "token-xyz" not in got
    assert got.count("[REDACTED]") >= 2


def test_inspect_wav_reports_canonical_pcm(tmp_path):
    path = tmp_path / "test.wav"
    make_speech_like_fixture(path)
    info = inspect_wav(path)
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert info["duration_seconds"] == 2.0
    assert info["rms"] > 100
    assert info["bytes"] > 1000


def test_prepare_test_wav_converts_to_canonical_pcm(tmp_path):
    if not __import__("shutil").which("ffmpeg"):
        pytest.skip("ffmpeg not installed; conversion smoke test needs ffmpeg")
    src, dst = tmp_path / "source.wav", tmp_path / "normalized.wav"
    make_speech_like_fixture(src)
    info = prepare_test_wav(src, dst)
    assert dst.exists()
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert info["duration_seconds"] >= 1.0


def test_prepare_test_wav_refuses_missing_input(tmp_path):
    if not __import__("shutil").which("ffmpeg"):
        pytest.skip("ffmpeg not installed; conversion smoke test needs ffmpeg")
    with pytest.raises(FileNotFoundError, match="AUDIO_FILE_NOT_FOUND"):
        prepare_test_wav(tmp_path / "does-not-exist.wav", tmp_path / "out.wav")


def test_stt_failure_preserves_provider_exception_fields(monkeypatch):
    class ProviderError(Exception):
        status_code = 403
        request_id = "request-test-id"

    provider_error = ProviderError("permission denied")

    class FakeSTT:
        def transcribe(self, audio, language_code, **kwargs):
            raise provider_error

    voice = Voice.__new__(Voice)
    voice.retries = 0
    voice.stt = FakeSTT()
    voice.stats = {"stt_calls": 0}
    monkeypatch.setattr("voiceops.voice.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Gnani STT failed") as raised:
        voice._stt_one("synthetic.wav", "en-IN")

    assert raised.value.__cause__ is provider_error
    assert raised.value.__cause__.status_code == 403
    assert raised.value.__cause__.request_id == "request-test-id"
    assert voice.stats["stt_calls"] == 1


def test_print_exception_omits_raw_provider_message(capsys):
    class ProviderError(Exception):
        status_code = 403
        request_id = "request-test-id"
        error_code = "permission_denied"

    print_exception("STT", ProviderError("transcript and credential must stay private"), "credential")

    output = capsys.readouterr().out
    assert "403" in output
    assert "request-test-id" in output
    assert "permission_denied" in output
    assert "transcript" not in output
    assert "credential" not in output


def test_stt_only_does_not_invoke_tts(monkeypatch, tmp_path, capsys):
    class FakeVoice:
        def __init__(self, **kwargs):
            self.stats = {"stt_calls": 1, "tts_calls": 0}

        def transcribe(self, *args, **kwargs):
            return {"transcript": "synthetic fixture transcript", "language_code": "en-IN"}

        def synthesize(self, *args, **kwargs):
            pytest.fail("STT-only mode must not call TTS")

    monkeypatch.setattr(smoke_test, "ROOT", tmp_path)
    monkeypatch.setattr(smoke_test, "load_env", lambda: None)
    monkeypatch.setattr(smoke_test, "prepare_test_wav", lambda source, target: {
        "duration_seconds": 2.0, "rms": 1000.0,
    })
    monkeypatch.setattr(smoke_test, "Voice", FakeVoice)
    monkeypatch.setenv("GNANI_API_KEY", "unit-test-placeholder")

    assert smoke_test.main(["--stt-only"]) == 0

    output = capsys.readouterr().out
    assert "STT_RESULT=PASS" in output
    assert "synthetic fixture transcript" not in output
    assert "TTS_RESULT" not in output
    assert "PROVIDER_REQUESTS_OBSERVED: STT=1 TTS=0" in output
    assert "unit-test-placeholder" not in output


def test_tts_only_does_not_repeat_stt(monkeypatch, tmp_path, capsys):
    class FakeVoice:
        def __init__(self, **kwargs):
            self.stats = {"stt_calls": 0, "tts_calls": 1}

        def transcribe(self, *args, **kwargs):
            pytest.fail("TTS-only mode must not call STT")

        def synthesize(self, text, lang, out_path):
            with wave.open(out_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 4800)
            return out_path

    monkeypatch.setattr(smoke_test, "ROOT", tmp_path)
    monkeypatch.setattr(smoke_test, "load_env", lambda: None)
    monkeypatch.setattr(smoke_test, "Voice", FakeVoice)
    monkeypatch.setenv("GNANI_API_KEY", "unit-test-placeholder")

    assert smoke_test.main(["--tts-only"]) == 0

    output = capsys.readouterr().out
    assert "STT_RESULT" not in output
    assert "TTS_RESULT=PASS" in output
    assert "PROVIDER_REQUESTS_OBSERVED: STT=0 TTS=1" in output
    assert "unit-test-placeholder" not in output
