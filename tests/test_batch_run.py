import csv, os, subprocess, sys, wave
from conftest import needs_data, ROOT
pytestmark = needs_data

def _sil(p):
    with wave.open(p, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00\x00" * 800)

def test_dry_run_batch_groups_conversation_and_survives_silence(tmp_path):
    a = tmp_path / "audio"; a.mkdir()
    for t, msg in enumerate(["Hello, mujhe order ke baare mein help chahiye.", "Order number hai 1859 9253.", "Last four digits 4265 hain."], 1):
        _sil(str(a / f"PC003_t{t}.wav")); (a / f"PC003_t{t}.txt").write_text(msg, encoding="utf-8")
    _sil(str(a / "empty.wav"))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "--audio-dir", str(a), "--out", str(out), "--dry-run", "--offline"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(out / "results.csv", encoding="utf-8")))
    assert len(rows) == 4 and all(not x["error"] for x in rows)
    assert (out / "PC003_t3.reply.wav").exists() and (out / "PC003_t3.reply.txt").exists()
    assert [x["decision"] for x in rows if x["conversation_id"] == "PC003"] == ["clarify", "clarify", "answer"]
    assert [x for x in rows if x["file"] == "empty.wav"][0]["decision"] == "clarify"

def test_stateless_by_default(tmp_path):
    a = tmp_path / "audio"; a.mkdir()
    for i in (1, 2):
        _sil(str(a / f"q_{i}.wav")); (a / f"q_{i}.txt").write_text("Can I use two coupons on one order?", encoding="utf-8")
    out = tmp_path / "out"
    subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "--audio-dir", str(a), "--out", str(out), "--dry-run", "--offline"], capture_output=True, text=True)
    rows = list(csv.DictReader(open(out / "results.csv", encoding="utf-8")))
    assert all("AI" in x["reply"] for x in rows)        # each file is a fresh call, so each reply opens with the AI intro

def test_demo_server_text_endpoint():
    import json, threading, urllib.request
    from http.server import ThreadingHTTPServer
    sys.path.insert(0, ROOT)
    import demo_server
    from voiceops.agent import Agent
    from voiceops.voice import Voice
    srv = ThreadingHTTPServer(("127.0.0.1", 0), demo_server.make_handler(Agent(use_llm=False, channel="voice"), Voice(dry_run=True)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/text?sid=t1", data=json.dumps({"text": "Can I use two coupons on one order?"}).encode(), method="POST")
        j = json.loads(urllib.request.urlopen(req, timeout=10).read())
        assert "AI" in j["reply"] and j["audio"] and j["decision"] == "answer"
        assert b"ZentroMart voice support" in urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read()
    finally: srv.shutdown()

def test_scripts_survive_a_cp1252_console():
    """Regression: the VS Code / Windows terminal used cp1252 and crashed when printing Hindi."""
    env = dict(os.environ, PYTHONIOENCODING="cp1252:strict")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "eval", "eval_conversations.py"), "--mode", "offline"], capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    assert r.returncode == 0 and "Traceback" not in r.stderr, r.stderr[-500:]
