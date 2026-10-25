#!/usr/bin/env python3
"""Live demo for the top-50 round: a web page with a microphone button.  python demo_server.py  ->  http://localhost:8000
Customer speaks -> Gnani STT -> agent -> Gnani TTS -> reply plays in the browser. Needs ffmpeg for browser recordings (webm).
Use --dry-run to test the page without Gnani. Browsers allow the microphone on http://localhost."""
import argparse, base64, json, os, sys, tempfile, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from voiceops.config import load_env
load_env()

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>ZentroMart voice support</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem;background:#faf6f1;color:#26323e}
button{font-size:1.1rem;padding:.8rem 1.4rem;border:0;border-radius:2rem;background:#26323e;color:#fff;cursor:pointer}button.rec{background:#c0392b}
.b{margin:.6rem 0;padding:.7rem 1rem;border-radius:1rem;background:#fff}.me{background:#e8f1f8}.meta{font-size:.75rem;color:#777}input{width:70%;padding:.6rem;border-radius:.6rem;border:1px solid #bbb}</style>
<h2>ZentroMart voice support</h2><p>Hold the button and speak, or type below. <button id=new>New call</button></p>
<p><button id=mic>Hold to talk</button></p><p><input id=txt placeholder="or type a message"><button id=send>Send</button></p><div id=log></div>
<script>
let sid=crypto.randomUUID(),rec,chunks=[];const log=document.getElementById('log'),mic=document.getElementById('mic');
function add(t,c,m){const d=document.createElement('div');d.className='b '+c;d.textContent=t;if(m){const s=document.createElement('div');s.className='meta';s.textContent=m;d.appendChild(s)}log.prepend(d)}
function show(j){if(j.error){add('Error: '+j.error,'');return}add(j.heard,'me');add(j.reply,'',j.decision+' | sources: '+j.chunks.join(', ')+' | '+j.latency+'s');if(j.audio){new Audio('data:audio/wav;base64,'+j.audio).play()}}
async function start(){const s=await navigator.mediaDevices.getUserMedia({audio:true});rec=new MediaRecorder(s);chunks=[];rec.ondataavailable=e=>chunks.push(e.data);
 rec.onstop=async()=>{s.getTracks().forEach(t=>t.stop());const r=await fetch('/turn?sid='+sid,{method:'POST',body:new Blob(chunks,{type:rec.mimeType})});show(await r.json())};rec.start();mic.classList.add('rec');mic.textContent='Listening...'}
function stop(){if(rec&&rec.state!='inactive')rec.stop();mic.classList.remove('rec');mic.textContent='Hold to talk'}
mic.onmousedown=mic.ontouchstart=e=>{e.preventDefault();start()};mic.onmouseup=mic.onmouseleave=mic.ontouchend=stop;
document.getElementById('send').onclick=async()=>{const t=document.getElementById('txt');if(!t.value)return;const r=await fetch('/text?sid='+sid,{method:'POST',body:JSON.stringify({text:t.value})});t.value='';show(await r.json())};
document.getElementById('new').onclick=()=>{sid=crypto.randomUUID();log.innerHTML=''};
</script>"""

def make_handler(agent, voice):
    from voiceops.speech_text import speechify
    from voiceops.prompts import T
    sessions = {}
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            b = PAGE.encode("utf-8"); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_POST(self):
            try:
                sid = self.path.split("sid=")[-1].split("&")[0]; sess = sessions.setdefault(sid, agent.new_session())
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path.startswith("/text"): heard = json.loads(body)["text"]
                else:
                    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f: f.write(body); path = f.name
                    try: heard = voice.transcribe(path)["transcript"]
                    finally: os.unlink(path)
                r = agent.respond(sess, heard) if heard.strip() else {"reply": T("unclear", sess.lang), "decision": "clarify", "language": sess.lang, "chunk_ids": [], "latency_s": 0}
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f: wav = f.name
                try:
                    voice.synthesize(speechify(r["reply"], r["language"]), r["language"], wav); audio = base64.b64encode(open(wav, "rb").read()).decode()
                finally: os.unlink(wav)
                self._json({"heard": heard or "(nothing heard)", "reply": r["reply"], "decision": r["decision"], "chunks": r["chunk_ids"], "latency": r.get("latency_s", 0), "audio": audio})
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
    return H

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--offline", action="store_true"); ap.add_argument("--dataset")
    a = ap.parse_args()
    if a.dataset: os.environ["DATASET_DIR"] = a.dataset
    from voiceops.agent import Agent
    from voiceops.voice import Voice
    agent = Agent(use_llm=not a.offline, channel="voice"); voice = Voice(dry_run=a.dry_run)
    print(f"Open http://localhost:{a.port}  (LLM: {agent.llm.name if agent.llm_on else 'offline rules'}, voice: {'dry-run' if a.dry_run else 'Gnani'})")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(agent, voice)).serve_forever()
