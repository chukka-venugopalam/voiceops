#!/usr/bin/env python3
"""One-shot readiness check:  python scripts/preflight.py [--ping] [--gnani]
   --ping  makes ONE tiny LLM call;  --gnani runs the Gnani smoke test (uses a few credits)."""
import argparse, os, subprocess, sys, importlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from voiceops.config import load_env, dataset_dir
load_env()
ap = argparse.ArgumentParser(); ap.add_argument("--ping", action="store_true"); ap.add_argument("--gnani", action="store_true", help="Run the bounded Gnani STT/TTS smoke test (uses provider credits)"); ap.add_argument("--models", action="store_true", help="list the model ids your LLM key can use"); a = ap.parse_args()
ok_all = True
def line(ok, msg, fix=""):
    global ok_all; ok_all &= bool(ok)
    print(("  [ OK ] " if ok else "  [FAIL] ") + msg + (f"\n         -> {fix}" if fix and not ok else ""))
print(f"Python {sys.version.split()[0]}")
line(sys.version_info >= (3, 9), "Python 3.9+", "install a newer Python")
for mod, pipname in (("gnani.stt", "gnani-vachana"), ("reportlab", "reportlab"), ("pytest", "pytest")):
    try: importlib.import_module(mod); line(True, f"package {pipname}")
    except Exception: line(False, f"package {pipname}", f"pip install {pipname}")
try: root = dataset_dir(); line(True, f"dataset found: {root}")
except FileNotFoundError: root = None; line(False, "dataset folder", "set DATASET_DIR in .env to the folder that contains kb/ and data/")
line(bool(os.environ.get("GNANI_API_KEY")), "GNANI_API_KEY is set", "add it to .env (get it at app.gnani.ai/voice)")
from voiceops.llm import get_llm
llm = get_llm()
line(llm.available, f"LLM configured ({llm.name} {llm.model})" if llm.available else "LLM key configured", "set LLM_PROVIDER, LLM_API_KEY and LLM_MODEL in .env (offline rules still work without it)")
if root and os.path.exists(os.path.join(root, "validate_dataset.py")):
    r = subprocess.run([sys.executable, "validate_dataset.py"], cwd=root, capture_output=True, text=True)
    line(r.returncode == 0, "dataset validator (validate_dataset.py)", (r.stdout + r.stderr)[-300:])
if root:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x"], cwd=ROOT, capture_output=True, text=True)
    line(r.returncode == 0, "unit tests: " + (r.stdout.strip().splitlines() or ["?"])[-1], r.stdout[-400:])
if a.ping and llm.available:
    try: out = llm.complete("Reply with one word.", [{"role": "user", "content": "Say OK"}], max_tokens=10); line(bool(out), f"LLM ping -> {out[:30]!r}")
    except Exception as e: line(False, "LLM ping", str(e)[:300])
if a.models and llm.available:
    try: ids = llm.list_models(); line(bool(ids), f"{len(ids)} models available; usable ones include: " + ", ".join(i for i in ids if "embed" in i.lower() or "embedding" in i.lower() or "flash" in i.lower() or "lite" in i.lower() or "haiku" in i.lower() or "mini" in i.lower() or "llama" in i.lower())[:600])
    except Exception as e: line(False, "list models", str(e)[:300])
if a.gnani:
    smoke_path = os.path.join(ROOT, "scripts", "gnani_smoke_test.py")
    if not os.path.isfile(smoke_path):
        line(False, "Gnani STT/TTS smoke-test script is not included in this release package",
             "This archive omits scripts/gnani_smoke_test.py; do not treat provider integration as verified by preflight.")
    else:
        r = subprocess.run([sys.executable, smoke_path], capture_output=True, text=True, cwd=ROOT)
        line(r.returncode == 0, "Gnani STT/TTS smoke test", (r.stdout + r.stderr)[-1400:])
print("\nREADY" if ok_all else "\nFix the [FAIL] items above, then run this again.")
sys.exit(0 if ok_all else 1)
