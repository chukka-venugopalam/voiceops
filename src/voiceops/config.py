import os, glob

def dataset_dir() -> str:
    """Folder containing kb/, data/, tools/, practice/. Set DATASET_DIR, or place the extracted
    dataset next to this repo / inside ./data_pack."""
    env = os.environ.get("DATASET_DIR")
    if env and os.path.isdir(os.path.join(env, "kb")):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", ".."))
    for pat in (os.path.join(root, "data_pack", "*"), os.path.join(root, "*"), os.path.join(root, "..", "*")):
        for cand in glob.glob(pat):
            if os.path.isdir(os.path.join(cand, "kb")) and os.path.isdir(os.path.join(cand, "data")):
                return cand
    raise FileNotFoundError("Dataset not found. Set DATASET_DIR to the folder that contains kb/ and data/.")


def fix_console():
    """Windows consoles (cp1252) cannot print Hindi: never crash on output, replace what can't be shown."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try: stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    try:      # piped input (not a real console) is read as UTF-8 too
        if not sys.stdin.isatty(): sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

def load_env(path=None):
    """Tiny .env loader (no dependency). Real environment variables win over the file."""
    import os
    fix_console()
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [path] if path else [os.path.join(os.getcwd(), ".env"), os.path.normpath(os.path.join(here, "..", "..", ".env"))]
    for c in cands:
        if c and os.path.isfile(c):
            for line in open(c, encoding="utf-8"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line: continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ: os.environ[k] = v
