"""Shared evaluation helpers: tool-call parsing/matching and result summaries."""
import json, os, re, statistics

def parse_tool_calls(spec: str):
    out = []
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", spec or ""):
        kw = {}
        for part in [p for p in args.split(",") if p.strip()]:
            k, _, v = part.partition("="); kw[k.strip()] = v.strip()
        out.append((name, kw))
    return out

def _digits(x): return "".join(ch for ch in str(x) if ch.isdigit())

def calls_match(expected_spec: str, made: list) -> dict:
    exp = parse_tool_calls(expected_spec); missing = []
    for name, kw in exp:
        ok = False
        for m in made:
            if m["tool"] != name: continue
            if name == "create_ticket": ok = ok or bool(m.get("ok")); continue
            if all(_digits(m["args"].get(k)) == _digits(v) for k, v in kw.items()): ok = True
        if not ok: missing.append(f"{name}({', '.join(f'{k}={v}' for k, v in kw.items())})")
    extra = [m for m in made if m["tool"] not in {n for n, _ in exp}]
    return {"ok": not missing, "missing": missing, "unexpected": [m["tool"] for m in extra]}

def sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s])

def pct(x, n): return round(100.0 * x / n, 1) if n else 0.0

def save(name, summary, items, outdir=None):
    outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval", "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f: json.dump({"summary": summary, "items": items}, f, ensure_ascii=False, indent=1, default=str)
    return path

def mean(xs): return round(statistics.mean(xs), 3) if xs else 0.0
