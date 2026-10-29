"""Conservative local sanitizer for eval JSON files.
Copies summary metrics and a small allowlist of per-item identifiers/metrics only.
It strips question/customer/answer/gold/retrieval text recursively by using a schema allowlist.
Never run with output pointed at a shared/public location without reviewing the result first.
"""
import argparse, json, re
from pathlib import Path

ITEM_FIELDS = {"id", "language", "recall", "score", "decision_ok", "content_ok", "tool_ok", "fully_correct", "turn", "intent_ok", "accuracy", "retrieval_recall", "correct", "passed"}
LANGS = {"en", "hi", "hi-en", "ta", "te", "kn", "ml", "bn", "gu", "mr", "pa", "?"}
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")

def scalar_metric(v):
    return v if isinstance(v, (int, float, bool)) and not isinstance(v, complex) else None

def clean_summary(obj):
    if not isinstance(obj, dict): return {}
    out = {}
    for key, value in obj.items():
        if not re.fullmatch(r"[A-Za-z0-9_:-]{1,80}", str(key)):
            continue
        if isinstance(value, dict):
            sub = {}
            for sk, sv in value.items():
                if not re.fullmatch(r"[A-Za-z0-9_:-]{1,40}", str(sk)): continue
                metric = scalar_metric(sv)
                if metric is not None: sub[sk] = metric
            if sub: out[key] = sub
        else:
            metric = scalar_metric(value)
            if metric is not None:
                out[key] = metric
            elif key == "split" and value in ("all", "diagnosis", "locked"):
                out[key] = value
    return out

def clean_item(obj):
    if not isinstance(obj, dict): return {}
    out = {}
    for key, value in obj.items():
        if key not in ITEM_FIELDS: continue
        if key == "id" and isinstance(value, str) and ID_RE.fullmatch(value):
            out[key] = value
        elif key == "language" and isinstance(value, str) and value in LANGS:
            out[key] = value
        else:
            metric = scalar_metric(value)
            if metric is not None: out[key] = metric
    return out

def clean(data):
    if not isinstance(data, dict): return {"summary": {}, "items": []}
    out = {"summary": clean_summary(data.get("summary", {}))}
    items = data.get("items", [])
    out["items"] = [r for row in items if (r := clean_item(row))] if isinstance(items, list) else []
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="eval/results")
    ap.add_argument("--dst", default="eval/results_public")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir(): raise SystemExit("ABORT: source directory does not exist")
    files = sorted(src.glob("*.json"))
    if not files: raise SystemExit("ABORT: no top-level JSON files found")
    conflicts = [dst / p.name for p in files if (dst / p.name).exists()]
    if conflicts and not args.overwrite:
        raise SystemExit("ABORT: destination already contains output files; choose a new --dst or pass --overwrite")
    # Parse and sanitize all source files before writing any output.
    outputs = []
    for path in files:
        with path.open(encoding="utf-8") as f: data = json.load(f)
        outputs.append((dst / path.name, clean(data)))
    dst.mkdir(parents=True, exist_ok=True)
    for path, data in outputs:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    print(f"SANITIZED_FILES={len(outputs)}")
    print("FIELDS_KEPT=summary numeric metrics; per-item id/language/numeric metrics only")
    print("QUESTION_CUSTOMER_REPLY_ANSWER_GOLD_GOT_TEXT_KEPT=0")
    print("REVIEW_OUTPUT_LOCALLY_BEFORE_ANY_PUBLICATION=1")

if __name__ == "__main__": main()
