"""Held-out retrieval evaluation with batched queries and a safe local output folder.
Input CSV: id,question,gold_ids,language. gold_ids is a semicolon-separated list of KB section IDs.
No questions, gold answers, or result-item text is copied to the public sanitizer output.
"""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voiceops.config import load_env, fix_console
from voiceops.kb import Retriever
from voiceops.scorer import recall
from voiceops.embed import EmbedIndex, RequestBudget

def side(gold):
    return int(hashlib.sha1(gold[0].encode()).hexdigest(), 16) % 2 if gold else 0

def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "question", "gold_ids", "language"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SystemExit("ABORT: CSV must contain id,question,gold_ids,language columns")
        rows = list(reader)
    if not rows:
        raise SystemExit("ABORT: held-out CSV is empty")
    seen = set()
    for row in rows:
        if not all((row.get(k) or "").strip() for k in required):
            raise SystemExit("ABORT: held-out CSV has a blank required value")
        if row["id"] in seen:
            raise SystemExit("ABORT: held-out CSV has duplicate row id")
        seen.add(row["id"])
    return rows

def selected_rows(rows, split):
    if split == "all": return rows
    expected = 0 if split == "diagnosis" else 1
    return [row for row in rows if side([x.strip() for x in row["gold_ids"].split(";") if x.strip()]) == expected]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "eval" / "heldout_retrieval.csv"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--split", choices=["all", "diagnosis", "locked"], default="all")
    ap.add_argument("--embed", action="store_true", help="explicitly use embeddings; may make API calls")
    ap.add_argument("--estimate-only", action="store_true", help="show minimum request estimate; makes NO API calls")
    ap.add_argument("--name", default="")
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results_v9_1_2"))
    ap.add_argument("--overwrite", action="store_true", help="explicitly overwrite this one output file")
    a = ap.parse_args()
    fix_console()
    if os.environ.get("VOICEOPS_NO_ENV_LOAD") != "1": load_env()
    rows = selected_rows(read_rows(a.csv), a.split)
    if not rows:
        raise SystemExit("ABORT: selected split has no rows")
    r = Retriever(use_embed=False)
    policy = [c for c in r.chunks if c["doc_type"] != "catalog"]
    if a.estimate_only:
        # For preflight, estimate over selected rows only; no Retriever embedding index is constructed.
        estimate = EmbedIndex.estimate_from_env([c["id"] for c in policy],
            [c["title"] + ". " + c["text"] for c in policy], [row["question"] for row in rows])
        print(json.dumps({"split": a.split, "rows": len(rows), "embedding_api_calls_made": 0, **estimate}, ensure_ascii=False))
        if not estimate.get("available"):
            raise SystemExit("ABORT: embedding estimate unavailable; no API calls made")
        return
    if a.embed:
        emb = EmbedIndex.from_env([c["id"] for c in policy], [c["title"] + ". " + c["text"] for c in policy],
                                  query_texts=[row["question"] for row in rows])
        if emb is None:
            raise SystemExit("ABORT: embeddings unavailable or request cap reached: " + EmbedIndex.last_error)
        r.embed, r.embed_errors = emb, 0
    items, by_lang = [], {}
    for row in rows:
        gold = [g.strip() for g in row["gold_ids"].split(";") if g.strip()]
        known = set(r.by_id)
        if not gold or any(g not in known for g in gold):
            raise SystemExit("ABORT: CSV contains an empty/unknown gold section ID; no metrics saved")
        got = [x["id"] for x in r.search(row["question"], k=a.k, catalog_k=0)]
        score = recall(got, gold)
        items.append({"id": row["id"], "language": row["language"], "gold": gold, "got": got, "recall": score})
        by_lang.setdefault(row["language"] or "?", []).append(score)
    n = len(items)
    summary = {"n": n, "k": a.k, "split": a.split, "embed": bool(a.embed),
               "mean_recall": round(sum(i["recall"] for i in items) / n, 4),
               "by_language": {l: round(sum(v) / len(v), 4) for l, v in sorted(by_lang.items())}}
    print(json.dumps(summary, ensure_ascii=False))
    for i in sorted((i for i in items if i["recall"] < 1), key=lambda x: x["recall"])[:15]:
        print(f"MISS id={i['id']} language={i['language']} recall={i['recall']:.2f} gold={i['gold']} got={i['got']}")
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = a.name or ("hybrid" if a.embed else "bm25")
    out = out_dir / f"heldout_retrieval_{name}_{a.split}.json"
    if out.exists() and not a.overwrite:
        raise SystemExit("ABORT: output already exists; choose a new --name or use --overwrite explicitly")
    with out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": items}, f, ensure_ascii=False, indent=2)
    print("RESULT_FILE=" + str(out))
    if a.embed:
        budget = RequestBudget()
        print(f"EMBED_REQUESTS_USED_TOTAL={budget.used()} LIMIT={budget.limit} REMAINING={budget.remaining()}")

if __name__ == "__main__":
    main()
