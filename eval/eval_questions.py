#!/usr/bin/env python3
"""Score the agent on single questions exactly like the answer key.
   python eval/eval_questions.py --mode prompt_only          # Step 1: LLM + role prompt only (needs LLM key)
   python eval/eval_questions.py --mode rag                  # Step 2: full agent (LLM + RAG + calculators + tools)
   python eval/eval_questions.py --mode offline              # no LLM: rules + extractive answers (no key needed)
   options: --field question|asr_text   --file path.csv (default practice/questions.csv, or eval/hidden_cases.csv)   --limit N"""
import argparse, csv, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from voiceops.config import load_env, dataset_dir
if os.environ.get("VOICEOPS_NO_ENV_LOAD") != "1":
    load_env()
from voiceops.agent import Agent
from voiceops.scorer import score_reply, decision_ok, recall
from voiceops.evalkit import pct, save, mean, sentences

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["prompt_only", "rag", "offline"], default="offline")
ap.add_argument("--field", default="question"); ap.add_argument("--file"); ap.add_argument("--limit", type=int); ap.add_argument("--name"); ap.add_argument("--allow-errors", action="store_true")
a = ap.parse_args()
os.environ.setdefault("TICKETS_PATH", os.path.join(tempfile.gettempdir(), "voiceops_eval_tickets.csv"))
path = a.file or os.path.join(dataset_dir(), "practice", "questions.csv")
rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))[: a.limit]
agent = Agent(use_llm=(a.mode != "offline"))
if a.mode != "offline" and not agent.llm_on: sys.exit("No LLM configured (set LLM_API_KEY / LLM_PROVIDER in .env) - or use --mode offline")
name = a.name or f"questions_{a.mode}_{a.field}" + ("_hidden" if "hidden" in path else "")
items = []
for r in rows:
    if agent.llm_errors >= 3 or sum(i['decision'] == 'error' for i in items) >= 3: print('ABORTING: 3 LLM errors in a row - see message below'); break
    text = r.get(a.field) or r["question"]
    try:
        if a.mode == "prompt_only":
            reply, decision, chunks, tickets = agent.respond_prompt_only(text), "answer", [], 0
        else:
            res = agent.respond(agent.new_session(), text)
            reply, decision, chunks = res["reply"], res["decision"], res["chunk_ids"]
            tickets = sum(1 for c in res["tool_calls"] if c["tool"] == "create_ticket" and c["ok"])
    except Exception as e:
        reply, decision, chunks, tickets = f"[ERROR {e}]", "error", [], 0
    sc = score_reply(reply, r["must_include"], r["must_not_include"])
    gold = [g.strip() for g in r["gold_evidence"].split(";") if g.strip()]
    items.append({"id": r.get("id"), "language": r["language"], "text": text, "reply": reply, "decision": decision, "expected": r["expected_decision"],
                  "content_ok": sc["passed"], "missing": sc["missing"], "forbidden": sc["forbidden"],
                  "decision_ok": decision_ok(decision, r["expected_decision"], r.get("acceptable_decisions", "")),
                  "recall": recall(chunks, gold) if gold else None, "chunks": chunks, "ticket_made": tickets > 0, "sentences": sentences(reply)})
n = len(items); rec = [i["recall"] for i in items if i["recall"] is not None]
bylang = {}
for i in items: bylang.setdefault(i["language"], []).append(i["content_ok"])
summary = {"mode": a.mode, "field": a.field, "file": os.path.basename(path), "n": n, "content_pass_pct": pct(sum(i["content_ok"] for i in items), n),
           "decision_acc_pct": pct(sum(i["decision_ok"] for i in items), n), "mean_retrieval_recall": mean(rec), "avg_sentences": mean([i["sentences"] for i in items]),
           "content_pass_by_language": {k: pct(sum(v), len(v)) for k, v in bylang.items()},
           "escalations_without_ticket": sum(1 for i in items if i["decision"] == "escalate" and not i["ticket_made"])}
print(f"\n=== {name} ===")
for k, v in summary.items(): print(f"  {k}: {v}")
print("  failures (content):")
for i in items:
    if not i["content_ok"]: print(f"   {i['id']} [{i['language']}] missing={i['missing']} forbidden={i['forbidden']} :: {i['text'][:60]!r}")
print(f"  llm_calls_made: {getattr(agent.llm, 'calls', 0)} (answers reused from cache: {getattr(agent.llm, 'cache_hits', 0)})")
print("saved:", save(name, summary, items))
errs = agent.llm_errors + sum(i["decision"] == "error" for i in items)
if errs and not a.allow_errors:
    last = agent.last_llm_error or next((i["reply"] for i in items if i["decision"] == "error"), "")
    print(f"\n!! {errs} LLM error(s). Last: {last}\n!! These results are NOT valid (answers fell back to offline rules). Fix the key / model / quota and re-run; answers already received are cached and cost nothing.")
    sys.exit(2)
