#!/usr/bin/env python3
"""Run the practice conversations turn by turn and check decision, tool calls, required facts and evidence.
   python eval/eval_conversations.py --mode offline|rag     [--field customer_message]  [--limit N conversations]"""
import argparse, csv, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from voiceops.config import load_env, dataset_dir
if os.environ.get("VOICEOPS_NO_ENV_LOAD") != "1":
    load_env()
from voiceops.agent import Agent
from voiceops.scorer import score_reply, decision_ok, recall
from voiceops.evalkit import calls_match, pct, save, mean

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["rag", "offline"], default="offline"); ap.add_argument("--file"); ap.add_argument("--limit", type=int); ap.add_argument("--name"); ap.add_argument("--allow-errors", action="store_true")
a = ap.parse_args()
os.environ.setdefault("TICKETS_PATH", os.path.join(tempfile.gettempdir(), "voiceops_eval_tickets.csv"))
path = a.file or os.path.join(dataset_dir(), "practice", "conversations.csv")
rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
convs = {}
for r in rows: convs.setdefault(r["conversation_id"], []).append(r)
agent = Agent(use_llm=(a.mode == "rag"))
if a.mode == "rag" and not agent.llm_on: sys.exit("No LLM configured - or use --mode offline")
items = []
for cid in list(convs)[: a.limit]:
    if agent.llm_errors >= 3 or sum(i['decision'] == 'error' for i in items) >= 3: print('ABORTING: 3 LLM errors in a row - see message below'); break
    sess = agent.new_session()
    for r in sorted(convs[cid], key=lambda x: int(x["turn"])):
        try: res = agent.respond(sess, r["customer_message"])
        except Exception as e: res = {"reply": f"[ERROR {e}]", "decision": "error", "tool_calls": [], "chunk_ids": []}
        sc = score_reply(res["reply"], r["must_include"], r["must_not_include"])
        tc = calls_match(r["expected_tool_calls"], res["tool_calls"])
        gold = [g.strip() for g in r["gold_evidence"].split(";") if g.strip()]
        d_ok = decision_ok(res["decision"], r["expected_decision"])
        items.append({"conv": cid, "turn": int(r["turn"]), "scenario": r["scenario"], "language": r["language"], "customer": r["customer_message"],
                      "reply": res["reply"], "decision": res["decision"], "expected": r["expected_decision"], "decision_ok": d_ok,
                      "content_ok": sc["passed"], "missing": sc["missing"], "forbidden": sc["forbidden"], "tools_ok": tc["ok"], "tools_missing": tc["missing"],
                      "tools_unexpected": tc["unexpected"], "recall": recall(res["chunk_ids"], gold) if gold and r["expected_decision"] == "answer" else None,
                      "all_ok": d_ok and sc["passed"] and tc["ok"]})
n = len(items); by_sc = {}
for i in items: by_sc.setdefault(i["scenario"], []).append(i["all_ok"])
summary = {"mode": a.mode, "turns": n, "all_checks_pass_pct": pct(sum(i["all_ok"] for i in items), n), "decision_acc_pct": pct(sum(i["decision_ok"] for i in items), n),
           "content_pass_pct": pct(sum(i["content_ok"] for i in items), n), "tool_calls_ok_pct": pct(sum(i["tools_ok"] for i in items), n),
           "mean_retrieval_recall_on_answer_turns": mean([i["recall"] for i in items if i["recall"] is not None]),
           "conversations_fully_correct": f"{sum(all(i['all_ok'] for i in items if i['conv'] == c) for c in convs)}/{len(set(i['conv'] for i in items))}",
           "pass_pct_by_scenario": {k: pct(sum(v), len(v)) for k, v in by_sc.items()}}
name = a.name or f"conversations_{a.mode}"
print(f"\n=== {name} ===")
for k, v in summary.items(): print(f"  {k}: {v}")
print("  failing turns:")
for i in items:
    if not i["all_ok"]:
        why = [x for x in ([f"decision {i['decision']}!={i['expected']}"] if not i["decision_ok"] else []) + ([f"missing {i['missing']}"] if i["missing"] else [])
               + ([f"forbidden {i['forbidden']}"] if i["forbidden"] else []) + ([f"tools {i['tools_missing']}"] if not i["tools_ok"] else [])]
        print(f"   {i['conv']}-T{i['turn']} [{i['scenario']}] {'; '.join(why)}")
print(f"  llm_calls_made: {getattr(agent.llm, 'calls', 0)} (answers reused from cache: {getattr(agent.llm, 'cache_hits', 0)})")
print("saved:", save(name, summary, items))
errs = agent.llm_errors + sum(i["decision"] == "error" for i in items)
if errs and not a.allow_errors:
    print(f"\n!! {errs} LLM error(s). Last: {agent.last_llm_error}\n!! These results are NOT valid (turns fell back to offline rules). Fix the key / model / quota and re-run; answers already received are cached and cost nothing.")
    sys.exit(2)
