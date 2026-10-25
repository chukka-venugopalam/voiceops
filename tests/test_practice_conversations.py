"""Regression guard: replays all practice conversations offline (rules only, no LLM)."""
import csv, os
from conftest import needs_data
from voiceops.config import dataset_dir
from voiceops.evalkit import calls_match
from voiceops.scorer import decision_ok
pytestmark = needs_data

def test_decisions_and_tool_calls_match_the_key(agent):
    rows = list(csv.DictReader(open(os.path.join(dataset_dir(), "practice", "conversations.csv"), encoding="utf-8")))
    convs = {}
    for r in rows: convs.setdefault(r["conversation_id"], []).append(r)
    bad = []
    for cid, turns in convs.items():
        s = agent.new_session()
        for r in sorted(turns, key=lambda x: int(x["turn"])):
            res = agent.respond(s, r["customer_message"])
            if not decision_ok(res["decision"], r["expected_decision"]) or not calls_match(r["expected_tool_calls"], res["tool_calls"])["ok"]:
                bad.append((cid, r["turn"], res["decision"], r["expected_decision"]))
    assert not bad, bad

def test_overdue_refund_reply_contains_rrn(agent):
    rows = [r for r in csv.DictReader(open(os.path.join(dataset_dir(), "practice", "conversations.csv"), encoding="utf-8")) if r["conversation_id"] == "PC010"]
    s = agent.new_session(); last = None
    for r in sorted(rows, key=lambda x: int(x["turn"])): last = agent.respond(s, r["customer_message"])
    assert last["decision"] == "escalate" and "315633825001" in last["reply"]
