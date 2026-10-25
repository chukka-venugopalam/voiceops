#!/usr/bin/env python3
"""Interactive text chat with the agent (no audio). Use it to try things before spending Gnani credits.
   python chat.py            # uses your LLM from .env (offline rules if no key)
   python chat.py --offline  # force rule-based mode"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from voiceops.config import load_env
load_env()
from voiceops.agent import Agent

ap = argparse.ArgumentParser()
ap.add_argument("--offline", action="store_true"); ap.add_argument("--dataset"); ap.add_argument("--verbose", "-v", action="store_true")
a = ap.parse_args()
if a.dataset: os.environ["DATASET_DIR"] = a.dataset
agent = Agent(use_llm=not a.offline)
sess = agent.new_session()
print(f"ZentroMart support agent ({'LLM: ' + agent.llm.name + ' ' + agent.llm.model if agent.llm_on else 'offline rules'}). Type 'quit' to exit, 'new' for a new call.\n")
while True:
    try: msg = input("You: ").strip()
    except (EOFError, KeyboardInterrupt): break
    if msg.lower() in ("quit", "exit"): break
    if msg.lower() == "new": sess = agent.new_session(); print("(new conversation)"); continue
    r = agent.respond(sess, msg)
    print(f"Agent [{r['decision']}]: {r['reply']}")
    if a.verbose: print("   tools:", [(c['tool'], c['args'], c['ok']) for c in r['tool_calls']], "| chunks:", r['chunk_ids'])
