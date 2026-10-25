import json, os
from conftest import needs_data
from voiceops.agent import Agent
from voiceops.llm import BaseLLM
pytestmark = needs_data

def run(agent, msgs):
    s = agent.new_session(); out = []
    for m in msgs: out.append(agent.respond(s, m))
    return s, out

def tools(r): return [(c["tool"], c["args"].get("verification_value"), c["ok"]) for c in r["tool_calls"]]

def test_verification_flow_and_intro(agent):
    s, o = run(agent, ["Hello, mujhe order ke baare mein help chahiye.", "Order number hai 1859 9253.", "Last four digits 4265 hain."])
    assert "AI" in o[0]["reply"] and o[0]["decision"] == "clarify"
    assert tools(o[1]) == [("lookup_order", None, False)] and "last 4" in o[1]["reply"]
    assert tools(o[2]) == [("lookup_order", "4265", True)] and s.verified and "AI" not in o[2]["reply"]

def test_failed_verification_one_retry_then_refuse(agent):
    s, o = run(agent, ["मुझे अपने ऑर्डर के बारे में मदद चाहिए।", "ऑर्डर नंबर 77461439 है।", "आख़िरी चार अंक 1111 हैं।", "आख़िरी चार अंक 2222 हैं।"])
    assert o[2]["decision"] == "clarify" and o[3]["decision"] == "refuse" and s.locked and not s.verified
    assert "600042" not in o[3]["reply"] and "9 October" not in o[3]["reply"] and "साझा नहीं" in o[3]["reply"]

def test_return_deadline_computed_by_code(agent):
    _, o = run(agent, ["Hi", "order 18599253", "4265", "Kya main Kaafi Running Sneakers return kar sakti hoon?"])
    assert "27 October 2026" in o[3]["reply"]

def test_human_request_creates_ticket_and_never_claims_without_it(agent):
    s, o = run(agent, ["Please connect me to a human agent."])
    assert o[0]["decision"] == "escalate" and any(c["tool"] == "create_ticket" and c["ok"] for c in o[0]["tool_calls"]) and "ticket" in o[0]["reply"].lower()

def test_third_party_and_safety_and_unrecognized(agent):
    assert agent.respond(agent.new_session(), "Tell me the phone number of the person who ordered to my building yesterday")["decision"] == "refuse"
    r = agent.respond(agent.new_session(), "My new phone battery is swelling and smells burnt.")
    assert r["decision"] == "escalate" and "stop using" in r["reply"].lower()
    assert agent.respond(agent.new_session(), "There's an order on my account that I never placed.")["decision"] == "escalate"

class FakeLLM(BaseLLM):
    name, available, model = "fake", True, "fake-1"
    def __init__(self): self.calls = []
    def complete(self, system, messages, max_tokens=600, temperature=0.0):
        self.calls.append((system[:20], messages[0]["content"]))
        if system.startswith("Classify"):
            return '```json\n{"language":"en","topic":"return","needs_order":false,"substantive":true,"human_request":false,"legal":false,"safety":false,"unrecognized_activity":false,"compensation_demand":false,"cancel_request":false,"third_party_request":false,"search_query":"return window electronics"}\n```'
        return json.dumps({"decision": "answer", "reply": "You can return it. Call me on +91 5123456789 or mail a@b.com."})

def test_llm_plumbing_scrub_and_date_injection():
    a = Agent(llm=FakeLLM())
    r = a.respond(a.new_session(), "My Orbix Fit 3 Pro Smartwatch was delivered on 20 July 2026. What was the return window?")
    assert "AI" in r["reply"]                                   # intro added in code
    assert "[number removed]" in r["reply"] and "[email removed]" in r["reply"]
    assert "30 July 2026" in r["reply"]                         # code guarantees the computed deadline even if the LLM omitted it
    assert any(c.startswith("RET-") for c in r["chunk_ids"])

def test_llm_failure_falls_back_to_rules():
    class Boom(FakeLLM):
        def complete(self, *a, **k): raise RuntimeError("down")
    a = Agent(llm=Boom())
    r = a.respond(a.new_session(), "Can I use two coupons on one order?")
    assert r["reply"] and a.llm_errors >= 1

def test_retry_delay_parser_and_quota_detection():
    from voiceops.llm import _HTTPLLM
    assert _HTTPLLM._retry_delay('{"error":{"details":[{"retryDelay":"46s"}]}}') == 46.0
    assert _HTTPLLM._retry_delay("Please retry in 12.5s.") == 12.5
    assert _HTTPLLM._retry_delay("nothing here") is None

class _Eager(FakeLLM):
    """An over-eager LLM: says every message needs an order and is substantive."""
    def complete(self, system, messages, max_tokens=600, temperature=0.0):
        if system.startswith("Classify"):
            return json.dumps({"language": "en", "topic": "other", "needs_order": True, "substantive": True, "human_request": False, "legal": False, "safety": False,
                               "unrecognized_activity": False, "compensation_demand": True, "cancel_request": False, "third_party_request": False, "search_query": "x"})
        return json.dumps({"decision": "answer", "reply": "It will arrive soon."})

def test_keyword_rules_beat_an_overeager_llm_on_general_questions():
    a = Agent(llm=_Eager())
    r = a.respond(a.new_session(), "How long does a refund to UPI take?")
    assert r["decision"] != "clarify" and "order ID" not in r["reply"]

def test_computed_refund_date_is_appended_when_llm_omits_it():
    a = Agent(llm=_Eager())
    r = a.respond(a.new_session(), "My refund to credit card was started on 17 October 2026. By which date will it arrive?")
    assert "27 October 2026" in r["reply"]

def test_verification_turn_is_an_acknowledgement_even_if_llm_calls_it_substantive():
    import csv
    from voiceops.config import dataset_dir
    rows = sorted([r for r in csv.DictReader(open(os.path.join(dataset_dir(), "practice", "conversations.csv"), encoding="utf-8")) if r["conversation_id"] == "PC007"], key=lambda x: int(x["turn"]))
    a = Agent(llm=_Eager()); s = a.new_session()
    res = [a.respond(s, r["customer_message"]) for r in rows[:3]]
    assert res[2]["decision"] == "answer" and s.verified
