"""Targeted regressions for the v5 offline-quality and intent fixes.

These tests use only the synthetic dataset and rules-only agent; they make no network calls.
"""
import csv
import os
import re

from conftest import needs_data
from voiceops.config import dataset_dir
from voiceops.textnorm import extract_ids

pytestmark = needs_data


def test_sku_suffix_is_not_misread_as_verification_code():
    q = "What warranty comes with the Voltra PowerCell 10000 mAh Power Bank, Green (SKU ELE-0008)?"
    assert extract_ids(q) == {"order_id": None, "verifier": None}
    assert extract_ids("4265")["verifier"] is None
    assert extract_ids("4265", expect_verifier=True)["verifier"] == "4265"
    assert extract_ids("Last four digits are 4265.")["verifier"] == "4265"


def test_general_assured_warranty_does_not_require_order_lookup(agent):
    q = "What warranty comes with the Voltra PowerCell 10000 mAh Power Bank, Green (SKU ELE-0008)?"
    r = agent.respond(agent.new_session(), q)
    assert r["decision"] == "answer"
    assert not any(c["tool"] == "lookup_order" for c in r["tool_calls"])
    assert "12" in r["reply"] and "My Orders" in r["reply"]


def test_cracked_high_value_laptop_creates_handover_ticket(agent):
    q = "The Kestrel Book 14 Laptop (SKU ELE-0023) I received yesterday has a crack. I want it sorted."
    r = agent.respond(agent.new_session(), q)
    assert r["decision"] == "escalate"
    assert any(c["tool"] == "create_ticket" and c["ok"] for c in r["tool_calls"])
    assert any(term in r["reply"].lower() for term in ("ticket", "colleague", "priority resolution"))


def test_policy_refund_timeline_is_answered_without_order_lookup(agent):
    r = agent.respond(agent.new_session(), "How long does a refund to UPI take?")
    assert r["decision"] == "answer"
    assert "1" in r["reply"] and "3" in r["reply"]
    assert not any(c["tool"] == "lookup_order" for c in r["tool_calls"])


def test_product_category_return_window_is_grounded_in_catalog(agent):
    r = agent.respond(agent.new_session(), "How many days do I get to return the Pagecraft A5 Ruled Notebook (Pack of 4)?")
    assert r["decision"] == "answer"
    assert "10" in r["reply"]


def test_nonreturnable_innerwear_question_does_not_ask_for_order_id(agent):
    r = agent.respond(agent.new_session(), "The Comfi Cotton Briefs (Pack of 3) doesn't fit. Can I send it back?")
    assert r["decision"] == "answer"
    assert "not returnable" in r["reply"].lower() or "cannot" in r["reply"].lower() or "nahi" in r["reply"].lower()
    assert not any(c["tool"] == "lookup_order" for c in r["tool_calls"])


def test_general_shipped_cancellation_policy_answers_without_order_lookup(agent):
    r = agent.respond(agent.new_session(), "My order is already on its way. Can I still stop it?")
    assert r["decision"] == "answer"
    assert "refuse" in r["reply"].lower()
    assert not any(c["tool"] == "lookup_order" for c in r["tool_calls"])


def test_vague_return_duration_asks_for_product_or_category(agent):
    r = agent.respond(agent.new_session(), "कितने दिन में वापस करना होगा?")
    assert r["decision"] == "clarify"
    assert any(s in r["reply"].lower() for s in ("category", "श्रेणी", "product", "उत्पाद"))


def test_unsupported_student_discount_is_not_invented(agent):
    r = agent.respond(agent.new_session(), "Is there a student discount?")
    assert r["decision"] == "answer"
    assert any(s in r["reply"].lower() for s in ("don't have information", "no information", "connect you", "colleague"))


def test_asr_product_spelling_and_punctuation_still_match_catalog():
    from voiceops.facts import find_products, load_products
    products = load_products(dataset_dir())

    rangoli = find_products(
        "so i bought the rangoli cotton straight kurtha in the sale on 5 october how long can i return it",
        products, allow_prefix_fuzzy=True,
    )
    assert rangoli and rangoli[0]["return_category"] == "fashion"

    comfi = find_products(
        "the comfi cotton briefs pack of 3 doesnt fit can i send it back",
        products, allow_prefix_fuzzy=True,
    )
    assert comfi and comfi[0]["return_category"] == "innerwear"

    kestrel = find_products(
        "mera order kestrel k7 smartphone to five six gb parso deliver hua kitne din tak return kar sakti hoon",
        products, allow_prefix_fuzzy=True,
    )
    assert kestrel and kestrel[0]["return_category"] == "electronics"


def test_split_asr_refund_phrase_is_classified_as_refund():
    from voiceops.intent import detect_topic
    assert detect_topic("if i paid with net banking how soon will the re fund show up") == "refund"


def test_shipped_prepaid_cancellation_explains_refund_after_return_to_warehouse(agent):
    """The official PC019 scenario expects REF-1's 7-10 day RTO refund-initiation timeline."""
    rows = list(csv.DictReader(open(os.path.join(dataset_dir(), "practice", "conversations.csv"), encoding="utf-8-sig")))
    turns = [r for r in rows if r["conversation_id"] == "PC019"]
    assert len(turns) == 4
    session = agent.new_session()
    result = None
    for row in sorted(turns, key=lambda x: int(x["turn"])):
        result = agent.respond(session, row["customer_message"])
    reply = result["reply"].lower()
    assert result["decision"] == "refuse"
    assert "7" in reply and ("warehouse" in reply or "वेयरहाउस" in reply)
    assert "10 days for home/kitchen" in reply
    assert "1 to 3 business days" not in reply  # do not confuse RTO-to-warehouse timing with UPI bank-credit timing
