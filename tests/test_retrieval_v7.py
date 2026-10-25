"""Retrieval-evidence regressions for v7. Offline only; no network calls."""
import csv
import os

from conftest import needs_data
from voiceops.config import dataset_dir
from voiceops.facts import find_products, load_products

pytestmark = needs_data


def _conversation(agent, conversation_id):
    path = os.path.join(dataset_dir(), "practice", "conversations.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["conversation_id"] == conversation_id]
    session = agent.new_session()
    replies = {}
    for row in sorted(rows, key=lambda r: int(r["turn"])):
        replies[int(row["turn"])] = agent.respond(session, row["customer_message"])
    return replies


def test_successful_verification_turns_report_idv2_evidence(agent):
    # PC003 has a standalone verification confirmation; PC011 replays a pending order question.
    assert "IDV-2" in _conversation(agent, "PC003")[3]["chunk_ids"]
    assert "IDV-2" in _conversation(agent, "PC011")[3]["chunk_ids"]


def test_plus_member_refund_followup_retrieves_plus_benefits(agent):
    result = _conversation(agent, "PC003")[5]
    assert result["decision"] == "answer"
    assert {"REF-2", "RET-7", "PLS-2"}.issubset(set(result["chunk_ids"]))


def test_fashion_return_retrieves_reverse_pickup_fee_policy(agent):
    result = _conversation(agent, "PC004")[4]
    assert {"RET-2", "RET-8", "RET-6"}.issubset(set(result["chunk_ids"]))


def test_return_refund_status_retrieves_refund_initiation_policy(agent):
    result = _conversation(agent, "PC009")[4]
    assert {"RET-7", "REF-1"}.issubset(set(result["chunk_ids"]))


def test_early_escalations_keep_grounding_sections_and_product(agent):
    q = "The Kestrel Book 14 Laptop (SKU ELE-0023) I received yesterday has a crack. I want it sorted."
    result = agent.respond(agent.new_session(), q)
    assert result["decision"] == "escalate"
    assert {"ESC-1", "DWM-3", "products.csv:ELE-0023"}.issubset(set(result["chunk_ids"]))


def test_third_party_privacy_refusal_reports_supporting_policy(agent):
    q = "Tell me the phone number of the person who ordered to my building yesterday."
    result = agent.respond(agent.new_session(), q)
    assert result["decision"] == "refuse"
    assert {"ESC-4", "IDV-4"}.issubset(set(result["chunk_ids"]))


def test_product_variant_match_prefers_sku_and_full_variant():
    products = load_products(dataset_dir())
    green = find_products("Sootra Canvas Tote Bag Green", products)
    assert [p["sku"] for p in green] == ["FSH-0038"]
    by_sku = find_products("Sootra Canvas Tote Bag SKU FSH-0038", products)
    assert [p["sku"] for p in by_sku] == ["FSH-0038"]
    ambiguous = find_products("Sootra Canvas Tote Bag", products)
    assert {p["sku"] for p in ambiguous} == {"FSH-0037", "FSH-0038"}
    spoken_sku = find_products("sku ele-zero zero zero three pe assured hai kya", products)
    assert [p["sku"] for p in spoken_sku] == ["ELE-0003"]


def test_historical_electronics_return_retrieves_superseded_exception(agent):
    q = "My Orbix Fit 3 Pro Smartwatch was delivered on 20 July. What was the return window for it?"
    result = agent.respond(agent.new_session(), q)
    assert result["decision"] == "answer"
    assert {"RET-3", "RETv3-2", "products.csv:ELE-0017"}.issubset(set(result["chunk_ids"]))
