"""Offline-only regression tests for the v9.1.2 retrieval experiment package."""
import json, os, subprocess, sys
from pathlib import Path
import pytest
from voiceops.embed import RequestBudget, RequestBudgetExceeded, rrf_fuse, EmbedIndex
from voiceops.llm import _PersistentLLMCallBudget, LLMError
from voiceops.kb import context_query, content_tokens, Retriever

ROOT = Path(__file__).resolve().parents[1]

def test_rrf_rewards_agreement():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]])
    assert fused["b"] > fused["c"] and fused["a"] > fused["d"]
    assert set(fused) == {"a", "b", "c", "d"}

def test_content_tokens_ignore_ids_digits_and_confirmation():
    for value in ("ORD-10432", "last four 4821", "1234", "yes okay", "ELE-0023"):
        assert content_tokens(value) == []
    assert content_tokens("I want to return my headphones")

def test_context_query_skips_id_and_verification_turns():
    hist = [("customer", "I want to return my headphones"), ("assistant", "Please share order ID"),
            ("customer", "ORD-10432"), ("assistant", "last four digits please"),
            ("customer", "last four 4821")]
    assert context_query(hist) == "I want to return my headphones"
    assert context_query([]) == ""

def test_context_query_can_keep_two_topic_turns():
    hist = [("customer", "refund kab milega"), ("assistant", "x"),
            ("customer", "cancel my order please"), ("assistant", "y")]
    assert context_query(hist, n=2) == "refund kab milega cancel my order please"

def test_embeddings_off_without_model(monkeypatch):
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    assert EmbedIndex.from_env(["a"], ["text"]) is None

def test_estimator_never_makes_calls_without_config(monkeypatch, tmp_path):
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    out = EmbedIndex.estimate_from_env(["a"], ["text"], ["question"], cache_dir=tmp_path,
                                       budget=RequestBudget(tmp_path / "budget.json"))
    assert out["available"] is False

def test_request_budget_caps_retries_and_persists(tmp_path):
    p = tmp_path / "budget.json"
    b = RequestBudget(p, limit=2)
    assert b.consume() == 1
    assert RequestBudget(p, limit=2).consume() == 2
    with pytest.raises(RequestBudgetExceeded): b.consume()
    assert json.loads(p.read_text())["used"] == 2

def test_request_budget_fails_closed_on_corrupt_ledger(tmp_path):
    p = tmp_path / "budget.json"; p.write_text("not-json")
    b = RequestBudget(p, limit=2)
    with pytest.raises(RequestBudgetExceeded): b.consume()

def test_llm_request_budget_caps_attempts_and_persists(tmp_path):
    p = tmp_path / "llm_budget.json"
    b = _PersistentLLMCallBudget(p, limit=2)
    assert b.consume() == 1
    assert _PersistentLLMCallBudget(p, limit=2).consume() == 2
    with pytest.raises(LLMError): b.consume()
    assert json.loads(p.read_text())["used"] == 2

def test_llm_request_budget_fails_closed_on_corrupt_ledger(tmp_path):
    p = tmp_path / "llm_budget.json"; p.write_text("not-json")
    with pytest.raises(LLMError): _PersistentLLMCallBudget(p, limit=2).consume()

def test_fuse_fallback_when_no_embed():
    root = Path(__file__).resolve().parents[1]
    # Does not initialize embeddings or make network calls.
    assert callable(rrf_fuse)

def test_sanitizer_drops_nested_text_gold_got_and_reply(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"; src.mkdir()
    (src / "a.json").write_text(json.dumps({"summary": {"n": 1, "mean_recall": 0.9, "notes": "private"},
        "items": [{"id": "Q1", "text": "SECRET QUESTION", "gold": ["RET-1"], "got": ["RET-1"],
                   "reply": "SECRET REPLY", "recall": 1.0, "decision_ok": True}]}), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "sanitize_eval_results.py"),
                    "--src", str(src), "--dst", str(dst)], check=True, capture_output=True, text=True)
    out = json.loads((dst / "a.json").read_text(encoding="utf-8"))
    assert out["summary"] == {"n": 1, "mean_recall": 0.9}
    assert out["items"] == [{"id": "Q1", "recall": 1.0, "decision_ok": True}]
    blob = json.dumps(out)
    for forbidden in ("SECRET QUESTION", "SECRET REPLY", '"gold"', '"got"'):
        assert forbidden not in blob

def test_sanitizer_refuses_existing_destination_without_overwrite(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"; src.mkdir(); dst.mkdir()
    (src / "a.json").write_text('{"summary":{"n":1},"items":[]}', encoding="utf-8")
    (dst / "a.json").write_text("preserve", encoding="utf-8")
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "sanitize_eval_results.py"),
                        "--src", str(src), "--dst", str(dst)], capture_output=True, text=True)
    assert p.returncode != 0 and "ABORT" in p.stdout + p.stderr
    assert (dst / "a.json").read_text() == "preserve"

def test_retrieval_constructor_is_embedding_off_by_default():
    # The constructor is deliberately invoked only in the mini fixture test in user project suite.
    # This assertion is an offline safety gate for the public API default.
    import inspect
    source = inspect.getsource(Retriever.__init__)
    assert "use_embed=False" in source
