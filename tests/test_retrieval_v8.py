"""Evidence-ranking regressions for v8. Uses the supplied gold_evidence labels; no network/LLM."""
import csv
import os

import pytest
from conftest import needs_data
from voiceops.config import dataset_dir
from voiceops.kb import Retriever
from voiceops.scorer import recall

pytestmark = needs_data


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _evidence_rows(rows, query_field):
    return [
        (row.get("id", ""), row.get(query_field) or row.get("question", ""),
         [part.strip() for part in row.get("gold_evidence", "").split(";") if part.strip()])
        for row in rows if row.get("gold_evidence", "").strip()
    ]


def _mean_top3(retriever, rows):
    return sum(recall([c["id"] for c in retriever.search(q)[:3]], gold) for _, q, gold in rows) / max(1, len(rows))


def test_official_question_evidence_recall_at_seven_is_complete():
    root = dataset_dir()
    retriever = Retriever(root)
    rows = _evidence_rows(_read(os.path.join(root, "practice", "questions.csv")), "question")
    misses = [(qid, gold, [c["id"] for c in retriever.search(q)[:7]])
              for qid, q, gold in rows
              if recall([c["id"] for c in retriever.search(q)[:7]], gold) < 1.0]
    assert not misses, f"Evidence missing from top 7: {misses}"
    assert _mean_top3(retriever, rows) >= 0.95


def test_asr_question_evidence_recall_at_seven_is_complete():
    root = dataset_dir()
    retriever = Retriever(root)
    rows = _evidence_rows(_read(os.path.join(root, "practice", "questions.csv")), "asr_text")
    misses = [(qid, gold, [c["id"] for c in retriever.search(q)[:7]])
              for qid, q, gold in rows
              if recall([c["id"] for c in retriever.search(q)[:7]], gold) < 1.0]
    assert not misses, f"ASR evidence missing from top 7: {misses}"
    assert _mean_top3(retriever, rows) >= 0.95


def test_hidden_case_evidence_recall_at_seven_is_complete():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    retriever = Retriever(dataset_dir())
    rows = _evidence_rows(_read(os.path.join(root, "eval", "hidden_cases.csv")), "question")
    misses = [(qid, gold, [c["id"] for c in retriever.search(q)[:7]])
              for qid, q, gold in rows
              if recall([c["id"] for c in retriever.search(q)[:7]], gold) < 1.0]
    assert not misses, f"Hidden-case evidence missing from top 7: {misses}"
    assert _mean_top3(retriever, rows) >= 0.95
